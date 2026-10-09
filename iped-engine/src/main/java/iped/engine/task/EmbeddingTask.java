package iped.engine.task;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;

import org.apache.http.HttpStatus;
import org.apache.tika.mime.MediaType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import iped.configuration.Configurable;
import iped.data.IItem;
import iped.engine.config.ConfigurationManager;
import iped.engine.config.EmbeddingTaskConfig;
import iped.engine.data.Item;
import iped.engine.embedding.EmbeddingServiceClient;
import iped.engine.embedding.EmbeddingServiceClient.EmbeddingServiceException;
import iped.engine.embedding.EmbeddingServiceClient.Input;
import iped.engine.embedding.EmbeddingServiceClient.Result;
import iped.engine.embedding.EmbeddingServiceClient.ServiceInfo;
import iped.engine.embedding.EmbeddingUtil;
import iped.engine.preview.PreviewRepositoryManager;
import iped.exception.IPEDException;
import iped.parsers.standard.StandardParser;
import iped.parsers.util.MetadataUtil;
import iped.properties.ExtraProperties;
import iped.utils.ImageUtil;

/**
 * Computes one multimodal embedding (EmbeddingGemma 2) per item, using the
 * local embedding service (scripts/embedding). Text, images, videos and audio
 * share the same vector space, enabling semantic search across modalities in
 * the analysis UI.
 *
 * Items are sent in batches (see 'batchSize'); they are held by this task until
 * their batch is embedded and then forwarded to the next task, in the same way
 * RemoteImageClassifierTask does. The vector is stored as a KnnVector extra
 * attribute, indexed by IndexItem as doc values, stored field and Lucene
 * KnnVectorField.
 */
public class EmbeddingTask extends AbstractTask {

    private static final Logger logger = LoggerFactory.getLogger(EmbeddingTask.class);

    private static final int MAX_RETRY = 5;
    private static final long MAX_BATCH_PAYLOAD = 96L * 1024 * 1024;
    private static final int CACHE_SIZE = 20000;

    private static final String STATUS_ERROR = "error";
    private static final String STATUS_SKIPPED_SIZE = "skippedSize";

    // shared by all worker instances
    private static final Object initLock = new Object();
    private static boolean initialized = false;
    private static boolean enabled = false;
    private static EmbeddingServiceClient client;
    private static ServiceInfo serviceInfo;
    private static final AtomicInteger activeInstances = new AtomicInteger();
    private static final AtomicBoolean statsLogged = new AtomicBoolean();

    // vectors of already embedded content, avoids embedding duplicates again
    private static final Map<String, float[]> cache = Collections.synchronizedMap(new LinkedHashMap<>(1024, 0.75f, true) {
        private static final long serialVersionUID = 1L;

        @Override
        protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
            return size() > CACHE_SIZE;
        }
    });

    // statistics per modality (EmbeddingUtil.MODALITIES order)
    private static final AtomicLongArray embedded = new AtomicLongArray(4);
    private static final AtomicLongArray failed = new AtomicLongArray(4);
    private static final AtomicLongArray fromCache = new AtomicLongArray(4);
    private static final AtomicLong skipped = new AtomicLong();
    private static final AtomicLong requestTime = new AtomicLong();
    private static final AtomicLong requests = new AtomicLong();

    private EmbeddingTaskConfig config;
    private boolean embedText, embedImages, embedVideos, embedAudio;

    private static class Pending {
        final IItem item;
        final String modality;
        final String cacheKey;
        final Input input;
        // set if another item with the same content (modality + hash) is already in this batch:
        // only the primary is sent to the service, duplicates reuse its vector
        final Pending primary;

        Pending(IItem item, String modality, String cacheKey, Input input, Pending primary) {
            this.item = item;
            this.modality = modality;
            this.cacheKey = cacheKey;
            this.input = input;
            this.primary = primary;
        }

        Input getInput() {
            return primary != null ? primary.input : input;
        }
    }

    // current batch of this worker and items already embedded waiting to be forwarded
    private final LinkedHashMap<String, Pending> pending = new LinkedHashMap<>();
    private final Map<String, Pending> pendingByCacheKey = new HashMap<>();
    private final LinkedList<IItem> sendToNext = new LinkedList<>();
    private long pendingPayload = 0;

    @Override
    public List<Configurable<?>> getConfigurables() {
        return Arrays.asList(new EmbeddingTaskConfig());
    }

    @Override
    public boolean isEnabled() {
        return config != null && config.isEnabled() && enabled;
    }

    @Override
    public void init(ConfigurationManager configurationManager) throws Exception {
        config = configurationManager.findObject(EmbeddingTaskConfig.class);
        synchronized (initLock) {
            if (!initialized) {
                initialized = true;
                // embeddings of reported items are restored from the source case by IPEDReader
                enabled = config.isEnabled() && !caseData.isIpedReport();
                if (enabled) {
                    connect();
                }
            }
        }
        if (!isEnabled()) {
            return;
        }
        activeInstances.incrementAndGet();
        List<String> modalities = serviceInfo.modalities;
        embedText = config.isEmbedText();
        embedImages = config.isEmbedImages() && modalities.contains(EmbeddingUtil.MODALITY_IMAGE);
        embedVideos = config.isEmbedVideos() && modalities.contains(EmbeddingUtil.MODALITY_VIDEO);
        embedAudio = config.isEmbedAudio() && modalities.contains(EmbeddingUtil.MODALITY_AUDIO);
    }

    private void connect() throws IPEDException {
        client = new EmbeddingServiceClient(config);
        try {
            serviceInfo = client.getInfo();
        } catch (IOException e) {
            enabled = false;
            throw new IPEDException("Embedding task is enabled but the embedding service is not available at "
                    + config.getServiceUrl() + " (" + e.getMessage()
                    + "). Start it with scripts/embedding/start_embedding_server.bat or disable 'enableEmbedding'.");
        }
        if (serviceInfo.dim != config.getDimensions()) {
            enabled = false;
            throw new IPEDException("Embedding service dimension (" + serviceInfo.dim
                    + ") differs from 'dimensions' in EmbeddingTaskConfig.txt (" + config.getDimensions() + ").");
        }
        logger.info("Connected to embedding service at {}: {}", config.getServiceUrl(), serviceInfo);
        String[] wanted = { config.isEmbedImages() ? EmbeddingUtil.MODALITY_IMAGE : null,
                config.isEmbedVideos() ? EmbeddingUtil.MODALITY_VIDEO : null,
                config.isEmbedAudio() ? EmbeddingUtil.MODALITY_AUDIO : null };
        for (String m : wanted) {
            if (m != null && !serviceInfo.modalities.contains(m)) {
                logger.warn("Embedding service was started without '{}' modality, those items will not be embedded.", m);
            }
        }
    }

    @Override
    protected boolean processQueueEnd() {
        return true;
    }

    @Override
    protected void process(IItem evidence) throws Exception {
        if (!pending.isEmpty() && (pending.size() >= config.getBatchSize() || pendingPayload >= MAX_BATCH_PAYLOAD
                || evidence.isQueueEnd())) {
            flush();
        }

        if (evidence.isQueueEnd() || !evidence.isToAddToCase() || evidence.isDir() || evidence.isRoot()) {
            return;
        }
        if (EmbeddingUtil.getVector(evidence) != null) {
            return;
        }
        if (config.isSkipHashDBFiles() && evidence.getExtraAttribute(ExtraProperties.HASHDB_STATUS) != null) {
            skipped.incrementAndGet();
            return;
        }

        String modality = getModality(evidence);
        if (modality == null) {
            return;
        }
        int m = EmbeddingUtil.modalityIndex(modality);

        String hash = evidence.getHash();
        String cacheKey = hash != null && !hash.isEmpty() ? modality + ":" + hash : null;
        if (cacheKey != null) {
            float[] vec = cache.get(cacheKey);
            if (vec != null) {
                setEmbedding(evidence, modality, vec);
                fromCache.incrementAndGet(m);
                return;
            }
            Pending primary = pendingByCacheKey.get(cacheKey);
            if (primary != null) {
                // same content already in this batch: hold the item and reuse the vector
                pending.put(Integer.toString(evidence.getId()), new Pending(evidence, modality, cacheKey, null, primary));
                return;
            }
        }

        Input input;
        try {
            input = createInput(evidence, modality);
        } catch (Exception e) {
            logger.warn("Failed to read {} content of {}: {}", modality, evidence.getPath(), e.toString());
            evidence.setExtraAttribute(EmbeddingUtil.EMBEDDING_STATUS, STATUS_ERROR);
            failed.incrementAndGet(m);
            return;
        }
        if (input == null) {
            return;
        }
        Pending p = new Pending(evidence, modality, cacheKey, input, null);
        pending.put(input.getId(), p);
        if (cacheKey != null) {
            pendingByCacheKey.put(cacheKey, p);
        }
        pendingPayload += input.getPayloadSize();
    }

    private String getModality(IItem evidence) {
        MediaType mediaType = evidence.getMediaType();
        if (mediaType == null) {
            return null;
        }
        String mime = mediaType.toString();
        if (MetadataUtil.isVideoType(mediaType)) {
            return embedVideos ? EmbeddingUtil.MODALITY_VIDEO : null;
        }
        if (MetadataUtil.isImageType(mediaType)) {
            // animated images are embedded from their frames, like videos, when videos are enabled
            if (embedVideos && MetadataUtil.isAnimationImage(evidence)) {
                return EmbeddingUtil.MODALITY_VIDEO;
            }
            return embedImages ? EmbeddingUtil.MODALITY_IMAGE : null;
        }
        if (mime.startsWith("audio/")) {
            return embedAudio ? EmbeddingUtil.MODALITY_AUDIO : null;
        }
        if (embedText && !MediaType.OCTET_STREAM.equals(mediaType) && ((Item) evidence).getTextCache() != null) {
            return EmbeddingUtil.MODALITY_TEXT;
        }
        return null;
    }

    private Input createInput(IItem evidence, String modality) throws Exception {
        String id = Integer.toString(evidence.getId());
        switch (modality) {
            case EmbeddingUtil.MODALITY_TEXT:
                String text = readText(evidence);
                return text == null ? null : Input.document(id, evidence.getName(), text);

            case EmbeddingUtil.MODALITY_IMAGE:
                byte[] img = readImage(evidence);
                return img == null ? null : Input.image(id, img);

            case EmbeddingUtil.MODALITY_VIDEO:
                List<byte[]> frames = readVideoFrames(evidence);
                if (frames != null && !frames.isEmpty()) {
                    return Input.videoFrames(id, frames);
                }
                Long len = evidence.getLength();
                if (len != null && len > 0 && len <= config.getMaxRawVideoBytes()) {
                    return Input.videoFile(id, readAll(evidence));
                }
                if (evidence.getThumb() != null && evidence.getThumb().length > 10) {
                    return Input.image(id, evidence.getThumb());
                }
                markSkippedSize(evidence);
                return null;

            case EmbeddingUtil.MODALITY_AUDIO:
                len = evidence.getLength();
                if (len == null || len == 0) {
                    return null;
                }
                if (len > config.getMaxAudioBytes()) {
                    markSkippedSize(evidence);
                    return null;
                }
                return Input.audio(id, readAll(evidence));

            default:
                return null;
        }
    }

    private void markSkippedSize(IItem evidence) {
        evidence.setExtraAttribute(EmbeddingUtil.EMBEDDING_STATUS, STATUS_SKIPPED_SIZE);
        skipped.incrementAndGet();
    }

    private String readText(IItem evidence) throws IOException {
        int max = config.getMaxTextChars();
        char[] cbuf = new char[max];
        int off = 0, i = 0;
        try (Reader reader = evidence.getTextReader()) {
            while (i != -1 && (off += i) < max) {
                i = reader.read(cbuf, off, max - off);
            }
        }
        String text = new String(cbuf, 0, off);
        // IPED appends item metadata to the extracted text, it is noise for semantic search
        int metadataStart = text.lastIndexOf(StandardParser.METADATA_HEADER);
        if (metadataStart != -1) {
            text = text.substring(0, metadataStart);
        }
        text = text.trim();
        return text.length() < config.getMinTextChars() ? null : text;
    }

    private byte[] readImage(IItem evidence) {
        int size = config.getImageSize();
        BufferedImage img = null;
        try (BufferedInputStream is = evidence.getBufferedInputStream()) {
            img = ImageUtil.getSubSampledImage(is, size, evidence.getMediaType().toString());
        } catch (Exception e) {
            logger.debug("Could not decode image {}: {}", evidence.getPath(), e.toString());
        }
        if (img != null) {
            try {
                return toJpeg(img, size);
            } catch (IOException e) {
                logger.debug("Could not encode image {}: {}", evidence.getPath(), e.toString());
            }
        }
        // formats not decoded by ImageIO (or corrupted) still may have a thumbnail
        byte[] thumb = evidence.getThumb();
        return thumb != null && thumb.length > 10 ? thumb : null;
    }

    private List<byte[]> readVideoFrames(IItem evidence) throws Exception {
        List<BufferedImage> frames = null;
        File viewFile = evidence.getViewFile();
        if (viewFile != null && viewFile.exists()) {
            frames = ImageUtil.getFrames(viewFile);
        } else if (evidence.hasPreview()) {
            try (InputStream is = PreviewRepositoryManager.get(output).readPreview(evidence, false)) {
                frames = ImageUtil.getFrames(is);
            }
        }
        if (frames == null || frames.isEmpty()) {
            return null;
        }
        int max = config.getMaxVideoFrames();
        List<byte[]> result = new ArrayList<>();
        for (int i = 0; i < Math.min(max, frames.size()); i++) {
            int idx = frames.size() <= max ? i : (int) Math.round(i * (frames.size() - 1) / (double) (max - 1));
            result.add(toJpeg(frames.get(idx), config.getImageSize()));
        }
        return result;
    }

    private static byte[] toJpeg(BufferedImage img, int size) throws IOException {
        img = ImageUtil.resizeImage(img, size, size, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageUtil.writeCompressedJPG(img, baos, 90);
        return baos.toByteArray();
    }

    private static byte[] readAll(IItem evidence) throws IOException {
        try (InputStream is = evidence.getBufferedInputStream()) {
            return is.readAllBytes();
        }
    }

    private void flush() throws InterruptedException {
        List<Input> inputs = new ArrayList<>(pending.size());
        for (Pending p : pending.values()) {
            if (p.primary == null) {
                inputs.add(p.input);
            }
        }
        Result result = sendWithRetry(inputs);
        for (Pending p : pending.values()) {
            int m = EmbeddingUtil.modalityIndex(p.modality);
            String inputId = p.getInput().getId();
            float[] vec = result != null ? result.vectors.get(inputId) : null;
            if (vec != null && vec.length == config.getDimensions()) {
                setEmbedding(p.item, p.modality, vec);
                if (p.primary != null) {
                    fromCache.incrementAndGet(m);
                } else {
                    if (p.cacheKey != null) {
                        cache.put(p.cacheKey, vec);
                    }
                    embedded.incrementAndGet(m);
                }
            } else {
                String error = result == null ? "service unavailable"
                        : vec != null ? "unexpected dimension " + vec.length : result.errors.get(inputId);
                logger.warn("Failed to embed {} {}: {}", p.modality, p.item.getPath(), error);
                p.item.setExtraAttribute(EmbeddingUtil.EMBEDDING_STATUS, STATUS_ERROR);
                failed.incrementAndGet(m);
            }
            sendToNext.add(p.item);
        }
        pending.clear();
        pendingByCacheKey.clear();
        pendingPayload = 0;
    }

    private Result sendWithRetry(List<Input> inputs) throws InterruptedException {
        long wait = 1000;
        for (int attempt = 0;; attempt++) {
            long t = System.currentTimeMillis();
            try {
                Result result = client.embed(inputs);
                requestTime.addAndGet(System.currentTimeMillis() - t);
                requests.incrementAndGet();
                return result;
            } catch (IOException e) {
                boolean clientError = e instanceof EmbeddingServiceException
                        && ((EmbeddingServiceException) e).getStatusCode() < HttpStatus.SC_INTERNAL_SERVER_ERROR;
                if (clientError || attempt >= MAX_RETRY) {
                    logger.error("Embedding request of {} items failed after {} attempt(s): {}", inputs.size(),
                            attempt + 1, e.toString());
                    return null;
                }
                logger.warn("Embedding request failed (attempt {}), retrying in {} ms: {}", attempt + 1, wait,
                        e.toString());
                Thread.sleep(wait);
                wait = Math.min(wait * 2, 30000);
            }
        }
    }

    private static void setEmbedding(IItem item, String modality, float[] vec) {
        item.setExtraAttribute(EmbeddingUtil.EMBEDDING, EmbeddingUtil.toKnnVector(vec));
        item.setExtraAttribute(EmbeddingUtil.EMBEDDING_MODALITY, modality);
    }

    @Override
    protected void sendToNextTask(IItem item) throws Exception {
        if (!isEnabled()) {
            super.sendToNextTask(item);
            return;
        }
        // forward items whose batch was already embedded (removed before the call, as
        // next tasks may create subitems that re-enter this task)
        while (!sendToNext.isEmpty()) {
            super.sendToNextTask(sendToNext.removeFirst());
        }
        Pending p = pending.get(Integer.toString(item.getId()));
        if (p == null || p.item != item || item.isQueueEnd()) {
            super.sendToNextTask(item);
        }
    }

    @Override
    public void finish() throws Exception {
        if (!isEnabled()) {
            return;
        }
        if (!statsLogged.getAndSet(true)) {
            logStatistics();
        }
        if (activeInstances.decrementAndGet() == 0) {
            // allows a new processing in the same JVM (e.g. case update started from the UI)
            synchronized (initLock) {
                cache.clear();
                if (client != null) {
                    client.close();
                    client = null;
                }
                initialized = false;
                statsLogged.set(false);
                resetStatistics();
            }
        }
    }

    private static void resetStatistics() {
        for (int i = 0; i < EmbeddingUtil.MODALITIES.length; i++) {
            embedded.set(i, 0);
            failed.set(i, 0);
            fromCache.set(i, 0);
        }
        skipped.set(0);
        requestTime.set(0);
        requests.set(0);
    }

    private static void logStatistics() {
        long total = 0;
        for (int i = 0; i < EmbeddingUtil.MODALITIES.length; i++) {
            total += embedded.get(i) + fromCache.get(i);
            logger.info("Embeddings of {} items: {} computed, {} from duplicates, {} failed", EmbeddingUtil.MODALITIES[i],
                    embedded.get(i), fromCache.get(i), failed.get(i));
        }
        logger.info("Total items with embeddings: {}; skipped: {}", total, skipped.get());
        if (requests.get() > 0) {
            logger.info("Embedding requests: {}; average request time (ms): {}", requests.get(),
                    requestTime.get() / requests.get());
        }
    }
}
