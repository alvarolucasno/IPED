package iped.engine.search;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.VectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import iped.engine.embedding.EmbeddingUtil;

/**
 * All item embeddings of an index, loaded once and kept in memory as int8
 * vectors (one byte per dimension plus a float scale per item), to score a
 * query against the whole case with a fast multithreaded scan.
 *
 * Vectors are read sequentially from the KnnVectorField raw data, which is much
 * cheaper than looking them up in doc values. If they do not fit in the memory
 * budget, scans stream them from the index instead (exact, slower).
 *
 * Scores are "standout" scores: 10 x robust z-score of the cosine similarity
 * relative to the other items of the same modality in the case, so a score of 30
 * means 3 robust standard deviations above the typical unrelated item. Raw
 * cosine is not comparable across queries and modalities: unrelated items
 * usually score 0.55-0.65 and that floor changes with the query.
 */
public class EmbeddingMatrix {

    private static final Logger logger = LoggerFactory.getLogger(EmbeddingMatrix.class);

    /** Score = SCORE_SCALE x robust z-score. */
    public static final float SCORE_SCALE = 10f;

    /** Modalities with fewer items use the spread estimated from all items. */
    static final int MIN_ITEMS_FOR_OWN_SPREAD = 30;

    /** Modalities up to this size get an exact median. */
    static final int EXACT_MEDIAN_MAX_ITEMS = 4096;

    /** Lower bound of the robust standard deviation (cosine units). */
    static final float MIN_SIGMA = 0.005f;

    private static final int NUM_MODALITIES = EmbeddingUtil.MODALITIES.length;
    private static final int HIST_BINS = 4096; // cosine in [-1, 1], bin width ~0.0005
    private static final float MAD_TO_SIGMA = 1.4826f;
    private static final int SCORE_CACHE_SIZE = 4;

    private static final Map<IndexReader, EmbeddingMatrix> instances = new WeakHashMap<>();
    private static final Object loadLock = new Object();

    final int dim;
    final int size;
    final int[] luceneIds; // ascending
    final byte[] modality;
    final int[] countPerModality = new int[NUM_MODALITIES];
    private final int[] slotOf; // luceneId -> slot, -1 if no embedding

    // in memory mode
    final byte[] quantized;
    final float[] scales;

    // streaming mode (weak: the matrix is the value of a weak map keyed by the reader)
    private final WeakReference<IndexReader> reader;
    private final int[] leafSlotStart;

    private final Map<String, float[]> scoreCache = new LinkedHashMap<String, float[]>(8, 0.75f, true) {
        private static final long serialVersionUID = 1L;

        @Override
        protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
            return size() > SCORE_CACHE_SIZE;
        }
    };

    /**
     * Returns the matrix of the reader, loading it on first use. Returns null if
     * the index has no embeddings.
     */
    public static EmbeddingMatrix get(IndexReader reader) throws IOException {
        synchronized (instances) {
            if (instances.containsKey(reader)) {
                return instances.get(reader);
            }
        }
        // load outside the map lock, only once
        synchronized (loadLock) {
            synchronized (instances) {
                if (instances.containsKey(reader)) {
                    return instances.get(reader);
                }
            }
            long t = System.currentTimeMillis();
            EmbeddingMatrix matrix = load(reader, getMemoryBudget());
            if (matrix != null) {
                logger.info("Loaded {} embeddings ({} dims, {}) in {} ms", matrix.size, matrix.dim,
                        matrix.quantized != null ? "in memory" : "streamed from index", System.currentTimeMillis() - t);
            }
            synchronized (instances) {
                instances.put(reader, matrix);
            }
            return matrix;
        }
    }

    private static long getMemoryBudget() {
        String prop = System.getProperty("iped.embedding.cacheMaxBytes");
        if (prop != null) {
            return Long.parseLong(prop);
        }
        return Runtime.getRuntime().maxMemory() / 4;
    }

    static EmbeddingMatrix load(IndexReader reader, long memoryBudget) throws IOException {
        List<LeafReaderContext> leaves = reader.leaves();
        int dim = 0;
        long upperBound = 0;
        for (LeafReaderContext ctx : leaves) {
            VectorValues vv = ctx.reader().getVectorValues(EmbeddingUtil.EMBEDDING);
            if (vv != null) {
                if (dim != 0 && dim != vv.dimension()) {
                    throw new IOException("Embeddings with different dimensions in the same index");
                }
                dim = vv.dimension();
                upperBound += vv.size();
            }
        }
        if (upperBound == 0) {
            return null;
        }
        // one array holds all vectors: beyond ~2.8M items of 768d it would exceed the array limit
        boolean inMemory = upperBound * (dim + 4) <= memoryBudget && upperBound * dim <= Integer.MAX_VALUE - 16;
        return new EmbeddingMatrix(reader, dim, (int) upperBound, inMemory);
    }

    private EmbeddingMatrix(IndexReader reader, int dim, int upperBound, boolean inMemory) throws IOException {
        this.reader = inMemory ? null : new WeakReference<>(reader);
        this.dim = dim;
        int[] ids = new int[upperBound];
        byte[] mods = new byte[upperBound];
        byte[] q = inMemory ? new byte[upperBound * dim] : null;
        float[] sc = inMemory ? new float[upperBound] : null;
        slotOf = new int[reader.maxDoc()];
        Arrays.fill(slotOf, -1);
        List<LeafReaderContext> leaves = reader.leaves();
        leafSlotStart = new int[leaves.size() + 1];
        int n = 0;
        for (int l = 0; l < leaves.size(); l++) {
            leafSlotStart[l] = n;
            LeafReaderContext ctx = leaves.get(l);
            LeafReader leaf = ctx.reader();
            VectorValues vv = leaf.getVectorValues(EmbeddingUtil.EMBEDDING);
            if (vv == null) {
                continue;
            }
            Bits live = leaf.getLiveDocs();
            SortedDocValues modDV = leaf.getSortedDocValues(EmbeddingUtil.EMBEDDING_MODALITY);
            int[] ordToModality = mapModalityOrds(modDV);
            for (int doc = vv.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = vv.nextDoc()) {
                if (live != null && !live.get(doc)) {
                    continue;
                }
                int m = 0; // items without modality are texts
                if (modDV != null && modDV.advanceExact(doc)) {
                    m = Math.max(0, ordToModality[modDV.ordValue()]);
                }
                if (inMemory) {
                    sc[n] = quantize(vv.vectorValue(), q, n * dim);
                }
                ids[n] = ctx.docBase + doc;
                mods[n] = (byte) m;
                slotOf[ids[n]] = n;
                countPerModality[m]++;
                n++;
            }
        }
        leafSlotStart[leaves.size()] = n;
        this.size = n;
        this.luceneIds = n == ids.length ? ids : Arrays.copyOf(ids, n);
        this.modality = n == mods.length ? mods : Arrays.copyOf(mods, n);
        this.quantized = q == null || n == upperBound ? q : Arrays.copyOf(q, n * dim);
        this.scales = sc == null || n == upperBound ? sc : Arrays.copyOf(sc, n);
    }

    /** For tests: in memory matrix from float vectors. */
    EmbeddingMatrix(float[][] vectors, byte[] modalities) {
        this(vectors[0].length, new byte[vectors.length * vectors[0].length], new float[vectors.length], modalities);
        for (int i = 0; i < size; i++) {
            scales[i] = quantize(vectors[i], quantized, i * dim);
        }
    }

    /** For tests: in memory matrix from already quantized vectors (lucene id = slot). */
    EmbeddingMatrix(int dim, byte[] quantized, float[] scales, byte[] modalities) {
        this.reader = null;
        this.leafSlotStart = null;
        this.dim = dim;
        this.size = scales.length;
        this.luceneIds = new int[size];
        this.modality = modalities;
        this.quantized = quantized;
        this.scales = scales;
        this.slotOf = new int[size];
        for (int i = 0; i < size; i++) {
            luceneIds[i] = i;
            slotOf[i] = i;
            countPerModality[modality[i]]++;
        }
    }

    private static int[] mapModalityOrds(SortedDocValues modDV) throws IOException {
        if (modDV == null) {
            return null;
        }
        int[] map = new int[modDV.getValueCount()];
        for (int ord = 0; ord < map.length; ord++) {
            map[ord] = EmbeddingUtil.modalityIndex(modDV.lookupOrd(ord).utf8ToString());
        }
        return map;
    }

    /** Symmetric int8 quantization with one scale per vector. */
    static float quantize(float[] v, byte[] dest, int offset) {
        float max = 0;
        for (float x : v) {
            max = Math.max(max, Math.abs(x));
        }
        float scale = max > 0 ? max / 127f : 1f;
        float inv = 1f / scale;
        for (int i = 0; i < v.length; i++) {
            dest[offset + i] = (byte) Math.round(v[i] * inv);
        }
        return scale;
    }

    public int getDimension() {
        return dim;
    }

    public int size() {
        return size;
    }

    public boolean isInMemory() {
        return quantized != null;
    }

    /** Slot of a lucene id, or -1 if the document has no embedding. */
    public int slotOf(int luceneId) {
        return luceneId >= 0 && luceneId < slotOf.length ? slotOf[luceneId] : -1;
    }

    /**
     * Standout scores of all items for the query, indexed by slot. Results of the
     * last queries are cached, so re-filtering with the same query costs nothing.
     */
    public float[] scores(float[] query) throws IOException {
        if (query.length != dim) {
            throw new IllegalArgumentException("Query embedding has " + query.length + " dimensions but the case has "
                    + dim + ". Check the embedding service '--dim' option.");
        }
        String key = Arrays.toString(query);
        synchronized (scoreCache) {
            float[] cached = scoreCache.get(key);
            if (cached != null) {
                return cached;
            }
        }
        float[] result = computeScores(query);
        synchronized (scoreCache) {
            scoreCache.put(key, result);
        }
        return result;
    }

    private float[] computeScores(float[] query) throws IOException {
        float[] cos = new float[size];
        int numThreads = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), size / 4096 + 1));
        int[][][] hist = new int[numThreads][][];
        IOException[] error = new IOException[1];
        Thread[] threads = new Thread[numThreads];
        for (int t = 0; t < numThreads; t++) {
            int threadIdx = t;
            hist[t] = new int[NUM_MODALITIES][HIST_BINS];
            threads[t] = new Thread("EmbeddingScan-" + t) {
                @Override
                public void run() {
                    try {
                        if (quantized != null) {
                            int chunk = (size + numThreads - 1) / numThreads;
                            int from = Math.min(size, threadIdx * chunk);
                            scanMemory(query, from, Math.min(size, from + chunk), cos, hist[threadIdx]);
                        } else {
                            IndexReader r = reader.get();
                            if (r == null) {
                                throw new IOException("Case index was closed");
                            }
                            List<LeafReaderContext> leaves = r.leaves();
                            for (int l = threadIdx; l < leaves.size(); l += numThreads) {
                                scanLeaf(query, leaves.get(l), leafSlotStart[l], cos, hist[threadIdx]);
                            }
                        }
                    } catch (IOException e) {
                        error[0] = e;
                    }
                }
            };
            threads[t].start();
        }
        try {
            for (Thread t : threads) {
                t.join();
            }
        } catch (InterruptedException e) {
            for (Thread t : threads) {
                t.interrupt();
            }
            Thread.currentThread().interrupt();
            throw new IOException("Semantic search interrupted", e);
        }
        if (error[0] != null) {
            throw error[0];
        }

        int[][] merged = new int[NUM_MODALITIES][HIST_BINS];
        for (int[][] h : hist) {
            for (int m = 0; m < NUM_MODALITIES; m++) {
                for (int b = 0; b < HIST_BINS; b++) {
                    merged[m][b] += h[m][b];
                }
            }
        }
        float[] center = new float[NUM_MODALITIES];
        float[] sigma;
        if (size <= EXACT_MEDIAN_MAX_ITEMS) {
            sigma = exactStats(cos, center);
        } else {
            Arrays.fill(center, Float.NaN);
            exactMediansOfSmallModalities(cos, center);
            sigma = robustStats(merged, countPerModality, center);
        }

        // cos -> standout score, in place
        float[] scale = new float[NUM_MODALITIES];
        for (int m = 0; m < NUM_MODALITIES; m++) {
            scale[m] = SCORE_SCALE / sigma[m];
        }
        for (int i = 0; i < size; i++) {
            int m = modality[i];
            cos[i] = (cos[i] - center[m]) * scale[m];
        }
        return cos;
    }

    private void scanMemory(float[] query, int from, int to, float[] out, int[][] hist) {
        final byte[] q = quantized;
        final int d = dim;
        for (int i = from; i < to; i++) {
            int off = i * d;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int j = 0;
            for (; j + 3 < d; j += 4) {
                s0 += query[j] * q[off + j];
                s1 += query[j + 1] * q[off + j + 1];
                s2 += query[j + 2] * q[off + j + 2];
                s3 += query[j + 3] * q[off + j + 3];
            }
            for (; j < d; j++) {
                s0 += query[j] * q[off + j];
            }
            float cos = (s0 + s1 + s2 + s3) * scales[i];
            out[i] = cos;
            hist[modality[i]][bin(cos)]++;
        }
    }

    private void scanLeaf(float[] query, LeafReaderContext ctx, int slot, float[] out, int[][] hist) throws IOException {
        LeafReader leaf = ctx.reader();
        VectorValues vv = leaf.getVectorValues(EmbeddingUtil.EMBEDDING);
        if (vv == null) {
            return;
        }
        Bits live = leaf.getLiveDocs();
        for (int doc = vv.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = vv.nextDoc()) {
            if (live != null && !live.get(doc)) {
                continue;
            }
            float cos = EmbeddingUtil.dot(query, vv.vectorValue());
            out[slot] = cos;
            hist[modality[slot]][bin(cos)]++;
            slot++;
        }
    }

    private static int bin(float cos) {
        int b = (int) ((cos + 1f) * (HIST_BINS / 2));
        return b < 0 ? 0 : b >= HIST_BINS ? HIST_BINS - 1 : b;
    }

    private static float binCenter(int b) {
        return (b + 0.5f) / (HIST_BINS / 2) - 1f;
    }

    /** Same statistics as {@link #robustStats}, computed exactly (small cases). */
    private float[] exactStats(float[] cos, float[] center) {
        float[][] values = new float[NUM_MODALITIES][];
        for (int m = 0; m < NUM_MODALITIES; m++) {
            values[m] = new float[countPerModality[m]];
        }
        int[] fill = new int[NUM_MODALITIES];
        for (int i = 0; i < size; i++) {
            values[modality[i]][fill[modality[i]]++] = cos[i];
        }
        float[] pooledDev = new float[size];
        float[] ownSigma = new float[NUM_MODALITIES];
        int p = 0;
        for (int m = 0; m < NUM_MODALITIES; m++) {
            float[] v = values[m];
            if (v.length == 0) {
                continue;
            }
            center[m] = median(v);
            for (int i = 0; i < v.length; i++) {
                v[i] = Math.abs(v[i] - center[m]);
                pooledDev[p++] = v[i];
            }
            ownSigma[m] = MAD_TO_SIGMA * median(v);
        }
        float pooledSigma = MAD_TO_SIGMA * median(pooledDev);
        float[] sigma = new float[NUM_MODALITIES];
        for (int m = 0; m < NUM_MODALITIES; m++) {
            sigma[m] = Math.max(MIN_SIGMA, countPerModality[m] >= MIN_ITEMS_FOR_OWN_SPREAD ? ownSigma[m] : pooledSigma);
        }
        return sigma;
    }

    /** Sorts the array and returns its median. */
    private static float median(float[] v) {
        Arrays.sort(v);
        int h = v.length / 2;
        return v.length % 2 == 1 ? v[h] : (v[h - 1] + v[h]) / 2;
    }

    /**
     * Histogram medians are only bin accurate and, with few items, are not the mean
     * of the two middle values: compute small modalities exactly.
     */
    private void exactMediansOfSmallModalities(float[] cos, float[] center) {
        float[][] values = new float[NUM_MODALITIES][];
        boolean any = false;
        for (int m = 0; m < NUM_MODALITIES; m++) {
            if (countPerModality[m] > 0 && countPerModality[m] <= EXACT_MEDIAN_MAX_ITEMS) {
                values[m] = new float[countPerModality[m]];
                any = true;
            }
        }
        if (!any) {
            return;
        }
        int[] fill = new int[NUM_MODALITIES];
        for (int i = 0; i < size; i++) {
            float[] v = values[modality[i]];
            if (v != null) {
                v[fill[modality[i]]++] = cos[i];
            }
        }
        for (int m = 0; m < NUM_MODALITIES; m++) {
            if (values[m] != null) {
                center[m] = median(values[m]);
            }
        }
    }

    /**
     * Median per modality (into center, unless already computed) and robust
     * standard deviation per modality (returned), from the score histograms. Small
     * modalities use the spread of all items around their own modality median.
     */
    static float[] robustStats(int[][] hist, int[] counts, float[] center) {
        int[] pooledDev = new int[HIST_BINS];
        float[] ownSigma = new float[NUM_MODALITIES];
        int total = 0;
        for (int m = 0; m < NUM_MODALITIES; m++) {
            if (counts[m] == 0) {
                continue;
            }
            total += counts[m];
            if (Float.isNaN(center[m])) {
                center[m] = histMedian(hist[m], counts[m], -1f);
            }
            int[] dev = new int[HIST_BINS];
            for (int b = 0; b < HIST_BINS; b++) {
                if (hist[m][b] != 0) {
                    int db = Math.min(HIST_BINS - 1, (int) (Math.abs(binCenter(b) - center[m]) * (HIST_BINS / 2)));
                    dev[db] += hist[m][b];
                    pooledDev[db] += hist[m][b];
                }
            }
            ownSigma[m] = MAD_TO_SIGMA * histMedian(dev, counts[m], 0f);
        }
        float pooledSigma = total > 0 ? MAD_TO_SIGMA * histMedian(pooledDev, total, 0f) : MIN_SIGMA;
        float[] sigma = new float[NUM_MODALITIES];
        for (int m = 0; m < NUM_MODALITIES; m++) {
            float s = counts[m] >= MIN_ITEMS_FOR_OWN_SPREAD ? ownSigma[m] : pooledSigma;
            sigma[m] = Math.max(MIN_SIGMA, s);
        }
        return sigma;
    }

    /** Median of a histogram whose first bin starts at 'origin', linear inside the bin. */
    private static float histMedian(int[] hist, int count, float origin) {
        float half = count / 2f;
        float width = 2f / HIST_BINS;
        int cum = 0;
        for (int b = 0; b < hist.length; b++) {
            if (cum + hist[b] >= half && hist[b] > 0) {
                return origin + (b + (half - cum) / hist[b]) * width;
            }
            cum += hist[b];
        }
        return origin + hist.length * width;
    }
}
