package iped.engine.config;

import iped.utils.UTF8Properties;

/**
 * Configuration of {@link iped.engine.task.EmbeddingTask} and of the semantic
 * search in the analysis UI. Loaded from conf/EmbeddingTaskConfig.txt and
 * enabled by "enableEmbedding" in IPEDConfig.txt.
 */
public class EmbeddingTaskConfig extends AbstractTaskPropertiesConfig {

    private static final long serialVersionUID = 1L;

    public static final String ENABLE_PROP = "enableEmbedding";
    private static final String CONFIG_FILE = "EmbeddingTaskConfig.txt";

    private static final String SERVICE_URL = "serviceUrl";
    private static final String DIMENSIONS = "dimensions";
    private static final String EMBED_TEXT = "embedText";
    private static final String EMBED_IMAGES = "embedImages";
    private static final String EMBED_VIDEOS = "embedVideos";
    private static final String EMBED_AUDIO = "embedAudio";
    private static final String MIN_TEXT_CHARS = "minTextChars";
    private static final String MAX_TEXT_CHARS = "maxTextChars";
    private static final String IMAGE_SIZE = "imageSize";
    private static final String MAX_VIDEO_FRAMES = "maxVideoFrames";
    private static final String MAX_RAW_VIDEO_BYTES = "maxRawVideoBytes";
    private static final String MAX_AUDIO_BYTES = "maxAudioBytes";
    private static final String BATCH_SIZE = "batchSize";
    private static final String SKIP_HASH_DB_FILES = "skipHashDBFiles";
    private static final String CONNECT_TIMEOUT = "connectTimeout";
    private static final String SOCKET_TIMEOUT = "socketTimeout";
    private static final String SEARCH_MAX_RESULTS = "searchMaxResultsPerModality";
    private static final String SEARCH_MIN_SCORE = "searchMinScore";

    private String serviceUrl = "http://127.0.0.1:8691";
    private int dimensions = 768;
    private boolean embedText = true;
    private boolean embedImages = true;
    private boolean embedVideos = true;
    private boolean embedAudio = true;
    private int minTextChars = 30;
    private int maxTextChars = 20000;
    private int imageSize = 768;
    private int maxVideoFrames = 16;
    private long maxRawVideoBytes = 200L * 1024 * 1024;
    private long maxAudioBytes = 64L * 1024 * 1024;
    private int batchSize = 16;
    private boolean skipHashDBFiles = false;
    private int connectTimeout = 30 * 1000;
    private int socketTimeout = 10 * 60 * 1000;
    private int searchMaxResultsPerModality = 500;
    private int searchMinScore = 0;

    @Override
    public String getTaskEnableProperty() {
        return ENABLE_PROP;
    }

    @Override
    public String getTaskConfigFileName() {
        return CONFIG_FILE;
    }

    public String getServiceUrl() {
        return serviceUrl;
    }

    public int getDimensions() {
        return dimensions;
    }

    public boolean isEmbedText() {
        return embedText;
    }

    public boolean isEmbedImages() {
        return embedImages;
    }

    public boolean isEmbedVideos() {
        return embedVideos;
    }

    public boolean isEmbedAudio() {
        return embedAudio;
    }

    public int getMinTextChars() {
        return minTextChars;
    }

    public int getMaxTextChars() {
        return maxTextChars;
    }

    public int getImageSize() {
        return imageSize;
    }

    public int getMaxVideoFrames() {
        return maxVideoFrames;
    }

    public long getMaxRawVideoBytes() {
        return maxRawVideoBytes;
    }

    public long getMaxAudioBytes() {
        return maxAudioBytes;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public boolean isSkipHashDBFiles() {
        return skipHashDBFiles;
    }

    public int getConnectTimeout() {
        return connectTimeout;
    }

    public int getSocketTimeout() {
        return socketTimeout;
    }

    public int getSearchMaxResultsPerModality() {
        return searchMaxResultsPerModality;
    }

    public int getSearchMinScore() {
        return searchMinScore;
    }

    @Override
    void processProperties(UTF8Properties properties) {
        String value = get(properties, SERVICE_URL);
        if (value != null) {
            serviceUrl = value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
            if (!serviceUrl.startsWith("http://") && !serviceUrl.startsWith("https://")) {
                serviceUrl = "http://" + serviceUrl;
            }
        }
        dimensions = getInt(properties, DIMENSIONS, dimensions, 1);
        embedText = getBool(properties, EMBED_TEXT, embedText);
        embedImages = getBool(properties, EMBED_IMAGES, embedImages);
        embedVideos = getBool(properties, EMBED_VIDEOS, embedVideos);
        embedAudio = getBool(properties, EMBED_AUDIO, embedAudio);
        minTextChars = getInt(properties, MIN_TEXT_CHARS, minTextChars, 1);
        maxTextChars = getInt(properties, MAX_TEXT_CHARS, maxTextChars, minTextChars);
        imageSize = getInt(properties, IMAGE_SIZE, imageSize, 64);
        maxVideoFrames = getInt(properties, MAX_VIDEO_FRAMES, maxVideoFrames, 1);
        maxRawVideoBytes = getLong(properties, MAX_RAW_VIDEO_BYTES, maxRawVideoBytes);
        maxAudioBytes = getLong(properties, MAX_AUDIO_BYTES, maxAudioBytes);
        batchSize = getInt(properties, BATCH_SIZE, batchSize, 1);
        skipHashDBFiles = getBool(properties, SKIP_HASH_DB_FILES, skipHashDBFiles);
        connectTimeout = getInt(properties, CONNECT_TIMEOUT, connectTimeout, 0);
        socketTimeout = getInt(properties, SOCKET_TIMEOUT, socketTimeout, 0);
        searchMaxResultsPerModality = getInt(properties, SEARCH_MAX_RESULTS, searchMaxResultsPerModality, 1);
        searchMinScore = Math.min(100, getInt(properties, SEARCH_MIN_SCORE, searchMinScore, 0));
    }

    private static String get(UTF8Properties properties, String key) {
        String value = properties.getProperty(key);
        return value == null || value.trim().isEmpty() ? null : value.trim();
    }

    private static int getInt(UTF8Properties properties, String key, int def, int min) {
        String value = get(properties, key);
        return value == null ? def : Math.max(min, Integer.parseInt(value));
    }

    private static long getLong(UTF8Properties properties, String key, long def) {
        String value = get(properties, key);
        return value == null ? def : Math.max(0, Long.parseLong(value));
    }

    private static boolean getBool(UTF8Properties properties, String key, boolean def) {
        String value = get(properties, key);
        return value == null ? def : Boolean.parseBoolean(value);
    }
}
