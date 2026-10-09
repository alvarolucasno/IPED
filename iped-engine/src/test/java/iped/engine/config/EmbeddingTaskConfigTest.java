package iped.engine.config;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

import iped.utils.UTF8Properties;

public class EmbeddingTaskConfigTest {

    @Test
    public void testDefaultsEmbedAllModalities() {
        EmbeddingTaskConfig config = new EmbeddingTaskConfig();
        assertEquals("enableEmbedding", config.getTaskEnableProperty());
        assertEquals("http://127.0.0.1:8691", config.getServiceUrl());
        assertEquals(768, config.getDimensions());
        assertTrue(config.isEmbedText());
        assertTrue(config.isEmbedImages());
        assertTrue(config.isEmbedVideos());
        assertTrue(config.isEmbedAudio());
    }

    @Test
    public void testProcessProperties() {
        EmbeddingTaskConfig config = new EmbeddingTaskConfig();
        UTF8Properties props = new UTF8Properties();
        props.setProperty("serviceUrl", " 10.0.0.5:9000/ ");
        props.setProperty("dimensions", "256");
        props.setProperty("embedAudio", "false");
        props.setProperty("embedVideos", "false");
        props.setProperty("batchSize", "0");
        props.setProperty("maxTextChars", "10");
        props.setProperty("minTextChars", "50");
        props.setProperty("searchMinScore", "250");
        props.setProperty("searchMaxResultsPerModality", "42");
        config.processProperties(props);

        assertEquals("http://10.0.0.5:9000", config.getServiceUrl());
        assertEquals(256, config.getDimensions());
        assertFalse(config.isEmbedAudio());
        assertFalse(config.isEmbedVideos());
        assertTrue(config.isEmbedText());
        assertEquals(1, config.getBatchSize());
        // max text chars can not be lower than min text chars
        assertEquals(50, config.getMaxTextChars());
        assertEquals(100, config.getSearchMinScore());
        assertEquals(42, config.getSearchMaxResultsPerModality());
    }

    @Test
    public void testShippedConfigFileMatchesDefaults() throws IOException {
        Path file = Paths.get("../iped-app/resources/config/conf/EmbeddingTaskConfig.txt");
        EmbeddingTaskConfig config = new EmbeddingTaskConfig();
        UTF8Properties props = new UTF8Properties();
        props.load(file.toFile());
        config.processProperties(props);

        EmbeddingTaskConfig defaults = new EmbeddingTaskConfig();
        assertEquals(defaults.getServiceUrl(), config.getServiceUrl());
        assertEquals(defaults.getDimensions(), config.getDimensions());
        assertEquals(defaults.getBatchSize(), config.getBatchSize());
        assertEquals(defaults.getMaxRawVideoBytes(), config.getMaxRawVideoBytes());
        assertEquals(defaults.getMaxAudioBytes(), config.getMaxAudioBytes());
        assertTrue(config.isEmbedText() && config.isEmbedImages() && config.isEmbedVideos() && config.isEmbedAudio());
    }
}
