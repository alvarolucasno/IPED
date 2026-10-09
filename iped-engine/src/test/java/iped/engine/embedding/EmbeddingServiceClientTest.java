package iped.engine.embedding;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import iped.engine.embedding.EmbeddingServiceClient.EmbeddingServiceException;
import iped.engine.embedding.EmbeddingServiceClient.Input;
import iped.engine.embedding.EmbeddingServiceClient.Result;
import iped.engine.embedding.EmbeddingServiceClient.ServiceInfo;

/**
 * Tests the client against an in-process mock of the embedding service contract.
 */
public class EmbeddingServiceClientTest {

    private HttpServer server;
    private EmbeddingServiceClient client;
    private final AtomicReference<JsonNode> lastRequest = new AtomicReference<>();
    private volatile String infoContract = EmbeddingServiceClient.CONTRACT;
    private volatile int embedStatus = 200;

    @Before
    public void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/info", ex -> reply(ex, 200, "{\"contract\":\"" + infoContract
                + "\",\"model\":\"google/embeddinggemma-2\",\"dim\":4,\"modalities\":[\"text\",\"image\",\"video\",\"audio\"],"
                + "\"device\":\"cuda\",\"dtype\":\"bfloat16\"}"));
        server.createContext("/embed", ex -> {
            JsonNode req = new ObjectMapper().readTree(ex.getRequestBody());
            lastRequest.set(req);
            if (embedStatus != 200) {
                reply(ex, embedStatus, "{\"detail\":\"overloaded\"}");
                return;
            }
            StringBuilder vectors = new StringBuilder();
            StringBuilder errors = new StringBuilder();
            for (JsonNode in : req.get("inputs")) {
                String id = in.get("id").asText();
                if (in.has("text") && in.get("text").asText().contains("fail")) {
                    errors.append(errors.length() > 0 ? "," : "").append('"').append(id).append("\":\"ValueError: bad\"");
                } else {
                    vectors.append(vectors.length() > 0 ? "," : "").append('"').append(id).append("\":[0.5,0.5,0.5,0.5]");
                }
            }
            reply(ex, 200, "{\"model\":\"m\",\"dim\":4,\"vectors\":{" + vectors + "},\"errors\":{" + errors + "}}");
        });
        server.start();
        client = new EmbeddingServiceClient("http://127.0.0.1:" + server.getAddress().getPort() + "/", 2000, 5000);
    }

    private static void reply(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    @After
    public void tearDown() throws IOException {
        client.close();
        server.stop(0);
    }

    @Test
    public void testInfo() throws IOException {
        ServiceInfo info = client.getInfo();
        assertEquals(4, info.dim);
        assertEquals(Arrays.asList("text", "image", "video", "audio"), info.modalities);
        assertEquals("google/embeddinggemma-2", info.model);
    }

    @Test(expected = IOException.class)
    public void testIncompatibleContractIsRejected() throws IOException {
        infoContract = "iped-embedding/v999";
        client.getInfo();
    }

    @Test
    public void testEmbedSerializesEveryModality() throws IOException {
        byte[] img = { 1, 2, 3 };
        byte[] frame = { 4, 5 };
        byte[] audio = { 6 };
        Result r = client.embed(Arrays.asList(Input.document("1", "a.txt", "conteúdo"), Input.query("2", "carro"),
                Input.image("3", img), Input.videoFrames("4", Arrays.asList(frame, frame)), Input.audio("5", audio),
                Input.document("6", "x", "fail please")));

        assertEquals(5, r.vectors.size());
        assertArrayEquals(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, r.vectors.get("4"), 1e-6f);
        assertTrue(r.errors.get("6").contains("bad"));

        JsonNode inputs = lastRequest.get().get("inputs");
        assertEquals("document", inputs.get(0).get("type").asText());
        assertEquals("a.txt", inputs.get(0).get("title").asText());
        assertEquals("conteúdo", inputs.get(0).get("text").asText());
        assertEquals("query", inputs.get(1).get("type").asText());
        assertEquals("image", inputs.get(2).get("type").asText());
        assertArrayEquals(img, Base64.getDecoder().decode(inputs.get(2).get("data").asText()));
        assertEquals("video", inputs.get(3).get("type").asText());
        assertEquals(2, inputs.get(3).get("frames").size());
        assertArrayEquals(frame, Base64.getDecoder().decode(inputs.get(3).get("frames").get(1).asText()));
        assertEquals("audio", inputs.get(4).get("type").asText());
    }

    @Test
    public void testEmbedQuery() throws IOException {
        assertEquals(4, client.embedQuery("comprovante de aluguel").length);
        assertEquals("query", lastRequest.get().get("inputs").get(0).get("type").asText());
    }

    @Test(expected = IOException.class)
    public void testEmbedOneFailsOnItemError() throws IOException {
        client.embedOne(Input.document("1", "x", "fail"));
    }

    @Test
    public void testHttpErrorExposesStatus() {
        embedStatus = 503;
        try {
            client.embedQuery("x");
            fail();
        } catch (EmbeddingServiceException e) {
            assertEquals(503, e.getStatusCode());
            assertTrue(e.getMessage().contains("overloaded"));
        } catch (IOException e) {
            fail("expected EmbeddingServiceException: " + e);
        }
    }

    @Test(expected = IOException.class)
    public void testServiceDown() throws IOException {
        server.stop(0);
        client.getInfo();
    }
}
