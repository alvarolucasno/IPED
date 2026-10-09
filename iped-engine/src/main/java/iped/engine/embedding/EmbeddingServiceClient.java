package iped.engine.embedding;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.http.HttpStatus;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.entity.ContentType;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import iped.engine.config.EmbeddingTaskConfig;

/**
 * Client of the local embedding service (scripts/embedding/embedding_server.py),
 * contract "iped-embedding/v1". Thread safe.
 */
public class EmbeddingServiceClient implements Closeable {

    public static final String CONTRACT = "iped-embedding/v1";

    private static final ObjectMapper mapper = new ObjectMapper();

    private final String baseUrl;
    private final RequestConfig requestConfig;
    private final CloseableHttpClient client;

    public EmbeddingServiceClient(EmbeddingTaskConfig config) {
        this(config.getServiceUrl(), config.getConnectTimeout(), config.getSocketTimeout());
    }

    public EmbeddingServiceClient(String baseUrl, int connectTimeout, int socketTimeout) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.requestConfig = RequestConfig.custom().setConnectTimeout(connectTimeout)
                .setConnectionRequestTimeout(connectTimeout).setSocketTimeout(socketTimeout).build();
        // the service is local by default: never route it through a system proxy
        this.client = HttpClients.custom().setMaxConnPerRoute(64).setMaxConnTotal(64).build();
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    /** One input of an /embed request. */
    public static class Input {
        final String id;
        final String type;
        String title, text;
        byte[] data;
        List<byte[]> frames;

        private Input(String id, String type) {
            this.id = id;
            this.type = type;
        }

        public String getId() {
            return id;
        }

        public String getType() {
            return type;
        }

        /** Approximate request payload size, used to bound batches. */
        public long getPayloadSize() {
            long size = (title != null ? title.length() : 0) + (text != null ? text.length() * 2L : 0);
            if (data != null) {
                size += data.length;
            }
            if (frames != null) {
                for (byte[] f : frames) {
                    size += f.length;
                }
            }
            return size * 4 / 3;
        }

        public static Input document(String id, String title, String text) {
            Input in = new Input(id, "document");
            in.title = title;
            in.text = text;
            return in;
        }

        public static Input query(String id, String text) {
            Input in = new Input(id, "query");
            in.text = text;
            return in;
        }

        public static Input image(String id, byte[] imageFile) {
            Input in = new Input(id, EmbeddingUtil.MODALITY_IMAGE);
            in.data = imageFile;
            return in;
        }

        public static Input videoFrames(String id, List<byte[]> jpegFrames) {
            Input in = new Input(id, EmbeddingUtil.MODALITY_VIDEO);
            in.frames = jpegFrames;
            return in;
        }

        public static Input videoFile(String id, byte[] videoFile) {
            Input in = new Input(id, EmbeddingUtil.MODALITY_VIDEO);
            in.data = videoFile;
            return in;
        }

        public static Input audio(String id, byte[] audioFile) {
            Input in = new Input(id, EmbeddingUtil.MODALITY_AUDIO);
            in.data = audioFile;
            return in;
        }

        ObjectNode toJson() {
            ObjectNode node = mapper.createObjectNode();
            node.put("id", id);
            node.put("type", type);
            if (title != null)
                node.put("title", title);
            if (text != null)
                node.put("text", text);
            if (data != null)
                node.put("data", data); // Jackson writes byte[] as base64
            if (frames != null) {
                ArrayNode array = node.putArray("frames");
                for (byte[] f : frames) {
                    array.add(f);
                }
            }
            return node;
        }
    }

    public static class Result {
        public final Map<String, float[]> vectors = new HashMap<>();
        public final Map<String, String> errors = new HashMap<>();
        public int dim;
        public String model;
    }

    public static class ServiceInfo {
        public String contract, model, device, dtype;
        public int dim;
        public List<String> modalities = new ArrayList<>();

        @Override
        public String toString() {
            return "model=" + model + " dim=" + dim + " modalities=" + modalities + " device=" + device + " dtype=" + dtype;
        }
    }

    public ServiceInfo getInfo() throws IOException {
        HttpGet get = new HttpGet(baseUrl + "/info");
        JsonNode json = execute(get);
        ServiceInfo info = new ServiceInfo();
        info.contract = json.path("contract").asText();
        info.model = json.path("model").asText();
        info.device = json.path("device").asText();
        info.dtype = json.path("dtype").asText();
        info.dim = json.path("dim").asInt();
        for (JsonNode m : json.path("modalities")) {
            info.modalities.add(m.asText());
        }
        if (!CONTRACT.equals(info.contract)) {
            throw new IOException("Incompatible embedding service contract '" + info.contract + "', expected " + CONTRACT);
        }
        return info;
    }

    public Result embed(List<Input> inputs) throws IOException {
        ObjectNode request = mapper.createObjectNode();
        ArrayNode array = request.putArray("inputs");
        for (Input in : inputs) {
            array.add(in.toJson());
        }
        HttpPost post = new HttpPost(baseUrl + "/embed");
        post.setEntity(new ByteArrayEntity(mapper.writeValueAsBytes(request), ContentType.APPLICATION_JSON));
        JsonNode json = execute(post);

        Result result = new Result();
        result.dim = json.path("dim").asInt();
        result.model = json.path("model").asText();
        Iterator<Entry<String, JsonNode>> it = json.path("vectors").fields();
        while (it.hasNext()) {
            Entry<String, JsonNode> e = it.next();
            JsonNode values = e.getValue();
            float[] vec = new float[values.size()];
            for (int i = 0; i < vec.length; i++) {
                vec[i] = (float) values.get(i).asDouble();
            }
            result.vectors.put(e.getKey(), vec);
        }
        it = json.path("errors").fields();
        while (it.hasNext()) {
            Entry<String, JsonNode> e = it.next();
            result.errors.put(e.getKey(), e.getValue().asText());
        }
        return result;
    }

    /** Embeds a natural language search query. */
    public float[] embedQuery(String query) throws IOException {
        return embedOne(Input.query("q", query));
    }

    public float[] embedOne(Input input) throws IOException {
        List<Input> list = new ArrayList<>();
        list.add(input);
        Result result = embed(list);
        float[] vec = result.vectors.get(input.getId());
        if (vec == null) {
            throw new IOException("Embedding service could not embed input: " + result.errors.get(input.getId()));
        }
        return vec;
    }

    private JsonNode execute(HttpRequestBase request) throws IOException {
        request.setConfig(requestConfig);
        try (CloseableHttpResponse response = client.execute(request);
                InputStream is = response.getEntity().getContent()) {
            byte[] body = is.readAllBytes();
            int status = response.getStatusLine().getStatusCode();
            if (status != HttpStatus.SC_OK) {
                String msg = new String(body, StandardCharsets.UTF_8);
                throw new EmbeddingServiceException(status, request.getURI() + " returned HTTP " + status + ": "
                        + (msg.length() > 500 ? msg.substring(0, 500) : msg));
            }
            return mapper.readTree(body);
        }
    }

    public static class EmbeddingServiceException extends IOException {
        private static final long serialVersionUID = 1L;
        private final int statusCode;

        public EmbeddingServiceException(int statusCode, String message) {
            super(message);
            this.statusCode = statusCode;
        }

        public int getStatusCode() {
            return statusCode;
        }
    }

    @Override
    public void close() throws IOException {
        client.close();
    }
}
