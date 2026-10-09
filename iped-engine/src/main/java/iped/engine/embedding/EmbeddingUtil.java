package iped.engine.embedding;

import java.nio.ByteBuffer;
import java.util.Collection;

import iped.data.IItem;
import iped.engine.task.index.IndexItem;
import iped.engine.task.index.IndexItem.KnnVector;

/**
 * Attribute names and vector conversions shared by processing and search.
 * Vectors are L2-normalized, so the dot product is the cosine similarity.
 */
public class EmbeddingUtil {

    /**
     * Item vector. Stored as a {@link KnnVector}, so IndexItem writes it to
     * SortedSetDocValues (used by semantic search), to a StoredField and to a
     * Lucene KnnVectorField (HNSW).
     */
    public static final String EMBEDDING = "embedding";

    /** Modality of the item vector: text, image, video or audio. */
    public static final String EMBEDDING_MODALITY = "embedding:modality";

    /** Set only when the item was eligible but could not be embedded. */
    public static final String EMBEDDING_STATUS = "embedding:status";

    public static final String MODALITY_TEXT = "text";
    public static final String MODALITY_IMAGE = "image";
    public static final String MODALITY_VIDEO = "video";
    public static final String MODALITY_AUDIO = "audio";

    public static final String[] MODALITIES = { MODALITY_TEXT, MODALITY_IMAGE, MODALITY_VIDEO, MODALITY_AUDIO };

    public static int modalityIndex(String modality) {
        if (modality != null) {
            for (int i = 0; i < MODALITIES.length; i++) {
                if (MODALITIES[i].equalsIgnoreCase(modality)) {
                    return i;
                }
            }
        }
        return -1;
    }

    public static KnnVector toKnnVector(float[] vector) {
        double[] array = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            array[i] = vector[i];
        }
        return new KnnVector(array);
    }

    /** Big endian float32, the same layout IndexItem uses for KnnVector doc values. */
    public static float[] fromBytes(byte[] bytes, int offset, int length) {
        float[] result = new float[length / 4];
        ByteBuffer bb = ByteBuffer.wrap(bytes, offset, length);
        for (int i = 0; i < result.length; i++) {
            result[i] = bb.getFloat();
        }
        return result;
    }

    public static byte[] toBytes(float[] vector) {
        return IndexItem.convFloatArrayToByteArray(vector);
    }

    /**
     * Returns the item vector whether it is still a KnnVector (during processing)
     * or bytes loaded back from the index, or null if the item has none.
     */
    public static float[] getVector(IItem item) {
        return item == null ? null : toFloatArray(item.getExtraAttribute(EMBEDDING));
    }

    public static float[] toFloatArray(Object value) {
        if (value instanceof Collection) {
            Collection<?> c = (Collection<?>) value;
            value = c.isEmpty() ? null : c.iterator().next();
        }
        if (value instanceof float[]) {
            return (float[]) value;
        }
        if (value instanceof KnnVector) {
            return IndexItem.convDoubleToFloatArray(((KnnVector) value).getArray());
        }
        if (value instanceof byte[]) {
            byte[] bytes = (byte[]) value;
            return fromBytes(bytes, 0, bytes.length);
        }
        return null;
    }

    public static float dot(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        for (; i < n; i++) {
            s0 += a[i] * b[i];
        }
        return s0 + s1 + s2 + s3;
    }

    public static float[] normalize(float[] v) {
        double norm = 0;
        for (float x : v) {
            norm += x * x;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
        }
        return v;
    }
}
