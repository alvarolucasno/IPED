package iped.engine.embedding;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.Arrays;

import org.junit.Test;

import iped.engine.data.Item;
import iped.engine.task.index.IndexItem.KnnVector;

public class EmbeddingUtilTest {

    private static final float EPS = 1e-6f;

    @Test
    public void testBytesRoundTripMatchesIndexLayout() {
        float[] v = { 0.5f, -0.25f, 1e-3f, 0f, 0.125f };
        byte[] bytes = EmbeddingUtil.toBytes(v);
        assertEquals(20, bytes.length);
        assertArrayEquals(v, EmbeddingUtil.fromBytes(bytes, 0, bytes.length), EPS);

        // doc values BytesRef may point into a larger shared buffer
        byte[] shared = new byte[bytes.length + 7];
        System.arraycopy(bytes, 0, shared, 3, bytes.length);
        assertArrayEquals(v, EmbeddingUtil.fromBytes(shared, 3, bytes.length), EPS);
    }

    @Test
    public void testToFloatArrayAcceptsAllStoredForms() {
        float[] v = { 0.6f, 0.8f };
        KnnVector knn = EmbeddingUtil.toKnnVector(v);
        assertArrayEquals(v, EmbeddingUtil.toFloatArray(knn), EPS);
        assertArrayEquals(v, EmbeddingUtil.toFloatArray(EmbeddingUtil.toBytes(v)), EPS);
        assertArrayEquals(v, EmbeddingUtil.toFloatArray(Arrays.asList(EmbeddingUtil.toBytes(v))), EPS);
        assertArrayEquals(v, EmbeddingUtil.toFloatArray(v), EPS);
        assertNull(EmbeddingUtil.toFloatArray(null));
        assertNull(EmbeddingUtil.toFloatArray("not a vector"));
    }

    @Test
    public void testGetVectorFromItem() {
        Item item = new Item();
        assertNull(EmbeddingUtil.getVector(item));
        item.setExtraAttribute(EmbeddingUtil.EMBEDDING, EmbeddingUtil.toKnnVector(new float[] { 1, 0 }));
        assertArrayEquals(new float[] { 1, 0 }, EmbeddingUtil.getVector(item), EPS);
    }

    @Test
    public void testDotIsCosineForNormalizedVectors() {
        float[] a = EmbeddingUtil.normalize(new float[] { 3, 4, 0, 0, 0 });
        float[] b = EmbeddingUtil.normalize(new float[] { 3, 4, 0, 0, 0 });
        float[] c = EmbeddingUtil.normalize(new float[] { 0, 0, 1, 1, 1 });
        assertEquals(1f, EmbeddingUtil.dot(a, b), EPS);
        assertEquals(0f, EmbeddingUtil.dot(a, c), EPS);
        // odd length exercises the unrolled loop tail
        assertEquals(0.6f, EmbeddingUtil.dot(a, new float[] { 1, 0, 0, 0, 0 }), EPS);
    }

    @Test
    public void testModalityIndex() {
        for (int i = 0; i < EmbeddingUtil.MODALITIES.length; i++) {
            assertEquals(i, EmbeddingUtil.modalityIndex(EmbeddingUtil.MODALITIES[i]));
        }
        assertEquals(1, EmbeddingUtil.modalityIndex("IMAGE"));
        assertEquals(-1, EmbeddingUtil.modalityIndex("smell"));
        assertEquals(-1, EmbeddingUtil.modalityIndex(null));
    }
}
