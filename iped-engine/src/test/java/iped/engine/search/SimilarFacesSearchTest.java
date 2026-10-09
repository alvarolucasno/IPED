package iped.engine.search;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import iped.engine.data.Item;
import iped.engine.task.index.IndexItem;
import iped.engine.task.index.IndexItem.KnnVector;

public class SimilarFacesSearchTest {

    private static float[] unit(int dim, int... hot) {
        float[] v = new float[dim];
        for (int h : hot) {
            v[h] = 1;
        }
        float n = (float) Math.sqrt(hot.length);
        for (int i = 0; i < dim; i++) {
            v[i] /= n;
        }
        return v;
    }

    @Test
    public void testDlibUsesEuclideanDistance() {
        float[] a = new float[128];
        float[] b = new float[128];
        b[0] = 0.3f; // distance 0.3 -> score 70
        assertEquals(70, SimilarFacesSearch.score(a, b), 1e-3);
        b[0] = 2f; // distance above 1 -> 0
        assertEquals(0, SimilarFacesSearch.score(a, b), 1e-3);
    }

    @Test
    public void testArcFaceUsesCosine() {
        assertEquals(100, SimilarFacesSearch.score(unit(512, 0), unit(512, 0)), 1e-3);
        // cos = 1/sqrt(2)
        assertEquals(70.71, SimilarFacesSearch.score(unit(512, 0), unit(512, 0, 1)), 0.01);
        // negative similarity is clamped to 0
        float[] neg = unit(512, 0);
        neg[0] = -1;
        assertEquals(0, SimilarFacesSearch.score(unit(512, 0), neg), 1e-3);
    }

    @Test
    public void testDifferentModelsAreNotCompared() {
        assertEquals(-1, SimilarFacesSearch.score(new float[128], new float[512]), 0);
    }

    @Test
    public void testModelDetectionAndDefaults() {
        Item arcface = new Item();
        arcface.setExtraAttribute(SimilarFacesSearch.FACE_FEATURES,
                Arrays.asList(new KnnVector(new double[512]), new KnnVector(new double[512])));
        assertTrue(SimilarFacesSearch.isCosineModel(arcface));
        assertEquals(SimilarFacesSearch.DEFAULT_MIN_SCORE_COSINE, SimilarFacesSearch.getDefaultMinScore(arcface));

        // faces loaded from the index are byte[]
        Item dlib = new Item();
        dlib.setExtraAttribute(SimilarFacesSearch.FACE_FEATURES,
                Arrays.asList(IndexItem.convFloatArrayToByteArray(new float[128])));
        assertFalse(SimilarFacesSearch.isCosineModel(dlib));
        assertEquals(SimilarFacesSearch.DEFAULT_MIN_SCORE_DLIB, SimilarFacesSearch.getDefaultMinScore(dlib));
    }

    @Test
    public void testMatchLocationsWithArcFace() {
        Item ref = new Item();
        ref.setExtraAttribute(SimilarFacesSearch.FACE_FEATURES,
                Arrays.asList(IndexItem.convFloatArrayToByteArray(unit(512, 7))));
        Item group = new Item();
        group.setExtraAttribute(SimilarFacesSearch.FACE_FEATURES,
                Arrays.asList(IndexItem.convFloatArrayToByteArray(unit(512, 1)),
                        IndexItem.convFloatArrayToByteArray(unit(512, 7, 8))));
        group.setExtraAttribute(SimilarFacesSearch.FACE_LOCATIONS, Arrays.asList("[1, 2, 3, 4]", "[5, 6, 7, 8]"));
        SimilarFacesSearch.setSelectedIdxs(null);
        SimilarFacesSearch.setMinScore(SimilarFacesSearch.DEFAULT_MIN_SCORE_COSINE);
        List<String> matches = SimilarFacesSearch.getMatchLocations(ref, group);
        // second face has cos 0.71 with the reference, first one 0
        assertEquals(Arrays.asList("[5, 6, 7, 8]"), matches);
    }
}
