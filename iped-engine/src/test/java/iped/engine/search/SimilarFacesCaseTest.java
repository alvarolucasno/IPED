package iped.engine.search;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import iped.data.IItem;
import iped.data.IItemId;
import iped.engine.data.IPEDMultiSource;
import iped.engine.data.IPEDSource;
import iped.properties.ExtraProperties;

/**
 * End-to-end test of face recognition with InsightFace buffalo_l over a case
 * processed with enableFaceRecognition=true from photos of two people: two
 * portraits of person A, two of person B and one photo with A facing the camera
 * and B turned back. Opt-in:
 *
 * mvn -pl iped-engine test -Dtest=SimilarFacesCaseTest -Diped.facecase=C:\path\to\case
 */
public class SimilarFacesCaseTest {

    private static IPEDMultiSource ipedCase;
    private static Map<String, IItemId> ids = new HashMap<>();

    @BeforeClass
    public static void open() throws IOException {
        String casePath = System.getProperty("iped.facecase");
        assumeTrue("set -Diped.facecase to run", casePath != null);
        ipedCase = new IPEDMultiSource(Collections.singletonList(new IPEDSource(new File(casePath))));
        MultiSearchResult all = new IPEDSearcher(ipedCase, "*:*").multiSearch();
        for (int i = 0; i < all.getLength(); i++) {
            ids.put(ipedCase.getItemByItemId(all.getItem(i)).getName(), all.getItem(i));
        }
    }

    @AfterClass
    public static void close() {
        if (ipedCase != null) {
            ipedCase.close();
        }
    }

    private static IItem item(String name) {
        IItemId id = ids.get(name);
        assertNotNull(name, id);
        return ipedCase.getItemByItemId(id);
    }

    private static Map<String, Float> similarTo(String name) throws IOException {
        SimilarFacesSearch.setSelectedIdxs(null);
        SimilarFacesSearch.setMode(0);
        SimilarFacesSearch.setMinScore(SimilarFacesSearch.DEFAULT_MIN_SCORE_COSINE);
        MultiSearchResult result = new SimilarFacesSearch(ipedCase, item(name)).search();
        Map<String, Float> found = new HashMap<>();
        System.out.println("similar to " + name);
        for (int i = 0; i < result.getLength(); i++) {
            String n = ipedCase.getItemByItemId(result.getItem(i)).getName();
            found.put(n, result.getScore(i));
            System.out.printf("  %5.1f %s%n", result.getScore(i), n);
        }
        return found;
    }

    @Test
    public void testFacesAreDetectedWithArcFaceEmbeddings() {
        for (String name : new String[] { "pessoaA_1.jpg", "pessoaA_2.jpg", "pessoaB_1.jpg", "pessoaB_2.jpg",
                "grupo_AB.jpg" }) {
            IItem item = item(name);
            Object count = item.getExtraAttribute(ExtraProperties.FACE_COUNT);
            assertEquals(name, 1, ((Number) count).intValue());
            assertTrue(name, SimilarFacesSearch.isCosineModel(item));
            Object features = item.getExtraAttribute(SimilarFacesSearch.FACE_FEATURES);
            Object one = features instanceof Collection ? ((Collection<?>) features).iterator().next() : features;
            assertEquals(name, 512 * 4, ((byte[]) one).length);
        }
    }

    @Test
    public void testSamePersonIsFoundAndOtherPersonIsNot() throws IOException {
        Map<String, Float> a = similarTo("pessoaA_1.jpg");
        assertTrue(a.containsKey("pessoaA_2.jpg"));
        assertTrue(a.containsKey("grupo_AB.jpg")); // A facing the camera in a group photo
        assertFalse(a.containsKey("pessoaB_1.jpg"));
        assertFalse(a.containsKey("pessoaB_2.jpg"));

        Map<String, Float> b = similarTo("pessoaB_1.jpg");
        assertTrue(b.containsKey("pessoaB_2.jpg"));
        assertFalse(b.containsKey("pessoaA_1.jpg"));
        assertFalse(b.containsKey("grupo_AB.jpg")); // B is turned back in the group photo
    }
}
