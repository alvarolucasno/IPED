package iped.engine.search;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.util.BytesRef;

import iped.data.IItem;
import iped.data.IItemId;
import iped.engine.data.IPEDMultiSource;
import iped.engine.data.IPEDSource;
import iped.engine.task.index.IndexItem.KnnVector;

/**
 * Similar faces search. Face embeddings come from FaceRecognitionTask and the
 * metric depends on the model that produced them:
 * <ul>
 * <li>dlib (128-d): euclidean distance, score = (1 - distance) x 100;</li>
 * <li>InsightFace buffalo_l ArcFace (512-d, L2 normalized): cosine similarity,
 * score = cosine x 100.</li>
 * </ul>
 * Faces are only compared with faces of the same dimension (model).
 */
public class SimilarFacesSearch {

    public static final String FACE_FEATURES = "face_encodings";
    public static final String FACE_LOCATIONS = "face_locations";

    /** Embedding size of the legacy dlib model. */
    public static final int DLIB_DIMENSION = 128;

    /** Default minimum scores: dlib distance 0.5, ArcFace cosine 0.4. */
    public static final int DEFAULT_MIN_SCORE_DLIB = 50;
    public static final int DEFAULT_MIN_SCORE_COSINE = 40;

    private static float minScore = DEFAULT_MIN_SCORE_DLIB;
    private static int mode = 0; // Mode 0 = OR, Mode 1 = AND
    private static Set<Integer> selectedIdxs;

    private IPEDMultiSource ipedCase;
    private float[][] refSimilarityFeatures;

    public SimilarFacesSearch(IPEDSource ipedCase, IItem refImage) {
        this.ipedCase = ipedCase instanceof IPEDMultiSource ? (IPEDMultiSource) ipedCase
                : new IPEDMultiSource(Collections.singletonList(ipedCase));
        this.refSimilarityFeatures = getFaceFeatures(refImage, selectedIdxs);
    }

    public MultiSearchResult search() throws IOException {
        IPEDSearcher searcher = new IPEDSearcher(ipedCase, "*:*");
        MultiSearchResult result = searcher.multiSearch();
        return filter(result);
    }

    public MultiSearchResult filter(MultiSearchResult result) throws IOException {
        score(result);
        // filter keeps scores strictly greater than the threshold
        return ImageSimilarityLowScoreFilter.filter(result, Math.max(0, minScore - 1e-3f));
    }

    public static final int getMinScore() {
        return Math.round(minScore);
    }

    public static final void setMinScore(int minScore) {
        SimilarFacesSearch.minScore = minScore;
    }

    public static final void setMode(int mode) {
        SimilarFacesSearch.mode = mode;
    }

    public static final void setSelectedIdxs(Set<Integer> selectedIdxs) {
        SimilarFacesSearch.selectedIdxs = selectedIdxs;
    }

    /** True if the item faces are compared by cosine similarity (not dlib). */
    public static boolean isCosineModel(IItem item) {
        float[][] faces = getFaceFeatures(item, null);
        return faces.length > 0 && isCosine(faces[0].length);
    }

    public static int getDefaultMinScore(IItem item) {
        return isCosineModel(item) ? DEFAULT_MIN_SCORE_COSINE : DEFAULT_MIN_SCORE_DLIB;
    }

    static boolean isCosine(int dimension) {
        return dimension != DLIB_DIMENSION;
    }

    /**
     * Similarity score (0-100) between two faces of the same model, or -1 if they
     * can not be compared (different models).
     */
    static float score(float[] a, float[] b) {
        if (a.length != b.length) {
            return -1;
        }
        if (isCosine(a.length)) {
            float dot = 0;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
            }
            return Math.max(0, dot * 100);
        }
        return squaredDistToScore(distance(a, b, Float.MAX_VALUE));
    }

    private static float squaredDistToScore(float squaredDist) {
        return Math.max(0, (1 - (float) Math.sqrt(squaredDist)) * 100);
    }

    private void score(MultiSearchResult result) throws IOException {

        LeafReader leafReader = ipedCase.getLeafReader();
        int numThreads = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[numThreads];
        int len = result.getLength();
        int itemsPerThread = (len + numThreads - 1) / numThreads;
        for (int k = 0; k < numThreads; k++) {
            int threadIdx = k;
            (threads[k] = new Thread() {
                public void run() {
                    SortedSetDocValues similarityFeaturesValues = null;
                    try {
                        similarityFeaturesValues = leafReader.getSortedSetDocValues(FACE_FEATURES);
                    } catch (IOException e) {
                        e.printStackTrace();
                        return;
                    }
                    if (similarityFeaturesValues == null) {
                        return;
                    }
                    int i0 = Math.min(len, itemsPerThread * threadIdx);
                    int i1 = Math.min(len, i0 + itemsPerThread);
                    int numRefFaces = refSimilarityFeatures.length;
                    float[] bestPerRefFace = new float[numRefFaces];
                    for (int i = i0; i < i1; i++) {
                        if (i % 1000 == 0 && this.isInterrupted()) {
                            return;
                        }
                        IItemId itemId = result.getItem(i);
                        int luceneId = ipedCase.getLuceneId(itemId);
                        long ordinal;
                        float score = 0;
                        Arrays.fill(bestPerRefFace, 0);
                        try {
                            boolean hasVal = similarityFeaturesValues.advanceExact(luceneId);
                            while (hasVal && (ordinal = similarityFeaturesValues
                                    .nextOrd()) != SortedSetDocValues.NO_MORE_ORDS) {
                                BytesRef bytesRef = similarityFeaturesValues.lookupOrd(ordinal);
                                float[] currentFeatures = convToFloatVec(bytesRef.bytes, bytesRef.offset,
                                        bytesRef.length);
                                // each face of the item is assigned to its most similar reference face
                                int faceIdx = -1;
                                float best = 0;
                                for (int j = 0; j < numRefFaces; j++) {
                                    float s = score(refSimilarityFeatures[j], currentFeatures);
                                    if (s > best) {
                                        best = s;
                                        faceIdx = j;
                                    }
                                }
                                if (faceIdx != -1 && best > bestPerRefFace[faceIdx]) {
                                    bestPerRefFace[faceIdx] = best;
                                }
                            }
                            if (numRefFaces > 0) {
                                if (mode == 0) {
                                    // OR mode: any reference face
                                    for (float s : bestPerRefFace) {
                                        score = Math.max(score, s);
                                    }
                                } else {
                                    // AND mode: all reference faces
                                    score = Float.MAX_VALUE;
                                    for (float s : bestPerRefFace) {
                                        score = Math.min(score, s);
                                    }
                                }
                                if (score < minScore) {
                                    score = 0;
                                }
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        } finally {
                            result.setScore(i, score);
                        }
                    }
                }
            }).start();
        }
        boolean canceled = false;
        for (Thread thread : threads) {
            if (thread != null) {
                if (!canceled) {
                    try {
                        thread.join();
                    } catch (InterruptedException e) {
                        canceled = true;
                    }
                }
                if (canceled) {
                    thread.interrupt();
                }
            }
        }

    }

    private static float[] convToFloatVec(byte[] bytes, int offset, int length) {
        float[] result = new float[length / 4];
        ByteBuffer bb = ByteBuffer.wrap(bytes, offset, length);
        for (int i = 0; i < result.length; i++) {
            result[i] = bb.getFloat();
        }
        return result;
    }

    private static float[] convToFloatVec(Object value) {
        if (value instanceof byte[]) {
            byte[] bytes = (byte[]) value;
            return convToFloatVec(bytes, 0, bytes.length);
        }
        if (value instanceof KnnVector) {
            double[] array = ((KnnVector) value).getArray();
            float[] result = new float[array.length];
            for (int i = 0; i < array.length; i++) {
                result[i] = (float) array[i];
            }
            return result;
        }
        return null;
    }

    public static float distance(float[] a, float[] b, float cut) {
        float distance = 0;
        for (int i = 0; i < a.length && distance <= cut;) {
            float d = a[i] - b[i++];
            distance += d * d + (d = a[i] - b[i++]) * d + (d = a[i] - b[i++]) * d + (d = a[i] - b[i++]) * d;
        }
        return distance;
    }

    private static float[][] getFaceFeatures(IItem item, Set<Integer> idxs) {
        if (item == null) {
            return new float[0][0];
        }
        Object value = item.getExtraAttribute(FACE_FEATURES);
        if (value == null) {
            return new float[0][0];
        }
        if (value instanceof Collection) {
            List<float[]> l = new ArrayList<float[]>();
            Iterator<?> it = ((Collection<?>) value).iterator();
            int idx = 0;
            while (it.hasNext()) {
                Object o = it.next();
                if (idxs == null || idxs.isEmpty() || idxs.contains(idx)) {
                    float[] f = convToFloatVec(o);
                    if (f != null) {
                        l.add(f);
                    }
                }
                idx++;
            }
            return l.toArray(new float[0][]);
        }
        float[] f = convToFloatVec(value);
        return f == null ? new float[0][0] : new float[][] { f };
    }

    public static List<String> getMatchLocations(IItem refItem, IItem matchItem) {
        ArrayList<String> matchLocations = new ArrayList<>();
        Object location = matchItem.getExtraAttribute(SimilarFacesSearch.FACE_LOCATIONS);
        if (location instanceof List) {
            float[][] refFeatures = getFaceFeatures(refItem, selectedIdxs);
            float[][] matchFeatures = getFaceFeatures(matchItem, null);
            for (int i = 0; i < matchFeatures.length; i++) {
                float[] mi = matchFeatures[i];
                for (int j = 0; j < refFeatures.length; j++) {
                    if (score(mi, refFeatures[j]) >= minScore) {
                        matchLocations.add((String) ((List<?>) location).get(i));
                        break;
                    }
                }
            }
        } else {
            matchLocations.add((String) location);
        }
        return matchLocations;
    }
}
