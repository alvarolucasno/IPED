package iped.engine.search;

import java.io.IOException;
import java.util.Arrays;
import java.util.Set;

import iped.data.IItemId;
import iped.engine.data.IPEDSource;
import iped.engine.embedding.EmbeddingUtil;

/**
 * Semantic search filter: keeps the items whose embedding stands out for the
 * query vector (from a text query, an external file or a reference item).
 *
 * The score of an item is its standout score (see {@link EmbeddingMatrix}): how
 * many robust standard deviations (x10) its cosine similarity is above the
 * typical item of the same modality in the case. Only items reaching
 * {@code minScore} are kept, so a query with no relevant item in a modality
 * returns nothing for that modality instead of its least unrelated items.
 *
 * Similarities of the whole case are computed once per query by a multithreaded
 * scan over an in memory int8 matrix and cached, so applying the filter to any
 * result set (combined with the other UI filters) is a lookup per item.
 */
public class SemanticSearch {

    /** Default minimum standout score: 3 robust standard deviations. */
    public static final int DEFAULT_MIN_SCORE = 30;

    /** Score of the reference item in a search by item, shown as "REF". */
    public static final float REF_SCORE = 1000;

    private final IPEDSource ipedCase;
    private final float[] query;
    private final boolean[] modalities;
    private final int maxResultsPerModality;
    private final float minScore;
    private IItemId reference;

    /**
     * @param query
     *            normalized query vector
     * @param modalities
     *            modalities to return (see {@link EmbeddingUtil#MODALITIES}), null
     *            for all
     * @param maxResultsPerModality
     *            best results kept per modality
     * @param minScore
     *            minimum standout score (10 x robust z-score)
     */
    public SemanticSearch(IPEDSource ipedCase, float[] query, Set<String> modalities, int maxResultsPerModality,
            float minScore) {
        this.ipedCase = ipedCase;
        this.query = query;
        this.modalities = new boolean[EmbeddingUtil.MODALITIES.length];
        for (int i = 0; i < this.modalities.length; i++) {
            this.modalities[i] = modalities == null || modalities.contains(EmbeddingUtil.MODALITIES[i]);
        }
        this.maxResultsPerModality = maxResultsPerModality;
        this.minScore = minScore;
    }

    /** Marks the reference item of a search by item, it is always kept, as "REF". */
    public SemanticSearch setReference(IItemId reference) {
        this.reference = reference;
        return this;
    }

    /** Loads the case embeddings in background, so the first search is fast. */
    public static void warmUp(IPEDSource ipedCase) {
        Thread t = new Thread("SemanticSearchWarmUp") {
            @Override
            public void run() {
                try {
                    EmbeddingMatrix.get(ipedCase.getReader());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        t.setDaemon(true);
        t.start();
    }

    public MultiSearchResult search() throws IOException {
        return filter(new IPEDSearcher(ipedCase, "*:*").multiSearch());
    }

    public MultiSearchResult filter(MultiSearchResult result) throws IOException {
        EmbeddingMatrix matrix = EmbeddingMatrix.get(ipedCase.getReader());
        if (matrix == null) {
            return new MultiSearchResult(new IItemId[0], new float[0]);
        }
        float[] scores = matrix.scores(query);
        int refSlot = reference != null ? matrix.slotOf(ipedCase.getLuceneId(reference)) : -1;

        // candidates above the cut, packed as (score bits, result index) to sort primitives
        int len = result.getLength();
        long[] candidates = new long[Math.min(len, matrix.size())];
        int n = 0;
        for (int i = 0; i < len; i++) {
            int slot = matrix.slotOf(ipedCase.getLuceneId(result.getItem(i)));
            if (slot < 0 || !modalities[matrix.modality[slot]]) {
                continue;
            }
            float score = slot == refSlot ? REF_SCORE : scores[slot];
            if (score >= minScore) {
                // scores may be negative: flip the sign bit order so longs sort like floats
                int bits = Float.floatToIntBits(score);
                bits ^= (bits >> 31) & 0x7fffffff;
                candidates[n++] = ((long) bits << 32) | i;
            }
        }
        Arrays.sort(candidates, 0, n);

        int[] perModality = new int[EmbeddingUtil.MODALITIES.length];
        IItemId[] ids = new IItemId[n];
        float[] selScores = new float[n];
        int k = 0;
        for (int c = n - 1; c >= 0; c--) {
            int i = (int) candidates[c];
            IItemId itemId = result.getItem(i);
            int slot = matrix.slotOf(ipedCase.getLuceneId(itemId));
            int m = matrix.modality[slot];
            if (perModality[m] >= maxResultsPerModality && slot != refSlot) {
                continue;
            }
            perModality[m]++;
            ids[k] = itemId;
            selScores[k++] = slot == refSlot ? REF_SCORE : scores[slot];
        }
        return new MultiSearchResult(Arrays.copyOf(ids, k), Arrays.copyOf(selScores, k));
    }
}
