package iped.engine.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.util.BytesRef;

import iped.data.IItemId;
import iped.engine.data.IPEDMultiSource;
import iped.engine.data.IPEDSource;
import iped.engine.embedding.EmbeddingUtil;

/**
 * Ranks items by cosine similarity between their embedding and a query vector
 * (from a text query, an external file or a reference item). Works on any
 * result set, so it composes with the other UI filters, and on multicases.
 *
 * Similarities of different modalities have different ranges (text-to-text is
 * usually higher than text-to-image), so the best results are kept per
 * modality instead of globally, letting images, videos and audio show up
 * together with documents.
 */
public class SemanticSearch {

    private final IPEDMultiSource ipedCase;
    private final float[] query;
    private final boolean[] modalities;
    private final int maxResultsPerModality;
    private final float minScore;

    /**
     * @param query
     *            normalized query vector
     * @param modalities
     *            modalities to return (see {@link EmbeddingUtil#MODALITIES}), null
     *            for all
     * @param maxResultsPerModality
     *            best results kept per modality
     * @param minScore
     *            minimum cosine similarity x 100
     */
    public SemanticSearch(IPEDSource ipedCase, float[] query, Set<String> modalities, int maxResultsPerModality,
            float minScore) {
        this.ipedCase = ipedCase instanceof IPEDMultiSource ? (IPEDMultiSource) ipedCase
                : new IPEDMultiSource(Collections.singletonList(ipedCase));
        this.query = query;
        this.modalities = new boolean[EmbeddingUtil.MODALITIES.length];
        for (int i = 0; i < this.modalities.length; i++) {
            this.modalities[i] = modalities == null || modalities.contains(EmbeddingUtil.MODALITIES[i]);
        }
        this.maxResultsPerModality = maxResultsPerModality;
        this.minScore = minScore;
    }

    public MultiSearchResult search() throws IOException {
        IPEDSearcher searcher = new IPEDSearcher(ipedCase, "*:*");
        return filter(searcher.multiSearch());
    }

    public MultiSearchResult filter(MultiSearchResult result) throws IOException {
        int len = result.getLength();
        float[] scores = new float[len];
        byte[] modality = new byte[len];
        Arrays.fill(modality, (byte) -1);

        // doc values iterators only move forward: visit items in lucene id order
        long[] order = new long[len];
        for (int i = 0; i < len; i++) {
            order[i] = ((long) ipedCase.getLuceneId(result.getItem(i)) << 32) | i;
        }
        Arrays.sort(order);

        score(order, scores, modality);
        return selectBest(result, scores, modality);
    }

    private void score(long[] order, float[] scores, byte[] modality) throws IOException {
        LeafReader leafReader = ipedCase.getLeafReader();
        int numThreads = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), order.length / 2000 + 1));
        int perThread = (order.length + numThreads - 1) / numThreads;
        Thread[] threads = new Thread[numThreads];
        IOException[] error = new IOException[1];
        for (int k = 0; k < numThreads; k++) {
            int i0 = Math.min(order.length, k * perThread);
            int i1 = Math.min(order.length, i0 + perThread);
            threads[k] = new Thread("SemanticSearch-" + k) {
                @Override
                public void run() {
                    try {
                        // each thread needs its own doc values iterators
                        SortedSetDocValues vectors = leafReader.getSortedSetDocValues(EmbeddingUtil.EMBEDDING);
                        SortedDocValues modalities = leafReader.getSortedDocValues(EmbeddingUtil.EMBEDDING_MODALITY);
                        if (vectors == null) {
                            return;
                        }
                        int[] ordToModality = modalities != null ? mapModalityOrds(modalities) : null;
                        for (int j = i0; j < i1; j++) {
                            if ((j & 1023) == 0 && isInterrupted()) {
                                return;
                            }
                            int luceneId = (int) (order[j] >>> 32);
                            int i = (int) order[j];
                            if (!vectors.advanceExact(luceneId)) {
                                continue;
                            }
                            int m = 0; // items embedded before modality was stored are treated as text
                            if (modalities != null && modalities.advanceExact(luceneId)) {
                                m = ordToModality[modalities.ordValue()];
                            }
                            if (m < 0 || !SemanticSearch.this.modalities[m]) {
                                continue;
                            }
                            BytesRef bytes = vectors.lookupOrd(vectors.nextOrd());
                            float[] vec = EmbeddingUtil.fromBytes(bytes.bytes, bytes.offset, bytes.length);
                            if (vec.length != query.length) {
                                continue;
                            }
                            float score = EmbeddingUtil.dot(query, vec) * 100;
                            if (score >= minScore) {
                                scores[i] = score;
                                modality[i] = (byte) m;
                            }
                        }
                    } catch (IOException e) {
                        error[0] = e;
                    }
                }
            };
            threads[k].start();
        }
        try {
            for (Thread t : threads) {
                t.join();
            }
        } catch (InterruptedException e) {
            for (Thread t : threads) {
                t.interrupt();
            }
            Thread.currentThread().interrupt();
        }
        if (error[0] != null) {
            throw error[0];
        }
    }

    private static int[] mapModalityOrds(SortedDocValues modalities) throws IOException {
        int[] map = new int[modalities.getValueCount()];
        for (int ord = 0; ord < map.length; ord++) {
            map[ord] = EmbeddingUtil.modalityIndex(modalities.lookupOrd(ord).utf8ToString());
        }
        return map;
    }

    private MultiSearchResult selectBest(MultiSearchResult result, float[] scores, byte[] modality) {
        List<Integer> selected = new ArrayList<>();
        for (int m = 0; m < EmbeddingUtil.MODALITIES.length; m++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < scores.length; i++) {
                if (modality[i] == m) {
                    candidates.add(i);
                }
            }
            candidates.sort((a, b) -> Float.compare(scores[b], scores[a]));
            selected.addAll(candidates.subList(0, Math.min(maxResultsPerModality, candidates.size())));
        }
        selected.sort((a, b) -> Float.compare(scores[b], scores[a]));
        IItemId[] ids = new IItemId[selected.size()];
        float[] selScores = new float[selected.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = result.getItem(selected.get(i));
            selScores[i] = scores[selected.get(i)];
        }
        return new MultiSearchResult(ids, selScores);
    }
}
