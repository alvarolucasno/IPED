package iped.engine.search;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.util.Random;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;
import org.junit.Test;

import iped.engine.embedding.EmbeddingUtil;

public class EmbeddingMatrixTest {

    private static final int DIM = 64;
    private static final byte TEXT = 0, IMAGE = 1, VIDEO = 2;

    /** Unit vector whose cosine with e0 (the query) is exactly c. */
    private static float[] withCosine(double c, Random rnd) {
        float[] v = new float[DIM];
        double norm = 0;
        for (int i = 1; i < DIM; i++) {
            v[i] = (float) rnd.nextGaussian();
            norm += v[i] * v[i];
        }
        double s = Math.sqrt(1 - c * c) / Math.sqrt(norm);
        for (int i = 1; i < DIM; i++) {
            v[i] *= s;
        }
        v[0] = (float) c;
        return v;
    }

    private static float[] query() {
        float[] q = new float[DIM];
        q[0] = 1;
        return q;
    }

    @Test
    public void testHitStandsOutAndUnrelatedItemsDoNot() throws IOException {
        Random rnd = new Random(42);
        int n = 2001;
        float[][] vectors = new float[n][];
        byte[] mods = new byte[n];
        // texts: unrelated cosine ~ N(0.60, 0.02); images: unrelated ~ N(0.55, 0.015)
        for (int i = 0; i < 1000; i++) {
            vectors[i] = withCosine(0.60 + 0.02 * rnd.nextGaussian(), rnd);
            mods[i] = TEXT;
        }
        for (int i = 1000; i < 2000; i++) {
            vectors[i] = withCosine(0.55 + 0.015 * rnd.nextGaussian(), rnd);
            mods[i] = IMAGE;
        }
        // the only relevant image: cosine 0.65, lower than many unrelated texts
        vectors[2000] = withCosine(0.65, rnd);
        mods[2000] = IMAGE;

        EmbeddingMatrix matrix = new EmbeddingMatrix(vectors, mods);
        float[] scores = matrix.scores(query());

        // (0.65 - 0.55) / 0.015 = 6.7 deviations
        assertEquals(67, scores[2000], 4);
        int textsAboveCut = 0, imagesAboveCut = 0;
        for (int i = 0; i < 2000; i++) {
            if (scores[i] >= SemanticSearch.DEFAULT_MIN_SCORE) {
                if (mods[i] == TEXT)
                    textsAboveCut++;
                else
                    imagesAboveCut++;
            }
        }
        // a normal tail above 3 deviations is ~0.13%: about one item per thousand
        assertTrue("texts above cut: " + textsAboveCut, textsAboveCut <= 4);
        assertTrue("images above cut: " + imagesAboveCut, imagesAboveCut <= 4);

        // robust center and spread per modality: a text at the text mean scores ~0
        assertEquals(0, scoreOf(matrix, withCosine(0.60, rnd), TEXT, vectors, mods), 2);
    }

    private static float scoreOf(EmbeddingMatrix base, float[] extra, byte modality, float[][] vectors, byte[] mods)
            throws IOException {
        float[][] v = new float[vectors.length + 1][];
        System.arraycopy(vectors, 0, v, 0, vectors.length);
        v[vectors.length] = extra;
        byte[] m = new byte[mods.length + 1];
        System.arraycopy(mods, 0, m, 0, mods.length);
        m[mods.length] = modality;
        return new EmbeddingMatrix(v, m).scores(query())[vectors.length];
    }

    @Test
    public void testSmallModalityUsesPooledSpread() throws IOException {
        Random rnd = new Random(7);
        int n = 502;
        float[][] vectors = new float[n][];
        byte[] mods = new byte[n];
        for (int i = 0; i < 500; i++) {
            vectors[i] = withCosine(0.60 + 0.02 * rnd.nextGaussian(), rnd);
        }
        // two videos only: own spread would be meaningless
        vectors[500] = withCosine(0.70, rnd);
        vectors[501] = withCosine(0.50, rnd);
        mods[500] = mods[501] = VIDEO;
        float[] scores = new EmbeddingMatrix(vectors, mods).scores(query());
        // center = 0.60 (median of the two), spread = pooled ~0.02 -> +-5 deviations
        assertEquals(50, scores[500], 6);
        assertEquals(-50, scores[501], 6);
    }

    @Test
    public void testLargeCaseUsesHistogramStats() throws IOException {
        Random rnd = new Random(11);
        int n = 9003;
        float[][] vectors = new float[n][];
        byte[] mods = new byte[n];
        for (int i = 0; i < 6000; i++) {
            vectors[i] = withCosine(0.60 + 0.02 * rnd.nextGaussian(), rnd);
        }
        for (int i = 6000; i < 9000; i++) {
            vectors[i] = withCosine(0.55 + 0.015 * rnd.nextGaussian(), rnd);
            mods[i] = IMAGE;
        }
        vectors[9000] = withCosine(0.70, rnd); // text hit: +5 deviations
        vectors[9001] = withCosine(0.70, rnd); // two videos, center 0.60
        vectors[9002] = withCosine(0.50, rnd);
        mods[9001] = mods[9002] = VIDEO;
        assertTrue(n > EmbeddingMatrix.EXACT_MEDIAN_MAX_ITEMS);
        float[] scores = new EmbeddingMatrix(vectors, mods).scores(query());
        assertEquals(50, scores[9000], 4);
        // pooled spread is between the text (0.02) and image (0.015) spreads
        assertTrue("video score " + scores[9001], scores[9001] > 48 && scores[9001] < 70);
        assertEquals(-scores[9001], scores[9002], 2);
    }

    @Test
    public void testQuantizationIsAccurate() {
        Random rnd = new Random(1);
        float[] v = EmbeddingUtil.normalize(new float[768]);
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        EmbeddingUtil.normalize(v);
        float[] q = new float[768];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) rnd.nextGaussian();
        }
        EmbeddingUtil.normalize(q);
        byte[] dest = new byte[768];
        float scale = EmbeddingMatrix.quantize(v, dest, 0);
        float approx = 0;
        for (int i = 0; i < 768; i++) {
            approx += q[i] * dest[i];
        }
        assertEquals(EmbeddingUtil.dot(q, v), approx * scale, 1e-3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDimensionMismatchIsReported() throws IOException {
        new EmbeddingMatrix(new float[][] { query() }, new byte[1]).scores(new float[DIM / 2]);
    }

    @Test
    public void testLoadFromIndexWithSegmentsDeletionsAndStreamingMode() throws IOException {
        Random rnd = new Random(3);
        try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
            float[][] expected = new float[300][];
            try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
                for (int i = 0; i < 300; i++) {
                    Document doc = new Document();
                    doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
                    if (i % 3 != 0) { // documents without embedding are skipped
                        expected[i] = withCosine(0.6 + 0.02 * rnd.nextGaussian(), rnd);
                        doc.add(new KnnVectorField(EmbeddingUtil.EMBEDDING, expected[i]));
                        String m = i % 2 == 0 ? "image" : "text";
                        doc.add(new SortedDocValuesField(EmbeddingUtil.EMBEDDING_MODALITY, new BytesRef(m)));
                    }
                    w.addDocument(doc);
                    if (i % 100 == 99) {
                        w.commit(); // several segments
                    }
                }
                w.deleteDocuments(new Term("id", "1"));
                w.commit();
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertTrue(reader.leaves().size() > 1);
                EmbeddingMatrix memory = EmbeddingMatrix.load(reader, Long.MAX_VALUE);
                EmbeddingMatrix streaming = EmbeddingMatrix.load(reader, 0);
                assertTrue(memory.isInMemory());
                assertFalse(streaming.isInMemory());
                assertEquals(199, memory.size()); // 200 embedded minus 1 deleted
                assertEquals(DIM, memory.getDimension());

                float[] a = memory.scores(query());
                float[] b = streaming.scores(query());
                for (int i = 0; i < memory.size(); i++) {
                    assertEquals(b[i], a[i], 1.0f);
                }
                // doc ids map back to the right slots and modalities
                for (int doc = 0; doc < reader.maxDoc(); doc++) {
                    int id = Integer.parseInt(reader.document(doc).get("id"));
                    int slot = memory.slotOf(doc);
                    if (id % 3 == 0 || id == 1) {
                        assertEquals(-1, slot);
                    } else {
                        assertEquals(id % 2 == 0 ? IMAGE : TEXT, memory.modality[slot]);
                    }
                }
                // cached per query
                assertTrue(memory.scores(query()) == a);
            }
        }
    }

    @Test
    public void testIndexWithoutEmbeddings() throws IOException {
        try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
            try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
                Document doc = new Document();
                doc.add(new StringField("id", "1", Field.Store.YES));
                w.addDocument(doc);
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertNull(EmbeddingMatrix.get(reader));
            }
        }
    }

    /**
     * Scan throughput with one million 768d items. Run with
     * -Diped.benchmark=true (needs ~1 GB of heap).
     */
    @Test
    public void benchmarkOneMillionItems() throws IOException {
        assumeTrue(Boolean.getBoolean("iped.benchmark"));
        int n = 1_000_000, dim = 768;
        Random rnd = new Random(0);
        byte[] q = new byte[n * dim];
        rnd.nextBytes(q);
        float[] scales = new float[n];
        byte[] mods = new byte[n];
        for (int i = 0; i < n; i++) {
            scales[i] = 1f / (127 * 16);
            mods[i] = (byte) (i % 4);
        }
        EmbeddingMatrix matrix = new EmbeddingMatrix(dim, q, scales, mods);
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++) {
            query[i] = (float) rnd.nextGaussian();
        }
        EmbeddingUtil.normalize(query);
        for (int run = 0; run < 5; run++) {
            query[0] += 1e-3f; // new query: no cache
            long t = System.nanoTime();
            assertNotNull(matrix.scores(query));
            System.out.printf("1M x 768 scan + stats: %.1f ms (%d threads)%n", (System.nanoTime() - t) / 1e6,
                    Runtime.getRuntime().availableProcessors());
        }
        long t = System.nanoTime();
        matrix.scores(query);
        System.out.printf("cached query: %.3f ms%n", (System.nanoTime() - t) / 1e6);
    }
}
