package iped.engine.search;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import iped.data.IItem;
import iped.data.IItemId;
import iped.engine.data.IPEDMultiSource;
import iped.engine.data.IPEDSource;
import iped.engine.embedding.EmbeddingServiceClient;
import iped.engine.embedding.EmbeddingUtil;

/**
 * End-to-end test of semantic search over a case processed with
 * enableEmbedding=true from the synthetic dataset (Portuguese documents, photos,
 * videos and spoken audio). It needs the processed case and the embedding
 * service running, so it only runs when explicitly requested:
 *
 * mvn -pl iped-engine test -Dtest=SemanticSearchCaseTest -Diped.testcase=C:\path\to\case
 * [-Diped.embedding.url=http://127.0.0.1:8691]
 */
public class SemanticSearchCaseTest {

    private static IPEDMultiSource ipedCase;
    private static EmbeddingServiceClient client;

    @BeforeClass
    public static void open() throws IOException {
        String casePath = System.getProperty("iped.testcase");
        assumeTrue("set -Diped.testcase to run", casePath != null);
        ipedCase = new IPEDMultiSource(Collections.singletonList(new IPEDSource(new File(casePath))));
        client = new EmbeddingServiceClient(System.getProperty("iped.embedding.url", "http://127.0.0.1:8691"), 10000,
                120000);
    }

    @AfterClass
    public static void close() throws IOException {
        if (client != null) {
            client.close();
        }
        if (ipedCase != null) {
            ipedCase.close();
        }
    }

    /** Ranking only: no standout cut. */
    private static List<String> rank(float[] query, String modality, int k) throws IOException {
        return search(new SemanticSearch(ipedCase, query, modality == null ? null : Collections.singleton(modality), k,
                Float.NEGATIVE_INFINITY));
    }

    /** What the analyst sees with the default filter: only items that stand out. */
    private static List<String> filter(String query) throws IOException {
        System.out.println("[filter, min score " + SemanticSearch.DEFAULT_MIN_SCORE + "] " + query);
        return search(new SemanticSearch(ipedCase, client.embedQuery(query), null, 500,
                SemanticSearch.DEFAULT_MIN_SCORE));
    }

    private static List<String> search(SemanticSearch search) throws IOException {
        MultiSearchResult result = search.search();
        List<String> names = new ArrayList<>();
        StringBuilder log = new StringBuilder();
        for (int i = 0; i < result.getLength(); i++) {
            String name = ipedCase.getItemByItemId(result.getItem(i)).getName();
            names.add(name);
            log.append(String.format("  %5.1f %s%n", result.getScore(i), name));
        }
        System.out.print(log);
        return names;
    }

    private static List<String> rank(String query, String modality, int k) throws IOException {
        System.out.println("[" + modality + "] " + query);
        return rank(client.embedQuery(query), modality, k);
    }

    private static void assertTop(String expected, String query, String modality) throws IOException {
        assertEquals(query, expected, rank(query, modality, 3).get(0));
    }

    @Test
    public void testEveryItemOfTheDatasetHasAnEmbedding() throws IOException {
        MultiSearchResult all = new SemanticSearch(ipedCase, client.embedQuery("qualquer coisa"), null, 1000,
                Float.NEGATIVE_INFINITY).search();
        assertEquals(32, all.getLength());
        Set<String> names = new HashSet<>();
        for (int i = 0; i < all.getLength(); i++) {
            names.add(ipedCase.getItemByItemId(all.getItem(i)).getName());
        }
        for (String expected : Arrays.asList("contrato_locacao.docx", "relatorio_viagem.pdf", "email_banco.eml",
                "cronica_jogo.html", "IMG_0101.jpg", "IMG_0110.jpg", "VID_20250301_0201.mp4",
                "VID_20250302_0202.mp4", "PTT-20250312-WA0001.opus", "gravacao_ligacao_03.mp3", "gravacao_rua.wav")) {
            assertEquals(expected, true, names.contains(expected));
        }
    }

    @Test
    public void testTextQueriesFindDocuments() throws IOException {
        assertTop("comprovante_pagamento.txt", "recibo de pagamento do aluguel do apartamento", EmbeddingUtil.MODALITY_TEXT);
        assertTop("email_banco.eml", "golpe do falso funcionário do banco pedindo senha e código", EmbeddingUtil.MODALITY_TEXT);
        assertTop("mensagem_cobranca.txt", "ameaça para cobrar uma dívida", EmbeddingUtil.MODALITY_TEXT);
        assertTop("anotacoes.txt", "tráfico de drogas chegando de navio", EmbeddingUtil.MODALITY_TEXT);
        assertTop("anuncio_veiculo.txt", "carro roubado com documentação falsa", EmbeddingUtil.MODALITY_TEXT);
        assertTop("relatorio_viagem.pdf", "passeio turístico na Bahia", EmbeddingUtil.MODALITY_TEXT);
    }

    @Test
    public void testTextQueriesFindImages() throws IOException {
        assertTop("IMG_0101.jpg", "carro esportivo vermelho estacionado", EmbeddingUtil.MODALITY_IMAGE);
        assertTop("IMG_0104.jpg", "arma de fogo", EmbeddingUtil.MODALITY_IMAGE);
        assertTop("IMG_0105.jpg", "cédula de dinheiro antiga", EmbeddingUtil.MODALITY_IMAGE);
        assertTop("IMG_0106.jpg", "bolo de aniversário com velas", EmbeddingUtil.MODALITY_IMAGE);
        assertTop("IMG_0110.jpg", "estádio de futebol lotado", EmbeddingUtil.MODALITY_IMAGE);
    }

    @Test
    public void testTextQueriesFindVideos() throws IOException {
        assertTop("VID_20250302_0202.mp4", "viatura da polícia", EmbeddingUtil.MODALITY_VIDEO);
        assertTop("VID_20250301_0201.mp4", "praia com pedras ao pôr do sol", EmbeddingUtil.MODALITY_VIDEO);
    }

    @Test
    public void testTextQueriesFindSpokenAudio() throws IOException {
        assertTop("PTT-20250312-WA0001.opus", "áudio combinando entrega de mercadoria e pagamento em dinheiro",
                EmbeddingUtil.MODALITY_AUDIO);
        assertTop("gravacao_ligacao_03.mp3", "ameaça por telefone para cobrar dívida", EmbeddingUtil.MODALITY_AUDIO);
        assertTop("audio_receita_04.mp3", "receita de bolo", EmbeddingUtil.MODALITY_AUDIO);
        assertTop("gravacao_rua.wav", "sirene", EmbeddingUtil.MODALITY_AUDIO);
    }

    @Test
    public void testSimilarToItemCrossesModalities() throws IOException {
        IItem beach = findByName("beach photo", "IMG_0103.jpg");
        float[] vector = EmbeddingUtil.getVector(beach);
        assertNotNull(vector);
        System.out.println("[video] similar to IMG_0103.jpg");
        assertEquals("VID_20250301_0201.mp4", rank(vector, EmbeddingUtil.MODALITY_VIDEO, 2).get(0));
    }

    @Test
    public void testResultsAreBalancedPerModality() throws IOException {
        List<String> names = rank("dinheiro", null, 1);
        assertEquals(4, names.size());
    }

    @Test
    public void testFilterDropsModalitiesWithoutRelevantItems() throws IOException {
        // no text, video or audio of the case is about dogs: only photos may pass the cut
        for (String query : Arrays.asList("cachorro", "dog", "foto de um cachorro brincando")) {
            List<String> names = filter(query);
            for (String name : names) {
                assertTrue(query + " -> " + name, name.endsWith(".jpg"));
            }
            assertTrue(query, names.size() <= 2);
        }
        assertEquals(Arrays.asList("IMG_0108.jpg"), filter("dog"));
        assertEquals("IMG_0108.jpg", filter("foto de um cachorro brincando").get(0));
        // nothing about sheet music in the case. Known borderline false positive: the
        // ornate 1923 banknote engraving (IMG_0105) scores ~32, just above the cut.
        for (String name : filter("partitura musical")) {
            assertEquals("IMG_0105.jpg", name);
        }
    }

    @Test
    public void testFilterKeepsClearHits() throws IOException {
        assertTrue(filter("recibo de pagamento do aluguel do apartamento").contains("comprovante_pagamento.txt"));
        assertTrue(filter("bolo de aniversário com velas").contains("IMG_0106.jpg"));
        assertTrue(filter("estádio de futebol lotado").contains("IMG_0110.jpg"));
        assertTrue(filter("sirene").contains("gravacao_rua.wav"));
        assertTrue(filter("receita de bolo").contains("audio_receita_04.mp3"));
    }

    @Test
    public void testReferenceItemIsShownAsRef() throws IOException {
        IItemId beachId = findIdByName("IMG_0103.jpg");
        float[] vector = EmbeddingUtil.getVector(ipedCase.getItemByItemId(beachId));
        System.out.println("[filter] similar to IMG_0103.jpg");
        SemanticSearch search = new SemanticSearch(ipedCase, vector, null, 500, SemanticSearch.DEFAULT_MIN_SCORE)
                .setReference(beachId);
        MultiSearchResult result = search.search();
        assertEquals(beachId, result.getItem(0));
        assertEquals(SemanticSearch.REF_SCORE, result.getScore(0), 0);
        List<String> names = search(search);
        // the sea poem stands out among texts
        assertTrue(names.contains("poema_mar.txt"));
        // the beach video ranks first among videos, but with only two videos in the
        // case the modality baseline is their midpoint, so it can not pass the cut
        // (conservative by design: small modalities do not borrow other baselines)
        assertEquals("VID_20250301_0201.mp4", rank(vector, EmbeddingUtil.MODALITY_VIDEO, 1).get(0));
    }

    private static IItemId findIdByName(String name) throws IOException {
        MultiSearchResult all = new IPEDSearcher(ipedCase, "*:*").multiSearch();
        for (int i = 0; i < all.getLength(); i++) {
            if (name.equals(ipedCase.getItemByItemId(all.getItem(i)).getName())) {
                return all.getItem(i);
            }
        }
        throw new AssertionError("not found: " + name);
    }

    private static IItem findByName(String what, String name) throws IOException {
        return ipedCase.getItemByItemId(findIdByName(name));
    }
}
