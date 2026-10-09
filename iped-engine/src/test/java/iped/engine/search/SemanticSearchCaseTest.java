package iped.engine.search;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
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

    private static List<String> rank(float[] query, String modality, int k) throws IOException {
        Set<String> modalities = modality == null ? null : Collections.singleton(modality);
        MultiSearchResult result = new SemanticSearch(ipedCase, query, modalities, k, 0).search();
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
        MultiSearchResult all = new SemanticSearch(ipedCase, client.embedQuery("qualquer coisa"), null, 1000, -100)
                .search();
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

    private static IItem findByName(String what, String name) throws IOException {
        MultiSearchResult all = new IPEDSearcher(ipedCase, "*:*").multiSearch();
        for (int i = 0; i < all.getLength(); i++) {
            IItem item = ipedCase.getItemByItemId(all.getItem(i));
            if (name.equals(item.getName())) {
                return item;
            }
        }
        throw new AssertionError(what + " not found: " + name);
    }
}
