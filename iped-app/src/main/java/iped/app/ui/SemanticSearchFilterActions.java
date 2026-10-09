package iped.app.ui;

import java.awt.Cursor;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.RowSorter;
import javax.swing.RowSorter.SortKey;
import javax.swing.SortOrder;
import javax.swing.SwingWorker;

import org.apache.lucene.index.FieldInfo;

import iped.app.ui.SemanticSearchFilterer.SemanticSearchFilter;
import iped.data.IItem;
import iped.data.IItemId;
import iped.engine.config.ConfigurationManager;
import iped.engine.config.EmbeddingTaskConfig;
import iped.engine.embedding.EmbeddingServiceClient;
import iped.engine.embedding.EmbeddingServiceClient.Input;
import iped.engine.embedding.EmbeddingUtil;
import iped.engine.search.SemanticSearch;

/**
 * Semantic search actions of the analysis UI: by natural language text, by an
 * external file (image, video, audio or text) or by the highlighted item.
 */
public class SemanticSearchFilterActions {

    private static final int SCORE_COL = 2;

    private static final List<String> IMAGE_EXT = Arrays.asList("jpg", "jpeg", "png", "gif", "bmp", "webp", "tif",
            "tiff", "heic", "heif");
    private static final List<String> VIDEO_EXT = Arrays.asList("mp4", "avi", "mov", "mkv", "3gp", "webm", "wmv",
            "m4v", "mpg", "mpeg", "flv");
    private static final List<String> AUDIO_EXT = Arrays.asList("mp3", "wav", "ogg", "opus", "m4a", "aac", "flac",
            "amr", "wma", "oga");

    private static SemanticSearchDialog.Options lastOptions;
    private static List<? extends SortKey> prevSortKeys;

    private interface Embedder {
        float[] embed(EmbeddingServiceClient client) throws IOException;
    }

    public static EmbeddingTaskConfig getConfig() {
        EmbeddingTaskConfig config = null;
        try {
            config = ConfigurationManager.get().findObject(EmbeddingTaskConfig.class);
        } catch (Exception e) {
            // configuration not loaded
        }
        return config != null ? config : new EmbeddingTaskConfig();
    }

    /** True if the opened case (or any case of a multicase) has item embeddings. */
    public static boolean isAvailable() {
        App app = App.get();
        if (app.appCase == null) {
            return false;
        }
        try {
            FieldInfo fi = app.appCase.getLeafReader().getFieldInfos().fieldInfo(EmbeddingUtil.EMBEDDING);
            return fi != null;
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean checkAvailable() {
        if (isAvailable()) {
            // loads the case embeddings while the user types the query
            SemanticSearch.warmUp(App.get().appCase);
            return true;
        }
        JOptionPane.showMessageDialog(App.get(), Messages.getString("SemanticSearch.NotAvailable"),
                Messages.getString("SemanticSearch.DialogTitle"), JOptionPane.INFORMATION_MESSAGE);
        return false;
    }

    private static SemanticSearchDialog.Options getLastOptions() {
        if (lastOptions == null) {
            EmbeddingTaskConfig config = getConfig();
            lastOptions = new SemanticSearchDialog.Options(config.getSearchMaxResultsPerModality(),
                    config.getSearchMinScore());
        }
        return lastOptions;
    }

    /** Opens the query dialog and runs a search by text or by an external file. */
    public static void searchByText() {
        if (!checkAvailable()) {
            return;
        }
        SemanticSearchDialog.Options opt = new SemanticSearchDialog(App.get(), getLastOptions()).showDialog();
        if (opt == null) {
            return;
        }
        lastOptions = opt;
        if (opt.external) {
            searchByExternalFile(opt);
        } else {
            run(client -> client.embedQuery(opt.query), opt.query, null, null, opt);
        }
    }

    public static void searchByExternalFile() {
        if (checkAvailable()) {
            searchByExternalFile(getLastOptions());
        }
    }

    private static void searchByExternalFile(SemanticSearchDialog.Options opt) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle(Messages.getString("SemanticSearch.ExternalTitle"));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        if (fileChooser.showOpenDialog(App.get()) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File file = fileChooser.getSelectedFile();
        EmbeddingTaskConfig config = getConfig();
        run(client -> client.embedOne(createInput(file, config)), file.getName(), null, null, opt);
    }

    static Input createInput(File file, EmbeddingTaskConfig config) throws IOException {
        String name = file.getName();
        String ext = name.contains(".") ? name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT) : "";
        if (IMAGE_EXT.contains(ext)) {
            return Input.image("ext", Files.readAllBytes(file.toPath()));
        }
        if (VIDEO_EXT.contains(ext)) {
            return Input.videoFile("ext", Files.readAllBytes(file.toPath()));
        }
        if (AUDIO_EXT.contains(ext)) {
            return Input.audio("ext", Files.readAllBytes(file.toPath()));
        }
        String text = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        if (text.length() > config.getMaxTextChars()) {
            text = text.substring(0, config.getMaxTextChars());
        }
        return Input.document("ext", name, text);
    }

    /** Ranks items by similarity to the highlighted item, no service needed. */
    public static void searchSimilarToSelected() {
        App app = App.get();
        int selIdx = app.resultsTable.getSelectedRow();
        if (selIdx == -1) {
            return;
        }
        IItemId itemId = app.ipedResult.getItem(app.resultsTable.convertRowIndexToModel(selIdx));
        IItem item = itemId != null ? app.appCase.getItemByItemId(itemId) : null;
        float[] vector = EmbeddingUtil.getVector(item);
        if (vector == null) {
            JOptionPane.showMessageDialog(app, Messages.getString("SemanticSearch.NoEmbedding"),
                    Messages.getString("SemanticSearch.DialogTitle"), JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        SemanticSearchDialog.Options opt = getLastOptions().copy();
        opt.modalities.addAll(Arrays.asList(EmbeddingUtil.MODALITIES));
        apply(new SemanticSearchFilter(vector, item.getName(), itemId, item, opt.modalities,
                opt.maxResultsPerModality, opt.minScore));
    }

    public static boolean canSearchSimilarTo(IItem item) {
        return item != null && item.getExtraAttribute(EmbeddingUtil.EMBEDDING) != null;
    }

    private static void run(Embedder embedder, String description, IItemId refId, IItem ref,
            SemanticSearchDialog.Options opt) {
        App app = App.get();
        EmbeddingTaskConfig config = getConfig();
        app.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        new SwingWorker<float[], Void>() {
            @Override
            protected float[] doInBackground() throws Exception {
                try (EmbeddingServiceClient client = new EmbeddingServiceClient(config.getServiceUrl(),
                        config.getConnectTimeout() > 0 ? Math.min(config.getConnectTimeout(), 10000) : 10000,
                        config.getSocketTimeout())) {
                    return embedder.embed(client);
                }
            }

            @Override
            protected void done() {
                app.setCursor(Cursor.getDefaultCursor());
                try {
                    float[] vector = get();
                    apply(new SemanticSearchFilter(vector, description, refId, ref, opt.modalities,
                            opt.maxResultsPerModality, opt.minScore));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause() != null ? e.getCause() : e;
                    cause.printStackTrace();
                    JOptionPane.showMessageDialog(app,
                            MessageFormat.format(Messages.getString("SemanticSearch.ServiceError"),
                                    config.getServiceUrl(), cause.getMessage()),
                            Messages.getString("SemanticSearch.DialogTitle"), JOptionPane.ERROR_MESSAGE);
                }
            }
        }.execute();
    }

    private static void apply(SemanticSearchFilter filter) {
        App app = App.get();
        app.semanticSearchFilterer.setFilter(filter);
        List<? extends SortKey> sortKeys = app.resultsTable.getRowSorter().getSortKeys();
        if (sortKeys == null || sortKeys.isEmpty() || sortKeys.get(0).getColumn() != SCORE_COL
                || sortKeys.get(0).getSortOrder() != SortOrder.DESCENDING) {
            if (prevSortKeys == null) {
                prevSortKeys = sortKeys;
            }
            List<RowSorter.SortKey> sortScore = new ArrayList<>();
            sortScore.add(new RowSorter.SortKey(SCORE_COL, SortOrder.DESCENDING));
            ((ResultTableRowSorter) app.resultsTable.getRowSorter()).setSortKeysSuper(sortScore);
        }
        app.semanticSearchFilterPanel.setDescription(filter.getDescription());
        app.semanticSearchFilterPanel.setVisible(true);
        app.appletListener.updateFileListing();
    }

    public static void clear(boolean updateResults) {
        App app = App.get();
        if (app.semanticSearchFilterer == null || app.semanticSearchFilterPanel == null) {
            return;
        }
        boolean wasActive = app.semanticSearchFilterer.getCurrentFilter() != null
                || app.semanticSearchFilterPanel.isVisible();
        app.semanticSearchFilterer.setFilter(null);
        app.semanticSearchFilterPanel.setVisible(false);
        if (prevSortKeys != null) {
            List<? extends SortKey> sortKeys = app.resultsTable.getRowSorter().getSortKeys();
            if (sortKeys != null && !sortKeys.isEmpty() && sortKeys.get(0).getColumn() == SCORE_COL) {
                ((ResultTableRowSorter) app.resultsTable.getRowSorter()).setSortKeysSuper(prevSortKeys);
            }
            prevSortKeys = null;
        }
        if (updateResults && wasActive) {
            app.appletListener.updateFileListing();
        }
    }
}
