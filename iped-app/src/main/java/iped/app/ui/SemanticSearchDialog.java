package iped.app.ui;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.KeyEvent;
import java.util.LinkedHashSet;
import java.util.Set;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.KeyStroke;
import javax.swing.SpinnerNumberModel;

import iped.engine.embedding.EmbeddingUtil;

/**
 * Asks for the semantic search query and options.
 */
public class SemanticSearchDialog extends JDialog {

    private static final long serialVersionUID = 1L;

    public static class Options {
        public String query = "";
        public Set<String> modalities = new LinkedHashSet<>();
        public int maxResultsPerModality;
        public int minScore;
        /** true if the user asked to pick an external file instead of typing a query */
        public boolean external;

        public Options(int maxResultsPerModality, int minScore) {
            this.maxResultsPerModality = maxResultsPerModality;
            this.minScore = minScore;
            for (String m : EmbeddingUtil.MODALITIES) {
                modalities.add(m);
            }
        }

        public Options copy() {
            Options o = new Options(maxResultsPerModality, minScore);
            o.query = query;
            o.modalities = new LinkedHashSet<>(modalities);
            return o;
        }
    }

    private final JTextArea queryArea = new JTextArea(3, 40);
    private final JCheckBox[] modalityBoxes = new JCheckBox[EmbeddingUtil.MODALITIES.length];
    private final JSpinner maxResults;
    private final JSpinner minScore;
    private Options result;

    public SemanticSearchDialog(Frame owner, Options defaults) {
        super(owner, Messages.getString("SemanticSearch.DialogTitle"), true);

        queryArea.setLineWrap(true);
        queryArea.setWrapStyleWord(true);
        queryArea.setText(defaults.query);
        queryArea.selectAll();
        // Enter searches, Shift+Enter breaks the line
        queryArea.getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, 0), "search");
        queryArea.getInputMap().put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, KeyEvent.SHIFT_DOWN_MASK), "insert-break");
        queryArea.getActionMap().put("search", new javax.swing.AbstractAction() {
            private static final long serialVersionUID = 1L;

            @Override
            public void actionPerformed(java.awt.event.ActionEvent e) {
                accept(false);
            }
        });

        maxResults = new JSpinner(new SpinnerNumberModel(defaults.maxResultsPerModality, 1, 1000000, 50));
        minScore = new JSpinner(new SpinnerNumberModel(defaults.minScore, 0, 100, 5));

        JPanel form = new JPanel(new GridBagLayout());
        form.setBorder(BorderFactory.createEmptyBorder(10, 10, 4, 10));
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;
        c.gridwidth = 2;
        c.anchor = GridBagConstraints.WEST;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(2, 0, 2, 0);
        form.add(new JLabel(Messages.getString("SemanticSearch.QueryLabel")), c);
        c.gridy++;
        c.fill = GridBagConstraints.BOTH;
        c.weightx = c.weighty = 1;
        form.add(new JScrollPane(queryArea), c);
        c.weightx = c.weighty = 0;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridy++;
        c.insets = new Insets(8, 0, 2, 0);
        JPanel modalities = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        modalities.add(new JLabel(Messages.getString("SemanticSearch.Modalities") + "  "));
        for (int i = 0; i < modalityBoxes.length; i++) {
            String m = EmbeddingUtil.MODALITIES[i];
            modalityBoxes[i] = new JCheckBox(Messages.getString("SemanticSearch.Modality." + m),
                    defaults.modalities.contains(m));
            modalities.add(modalityBoxes[i]);
        }
        form.add(modalities, c);
        c.gridwidth = 1;
        c.insets = new Insets(2, 0, 2, 8);
        c.gridy++;
        form.add(new JLabel(Messages.getString("SemanticSearch.MaxResults")), c);
        c.gridx = 1;
        form.add(maxResults, c);
        c.gridx = 0;
        c.gridy++;
        form.add(new JLabel(Messages.getString("SemanticSearch.MinScore")), c);
        c.gridx = 1;
        form.add(minScore, c);
        minScore.setToolTipText(Messages.getString("SemanticSearch.MinScoreTip"));

        JButton search = new JButton(Messages.getString("SemanticSearch.Search"));
        JButton external = new JButton(Messages.getString("SemanticSearch.ExternalButton"));
        JButton cancel = new JButton(Messages.getString("SemanticSearch.Cancel"));
        search.addActionListener(e -> accept(false));
        external.addActionListener(e -> accept(true));
        cancel.addActionListener(e -> dispose());
        external.setToolTipText(Messages.getString("SemanticSearch.ExternalTip"));

        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttons.add(external);
        buttons.add(search);
        buttons.add(cancel);

        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(form, BorderLayout.CENTER);
        getContentPane().add(buttons, BorderLayout.SOUTH);
        getRootPane().setDefaultButton(search);
        getRootPane().registerKeyboardAction(e -> dispose(), KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE, 0),
                JComponent.WHEN_IN_FOCUSED_WINDOW);
        pack();
        setLocationRelativeTo(owner);
    }

    private void accept(boolean externalFile) {
        Options o = new Options((Integer) maxResults.getValue(), (Integer) minScore.getValue());
        o.query = queryArea.getText().trim();
        o.external = externalFile;
        o.modalities.clear();
        for (int i = 0; i < modalityBoxes.length; i++) {
            if (modalityBoxes[i].isSelected()) {
                o.modalities.add(EmbeddingUtil.MODALITIES[i]);
            }
        }
        if (o.modalities.isEmpty() || (!externalFile && o.query.isEmpty())) {
            queryArea.requestFocusInWindow();
            return;
        }
        result = o;
        dispose();
    }

    /** Shows the dialog and returns the chosen options, or null if canceled. */
    public Options showDialog() {
        setVisible(true);
        return result;
    }
}
