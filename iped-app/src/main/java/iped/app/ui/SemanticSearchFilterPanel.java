package iped.app.ui;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;

import iped.viewers.api.ClearFilterListener;

/**
 * Shows the active semantic search next to the filter combo. Clicking it removes
 * the semantic search filter.
 */
public class SemanticSearchFilterPanel extends JPanel implements ClearFilterListener {

    private static final long serialVersionUID = 1L;
    private static final int MAX_CHARS = 32;

    private final JLabel label = new JLabel();

    public SemanticSearchFilterPanel() {
        setLayout(new BoxLayout(this, BoxLayout.LINE_AXIS));
        setBorder(BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(new Color(200, 30, 30), 2, true),
                BorderFactory.createEmptyBorder(1, 4, 1, 4)));
        add(label);
        setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                SemanticSearchFilterActions.clear(true);
            }
        });
    }

    public void setDescription(String description) {
        String text = description == null ? "" : description;
        if (text.length() > MAX_CHARS) {
            text = text.substring(0, MAX_CHARS - 1) + "…";
        }
        label.setText(Messages.getString("SemanticSearch.PanelLabel") + " " + text + "  ✕");
        setToolTipText("<html>" + Messages.getString("SemanticSearch.FilterTip") + "<br><b>"
                + escape(description == null ? "" : description) + "</b></html>");
        setMaximumSize(getPreferredSize());
        revalidate();
        repaint();
    }

    private static String escape(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    @Override
    public void clearFilter() {
        SemanticSearchFilterActions.clear(false);
    }
}
