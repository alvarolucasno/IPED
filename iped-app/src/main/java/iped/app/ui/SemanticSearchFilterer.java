package iped.app.ui;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import iped.data.IItem;
import iped.data.IItemId;
import iped.engine.search.MultiSearchResult;
import iped.exception.ParseException;
import iped.exception.QueryNodeException;
import iped.engine.search.SemanticSearch;
import iped.search.IMultiSearchResult;
import iped.viewers.api.IFilter;
import iped.viewers.api.IItemRef;
import iped.viewers.api.IQuantifiableFilter;
import iped.viewers.api.IResultSetFilter;
import iped.viewers.api.IResultSetFilterer;

/**
 * Applies the current semantic search (if any) to the result set, ranking
 * items by similarity to a text query, an external file or a reference item.
 */
public class SemanticSearchFilterer implements IResultSetFilterer {

    private SemanticSearchFilter filter;

    public static class SemanticSearchFilter implements IResultSetFilter, IItemRef, IQuantifiableFilter {

        private final float[] queryVector;
        private final String description;
        private final IItemId refItemId;
        private final IItem refItem;
        private final Set<String> modalities;
        private final int maxResultsPerModality;
        private int minScore;

        public SemanticSearchFilter(float[] queryVector, String description, IItemId refItemId, IItem refItem,
                Set<String> modalities, int maxResultsPerModality, int minScore) {
            this.queryVector = queryVector;
            this.description = description;
            this.refItemId = refItemId;
            this.refItem = refItem;
            this.modalities = modalities;
            this.maxResultsPerModality = maxResultsPerModality;
            this.minScore = minScore;
        }

        @Override
        public IMultiSearchResult filterResult(IMultiSearchResult src)
                throws ParseException, QueryNodeException, IOException {
            SemanticSearch search = new SemanticSearch(App.get().appCase, queryVector, modalities,
                    maxResultsPerModality, minScore);
            return search.filter((MultiSearchResult) src);
        }

        public String getDescription() {
            return description;
        }

        @Override
        public String toString() {
            return Messages.getString("FilterValue.SemanticSearch") + " " + description;
        }

        @Override
        public IItem getItemRef() {
            return refItem;
        }

        @Override
        public IItemId getItemRefId() {
            return refItemId;
        }

        @Override
        public int getQuantityValue() {
            return minScore;
        }

        @Override
        public void setQuantityValue(int value) {
            minScore = value;
        }
    }

    public void setFilter(SemanticSearchFilter filter) {
        this.filter = filter;
    }

    public SemanticSearchFilter getCurrentFilter() {
        return filter;
    }

    @Override
    public List<IFilter> getDefinedFilters() {
        List<IFilter> result = new ArrayList<>();
        if (filter != null) {
            result.add(filter);
        }
        return result;
    }

    @Override
    public IFilter getFilter() {
        return filter;
    }

    @Override
    public boolean hasFilters() {
        return filter != null;
    }

    @Override
    public boolean hasFiltersApplied() {
        return filter != null;
    }

    @Override
    public void clearFilter() {
        filter = null;
        SemanticSearchFilterActions.clear(false);
    }
}
