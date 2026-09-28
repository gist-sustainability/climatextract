"""Unit tests for percentile / top_k / min_k page selection in semantic search.

``_select_pages_by_similarity`` picks which pages of a PDF are sent to the LLM:
1. Percentile filter: keep pages scoring at or above the Nth percentile of this
   PDF's scores (0 or None turns the filter off).
2. Cap: keep at most ``similarity_top_k`` pages.
3. Floor: if fewer than ``similarity_min_k`` pages remain, return the
   ``similarity_min_k`` best pages overall (never more than ``similarity_top_k``).

Each test uses ``scored_pages(n)``, where page 1 is always the best match, so the
expected result is simply the first few page labels.
"""
import pandas as pd
import pytest

from climatextract.semantic_search import _select_pages_by_similarity


def scored_pages(n_pages: int) -> pd.DataFrame:
    """Pages 1..n with strictly decreasing similarity (page 1 is most similar)."""
    return pd.DataFrame({
        "page_label": [str(i) for i in range(1, n_pages + 1)],
        "similarity": [1.0 - i / 100 for i in range(n_pages)],
    })


def selected_labels(df: pd.DataFrame, **kwargs) -> list:
    """Run page selection and return the selected page labels in order."""
    return _select_pages_by_similarity(df, **kwargs)["page_label"].tolist()


def test_default_threshold_on_short_report_falls_back_to_floor():
    """Default settings on a short report: the 95th percentile keeps only 1 page.

    On 10 pages the cutoff is ~0.9955, so only page 1 passes. That is below
    min_k=4, so the floor applies and pages 1-4 are returned.
    """
    labels = selected_labels(scored_pages(10), similarity_top_k=7,
                             similarity_min_k=4, percentile_threshold=95)
    assert labels == ["1", "2", "3", "4"]


@pytest.mark.parametrize("percentile_threshold", [0, None])
def test_zero_or_none_disables_percentile_filter(percentile_threshold):
    """Regression test: 0 or None turns the percentile filter off.

    All 10 pages pass, so the top_k cap returns pages 1-7. Previously 0 and None
    silently became 95, which returned only pages 1-4.
    """
    labels = selected_labels(scored_pages(10), similarity_top_k=7,
                             similarity_min_k=4, percentile_threshold=percentile_threshold)
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]


def test_percentile_filter_is_capped_at_top_k():
    """The top_k cap applies after the percentile filter.

    The 50th percentile of 40 pages keeps 20 pages; the cap cuts them to 7.
    """
    labels = selected_labels(scored_pages(40), similarity_top_k=7,
                             similarity_min_k=4, percentile_threshold=50)
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]


def test_explicit_percentile_is_respected():
    """The percentile filter alone decides the result when cap and floor don't apply.

    On 100 pages the 95th percentile cutoff is ~0.9505, so pages 1-5 pass.
    That is under the cap of 7, and min_k=0 disables the floor.
    """
    labels = selected_labels(scored_pages(100), similarity_top_k=7,
                             similarity_min_k=0, percentile_threshold=95)
    assert labels == ["1", "2", "3", "4", "5"]


def test_min_k_equal_to_top_k_always_returns_top_k():
    """Setting min_k equal to top_k always returns exactly top_k pages.

    The percentile filter keeps only page 1, so the floor returns pages 1-7.
    """
    labels = selected_labels(scored_pages(10), similarity_top_k=7,
                             similarity_min_k=7, percentile_threshold=95)
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]


def test_min_k_is_capped_at_top_k():
    """A misconfigured min_k above top_k is lowered to top_k.

    With top_k=3 and min_k=5, the floor returns pages 1-3, not 1-5.
    """
    labels = selected_labels(scored_pages(10), similarity_top_k=3,
                             similarity_min_k=5, percentile_threshold=95)
    assert labels == ["1", "2", "3"]


def test_selection_ignores_input_order():
    """Pages are ranked by similarity, whatever order they arrive in.

    Shuffled input still returns pages 1-7, best first.
    """
    shuffled = scored_pages(10).sample(frac=1, random_state=0)
    labels = selected_labels(shuffled, similarity_top_k=7,
                             similarity_min_k=4, percentile_threshold=0)
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]


@pytest.mark.parametrize("percentile_threshold", [-1, 100.5, 150])
def test_out_of_range_percentile_raises(percentile_threshold):
    """Percentiles outside 0-100 raise a ValueError that names the valid range."""
    with pytest.raises(ValueError, match="between 0 and 100"):
        _select_pages_by_similarity(scored_pages(10), similarity_top_k=7,
                                    similarity_min_k=4, percentile_threshold=percentile_threshold)
