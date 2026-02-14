"""Unit tests for app/core/context_expander.py."""

from unittest.mock import patch

from app.core.context_expander import expand_context, _needs_expansion
from app.models.schemas import Chunk


def _chunk(
    source: str = "doc.pdf",
    page: int = 1,
    chunk_id: str | None = None,
    text: str = "some text",
) -> Chunk:
    """Helper to create a minimal chunk."""
    if chunk_id is None:
        chunk_id = f"{source}_p{page}_c0"
    return Chunk(chunk_id=chunk_id, text=text, source=source, page=page)


# ---------------------------------------------------------------------------
# _needs_expansion — pure threshold logic
# ---------------------------------------------------------------------------


class TestNeedsExpansion:
    def test_empty_ranked_no_expansion(self):
        assert _needs_expansion([], top_k=5) is False

    @patch("app.core.context_expander.settings")
    def test_strong_scores_no_expansion(self, mock_settings):
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        ranked = [(_chunk(), 0.9), (_chunk(page=2), 0.8)]
        assert _needs_expansion(ranked, top_k=2) is False

    @patch("app.core.context_expander.settings")
    def test_weak_score_triggers_expansion(self, mock_settings):
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        ranked = [(_chunk(), 0.3)]
        assert _needs_expansion(ranked, top_k=1) is True

    @patch("app.core.context_expander.settings")
    def test_low_coverage_triggers_expansion(self, mock_settings):
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        ranked = [(_chunk(), 0.9)]  # 1/5 = 0.2 coverage
        assert _needs_expansion(ranked, top_k=5) is True

    @patch("app.core.context_expander.settings")
    def test_zero_top_k_no_expansion(self, mock_settings):
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        ranked = [(_chunk(), 0.9)]
        assert _needs_expansion(ranked, top_k=0) is False


# ---------------------------------------------------------------------------
# expand_context — full expansion logic
# ---------------------------------------------------------------------------


class TestExpandContext:
    @patch("app.core.context_expander.settings")
    def test_no_expansion_when_strong(self, mock_settings):
        """Strong results are returned unchanged."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        ranked = [(_chunk(), 0.9), (_chunk(page=2), 0.85)]
        result = expand_context(ranked, top_k=2)
        assert result is ranked  # identity — no copy, no merge

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_expansion_adds_neighbors(self, mock_settings, mock_store):
        """When triggered, adjacent-page chunks are merged in."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        main_chunk = _chunk(page=3, chunk_id="doc.pdf_p3_c0")
        neighbor = _chunk(page=4, chunk_id="doc.pdf_p4_c0")
        ranked = [(main_chunk, 0.35)]  # weak score triggers expansion

        mock_store.get_chunks_by_source_and_pages.return_value = [neighbor]

        result = expand_context(ranked, top_k=5)

        assert len(result) == 2
        assert result[0][0] is main_chunk
        assert result[1][0] is neighbor
        assert result[1][1] == 0.35 * 0.8  # discounted

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_no_duplicate_neighbors(self, mock_settings, mock_store):
        """Chunks already in ranked are not added again."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        chunk_p2 = _chunk(page=2, chunk_id="doc.pdf_p2_c0")
        chunk_p3 = _chunk(page=3, chunk_id="doc.pdf_p3_c0")
        ranked = [(chunk_p2, 0.35), (chunk_p3, 0.30)]

        # Neighbors of p2 include p3 (already ranked).
        mock_store.get_chunks_by_source_and_pages.side_effect = [
            [chunk_p3],               # neighbors of p2: p3 is duplicate
            [chunk_p2],               # neighbors of p3: p2 is duplicate
        ]

        result = expand_context(ranked, top_k=5)
        assert len(result) == 2  # no new chunks added

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_page_1_no_page_0(self, mock_settings, mock_store):
        """Page 1 chunks only expand forward — no page 0 lookup."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        ranked = [(_chunk(page=1, chunk_id="doc.pdf_p1_c0"), 0.35)]
        mock_store.get_chunks_by_source_and_pages.return_value = []

        expand_context(ranked, top_k=5)

        call_args = mock_store.get_chunks_by_source_and_pages.call_args
        pages_requested = call_args[1].get("pages") or call_args[0][1]
        assert 0 not in pages_requested
        assert 2 in pages_requested

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_cross_document_isolation(self, mock_settings, mock_store):
        """Neighbors are only looked up from the same source document."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        chunk_a = _chunk(source="a.pdf", page=2, chunk_id="a.pdf_p2_c0")
        chunk_b = _chunk(source="b.pdf", page=3, chunk_id="b.pdf_p3_c0")
        ranked = [(chunk_a, 0.35), (chunk_b, 0.30)]

        mock_store.get_chunks_by_source_and_pages.return_value = []

        expand_context(ranked, top_k=5)

        calls = mock_store.get_chunks_by_source_and_pages.call_args_list
        assert calls[0][0][0] == "a.pdf"
        assert calls[1][0][0] == "b.pdf"

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_neighbor_score_discounted(self, mock_settings, mock_store):
        """Neighbor scores are parent_score * discount factor."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.75

        main = _chunk(page=5, chunk_id="doc.pdf_p5_c0")
        neighbor = _chunk(page=6, chunk_id="doc.pdf_p6_c0")
        ranked = [(main, 0.38)]

        mock_store.get_chunks_by_source_and_pages.return_value = [neighbor]

        result = expand_context(ranked, top_k=5)
        neighbor_entry = [r for r in result if r[0].chunk_id == "doc.pdf_p6_c0"]
        assert len(neighbor_entry) == 1
        assert abs(neighbor_entry[0][1] - 0.38 * 0.75) < 1e-9

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_result_sorted_descending(self, mock_settings, mock_store):
        """Merged results are sorted by score descending."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        main = _chunk(page=3, chunk_id="doc.pdf_p3_c0")
        neighbor_p2 = _chunk(page=2, chunk_id="doc.pdf_p2_c0")
        neighbor_p4 = _chunk(page=4, chunk_id="doc.pdf_p4_c0")
        ranked = [(main, 0.35)]

        mock_store.get_chunks_by_source_and_pages.return_value = [
            neighbor_p2, neighbor_p4,
        ]

        result = expand_context(ranked, top_k=5)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    @patch("app.core.context_expander.vector_store")
    @patch("app.core.context_expander.settings")
    def test_no_neighbors_found_returns_ranked(self, mock_settings, mock_store):
        """If store has no adjacent chunks, return ranked unchanged."""
        mock_settings.expansion_score_threshold = 0.4
        mock_settings.expansion_coverage_ratio = 0.6
        mock_settings.expansion_neighbor_discount = 0.8

        ranked = [(_chunk(page=1, chunk_id="doc.pdf_p1_c0"), 0.35)]
        mock_store.get_chunks_by_source_and_pages.return_value = []

        result = expand_context(ranked, top_k=5)
        assert result is ranked
