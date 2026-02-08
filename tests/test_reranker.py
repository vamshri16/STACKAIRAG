"""Unit tests for app/core/reranker.py."""

from app.core.reranker import rerank
from app.models.schemas import Chunk


def _chunk(source: str, page: int, chunk_id: str = "x") -> Chunk:
    """Helper to create a minimal chunk."""
    return Chunk(chunk_id=chunk_id, text="some text", source=source, page=page)


class TestRerank:
    def test_empty_input(self):
        assert rerank([], top_k=5, threshold=0.7) == []

    def test_threshold_filters(self):
        scored = [
            (_chunk("a.pdf", 1), 0.9),
            (_chunk("b.pdf", 1), 0.5),  # below threshold
            (_chunk("c.pdf", 1), 0.8),
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert len(result) == 2
        assert all(score >= 0.7 for _, score in result)

    def test_all_below_threshold(self):
        scored = [
            (_chunk("a.pdf", 1), 0.3),
            (_chunk("b.pdf", 1), 0.5),
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert result == []

    def test_sort_order_descending(self):
        scored = [
            (_chunk("a.pdf", 1), 0.8),
            (_chunk("b.pdf", 1), 0.95),
            (_chunk("c.pdf", 1), 0.85),
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_dedup_same_page(self):
        scored = [
            (_chunk("doc.pdf", 1, "c0"), 0.9),
            (_chunk("doc.pdf", 1, "c1"), 0.85),  # same source + page → deduped
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert len(result) == 1
        assert result[0][1] == 0.9  # highest score kept

    def test_dedup_different_pages_kept(self):
        scored = [
            (_chunk("doc.pdf", 1), 0.9),
            (_chunk("doc.pdf", 2), 0.85),  # different page → kept
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert len(result) == 2

    def test_dedup_different_sources_kept(self):
        scored = [
            (_chunk("a.pdf", 1), 0.9),
            (_chunk("b.pdf", 1), 0.85),  # different source → kept
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert len(result) == 2

    def test_truncate_to_top_k(self):
        scored = [(_chunk(f"{i}.pdf", 1), 0.9 - i * 0.01) for i in range(10)]
        result = rerank(scored, top_k=3, threshold=0.0)
        assert len(result) == 3

    def test_fewer_than_top_k(self):
        scored = [
            (_chunk("a.pdf", 1), 0.9),
            (_chunk("b.pdf", 1), 0.8),
        ]
        result = rerank(scored, top_k=5, threshold=0.7)
        assert len(result) == 2

    def test_full_pipeline(self):
        """Filter + sort + dedup + truncate in combination."""
        scored = [
            (_chunk("a.pdf", 1, "a_c0"), 0.5),   # filtered (below 0.7)
            (_chunk("b.pdf", 1, "b_c0"), 0.95),   # kept (highest)
            (_chunk("b.pdf", 1, "b_c1"), 0.9),    # deduped (same source+page)
            (_chunk("c.pdf", 1, "c_c0"), 0.85),   # kept
            (_chunk("d.pdf", 1, "d_c0"), 0.8),    # kept
            (_chunk("e.pdf", 1, "e_c0"), 0.75),   # truncated (top_k=3)
        ]
        result = rerank(scored, top_k=3, threshold=0.7)
        assert len(result) == 3
        assert result[0][0].source == "b.pdf"
        assert result[0][1] == 0.95
        assert result[1][0].source == "c.pdf"
        assert result[2][0].source == "d.pdf"
