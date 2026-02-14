"""Unit tests for app/core/hallucination_filter.py."""

from unittest.mock import patch

import numpy as np

from app.core.hallucination_filter import compute_confidence
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    """Helper to create a minimal chunk."""
    return Chunk(chunk_id="x", text=text, source="test.pdf", page=1)


def _force_token_overlap():
    """Patch decorator: disable semantic path so tests exercise token-overlap."""
    return patch(
        "app.core.hallucination_filter._semantic_confidence",
        return_value=None,
    )


# ------------------------------------------------------------------
# Edge cases (no semantic/token-overlap needed)
# ------------------------------------------------------------------


class TestComputeConfidenceEdgeCases:
    def test_empty_answer(self):
        chunks = [_chunk("Machine learning is great.")]
        assert compute_confidence("", chunks) == 0.0

    def test_whitespace_answer(self):
        chunks = [_chunk("Machine learning is great.")]
        assert compute_confidence("   ", chunks) == 0.0

    def test_no_chunks(self):
        assert compute_confidence("Machine learning is great.", []) == 0.0

    def test_all_meta_returns_1(self):
        chunks = [_chunk("Some content here.")]
        answer = "I don't have enough information."
        assert compute_confidence(answer, chunks) == 1.0

    def test_short_fragments_skipped(self):
        chunks = [_chunk("Machine learning algorithms process data.")]
        answer = "OK."
        assert compute_confidence(answer, chunks) == 1.0


# ------------------------------------------------------------------
# Token-overlap path
# ------------------------------------------------------------------


class TestTokenOverlapConfidence:
    @_force_token_overlap()
    def test_fully_supported(self, _mock):
        chunks = [
            _chunk(
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data."
            ),
        ]
        answer = "Machine learning is a subset of artificial intelligence."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5

    @_force_token_overlap()
    def test_unsupported(self, _mock):
        chunks = [_chunk("Revenue increased by 20% in 2023.")]
        answer = "Quantum computing will revolutionize cryptography and encryption methods."
        score = compute_confidence(answer, chunks)
        assert score < 0.5

    @_force_token_overlap()
    def test_partially_supported(self, _mock):
        chunks = [
            _chunk("Machine learning uses algorithms to find patterns in data."),
        ]
        answer = (
            "Machine learning uses algorithms to find patterns. "
            "Quantum physics describes subatomic particle behavior."
        )
        score = compute_confidence(answer, chunks)
        assert 0.0 < score < 1.0

    @_force_token_overlap()
    def test_meta_statements_skipped(self, _mock):
        chunks = [_chunk("Machine learning uses algorithms.")]
        answer = "Based on the provided documents, machine learning uses algorithms."
        score = compute_confidence(answer, chunks)
        assert score > 0.0

    @_force_token_overlap()
    def test_multiple_chunks_best_overlap_used(self, _mock):
        chunks = [
            _chunk("Revenue increased by 20 percent."),
            _chunk("Machine learning algorithms find patterns in data."),
        ]
        answer = "Machine learning algorithms find patterns in data."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5

    @_force_token_overlap()
    def test_single_supported_sentence(self, _mock):
        chunks = [_chunk("Neural networks have interconnected layers of nodes.")]
        answer = "Neural networks have interconnected layers of nodes."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5


# ------------------------------------------------------------------
# Semantic path
# ------------------------------------------------------------------


class TestSemanticConfidence:
    def _mock_embeddings(self, texts):
        """Return fake embeddings: similar texts get similar vectors."""
        vecs = []
        for t in texts:
            # Deterministic pseudo-embedding based on token hash.
            np.random.seed(hash(t.lower().strip()) % 2**31)
            vecs.append(np.random.randn(64))
        return np.array(vecs)

    def test_semantic_supported(self):
        """Same text should produce high similarity → supported."""
        shared = "Machine learning uses algorithms to find patterns."
        chunks = [_chunk(shared)]
        answer = shared

        with patch(
            "app.core.hallucination_filter.get_embeddings_batch",
            side_effect=self._mock_embeddings,
        ):
            score = compute_confidence(answer, chunks)

        # Same text → identical embedding → cosine sim = 1.0 → supported.
        assert score >= 0.5

    def test_semantic_unsupported(self):
        """Completely different text should produce low similarity."""
        chunks = [_chunk("Revenue increased by 20% in 2023.")]
        answer = "Quantum computing will revolutionize cryptography."

        with patch(
            "app.core.hallucination_filter.get_embeddings_batch",
            side_effect=self._mock_embeddings,
        ):
            score = compute_confidence(answer, chunks)

        # Different seed → different vector → low similarity.
        assert score < 0.5

    def test_semantic_fallback_on_error(self):
        """If embedding call fails, should fall back to token-overlap."""
        from app.core.embeddings import EmbeddingError

        chunks = [_chunk("Neural networks have interconnected layers of nodes.")]
        answer = "Neural networks have interconnected layers of nodes."

        with patch(
            "app.core.hallucination_filter.get_embeddings_batch",
            side_effect=EmbeddingError("test error"),
        ):
            score = compute_confidence(answer, chunks)

        # Falls back to token-overlap, which should find high overlap.
        assert score >= 0.5
