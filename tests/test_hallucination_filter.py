"""Unit tests for app/core/hallucination_filter.py."""

from app.core.hallucination_filter import compute_confidence
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    """Helper to create a minimal chunk."""
    return Chunk(chunk_id="x", text=text, source="test.pdf", page=1)


class TestComputeConfidence:
    def test_empty_answer(self):
        chunks = [_chunk("Machine learning is great.")]
        assert compute_confidence("", chunks) == 0.0

    def test_whitespace_answer(self):
        chunks = [_chunk("Machine learning is great.")]
        assert compute_confidence("   ", chunks) == 0.0

    def test_no_chunks(self):
        assert compute_confidence("Machine learning is great.", []) == 0.0

    def test_fully_supported_answer(self):
        chunks = [
            _chunk(
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data."
            ),
        ]
        answer = "Machine learning is a subset of artificial intelligence."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5

    def test_unsupported_answer(self):
        chunks = [_chunk("Revenue increased by 20% in 2023.")]
        answer = "Quantum computing will revolutionize cryptography and encryption methods."
        score = compute_confidence(answer, chunks)
        assert score < 0.5

    def test_partially_supported(self):
        chunks = [
            _chunk("Machine learning uses algorithms to find patterns in data."),
        ]
        # One sentence supported, one not.
        answer = (
            "Machine learning uses algorithms to find patterns. "
            "Quantum physics describes subatomic particle behavior."
        )
        score = compute_confidence(answer, chunks)
        assert 0.0 < score < 1.0

    def test_meta_statements_skipped(self):
        """Meta-statements like 'Based on the...' should be skipped."""
        chunks = [_chunk("Machine learning uses algorithms.")]
        answer = "Based on the provided documents, machine learning uses algorithms."
        score = compute_confidence(answer, chunks)
        # The meta-statement is skipped; the factual part is evaluated.
        assert score > 0.0

    def test_all_meta_returns_1(self):
        """If ALL sentences are meta-statements, confidence should be 1.0."""
        chunks = [_chunk("Some content here.")]
        answer = "I don't have enough information."
        score = compute_confidence(answer, chunks)
        assert score == 1.0

    def test_short_fragments_skipped(self):
        """Fragments with < 3 tokens after tokenization should be skipped."""
        chunks = [_chunk("Machine learning algorithms process data.")]
        # "OK." has < 3 meaningful tokens after stopword removal.
        answer = "OK."
        score = compute_confidence(answer, chunks)
        # Skipped entirely → all meta → 1.0.
        assert score == 1.0

    def test_multiple_chunks_best_overlap_used(self):
        """The best overlap across all chunks should be used."""
        chunks = [
            _chunk("Revenue increased by 20 percent."),
            _chunk("Machine learning algorithms find patterns in data."),
        ]
        answer = "Machine learning algorithms find patterns in data."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5

    def test_single_supported_sentence(self):
        chunks = [_chunk("Neural networks have interconnected layers of nodes.")]
        answer = "Neural networks have interconnected layers of nodes."
        score = compute_confidence(answer, chunks)
        assert score >= 0.5
