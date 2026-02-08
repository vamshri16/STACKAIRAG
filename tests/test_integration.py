"""Integration tests — require MISTRAL_API_KEY set in the environment.

These tests hit the real Mistral API so they're skipped by default.
Run with: MISTRAL_API_KEY=your_key pytest -m integration
"""

import numpy as np
import pytest

from tests.conftest import requires_api_key, integration

from app.core.embeddings import get_embedding, get_embeddings_batch
from app.core.llm_client import generate, build_qa_prompt, format_context
from app.core.search import hybrid_search
from app.models.schemas import Chunk
from app.storage.vector_store import VectorStore


@integration
@requires_api_key
class TestEmbeddingsIntegration:
    def test_single_embedding(self):
        result = get_embedding("What is machine learning?")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] > 0

    def test_batch_embeddings(self):
        texts = [
            "Machine learning is a subset of AI.",
            "Revenue increased by 20%.",
        ]
        result = get_embeddings_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
        assert result.ndim == 2


@integration
@requires_api_key
class TestLLMIntegration:
    def test_generate_answer(self):
        chunk = Chunk(
            chunk_id="test_p1_c0",
            text="Machine learning enables computers to learn from data.",
            source="test.pdf",
            page=1,
        )
        context = format_context([(chunk, 0.95)])
        messages = build_qa_prompt("What is machine learning?", context)
        answer = generate(messages)
        assert isinstance(answer, str)
        assert len(answer) > 0


@integration
@requires_api_key
class TestSearchIntegration:
    def test_hybrid_search_with_loaded_store(self):
        """Requires vector store to have data. Skips if empty."""
        from app.storage.vector_store import vector_store

        if len(vector_store) == 0:
            pytest.skip("Vector store is empty — no data to search.")

        results = hybrid_search("machine learning", top_k=3)
        assert isinstance(results, list)
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)
