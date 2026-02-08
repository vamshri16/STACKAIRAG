"""Unit tests for app/storage/vector_store.py."""

import json

import numpy as np
import pytest

from app.models.schemas import Chunk, DocumentInfo
from app.storage.vector_store import VectorStore


class TestVectorStoreEmpty:
    def test_empty_length(self, fresh_vector_store: VectorStore):
        assert len(fresh_vector_store) == 0

    def test_empty_get_all(self, fresh_vector_store: VectorStore):
        chunks, embs = fresh_vector_store.get_all()
        assert chunks == []
        assert embs is None

    def test_clear(self, loaded_vector_store: VectorStore):
        loaded_vector_store.clear()
        assert len(loaded_vector_store) == 0
        assert loaded_vector_store.embeddings is None
        assert loaded_vector_store.chunks == []
        assert loaded_vector_store.documents == {}


class TestVectorStoreAdd:
    def test_add_sets_data(
        self,
        fresh_vector_store: VectorStore,
        sample_chunks: list[Chunk],
        fake_embeddings: np.ndarray,
    ):
        fresh_vector_store.add(sample_chunks, fake_embeddings)
        assert len(fresh_vector_store) == 3
        assert fresh_vector_store.embeddings is not None
        assert fresh_vector_store.embeddings.shape == (3, 8)

    def test_add_validates_mismatch(self, fresh_vector_store: VectorStore):
        chunks = [
            Chunk(chunk_id="a", text="hello", source="a.pdf", page=1),
        ]
        embeddings = np.random.randn(2, 4)  # 2 rows but 1 chunk
        with pytest.raises(ValueError, match="Mismatch"):
            fresh_vector_store.add(chunks, embeddings)

    def test_multiple_adds(self, fresh_vector_store: VectorStore):
        c1 = [Chunk(chunk_id="a", text="hello", source="a.pdf", page=1)]
        e1 = np.random.randn(1, 4)
        c2 = [Chunk(chunk_id="b", text="world", source="b.pdf", page=1)]
        e2 = np.random.randn(1, 4)

        fresh_vector_store.add(c1, e1)
        fresh_vector_store.add(c2, e2)

        assert len(fresh_vector_store) == 2
        assert fresh_vector_store.embeddings.shape == (2, 4)

    def test_parallel_arrays_invariant(self, loaded_vector_store: VectorStore):
        chunks, embs = loaded_vector_store.get_all()
        assert len(chunks) == embs.shape[0]
        assert chunks[0].chunk_id == "doc.pdf_p1_c0"


class TestVectorStoreDelete:
    def test_delete_by_source(self, loaded_vector_store: VectorStore):
        removed = loaded_vector_store.delete_by_source("doc.pdf")
        assert removed == 2
        assert len(loaded_vector_store) == 1
        assert loaded_vector_store.chunks[0].source == "other.pdf"

    def test_delete_nonexistent_source(self, loaded_vector_store: VectorStore):
        removed = loaded_vector_store.delete_by_source("nonexistent.pdf")
        assert removed == 0
        assert len(loaded_vector_store) == 3

    def test_delete_all_resets_embeddings(self, loaded_vector_store: VectorStore):
        loaded_vector_store.delete_by_source("doc.pdf")
        loaded_vector_store.delete_by_source("other.pdf")
        assert len(loaded_vector_store) == 0
        assert loaded_vector_store.embeddings is None


class TestVectorStorePersistence:
    def test_save_and_load(self, loaded_vector_store: VectorStore, tmp_path):
        save_path = str(tmp_path / "store")
        loaded_vector_store.save(save_path)

        new_store = VectorStore()
        new_store.load(save_path)

        assert len(new_store) == len(loaded_vector_store)
        assert new_store.chunks[0].chunk_id == loaded_vector_store.chunks[0].chunk_id
        np.testing.assert_array_almost_equal(
            new_store.embeddings, loaded_vector_store.embeddings
        )

    def test_save_empty_store(self, fresh_vector_store: VectorStore, tmp_path):
        save_path = str(tmp_path / "empty_store")
        fresh_vector_store.save(save_path)

        new_store = VectorStore()
        new_store.load(save_path)
        assert len(new_store) == 0

    def test_load_nonexistent_path(self, fresh_vector_store: VectorStore):
        fresh_vector_store.load("/nonexistent/path/that/does/not/exist")
        assert len(fresh_vector_store) == 0

    def test_load_consistency_check_resets(self, tmp_path):
        """If embeddings and chunks have different lengths, store resets."""
        save_path = str(tmp_path / "bad_store")
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save 3 embeddings but only 2 chunks.
        np.save(os.path.join(save_path, "embeddings.npy"), np.random.randn(3, 4))
        with open(os.path.join(save_path, "chunks.json"), "w") as f:
            json.dump(
                [
                    {"chunk_id": "a", "text": "hello", "source": "a.pdf", "page": 1},
                    {"chunk_id": "b", "text": "world", "source": "b.pdf", "page": 1},
                ],
                f,
            )

        store = VectorStore()
        store.load(save_path)
        # Mismatch detected → store should be cleared.
        assert len(store) == 0
        assert store.embeddings is None


class TestVectorStoreDocuments:
    def test_documents_dict(self, loaded_vector_store: VectorStore):
        from datetime import datetime, timezone

        doc = DocumentInfo(
            document_id="abc123",
            filename="doc.pdf",
            page_count=2,
            chunk_count=2,
            ingested_at=datetime.now(timezone.utc),
        )
        loaded_vector_store.documents["abc123"] = doc
        assert "abc123" in loaded_vector_store.documents
        assert loaded_vector_store.documents["abc123"].filename == "doc.pdf"
