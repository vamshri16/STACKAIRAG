"""In-memory vector store backed by NumPy.

Dumb by design — no business logic, no scoring, no ranking.
Just: add, get_all, delete, save, load, clear.

Replaces FAISS.  No external search libraries.
"""

import json
import logging
import os

import numpy as np

from app.models.schemas import Chunk, DocumentInfo

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILE = "embeddings.npy"
_CHUNKS_FILE = "chunks.json"
_DOCUMENTS_FILE = "documents.json"


class VectorStore:
    """NumPy-backed vector store.

    ``embeddings[i]`` is always the embedding for ``chunks[i]``.
    """

    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.documents: dict[str, DocumentInfo] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append *chunks* and their *embeddings* to the store.

        ``embeddings`` must have shape ``(len(chunks), dim)``.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but "
                f"{embeddings.shape[0]} embedding rows."
            )

        if self.embeddings is None:
            self.embeddings = embeddings.copy()
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)

    def get_all(self) -> tuple[list[Chunk], np.ndarray | None]:
        """Return all chunks and the embeddings matrix.

        Search logic belongs in ``core/``, not here.  The store just
        hands over the data.
        """
        return self.chunks, self.embeddings

    def delete_by_source(self, filename: str) -> int:
        """Remove all chunks whose ``source`` matches *filename*.

        Returns the number of chunks removed.
        """
        if not self.chunks:
            return 0

        keep_mask = [c.source != filename for c in self.chunks]
        removed = keep_mask.count(False)

        if removed == 0:
            return 0

        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]

        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep_mask]
            # If all rows were removed, reset to None.
            if self.embeddings.shape[0] == 0:
                self.embeddings = None

        # Remove the document record.
        doc_ids_to_remove = [
            doc_id
            for doc_id, info in self.documents.items()
            if info.filename == filename
        ]
        for doc_id in doc_ids_to_remove:
            del self.documents[doc_id]

        return removed

    def clear(self) -> None:
        """Remove everything.  Reset to empty state."""
        self.embeddings = None
        self.chunks = []
        self.documents = {}

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the store to *path*.

        Creates the directory if it does not exist.
        - ``embeddings.npy`` — NumPy matrix.
        - ``chunks.json`` — chunk metadata (JSON, human-readable).
        - ``documents.json`` — document records.
        """
        os.makedirs(path, exist_ok=True)

        # Embeddings.
        if self.embeddings is not None:
            np.save(os.path.join(path, _EMBEDDINGS_FILE), self.embeddings)
        else:
            # Remove stale file if the store is now empty.
            emb_path = os.path.join(path, _EMBEDDINGS_FILE)
            if os.path.exists(emb_path):
                os.remove(emb_path)

        # Chunks.
        chunks_data = [c.model_dump() for c in self.chunks]
        with open(os.path.join(path, _CHUNKS_FILE), "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Documents.
        docs_data = {
            doc_id: info.model_dump(mode="json")
            for doc_id, info in self.documents.items()
        }
        with open(os.path.join(path, _DOCUMENTS_FILE), "w") as f:
            json.dump(docs_data, f, indent=2, default=str)

        logger.info(
            "Vector store saved to %s (%d chunks, %d documents).",
            path, len(self.chunks), len(self.documents),
        )

    def load(self, path: str) -> None:
        """Load a previously saved store from *path*.

        If the directory does not exist or is missing files, the store
        remains empty — no error is raised.
        """
        emb_path = os.path.join(path, _EMBEDDINGS_FILE)
        chunks_path = os.path.join(path, _CHUNKS_FILE)
        docs_path = os.path.join(path, _DOCUMENTS_FILE)

        # Embeddings.
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = None

        # Chunks.
        if os.path.exists(chunks_path):
            with open(chunks_path) as f:
                chunks_data = json.load(f)
            self.chunks = [Chunk(**item) for item in chunks_data]
        else:
            self.chunks = []

        # Documents.
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                docs_data = json.load(f)
            self.documents = {
                doc_id: DocumentInfo(**info)
                for doc_id, info in docs_data.items()
            }
        else:
            self.documents = {}

        logger.info(
            "Vector store loaded from %s (%d chunks, %d documents).",
            path, len(self.chunks), len(self.documents),
        )

        # Consistency check: embeddings and chunks must have matching length.
        if self.embeddings is not None and self.embeddings.shape[0] != len(self.chunks):
            logger.error(
                "Embeddings/chunks mismatch: %d rows vs %d chunks. Resetting store.",
                self.embeddings.shape[0], len(self.chunks),
            )
            self.clear()


# ---------------------------------------------------------------------------
# Module-level singleton.
# ---------------------------------------------------------------------------
vector_store = VectorStore()
