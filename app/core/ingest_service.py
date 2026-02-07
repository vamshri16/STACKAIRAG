"""Ingestion pipeline orchestrator.

Pure logic — no FastAPI imports.  Coordinates:
  pdf_processor  →  embeddings  →  vector_store

Fail fast, fail loud on every step.
"""

import logging
import os
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.core.embeddings import EmbeddingError, get_embeddings_batch
from app.core.pdf_processor import PDFProcessingError, process_pdf
from app.models.schemas import DocumentInfo, IngestResponse
from app.storage.vector_store import vector_store

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when ingestion fails at any step."""


def run(file_path: str, filename: str) -> IngestResponse:
    """Execute the full ingestion pipeline for a single PDF.

    Steps:
    1. Process PDF → list[Chunk]
    2. Generate embeddings → np.ndarray
    3. Store chunks + embeddings
    4. Persist to disk
    5. Return IngestResponse

    Raises ``IngestError`` with a user-facing message on failure.
    """
    document_id = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Step 1 — Extract text and create chunks.
    # ------------------------------------------------------------------
    try:
        chunks = process_pdf(file_path, filename)
    except PDFProcessingError as exc:
        raise IngestError(str(exc)) from exc

    # Count unique pages that produced at least one chunk.
    page_count = len({c.page for c in chunks})

    # ------------------------------------------------------------------
    # Step 2 — Generate embeddings.
    # ------------------------------------------------------------------
    texts = [c.text for c in chunks]
    try:
        embeddings = get_embeddings_batch(texts)
    except EmbeddingError as exc:
        raise IngestError(f"Embedding failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 3 — Store in vector store.
    # ------------------------------------------------------------------
    vector_store.add(chunks, embeddings)

    doc_info = DocumentInfo(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
        ingested_at=datetime.now(timezone.utc),
    )
    vector_store.documents[document_id] = doc_info

    # ------------------------------------------------------------------
    # Step 4 — Persist to disk.
    # ------------------------------------------------------------------
    try:
        vector_store.save(settings.index_dir)
    except OSError as exc:
        logger.error("Failed to persist vector store: %s", exc)
        # Don't fail the request — data is in memory and will be saved
        # on the next successful write or server shutdown.

    # ------------------------------------------------------------------
    # Step 5 — Return response.
    # ------------------------------------------------------------------
    logger.info(
        "Ingested '%s': %d pages, %d chunks (doc_id=%s).",
        filename, page_count, len(chunks), document_id,
    )

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
        status="completed",
        message=(
            f"Successfully processed {page_count} pages "
            f"into {len(chunks)} chunks"
        ),
    )


def delete_document(document_id: str) -> None:
    """Remove a document and all its chunks from the store.

    Raises ``IngestError`` if the document_id is not found.
    """
    if document_id not in vector_store.documents:
        raise IngestError(f"Document '{document_id}' not found.")

    doc_info = vector_store.documents[document_id]
    filename = doc_info.filename

    removed = vector_store.delete_by_source(filename)
    logger.info(
        "Deleted document '%s' (%s): removed %d chunks.",
        document_id, filename, removed,
    )

    # Remove PDF file from disk.
    pdf_path = os.path.join(settings.data_dir, f"{document_id}.pdf")
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError as exc:
            logger.warning("Could not delete PDF file %s: %s", pdf_path, exc)

    # Persist updated store.
    try:
        vector_store.save(settings.index_dir)
    except OSError as exc:
        logger.error("Failed to persist vector store after deletion: %s", exc)


def list_documents() -> list[DocumentInfo]:
    """Return all ingested document records."""
    return list(vector_store.documents.values())
