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

    Processes the file, stores results, and persists to disk.
    Raises ``IngestError`` with a user-facing message on failure.
    """
    response = _run_single(file_path, filename)

    # Persist to disk.
    try:
        vector_store.save(settings.index_dir)
    except OSError as exc:
        logger.error("Failed to persist vector store: %s", exc)

    return response


def _run_single(file_path: str, filename: str) -> IngestResponse:
    """Core ingestion logic for one PDF — no disk persistence.

    Steps:
    1. Process PDF → list[Chunk]
    2. Generate embeddings → np.ndarray
    3. Store chunks + embeddings in memory
    4. Return IngestResponse

    Raises ``IngestError`` with a user-facing message on failure.
    """
    document_id = uuid.uuid4().hex

    # Step 1 — Extract text and create chunks.
    try:
        chunks = process_pdf(file_path, filename)
    except PDFProcessingError as exc:
        raise IngestError(str(exc)) from exc

    page_count = len({c.page for c in chunks})

    # Step 2 — Generate embeddings.
    texts = [c.text for c in chunks]
    try:
        embeddings = get_embeddings_batch(texts)
    except EmbeddingError as exc:
        raise IngestError(f"Embedding failed: {exc}") from exc

    # Step 3 — Store in vector store (memory only).
    vector_store.add(chunks, embeddings)

    doc_info = DocumentInfo(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
        ingested_at=datetime.now(timezone.utc),
    )
    vector_store.documents[document_id] = doc_info

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


def run_batch(
    file_entries: list[tuple[str, str]],
) -> list[IngestResponse]:
    """Process multiple PDFs sequentially, persist once at the end.

    Each entry is (file_path, original_filename).
    Returns one IngestResponse per file — failures are recorded but
    don't stop the rest of the batch.
    """
    results: list[IngestResponse] = []

    for file_path, filename in file_entries:
        try:
            response = _run_single(file_path, filename)
            results.append(response)
        except IngestError as exc:
            logger.warning("Batch ingest failed for '%s': %s", filename, exc)
            results.append(IngestResponse(
                document_id="",
                filename=filename,
                page_count=0,
                chunk_count=0,
                status="failed",
                message=str(exc),
            ))

    # Persist once after all files are processed.
    if any(r.status == "completed" for r in results):
        try:
            vector_store.save(settings.index_dir)
        except OSError as exc:
            logger.error("Failed to persist vector store after batch: %s", exc)

    return results


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
