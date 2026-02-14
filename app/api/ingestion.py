"""Ingestion API endpoints.

Thin HTTP layer — no business logic, no PDF parsing, no chunking.
Just: receive request → call core → return response.
"""

import logging
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.core.ingest_service import (
    IngestError,
    delete_document,
    list_documents,
    run as run_ingest,
    run_batch,
)
from app.models.schemas import BatchIngestResponse, DocumentInfo, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/ingest", response_model=IngestResponse)
def ingest(file: UploadFile) -> IngestResponse:
    """Upload a PDF file, process it, embed it, store it."""

    # --- Validate file type ------------------------------------------------
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # --- Save uploaded file to disk ----------------------------------------
    os.makedirs(settings.data_dir, exist_ok=True)
    temp_name = f"_upload_{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(settings.data_dir, temp_name)

    try:
        contents = file.file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(temp_path, "wb") as f:
            f.write(contents)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # --- Run ingestion pipeline --------------------------------------------
    try:
        response = run_ingest(temp_path, filename)
    except IngestError as exc:
        # Clean up the temp file on failure.
        _safe_remove(temp_path)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _safe_remove(temp_path)
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during ingestion: {exc}",
        )

    # --- Rename temp file to {document_id}.pdf -----------------------------
    final_path = os.path.join(settings.data_dir, f"{response.document_id}.pdf")
    try:
        os.rename(temp_path, final_path)
    except OSError:
        # Not critical — the temp file still exists, just has a different name.
        logger.warning("Could not rename %s to %s", temp_path, final_path)

    return response


@router.post("/api/ingest/batch", response_model=BatchIngestResponse)
def ingest_batch(files: list[UploadFile]) -> BatchIngestResponse:
    """Upload multiple PDF files for ingestion.

    Each file is processed sequentially. One failure does not stop the batch.
    Disk persistence happens once after all files are processed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    os.makedirs(settings.data_dir, exist_ok=True)

    # Save all uploads to disk, collecting (temp_path, filename) pairs.
    saved: list[tuple[str, str]] = []  # (temp_path, filename) — "" path = skip
    for file in files:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            saved.append(("", filename))
            continue

        temp_path = os.path.join(
            settings.data_dir, f"_upload_{uuid.uuid4().hex}.pdf"
        )
        try:
            contents = file.file.read()
            if not contents:
                saved.append(("", filename))
                continue
            with open(temp_path, "wb") as f:
                f.write(contents)
            saved.append((temp_path, filename))
        except Exception as exc:
            logger.warning("Failed to save '%s': %s", filename, exc)
            saved.append(("", filename))

    file_entries = [(path, name) for path, name in saved if path]
    skipped = [(path, name) for path, name in saved if not path]

    # Run batch ingestion (sequential, one persist at the end).
    results = run_batch(file_entries) if file_entries else []

    # Rename successful temp files to {document_id}.pdf, clean up failures.
    for (temp_path, _), result in zip(file_entries, results):
        if result.status == "completed" and result.document_id:
            final_path = os.path.join(
                settings.data_dir, f"{result.document_id}.pdf"
            )
            try:
                os.rename(temp_path, final_path)
            except OSError:
                logger.warning("Could not rename %s to %s", temp_path, final_path)
        else:
            _safe_remove(temp_path)

    # Add failure entries for skipped files.
    for _, name in skipped:
        reason = (
            "Only PDF files are accepted."
            if not name.lower().endswith(".pdf")
            else "Empty file."
        )
        results.append(IngestResponse(
            document_id="",
            filename=name,
            page_count=0,
            chunk_count=0,
            status="failed",
            message=reason,
        ))

    succeeded = sum(1 for r in results if r.status == "completed")
    return BatchIngestResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


@router.get("/api/documents", response_model=list[DocumentInfo])
def get_documents() -> list[DocumentInfo]:
    """List all ingested documents."""
    return list_documents()


@router.delete("/api/documents/{document_id}")
def remove_document(document_id: str) -> dict:
    """Delete a document and all its chunks."""
    try:
        delete_document(document_id)
    except IngestError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {"status": "deleted", "document_id": document_id}


def _safe_remove(path: str) -> None:
    """Delete a file without raising if it doesn't exist."""
    try:
        os.remove(path)
    except OSError:
        pass
