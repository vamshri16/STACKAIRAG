"""Ingestion API endpoints.

Thin HTTP layer — no business logic, no PDF parsing, no chunking.
Just: receive request → call core → return response.
"""

import os
import logging

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.core.ingest_service import (
    IngestError,
    delete_document,
    list_documents,
    run as run_ingest,
)
from app.models.schemas import DocumentInfo, IngestResponse

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

    # Use a temp name until we have the document_id, but since ingest_service
    # generates the ID, we save with the original filename for now and let the
    # service handle it.  Actually, we need a unique on-disk name to avoid
    # collisions.  We'll re-save after ingest with the document_id.
    import uuid

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
