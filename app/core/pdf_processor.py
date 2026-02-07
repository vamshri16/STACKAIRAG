"""PDF text extraction and chunking.

Pure logic — no FastAPI imports, no request/response objects.
Every function is testable in isolation.

Pipeline:  PDF → pages → cleaned text → chunks
Each step = one function.
"""

import logging

import PyPDF2
import pdfplumber

from app.config import settings
from app.models.schemas import Chunk
from app.utils.text_utils import clean_text, split_into_chunks

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """Read a PDF and return a list of ``(page_number, raw_text)`` tuples.

    Strategy:
    1. Try PyPDF2 first (fast).
    2. If a page yields no text, fall back to pdfplumber for that page.
    3. Page numbers are 1-indexed to match what users see in their viewer.

    Raises ``PDFProcessingError`` on unrecoverable failures.
    """
    pages: list[tuple[int, str]] = []

    # --- Attempt with PyPDF2 ------------------------------------------------
    try:
        reader = PyPDF2.PdfReader(file_path)
    except PyPDF2.errors.PdfReadError as exc:
        raise PDFProcessingError(f"Corrupted or invalid PDF: {exc}") from exc
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise PDFProcessingError("PDF has no pages")

    for i, page in enumerate(reader.pages):
        page_number = i + 1  # 1-indexed
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if text.strip():
            pages.append((page_number, text))
            continue

        # --- Fallback to pdfplumber for this page ---------------------------
        try:
            with pdfplumber.open(file_path) as pdf:
                if i < len(pdf.pages):
                    plumber_text = pdf.pages[i].extract_text() or ""
                    if plumber_text.strip():
                        pages.append((page_number, plumber_text))
                        continue
        except Exception:
            pass

        # Both extractors returned nothing for this page — skip it.
        logger.warning("No text extracted from page %d of %s", page_number, file_path)

    return pages


def process_pdf(file_path: str, filename: str) -> list[Chunk]:
    """Full ingestion pipeline: extract → clean → chunk → attach metadata.

    Returns a list of ``Chunk`` objects with deterministic IDs.
    Raises ``PDFProcessingError`` if the PDF is empty or yields no chunks.
    """
    # Step 1 — extract raw text per page.
    pages = extract_text_from_pdf(file_path)

    if not pages:
        raise PDFProcessingError(
            f"Could not extract any text from '{filename}'. "
            "The PDF may be scanned (image-only) or empty."
        )

    # Step 2 — clean text and split into chunks.
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        page_chunks = split_into_chunks(
            cleaned,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue

            chunk = Chunk(
                chunk_id=f"{filename}_p{page_number}_c{chunk_index}",
                text=chunk_text,
                source=filename,
                page=page_number,
            )
            chunks.append(chunk)
            chunk_index += 1

    # Step 3 — fail fast if no chunks were produced.
    if not chunks:
        raise PDFProcessingError(
            f"PDF '{filename}' produced no text chunks after processing."
        )

    return chunks
