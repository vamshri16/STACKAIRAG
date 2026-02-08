"""Shared test fixtures for the RAG pipeline test suite."""

import os
import struct

import numpy as np
import pytest

from app.config import settings
from app.models.schemas import Chunk, DocumentInfo
from app.storage.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_api_key = pytest.mark.skipif(
    not settings.mistral_api_key,
    reason="MISTRAL_API_KEY not configured — skipping live API test",
)

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three chunks from two different PDFs."""
    return [
        Chunk(
            chunk_id="doc.pdf_p1_c0",
            text=(
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data. It uses algorithms "
                "to find patterns and make predictions without being "
                "explicitly programmed for each task."
            ),
            source="doc.pdf",
            page=1,
        ),
        Chunk(
            chunk_id="doc.pdf_p2_c1",
            text=(
                "Neural networks consist of interconnected layers of nodes. "
                "Each layer transforms the input data. Deep learning uses "
                "many layers to learn complex representations."
            ),
            source="doc.pdf",
            page=2,
        ),
        Chunk(
            chunk_id="other.pdf_p1_c0",
            text=(
                "Revenue increased by 20% in 2023 compared to the previous "
                "year. The company reported total earnings of $5 million "
                "driven by strong product demand."
            ),
            source="other.pdf",
            page=1,
        ),
    ]


@pytest.fixture
def fake_embeddings() -> np.ndarray:
    """Deterministic 3x8 embeddings matrix (seeded random)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 8))


@pytest.fixture
def fresh_vector_store() -> VectorStore:
    """Empty VectorStore instance (not the module-level singleton)."""
    return VectorStore()


@pytest.fixture
def loaded_vector_store(
    sample_chunks: list[Chunk],
    fake_embeddings: np.ndarray,
) -> VectorStore:
    """VectorStore pre-loaded with sample chunks and embeddings."""
    store = VectorStore()
    store.add(sample_chunks, fake_embeddings)
    return store


# ---------------------------------------------------------------------------
# Test PDF fixture
# ---------------------------------------------------------------------------


def _build_minimal_pdf(text_pages: list[str]) -> bytes:
    """Build a minimal valid PDF from scratch (no external library needed).

    Creates a bare-bones PDF with one or more pages containing the given text.
    This avoids depending on reportlab or fpdf for tests.
    """
    objects: list[bytes] = []
    offsets: list[int] = []

    def add_obj(content: bytes) -> int:
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # Object 1 — Catalog
    catalog_num = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

    # Object 2 — Pages (placeholder, updated later)
    pages_num = add_obj(b"PLACEHOLDER")

    # Object 3 — Font
    font_num = add_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    page_obj_nums: list[int] = []
    for page_text in text_pages:
        # Content stream
        encoded = page_text.encode("latin-1", errors="replace")
        stream_content = b"BT /F1 12 Tf 72 720 Td (" + encoded + b") Tj ET"
        stream_obj = add_obj(
            b"<< /Length "
            + str(len(stream_content)).encode()
            + b" >>\nstream\n"
            + stream_content
            + b"\nendstream"
        )

        # Page object
        page_obj = add_obj(
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents "
            + str(stream_obj).encode()
            + b" 0 R "
            b"/Resources << /Font << /F1 "
            + str(font_num).encode()
            + b" 0 R >> >> >>"
        )
        page_obj_nums.append(page_obj)

    # Fix up Pages object
    kids = b" ".join(str(n).encode() + b" 0 R" for n in page_obj_nums)
    objects[pages_num - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(page_obj_nums)).encode() + b" >>"
    )

    # Serialize
    buf = bytearray(b"%PDF-1.4\n")
    for i, obj_content in enumerate(objects):
        offsets.append(len(buf))
        obj_num = i + 1
        buf += f"{obj_num} 0 obj\n".encode()
        buf += obj_content
        buf += b"\nendobj\n"

    xref_offset = len(buf)
    buf += b"xref\n"
    buf += f"0 {len(objects) + 1}\n".encode()
    buf += b"0000000000 65535 f \n"
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode()

    buf += b"trailer\n"
    buf += (
        b"<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\n"
    )
    buf += b"startxref\n"
    buf += str(xref_offset).encode() + b"\n"
    buf += b"%%EOF\n"

    return bytes(buf)


@pytest.fixture
def test_pdf_path(tmp_path) -> str:
    """Create a tiny valid PDF with 2 pages of text."""
    pdf_bytes = _build_minimal_pdf([
        "This is page one about machine learning and artificial intelligence.",
        "This is page two about natural language processing and deep learning.",
    ])
    path = tmp_path / "test_sample.pdf"
    path.write_bytes(pdf_bytes)
    return str(path)


@pytest.fixture
def empty_pdf_path(tmp_path) -> str:
    """Create a 0-byte file pretending to be a PDF."""
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture
def corrupt_pdf_path(tmp_path) -> str:
    """Create a file with random bytes (not a valid PDF)."""
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is definitely not a PDF file content!!!")
    return str(path)
