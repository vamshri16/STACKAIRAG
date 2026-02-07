from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single piece of text from a PDF.

    THIS SCHEMA IS LOCKED. Do not change it.
    Retrieval, citations, and prompts all depend on this structure.
    """

    chunk_id: str   # Deterministic: "{filename}_p{page}_c{index}"
    text: str       # The actual text content
    source: str     # Original PDF filename
    page: int       # Page number (1-indexed)


class DocumentInfo(BaseModel):
    """Tracks an ingested document."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    ingested_at: datetime


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Returned after a PDF is uploaded and processed."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str     # "completed" or "failed"
    message: str


class QueryRequest(BaseModel):
    """Request body for POST /api/query (Phase 3)."""

    query: str
    top_k: int = 5
    include_sources: bool = True
    session_id: str | None = None


class Source(BaseModel):
    """A single source citation in a query response."""

    chunk_id: str
    source: str
    page: int
    text: str
    score: float


class QueryResponse(BaseModel):
    """Returned from POST /api/query (Phase 3)."""

    answer: str
    sources: list[Source]
    confidence: float
    intent: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Returned from GET /api/health."""

    status: str
    mistral_configured: bool
    documents_count: int
    chunks_count: int
    data_dir: str
    index_dir: str
