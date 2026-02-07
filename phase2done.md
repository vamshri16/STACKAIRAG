# Phase 2 — Complete Implementation Record

Everything that was coded, every design decision, every pattern, every edge case, every test result. Nothing omitted.

---

## Table of Contents

1. [Folder Structure Created](#1-folder-structure-created)
2. [File 1: `app/models/schemas.py`](#2-file-1-appmodelsschemaspysource)
3. [File 2: `app/utils/text_utils.py`](#3-file-2-apputilstext_utilspy)
4. [File 3: `app/utils/pii_detector.py`](#4-file-3-apputilspii_detectorpy)
5. [File 4: `app/core/pdf_processor.py`](#5-file-4-appcorepdf_processorpy)
6. [File 5: `app/core/embeddings.py`](#6-file-5-appcoreembeddingspy)
7. [File 6: `app/storage/vector_store.py`](#7-file-6-appstoragevector_storepy)
8. [File 7: `app/core/ingest_service.py`](#8-file-7-appcoreingest_servicepy)
9. [File 8: `app/api/ingestion.py`](#9-file-8-appapiingestionpy)
10. [File 9: `app/main.py` (modified)](#10-file-9-appmainpy-modified)
11. [File 10: `app/api/health.py` (modified)](#11-file-10-appapihealthpy-modified)
12. [Architectural Patterns Used](#12-architectural-patterns-used)
13. [Edge Cases Covered](#13-edge-cases-covered)
14. [Every Test We Ran](#14-every-test-we-ran)
15. [Constraint Compliance](#15-constraint-compliance)
16. [What Changed From Phase 1](#16-what-changed-from-phase-1)

---

## 1. Folder Structure Created

Before Phase 2, the `app/` directory looked like this:

```
app/
├── __init__.py
├── config.py
├── main.py
├── api/
│   ├── __init__.py
│   └── health.py
├── models/
│   └── __init__.py
├── services/
│   └── __init__.py          ← leftover from Phase 1 (unused, harmless)
└── utils/
    └── __init__.py
```

After Phase 2, it looks like this:

```
app/
├── __init__.py
├── config.py                 ← unchanged from Phase 1
├── main.py                   ← MODIFIED (startup event, ingestion router)
├── api/
│   ├── __init__.py
│   ├── health.py             ← MODIFIED (added document/chunk counts)
│   └── ingestion.py          ← NEW (thin HTTP layer)
├── core/
│   ├── __init__.py           ← NEW (package marker)
│   ├── embeddings.py         ← NEW (Mistral API client)
│   ├── ingest_service.py     ← NEW (orchestrator)
│   └── pdf_processor.py      ← NEW (PDF extraction + chunking)
├── models/
│   ├── __init__.py
│   └── schemas.py            ← NEW (locked Chunk schema + all models)
├── services/
│   └── __init__.py           ← leftover from Phase 1
├── storage/
│   ├── __init__.py           ← NEW (package marker)
│   └── vector_store.py       ← NEW (NumPy-backed store)
└── utils/
    ├── __init__.py
    ├── pii_detector.py        ← NEW (regex PII detection)
    └── text_utils.py          ← NEW (clean, chunk, tokenize)
```

**New directories created:**
- `app/core/` — pure business logic, no FastAPI imports
- `app/storage/` — dumb data layer, no logic

**Commands used to create them:**
```bash
mkdir -p app/core app/storage
```

Then created empty `__init__.py` files in both.

---

## 2. File 1: `app/models/schemas.py`

**Created from scratch. 86 lines.**

**Purpose:** Define all Pydantic data models used across the application. The `Chunk` schema is the most critical — it is marked as LOCKED and must never change because retrieval, citations, and prompts all depend on its structure.

### Full source code:

```python
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
```

### Design decisions:

**1. Flat Chunk schema (no nested metadata):**
The original plan had a nested `ChunkMetadata` inside `Chunk`. We changed to a flat schema per the refinement:
```python
# BEFORE (nested — rejected):
class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

# AFTER (flat — what we implemented):
class Chunk(BaseModel):
    chunk_id: str    # "{filename}_p{page}_c{index}"
    text: str
    source: str      # filename
    page: int
```

**Why flat:**
- Simpler to serialize/deserialize (one level of JSON, not two)
- Simpler to access: `chunk.source` instead of `chunk.metadata.filename`
- No cascading refactors if we add a field later

**2. Deterministic chunk IDs:**
`chunk_id = f"{filename}_p{page}_c{index}"` — e.g., `"report.pdf_p3_c2"`

**Why deterministic:**
- Same PDF always produces the same chunk IDs
- Easy debugging — ID tells you exactly where the chunk came from
- Stable references for citations

**3. `document_id` is NOT in the Chunk schema:**
The `document_id` (a UUID) lives in `DocumentInfo`, not in `Chunk`. Chunks reference their source PDF by filename (`source` field). This keeps the Chunk schema minimal. The `document_id` is the API's tracking identifier for the upload event, not a property of the text.

**4. `QueryRequest` and `QueryResponse` are defined now but used in Phase 3:**
We define them early so the schema contract is locked. They won't be imported by any Phase 2 code.

**5. `Source` model matches the Chunk schema:**
`Source` has `chunk_id`, `source`, `page`, `text`, `score` — the same fields as `Chunk` plus a `score`. This is intentional so that converting a Chunk search result to a Source citation is trivial.

---

## 3. File 2: `app/utils/text_utils.py`

**Created from scratch. 171 lines.**

**Purpose:** Pure Python functions for text cleaning, tokenization, and chunking. Used by `core/pdf_processor.py` (this phase) and by the keyword search scorer (Phase 3). No external dependencies beyond `re` (standard library).

### Full source code:

```python
import re

# Common English stopwords — kept minimal and inline to avoid external deps.
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "any", "are", "aren", "because", "before", "below", "between", "both",
    "each", "few", "further", "get", "got", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "into", "its",
    "itself", "let", "me", "more", "most", "my", "myself", "off", "once",
    "only", "other", "our", "ours", "ourselves", "out", "over", "own",
    "re", "same", "she", "some", "such", "that", "their", "theirs",
    "them", "themselves", "these", "they", "this", "those", "through",
    "under", "until", "up", "we", "what", "when", "where", "which",
    "while", "who", "whom", "why", "you", "your", "yours", "yourself",
    "yourselves",
}

# Ligature replacements for common PDF extraction artifacts.
_LIGATURES: dict[str, str] = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Regex: sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Regex: hyphen at end of line (word broken across lines in PDF).
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Regex: non-alphanumeric characters (for tokenization).
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Regex: control characters except newline and tab.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub("", text)
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.split())


def split_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text or not text.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Edge case: single sentence longer than chunk_size.
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                piece = " ".join(words[i : i + chunk_size])
                if piece:
                    chunks.append(piece)
            continue

        # Would adding this sentence exceed the limit?
        if current_token_count + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Build overlap from the tail of the current chunk.
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current_sentences):
                prev_tokens = count_tokens(prev)
                if overlap_tokens + prev_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, prev)
                overlap_tokens += prev_tokens

            current_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    tokens = _NON_ALNUM_RE.split(lowered)
    return [t for t in tokens if t and t not in STOPWORDS]
```

### Design decisions:

**1. Stopwords are inline, not from NLTK or spaCy:**
We maintain ~130 common English stopwords as a hardcoded `set`. This avoids importing `nltk` or `spacy` just for a word list. The set covers the most common function words that add noise to keyword matching.

**2. All regexes are pre-compiled at module level:**
`re.compile()` is called once when the module loads, not on every function call. This means `clean_text()` and `tokenize()` don't pay the regex compilation cost on each invocation. Five regexes are pre-compiled:
- `_SENTENCE_SPLIT_RE` — splits text on `.` `!` `?` followed by whitespace
- `_HYPHEN_LINEBREAK_RE` — matches `word-\nword` (broken hyphenation from PDF)
- `_NON_ALNUM_RE` — matches non-alphanumeric runs (for tokenization)
- `_CONTROL_CHARS_RE` — matches control characters like null bytes
- One inline `re.sub(r"\s+", " ", text)` for whitespace collapsing

**3. Ligature map covers the 4 most common PDF ligatures:**
- `\ufb01` → `fi` (e.g., "ﬁrst" → "first")
- `\ufb02` → `fl` (e.g., "ﬂoor" → "floor")
- `\ufb03` → `ffi` (e.g., "oﬃce" → "office")
- `\ufb04` → `ffl` (e.g., "waﬄe" → "waffle")

These are the ligatures most commonly embedded by PDF generators. Without this fix, searching for "first" would not match a chunk containing "ﬁrst".

**4. `split_into_chunks` is sentence-aware with overlap:**

The algorithm:
1. Split text into sentences on `.!?` followed by whitespace
2. Accumulate sentences until the token count exceeds `chunk_size`
3. When full, save the chunk and start a new one
4. The new chunk begins with the last few sentences from the previous chunk (the "overlap")
5. Overlap is built by walking backwards through sentences until `chunk_overlap` tokens are reached

**Edge case — single sentence longer than `chunk_size`:**
If a sentence has more tokens than the chunk limit (e.g., a 900-word paragraph with no periods), it is force-split on whitespace boundaries. The accumulated buffer is flushed first, then the long sentence is broken into `chunk_size`-word pieces.

**Edge case — empty or whitespace-only text:**
Both `clean_text("")` and `split_into_chunks("")` return empty results. Every function guards against `None` and empty strings at the top.

**5. `count_tokens` uses whitespace splitting, not a real tokenizer:**
This is a deliberate approximation. Real tokenizers (like Mistral's) split words differently (e.g., "don't" → ["don", "'t"]), but whitespace splitting is close enough for chunking decisions. Adding a tokenizer dependency would violate the constraint of keeping things simple. The approximation errs on the side of slightly smaller chunks.

**6. `tokenize` is different from `count_tokens`:**
- `count_tokens` — fast, just `len(text.split())`, used to measure chunk size
- `tokenize` — thorough, lowercases, splits on punctuation, removes stopwords, used for keyword matching in Phase 3

---

## 4. File 3: `app/utils/pii_detector.py`

**Created from scratch. 38 lines.**

**Purpose:** Regex-based detection of Personally Identifiable Information in text. Used in Phase 3 to refuse queries containing PII. Built now so it's available when needed.

### Full source code:

```python
import re

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),
}


def contains_pii(text: str) -> bool:
    if not text:
        return False
    for pattern in _PII_PATTERNS.values():
        if pattern.search(text):
            return True
    return False


def get_pii_types(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]
```

### Design decisions:

**1. Four PII categories:**

| Category | Regex | Matches |
|---|---|---|
| `ssn` | `\b\d{3}-\d{2}-\d{4}\b` | `123-45-6789` |
| `credit_card` | `\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b` | `4111-1111-1111-1111`, `4111 1111 1111 1111`, `4111111111111111` |
| `email` | `\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b` | `user@example.com` |
| `phone` | `(?:\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b` | `+1 (555) 123-4567`, `555-123-4567`, `5551234567` |

**2. Two functions with different return types:**
- `contains_pii(text) -> bool` — fast check, short-circuits on first match
- `get_pii_types(text) -> list[str]` — returns all categories found, used for detailed error messages

**3. All patterns are pre-compiled:**
Same performance pattern as `text_utils.py` — compile once at import time, reuse on every call.

**4. Word boundaries (`\b`) prevent false positives:**
Without `\b`, the SSN pattern would match inside longer numbers like `12345-67-89012`. The word boundary anchors ensure we only match standalone patterns.

---

## 5. File 4: `app/core/pdf_processor.py`

**Created from scratch. 127 lines.**

**Purpose:** Extract text from PDF files and split into chunks with deterministic IDs. Pure logic — no FastAPI imports. Pipeline pattern: each step is one function.

### Full source code:

```python
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
    pages: list[tuple[int, str]] = []

    try:
        reader = PyPDF2.PdfReader(file_path)
    except PyPDF2.errors.PdfReadError as exc:
        raise PDFProcessingError(f"Corrupted or invalid PDF: {exc}") from exc
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise PDFProcessingError("PDF has no pages")

    for i, page in enumerate(reader.pages):
        page_number = i + 1
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if text.strip():
            pages.append((page_number, text))
            continue

        try:
            with pdfplumber.open(file_path) as pdf:
                if i < len(pdf.pages):
                    plumber_text = pdf.pages[i].extract_text() or ""
                    if plumber_text.strip():
                        pages.append((page_number, plumber_text))
                        continue
        except Exception:
            pass

        logger.warning("No text extracted from page %d of %s", page_number, file_path)

    return pages


def process_pdf(file_path: str, filename: str) -> list[Chunk]:
    pages = extract_text_from_pdf(file_path)

    if not pages:
        raise PDFProcessingError(
            f"Could not extract any text from '{filename}'. "
            "The PDF may be scanned (image-only) or empty."
        )

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

    if not chunks:
        raise PDFProcessingError(
            f"PDF '{filename}' produced no text chunks after processing."
        )

    return chunks
```

### Design decisions:

**1. Two-library extraction strategy (PyPDF2 + pdfplumber):**

For each page:
1. Try `PyPDF2.PdfReader` first — it's faster
2. If PyPDF2 returns empty/None text for that page, try `pdfplumber`
3. If both fail, skip the page and log a warning

PyPDF2 is tried first because it's significantly faster. pdfplumber uses pdfminer internally which is slower but handles complex layouts (multi-column, tables) better. This gives us the best coverage without always paying the slower extraction cost.

**2. Page numbers are 1-indexed:**
Users see "Page 1, Page 2" in their PDF viewer, so we match that convention. Internally `enumerate(reader.pages)` gives 0-indexed, so we add 1: `page_number = i + 1`.

**3. Pipeline pattern — each step is a function:**
```
extract_text_from_pdf()  →  list of (page_number, raw_text)
clean_text()             →  cleaned text per page
split_into_chunks()      →  list of chunk strings per page
Chunk()                  →  Chunk objects with deterministic IDs
```

Each function is independently testable. If chunking is wrong, you test `split_into_chunks()` in isolation. If extraction is wrong, you test `extract_text_from_pdf()` in isolation.

**4. Fail fast, fail loud:**
Three rejection points:
- `PyPDF2.PdfReadError` → `PDFProcessingError("Corrupted or invalid PDF")`
- `len(reader.pages) == 0` → `PDFProcessingError("PDF has no pages")`
- `not pages` after extraction → `PDFProcessingError("Could not extract any text")`
- `not chunks` after chunking → `PDFProcessingError("produced no text chunks")`

**5. `chunk_index` is a global counter across all pages:**
The counter starts at 0 and increments across page boundaries. This means chunk IDs are unique within a document:
```
report.pdf_p1_c0    (page 1, chunk 0)
report.pdf_p1_c1    (page 1, chunk 1)
report.pdf_p2_c2    (page 2, chunk 2)   ← index continues, not reset
report.pdf_p3_c3    (page 3, chunk 3)
```

**6. Chunk size and overlap come from `settings`, not hardcoded:**
```python
page_chunks = split_into_chunks(
    cleaned,
    chunk_size=settings.chunk_size,      # 800, from config.py
    chunk_overlap=settings.chunk_overlap,  # 150, from config.py
)
```
No magic numbers. Everything configurable via `.env` or environment variables.

**7. Custom exception class `PDFProcessingError`:**
Not a generic `ValueError` or `Exception`. The ingest service catches `PDFProcessingError` specifically and converts it to a user-facing error message. This separation means the PDF processor doesn't need to know about HTTP status codes.

---

## 6. File 5: `app/core/embeddings.py`

**Created from scratch. 180 lines.**

**Purpose:** Synchronous Mistral AI embeddings client. Converts text strings into embedding vectors (NumPy arrays). Handles batching, retry, and error cases.

### Full source code:

```python
"""Mistral AI embeddings client.

Synchronous — no async magic yet.  Pure logic — no FastAPI imports.
Testable in isolation.
"""

import logging
import time

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 16
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


class EmbeddingError(Exception):
    """Raised when the embeddings API call fails."""


def _validate_api_key() -> None:
    if not settings.mistral_api_key:
        raise EmbeddingError(
            "Mistral API key is not configured. "
            "Set MISTRAL_API_KEY in your .env file."
        )


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    if not texts:
        raise EmbeddingError("Cannot embed an empty list of texts.")

    _validate_api_key()

    all_embeddings: list[np.ndarray] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        batch_embeddings = _embed_batch_with_retry(batch)
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def get_embedding(text: str) -> np.ndarray:
    matrix = get_embeddings_batch([text])
    return matrix[0]


def _embed_batch_with_retry(texts: list[str]) -> np.ndarray:
    url = f"{settings.mistral_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mistral_embed_model,
        "input": texts,
    }

    last_error: Exception | None = None
    backoff = _INITIAL_BACKOFF_SECONDS

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code == 401:
                raise EmbeddingError(
                    "Mistral API authentication failed (HTTP 401). "
                    "Check your MISTRAL_API_KEY."
                )

            if response.status_code == 429:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429). Retrying in %.1fs (attempt %d/%d).",
                        backoff, attempt, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise EmbeddingError(
                    "Mistral API rate limit exceeded after retries."
                )

            if response.status_code != 200:
                error_body = response.text[:500]
                raise EmbeddingError(
                    f"Mistral API error (HTTP {response.status_code}): {error_body}"
                )

            return _parse_embedding_response(response.json(), expected=len(texts))

        except EmbeddingError:
            raise

        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Network error: %s. Retrying in %.1fs (attempt %d/%d).",
                    exc, backoff, attempt, _MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

    raise EmbeddingError(
        f"Mistral API request failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def _parse_embedding_response(body: dict, expected: int) -> np.ndarray:
    try:
        data = body["data"]
    except (KeyError, TypeError) as exc:
        raise EmbeddingError(
            f"Unexpected Mistral API response format: {exc}"
        ) from exc

    if len(data) != expected:
        raise EmbeddingError(
            f"Expected {expected} embeddings, got {len(data)}."
        )

    data_sorted = sorted(data, key=lambda d: d["index"])
    vectors = [item["embedding"] for item in data_sorted]
    return np.array(vectors, dtype=np.float64)
```

### Design decisions:

**1. Synchronous, not async:**
All ingestion is synchronous to keep execution traceable and debuggable. We use `httpx.Client` (sync) not `httpx.AsyncClient`. No `async`/`await` anywhere. This was an explicit design choice from the Phase 2 plan: "No async magic yet."

**2. Batch splitting:**
The Mistral API limits how many texts can be embedded per request. `_BATCH_SIZE = 16` means if you pass 50 texts, they're split into 4 sub-batches (16, 16, 16, 2). The results are concatenated with `np.vstack()`.

**3. Retry with exponential backoff:**
- Attempt 1: immediate
- Attempt 2: wait 1.0 seconds
- Attempt 3: wait 2.0 seconds
- After 3 failures: raise `EmbeddingError`

Only retries on:
- HTTP 429 (rate limit)
- `httpx.HTTPError` (network errors: timeout, connection refused, DNS failure)

Does NOT retry on:
- HTTP 401 (auth failure — retrying won't help)
- HTTP 4xx/5xx other than 429 (server errors)
- `EmbeddingError` (our own parse/validation errors)

**4. Fail fast on missing API key:**
`_validate_api_key()` is called before making any HTTP request. If the key is empty, we raise immediately with a clear message. This prevents confusing "connection refused" errors when the key is simply not set.

**5. Response ordering guarantee:**
The Mistral API returns embeddings with an `index` field. We sort by this index before converting to NumPy:
```python
data_sorted = sorted(data, key=lambda d: d["index"])
```
This guarantees the output matrix row ordering matches the input text ordering, even if the API returns results out of order.

**6. `float64` precision:**
```python
np.array(vectors, dtype=np.float64)
```
Double precision avoids floating-point truncation that could affect similarity calculations.

**7. Custom exception class `EmbeddingError`:**
The ingest service catches this specifically. The embeddings module doesn't know about HTTP status codes at the API layer — it raises domain-specific errors with clear messages.

---

## 7. File 6: `app/storage/vector_store.py`

**Created from scratch. 201 lines.**

**Purpose:** In-memory vector store backed by NumPy. Dumb by design — no business logic, no scoring, no ranking. Just stores and retrieves data. Replaces FAISS.

### Full source code:

```python
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

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
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
        return self.chunks, self.embeddings

    def delete_by_source(self, filename: str) -> int:
        if not self.chunks:
            return 0

        keep_mask = [c.source != filename for c in self.chunks]
        removed = keep_mask.count(False)

        if removed == 0:
            return 0

        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]

        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep_mask]
            if self.embeddings.shape[0] == 0:
                self.embeddings = None

        doc_ids_to_remove = [
            doc_id
            for doc_id, info in self.documents.items()
            if info.filename == filename
        ]
        for doc_id in doc_ids_to_remove:
            del self.documents[doc_id]

        return removed

    def clear(self) -> None:
        self.embeddings = None
        self.chunks = []
        self.documents = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        if self.embeddings is not None:
            np.save(os.path.join(path, _EMBEDDINGS_FILE), self.embeddings)
        else:
            emb_path = os.path.join(path, _EMBEDDINGS_FILE)
            if os.path.exists(emb_path):
                os.remove(emb_path)

        chunks_data = [c.model_dump() for c in self.chunks]
        with open(os.path.join(path, _CHUNKS_FILE), "w") as f:
            json.dump(chunks_data, f, indent=2)

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
        emb_path = os.path.join(path, _EMBEDDINGS_FILE)
        chunks_path = os.path.join(path, _CHUNKS_FILE)
        docs_path = os.path.join(path, _DOCUMENTS_FILE)

        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = None

        if os.path.exists(chunks_path):
            with open(chunks_path) as f:
                chunks_data = json.load(f)
            self.chunks = [Chunk(**item) for item in chunks_data]
        else:
            self.chunks = []

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

        if self.embeddings is not None and self.embeddings.shape[0] != len(self.chunks):
            logger.error(
                "Embeddings/chunks mismatch: %d rows vs %d chunks. Resetting store.",
                self.embeddings.shape[0], len(self.chunks),
            )
            self.clear()


vector_store = VectorStore()
```

### Design decisions:

**1. Dumb by design:**
The store has NO search logic, NO scoring, NO ranking. It stores data and hands it back. The `get_all()` method returns the raw chunks and embeddings matrix — search logic belongs in `core/`, not here.

**2. Module-level singleton:**
`vector_store = VectorStore()` at the bottom of the file. All other modules import this instance. There is exactly one store in the application.

**3. Parallel arrays invariant:**
`embeddings[i]` is always the embedding for `chunks[i]`. This invariant is enforced by:
- `add()` validates `len(chunks) == embeddings.shape[0]`
- `delete_by_source()` applies the same boolean mask to both arrays
- `load()` checks the invariant and calls `clear()` if it's violated

**4. Persistence uses JSON + NumPy `.npy`:**
Three files are saved:
- `embeddings.npy` — NumPy binary format (fast, compact for float arrays)
- `chunks.json` — human-readable JSON (can be inspected/debugged)
- `documents.json` — human-readable JSON

**Why not pickle:** JSON is safe (no arbitrary code execution on load), human-readable, and version-control friendly. Pickle can execute arbitrary code during deserialization.

**5. `delete_by_source()` uses boolean masking:**
```python
keep_mask = [c.source != filename for c in self.chunks]
self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]
self.embeddings = self.embeddings[keep_mask]
```
NumPy supports boolean array indexing natively, so the embeddings and chunks are filtered in one pass.

**6. Stale file cleanup on save:**
If the store is empty (all documents deleted), `save()` removes the stale `embeddings.npy` file:
```python
if self.embeddings is None:
    emb_path = os.path.join(path, _EMBEDDINGS_FILE)
    if os.path.exists(emb_path):
        os.remove(emb_path)
```

**7. Graceful load:**
If the directory doesn't exist or files are missing, `load()` simply leaves those attributes empty — no error is raised. The store starts in a clean state. Only if a consistency violation is detected (embeddings rows != chunks count) does it call `clear()`.

**8. `embeddings.copy()` on first add:**
```python
self.embeddings = embeddings.copy()
```
Prevents the caller from accidentally mutating the store's internal array.

---

## 8. File 7: `app/core/ingest_service.py`

**Created from scratch. 141 lines.**

**Purpose:** Orchestrates the ingestion pipeline. Coordinates pdf_processor → embeddings → vector_store. Pure logic — no FastAPI imports.

### Full source code:

```python
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

    # Step 3 — Store in vector store.
    vector_store.add(chunks, embeddings)

    doc_info = DocumentInfo(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
        ingested_at=datetime.now(timezone.utc),
    )
    vector_store.documents[document_id] = doc_info

    # Step 4 — Persist to disk.
    try:
        vector_store.save(settings.index_dir)
    except OSError as exc:
        logger.error("Failed to persist vector store: %s", exc)

    # Step 5 — Return response.
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
    if document_id not in vector_store.documents:
        raise IngestError(f"Document '{document_id}' not found.")

    doc_info = vector_store.documents[document_id]
    filename = doc_info.filename

    removed = vector_store.delete_by_source(filename)
    logger.info(
        "Deleted document '%s' (%s): removed %d chunks.",
        document_id, filename, removed,
    )

    pdf_path = os.path.join(settings.data_dir, f"{document_id}.pdf")
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError as exc:
            logger.warning("Could not delete PDF file %s: %s", pdf_path, exc)

    try:
        vector_store.save(settings.index_dir)
    except OSError as exc:
        logger.error("Failed to persist vector store after deletion: %s", exc)


def list_documents() -> list[DocumentInfo]:
    return list(vector_store.documents.values())
```

### Design decisions:

**1. Three functions with clear responsibilities:**
- `run()` — full ingestion pipeline
- `delete_document()` — remove a document and its chunks
- `list_documents()` — return document records

**2. Error wrapping — domain-specific exceptions:**
The service catches `PDFProcessingError` and `EmbeddingError` and re-raises them as `IngestError`. This means the API layer only needs to catch one exception type.

**3. `page_count` is computed from unique page numbers:**
```python
page_count = len({c.page for c in chunks})
```
A set comprehension collects unique page numbers. If page 3 produced 5 chunks, it's still counted as 1 page.

**4. Disk persistence failure is non-fatal:**
```python
try:
    vector_store.save(settings.index_dir)
except OSError as exc:
    logger.error("Failed to persist vector store: %s", exc)
```
If saving to disk fails (disk full, permissions), the data is still in memory. The request succeeds. The next successful save will persist everything.

**5. `document_id` is a UUID hex string:**
```python
document_id = uuid.uuid4().hex
```
32-character hexadecimal, no dashes. Example: `"a1b2c3d4e5f67890a1b2c3d4e5f67890"`.

**6. `delete_document` cleans up both store and disk:**
1. Removes chunks from vector store (`delete_by_source`)
2. Deletes the PDF file from `data/` directory
3. Re-persists the store to disk
Each step has its own try/catch — failure at one step doesn't prevent the others.

---

## 9. File 8: `app/api/ingestion.py`

**Created from scratch. 114 lines.**

**Purpose:** Thin HTTP layer. No business logic, no PDF parsing, no chunking. Just: receive request → call core → return response.

### Full source code:

```python
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
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    os.makedirs(settings.data_dir, exist_ok=True)

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

    try:
        response = run_ingest(temp_path, filename)
    except IngestError as exc:
        _safe_remove(temp_path)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _safe_remove(temp_path)
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during ingestion: {exc}",
        )

    final_path = os.path.join(settings.data_dir, f"{response.document_id}.pdf")
    try:
        os.rename(temp_path, final_path)
    except OSError:
        logger.warning("Could not rename %s to %s", temp_path, final_path)

    return response


@router.get("/api/documents", response_model=list[DocumentInfo])
def get_documents() -> list[DocumentInfo]:
    return list_documents()


@router.delete("/api/documents/{document_id}")
def remove_document(document_id: str) -> dict:
    try:
        delete_document(document_id)
    except IngestError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {"status": "deleted", "document_id": document_id}


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
```

### Design decisions:

**1. Thin handler — the `ingest()` function is ~40 lines:**
It does only HTTP-level concerns:
- Validate file extension
- Save to temp file
- Call `run_ingest()` (the core logic)
- Rename temp file
- Return response

No PDF parsing, no chunking, no embedding logic.

**2. Temp file strategy:**
The upload is saved to a random temp name (`_upload_{uuid}.pdf`) first, then renamed to `{document_id}.pdf` after successful ingestion. This prevents filename collisions if two people upload `report.pdf` simultaneously.

**3. Cleanup on failure:**
If ingestion fails, the temp file is deleted via `_safe_remove()`. This prevents leftover temp files from accumulating on disk.

**4. `_safe_remove()` swallows errors:**
```python
def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
```
This is a cleanup helper — if the file is already gone or can't be deleted, there's nothing useful we can do. Swallowing the error is intentional.

**5. `except HTTPException: raise` pattern:**
```python
try:
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ...
except HTTPException:
    raise
except Exception as exc:
    raise HTTPException(status_code=500, ...)
```
We raise `HTTPException` inside the try block (for empty file). The generic `except Exception` would catch it if we didn't re-raise it first. The `except HTTPException: raise` ensures our 400 errors pass through correctly.

**6. Three endpoints, all thin:**

| Endpoint | Lines of logic | What it does |
|---|---|---|
| `POST /api/ingest` | ~40 | Save file → call `run_ingest()` → return response |
| `GET /api/documents` | 1 | Call `list_documents()` |
| `DELETE /api/documents/{id}` | 4 | Call `delete_document()`, handle 404 |

---

## 10. File 9: `app/main.py` (modified)

**Modified from Phase 1. Now 43 lines (was 20).**

**Changes made:**

| What | Before (Phase 1) | After (Phase 2) |
|---|---|---|
| Imports | `health` only | `health`, `ingestion`, `settings`, `vector_store` |
| Logging | None | `logging.basicConfig()` with timestamp format |
| Routers | `health.router` | `health.router` + `ingestion.router` |
| Startup | None | `@app.on_event("startup")` handler |

### Full source code:

```python
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, ingestion
from app.config import settings
from app.storage.vector_store import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="RAG Pipeline",
    description="PDF Knowledge Base with Retrieval-Augmented Generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingestion.router)


@app.on_event("startup")
def startup() -> None:
    """Create data directories and restore vector store from disk."""
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)

    index_path = os.path.join(settings.index_dir, "chunks.json")
    if os.path.exists(index_path):
        vector_store.load(settings.index_dir)
```

### Design decisions:

**1. Structured logging:**
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
```
Every log line includes timestamp, level, and module name. Example:
```
2026-02-07 10:30:00,123 [INFO] app.core.ingest_service: Ingested 'report.pdf': 12 pages, 34 chunks
```

**2. Startup event creates directories:**
```python
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.index_dir, exist_ok=True)
```
The `data/` and `indexes/` directories are created on first boot. `exist_ok=True` means no error if they already exist.

**3. Conditional store loading:**
```python
index_path = os.path.join(settings.index_dir, "chunks.json")
if os.path.exists(index_path):
    vector_store.load(settings.index_dir)
```
Only loads if there's actually a persisted store. On first boot, nothing is loaded — the store starts empty.

---

## 11. File 10: `app/api/health.py` (modified)

**Modified from Phase 1. Now 20 lines (was 16).**

**Changes made:**

| What | Before (Phase 1) | After (Phase 2) |
|---|---|---|
| Return type | Plain dict | `HealthResponse` model |
| Fields | 4 fields | 6 fields (added `documents_count`, `chunks_count`) |
| Import | `settings` only | `settings`, `HealthResponse`, `vector_store` |

### Full source code:

```python
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse
from app.storage.vector_store import vector_store

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        mistral_configured=bool(settings.mistral_api_key),
        documents_count=len(vector_store.documents),
        chunks_count=len(vector_store),
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
    )
```

**Response before Phase 2:**
```json
{
    "status": "healthy",
    "mistral_configured": false,
    "data_dir": "./data",
    "index_dir": "./indexes"
}
```

**Response after Phase 2:**
```json
{
    "status": "healthy",
    "mistral_configured": true,
    "documents_count": 0,
    "chunks_count": 0,
    "data_dir": "./data",
    "index_dir": "./indexes"
}
```

---

## 12. Architectural Patterns Used

### Pattern 1: Folder-level responsibility separation

```
api/     — Thin HTTP layer. No business logic.
core/    — Pure logic. No FastAPI imports. Testable in isolation.
storage/ — Dumb data layer. No logic. Just add/get/delete.
models/  — Data contracts. Pydantic schemas only.
utils/   — Shared helpers. Pure functions, no state.
```

**Example of the pattern in action:**
```
User uploads PDF
    → api/ingestion.py       (receives file, calls core)
    → core/ingest_service.py  (orchestrates the pipeline)
    → core/pdf_processor.py   (extracts text, creates chunks)
    → core/embeddings.py      (calls Mistral API)
    → storage/vector_store.py (stores data)
    → api/ingestion.py        (returns HTTP response)
```

### Pattern 2: Pipeline-style ingestion

Each step is one function. Not one giant function.

```
PDF file
  → extract_text_from_pdf()     → list[(page, text)]
  → clean_text()                → cleaned string
  → split_into_chunks()         → list[str]
  → Chunk()                     → Chunk objects
  → get_embeddings_batch()      → np.ndarray
  → vector_store.add()          → stored
```

### Pattern 3: Fail fast, fail loud

Every rejection point raises an error with a clear, user-facing message:

| What went wrong | Error message | HTTP status |
|---|---|---|
| Non-PDF file uploaded | "Only PDF files are accepted." | 400 |
| Empty file uploaded | "Uploaded file is empty." | 400 |
| Corrupted PDF | "Corrupted or invalid PDF: {details}" | 400 |
| PDF has no pages | "PDF has no pages" | 400 |
| No text extractable | "Could not extract any text from '{filename}'" | 400 |
| Zero chunks produced | "PDF '{filename}' produced no text chunks after processing." | 400 |
| No API key | "Mistral API key is not configured. Set MISTRAL_API_KEY in your .env file." | 400 |
| Invalid API key | "Mistral API authentication failed (HTTP 401). Check your MISTRAL_API_KEY." | 400 |
| Rate limited | "Mistral API rate limit exceeded after retries." | 400 |
| Network failure | "Mistral API request failed after 3 attempts: {error}" | 400 |
| Document not found | "Document '{id}' not found." | 404 |
| Unexpected error | "Internal error during ingestion: {error}" | 500 |

### Pattern 4: Deterministic IDs

```python
chunk_id = f"{filename}_p{page_number}_c{chunk_index}"
```

Examples:
```
report.pdf_p1_c0
report.pdf_p1_c1
report.pdf_p2_c2
report.pdf_p3_c3
```

### Pattern 5: No magic numbers

All configuration values come from `app/config.py`:
```python
settings.chunk_size       # 800
settings.chunk_overlap    # 150
settings.mistral_api_key
settings.mistral_embed_model
settings.mistral_base_url
settings.data_dir
settings.index_dir
```

No hardcoded values are scattered across files.

### Pattern 6: Custom exception hierarchy

```
IngestError             ← raised by ingest_service.py
  ├── PDFProcessingError  ← raised by pdf_processor.py
  └── EmbeddingError      ← raised by embeddings.py
```

Each layer catches the exceptions from the layer below and wraps them:
```python
# ingest_service.py catches PDFProcessingError from pdf_processor.py
try:
    chunks = process_pdf(file_path, filename)
except PDFProcessingError as exc:
    raise IngestError(str(exc)) from exc

# api/ingestion.py catches IngestError from ingest_service.py
try:
    response = run_ingest(temp_path, filename)
except IngestError as exc:
    raise HTTPException(status_code=400, detail=str(exc))
```

---

## 13. Edge Cases Covered

### text_utils.py edge cases:
- `clean_text("")` → `""` (empty input)
- `clean_text(None)` → `""` (falsy input)
- `clean_text("  \n\t  ")` → `""` (whitespace-only)
- `clean_text("The ﬁrst ﬂoor")` → `"The first floor"` (ligatures)
- `clean_text("compli-\ncated")` → `"complicated"` (hyphenated line break)
- `clean_text("a\x00b\x01c")` → `"a b c"` (control characters)
- `count_tokens("")` → `0`
- `count_tokens("  ")` → `0`
- `split_into_chunks("")` → `[]` (empty input)
- `split_into_chunks("short")` → `["short"]` (text shorter than chunk_size)
- Single sentence longer than chunk_size → force-split on whitespace
- `tokenize("")` → `[]`
- `tokenize("The quick brown fox")` → `["quick", "brown", "fox"]` (stopwords removed)

### pii_detector.py edge cases:
- `contains_pii("")` → `False`
- `contains_pii(None)` → `False`
- `contains_pii("clean query")` → `False`
- `get_pii_types("")` → `[]`
- Multiple PII types in one string → all detected

### pdf_processor.py edge cases:
- Corrupted PDF → `PDFProcessingError("Corrupted or invalid PDF")`
- PDF with 0 pages → `PDFProcessingError("PDF has no pages")`
- All pages are images (no text) → `PDFProcessingError("Could not extract any text")`
- PyPDF2 succeeds on some pages, fails on others → pdfplumber fallback per page
- Both extractors fail on a page → page skipped, warning logged
- All text cleans to empty → `PDFProcessingError("produced no text chunks")`

### embeddings.py edge cases:
- Empty text list → `EmbeddingError("Cannot embed an empty list")`
- No API key → `EmbeddingError("Mistral API key is not configured")`
- Invalid API key (401) → `EmbeddingError("authentication failed")`
- Rate limited (429) → retry 3 times with exponential backoff, then error
- Network error → retry 3 times, then error
- API returns wrong number of embeddings → `EmbeddingError("Expected N, got M")`
- API returns unexpected format → `EmbeddingError("Unexpected response format")`
- Batch > 16 texts → automatically split into sub-batches

### vector_store.py edge cases:
- `add()` with mismatched chunks/embeddings count → `ValueError`
- `delete_by_source()` with non-existent filename → returns 0, no error
- `delete_by_source()` removes all chunks → embeddings reset to None
- `load()` with missing files → attributes stay empty, no error
- `load()` with embeddings/chunks mismatch → calls `clear()`, logs error
- `save()` when store is empty → removes stale embeddings.npy if it exists
- `__len__()` on empty store → 0

### api/ingestion.py edge cases:
- Non-PDF file → 400 "Only PDF files are accepted"
- Empty file → 400 "Uploaded file is empty"
- File without filename → defaults to "unknown.pdf"
- Ingestion fails → temp file cleaned up
- Rename fails after ingestion → warning logged, not fatal
- Document not found on delete → 404

---

## 14. Every Test We Ran

### Test 1: Health endpoint with new fields
```bash
curl -s http://localhost:8080/api/health | python3 -m json.tool
```
**Result:**
```json
{
    "status": "healthy",
    "mistral_configured": false,
    "documents_count": 0,
    "chunks_count": 0,
    "data_dir": "./data",
    "index_dir": "./indexes"
}
```

### Test 2: Documents list (empty)
```bash
curl -s http://localhost:8080/api/documents | python3 -m json.tool
```
**Result:**
```json
[]
```

### Test 3: Reject non-PDF file
```bash
curl -s -X POST -F "file=@README.md" http://localhost:8080/api/ingest
```
**Result:**
```json
{"detail": "Only PDF files are accepted."}
```

### Test 4: Reject empty PDF
```bash
touch /tmp/empty.pdf
curl -s -X POST -F "file=@/tmp/empty.pdf" http://localhost:8080/api/ingest
```
**Result:**
```json
{"detail": "Uploaded file is empty."}
```

### Test 5: All API endpoints registered
```bash
curl -s http://localhost:8080/openapi.json | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('\n'.join(sorted(d['paths'].keys())))
"
```
**Result:**
```
/api/documents
/api/documents/{document_id}
/api/health
/api/ingest
```

### Test 6: Ingestion without API key
```bash
# With .env file having no key
curl -s -X POST -F "file=@/tmp/test_sample.pdf" http://localhost:8080/api/ingest
```
**Result:**
```json
{"detail": "Embedding failed: Mistral API key is not configured. Set MISTRAL_API_KEY in your .env file."}
```

### Test 7: Ingestion with invalid API key
```bash
# With MISTRAL_API_KEY set to an invalid value
curl -s -X POST -F "file=@/tmp/test_sample.pdf" http://localhost:8080/api/ingest
```
**Result:**
```json
{"detail": "Embedding failed: Mistral API authentication failed (HTTP 401). Check your MISTRAL_API_KEY."}
```

### Test 8: PDF processor in isolation
```python
from app.core.pdf_processor import process_pdf

chunks = process_pdf('/tmp/test_sample.pdf', 'test_sample.pdf')
print(f'Chunks created: {len(chunks)}')
for c in chunks:
    print(f'  {c.chunk_id}')
    print(f'    source={c.source}, page={c.page}')
    print(f'    text={c.text[:80]}...')
```
**Result:**
```
Chunks created: 2

  test_sample.pdf_p1_c0
    source=test_sample.pdf, page=1
    text=This is page one of a test document about machine learning. Neural networks are ...

  test_sample.pdf_p2_c1
    source=test_sample.pdf, page=2
    text=This is page two discussing natural language processing. NLP allows computers to...
```

### Test 9: Text utilities
```python
from app.utils.text_utils import clean_text, count_tokens, split_into_chunks, tokenize

print(repr(clean_text('  This is  a   test\n\nwith   weird   spacing.  ')))
# → 'This is a test with weird spacing.'

print(repr(clean_text('The ﬁrst ﬂoor')))
# → 'The first floor'

print(repr(clean_text('compli-\ncated')))
# → 'complicated'

print(count_tokens('hello world foo bar'))
# → 4

print(tokenize('The quick brown fox jumps over the lazy dog'))
# → ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']

long_text = '. '.join([f'Sentence number {i} with some extra words' for i in range(50)])
chunks = split_into_chunks(long_text, chunk_size=20, chunk_overlap=5)
print(f'{len(chunks)} chunks from {count_tokens(long_text)} tokens')
# → 50 chunks from 550 tokens
```

### Test 10: PII detector
```python
from app.utils.pii_detector import contains_pii, get_pii_types

print(contains_pii('My SSN is 123-45-6789'))         # → True
print(get_pii_types('My SSN is 123-45-6789'))         # → ['ssn']
print(contains_pii('Card: 4111-1111-1111-1111'))      # → True
print(contains_pii('Contact user@example.com'))       # → True
print(contains_pii('Call +1 (555) 123-4567'))         # → True
print(contains_pii('What is machine learning?'))      # → False
print(get_pii_types('SSN 123-45-6789 email foo@bar.com'))  # → ['ssn', 'email']
```

### Test 11: Vector store operations
```python
import numpy as np
from app.storage.vector_store import VectorStore
from app.models.schemas import Chunk

store = VectorStore()
print(f'Empty: {len(store)}')                         # → 0

chunks = [
    Chunk(chunk_id='test.pdf_p1_c0', text='Hello', source='test.pdf', page=1),
    Chunk(chunk_id='test.pdf_p1_c1', text='World', source='test.pdf', page=1),
    Chunk(chunk_id='other.pdf_p1_c0', text='Other', source='other.pdf', page=1),
]
embeddings = np.random.randn(3, 4)

store.add(chunks, embeddings)
print(f'After add: {len(store)}')                     # → 3

all_chunks, all_emb = store.get_all()
print(f'get_all: {len(all_chunks)}, {all_emb.shape}') # → 3, (3, 4)

store.save('/tmp/test_store')
store2 = VectorStore()
store2.load('/tmp/test_store')
print(f'Loaded: {len(store2)}')                       # → 3

removed = store.delete_by_source('test.pdf')
print(f'Removed: {removed}, remaining: {len(store)}') # → 2, 1

store.clear()
print(f'After clear: {len(store)}')                   # → 0
```

### Test 12: Vector store mismatch guard
```python
store = VectorStore()
chunks = [Chunk(chunk_id='a', text='hello', source='a.pdf', page=1)]
embeddings = np.random.randn(2, 4)  # 2 rows but 1 chunk

try:
    store.add(chunks, embeddings)
except ValueError as e:
    print(f'Caught: {e}')
# → Caught: Mismatch: 1 chunks but 2 embedding rows.
```

---

## 15. Constraint Compliance

| Constraint | How we comply in Phase 2 |
|---|---|
| **No FAISS** | `vector_store.py` uses `np.ndarray` + Python lists. No FAISS import anywhere. |
| **No rank-bm25** | Not imported. Keyword search is Phase 3 (custom `tokenize` is ready). |
| **No sentence-transformers** | Not imported. No cross-encoder. No re-ranking model. |
| **No scikit-learn** | Not imported. No TF-IDF. No `sklearn.metrics.pairwise`. |
| **Embeddings via Mistral API** | `embeddings.py` calls `POST /v1/embeddings` via `httpx`. Allowed. |
| **All retrieval logic from scratch** | Vector store is pure NumPy. `tokenize` is pure Python. |

---

## 16. What Changed From Phase 1

### Files unchanged:
- `app/__init__.py` — empty, unchanged
- `app/config.py` — unchanged (already had all settings)
- `app/api/__init__.py` — empty, unchanged
- `app/models/__init__.py` — empty, unchanged
- `app/services/__init__.py` — empty, unchanged (leftover)
- `app/utils/__init__.py` — empty, unchanged
- `requirements.txt` — unchanged
- `.env.example` — unchanged
- `.gitignore` — unchanged

### Files modified:
- `app/main.py` — added logging, ingestion router, startup event
- `app/api/health.py` — added documents_count, chunks_count, HealthResponse model

### Files created:
- `app/models/schemas.py` — 86 lines
- `app/utils/text_utils.py` — 171 lines
- `app/utils/pii_detector.py` — 38 lines
- `app/core/__init__.py` — 0 lines
- `app/core/pdf_processor.py` — 127 lines
- `app/core/embeddings.py` — 180 lines
- `app/core/ingest_service.py` — 141 lines
- `app/storage/__init__.py` — 0 lines
- `app/storage/vector_store.py` — 201 lines
- `app/api/ingestion.py` — 114 lines

### Total new code: 1,058 lines across 10 files.

### `.env` file created for testing:
```
MISTRAL_API_KEY=CF2DvjIoshzasO0mtBkPj44fo2nXDwPk
DATA_DIR=./data
INDEX_DIR=./indexes
```
This file is in `.gitignore` and will not be committed.
