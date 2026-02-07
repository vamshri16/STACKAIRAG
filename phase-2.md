# Phase 2: Data Ingestion Pipeline

**Goal:** Build the complete path from "user uploads a PDF" to "chunks with embeddings are stored and ready for search." After this phase, you can POST a PDF to the API and it will be extracted, chunked, embedded, and persisted.

**This phase is NOT about querying or search.** It is solely about getting data in. Phase 2 is responsible for storing raw chunks and metadata only; embeddings are generated but not searched in this phase.

**Execution model:** All ingestion is synchronous to keep execution traceable and debuggable. No async magic yet.

---

## What Phase 2 Covers

```
                         ┌─────────┐
                         │  api/   │  Thin layer — just receives request
                         └────┬────┘
                              │
                              ▼
                         ┌─────────┐
                         │  core/  │  All logic lives here
                         └────┬────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ pdf_processor│   │  embeddings  │   │ingest_service│
  │ (core/)      │   │  (core/)     │   │  (core/)     │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │
                            ▼
                       ┌──────────┐
                       │ storage/ │  Dumb — just add/get/delete
                       └──────────┘


Pipeline (each step = one function):

PDF Upload (api/)
      │
      ▼
┌─────────────────┐
│  Save PDF to    │   api/ receives file, calls core/ingest_service
│  disk (data/)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract text   │   core/pdf_processor — PyPDF2 (primary),
│  per page       │   pdfplumber (fallback). Fail fast if empty.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clean text     │   utils/text_utils — strip whitespace, fix artifacts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Split into     │   utils/text_utils — sentence-aware chunking
│  chunks         │   (size/overlap from config, not hardcoded)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Assign         │   Deterministic IDs: "{filename}_p{page}_c{index}"
│  chunk IDs      │   Flat schema: chunk_id, text, source, page
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │   core/embeddings — Mistral API (synchronous)
│  embeddings     │   Fail fast if API key missing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Store chunks   │   storage/vector_store — dumb add()
│  + embeddings   │   Then persist to disk (indexes/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return         │   api/ returns document_id, chunk count, status
│  response       │
└─────────────────┘
```

---

## Step 1 — Pydantic Schemas (`app/models/schemas.py`)

**What:** Define all the data models (request bodies, response bodies, internal data structures) used across the application.

**Why this is first:** Every other module in Phase 2 needs to import these types. The PDF processor returns `Chunk` objects. The vector store stores `Chunk` objects. The API returns `IngestResponse` objects. If we don't define the data contracts first, every subsequent file would need to be edited later to match.

**What we will create:**

### Locked Chunk Schema

This is the canonical chunk structure. Every downstream system (retrieval, citations, prompts) depends on it. **Do not change this schema later.**

```python
class Chunk(BaseModel):
    """A single piece of text from a PDF. This schema is locked."""
    chunk_id: str       # Deterministic: "{filename}_p{page}_c{index}"
    text: str           # The actual text content
    source: str         # Original PDF filename ("report.pdf")
    page: int           # Which page this came from (1-indexed)
```

**Why deterministic IDs:**
- `chunk_id = f"{filename}_p{page}_c{index}"` — e.g., `"report.pdf_p3_c2"`
- Easy to debug — you can see exactly where a chunk came from by reading the ID
- Stable references — the same PDF always produces the same chunk IDs
- Clear citations — the ID itself tells you the source and page

**Why flat (no nested metadata object):**
- Simpler to serialize/deserialize
- Simpler to reason about
- No cascading changes if we add a field later
- Retrieval, citations, and prompts can access `chunk.source` and `chunk.page` directly

```python
# --- Document tracking ---

class DocumentInfo(BaseModel):
    """Tracks an ingested document."""
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    ingested_at: datetime

# --- API request/response models ---

class IngestResponse(BaseModel):
    """Returned after a PDF is uploaded and processed."""
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str            # "completed" or "failed"
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

**Why Pydantic models instead of plain dicts:**
- FastAPI auto-validates request bodies against these schemas — if a field is missing or the wrong type, the client gets a clear 422 error
- Auto-generates OpenAPI/Swagger documentation from these models
- IDE autocomplete — `chunk.page` is typed
- Serialization — `.model_dump()` converts to dict, `.model_dump_json()` to JSON string

**File:** `app/models/schemas.py`

---

## Step 2 — Text Utilities (`app/utils/text_utils.py`)

**What:** Pure Python functions for text cleaning, tokenization, and chunking. These are used by the PDF processor (this phase) and by the keyword search scorer (later phases).

**What we will create:**

### Function 1: `clean_text(text: str) -> str`

**Purpose:** Take raw text extracted from a PDF and normalize it so it is consistent and usable.

**What it does:**
1. Replace multiple consecutive whitespace characters (spaces, tabs, newlines) with a single space
2. Strip leading/trailing whitespace
3. Fix common PDF extraction artifacts:
   - Ligatures: `ﬁ` → `fi`, `ﬂ` → `fl`
   - Broken hyphens at line endings: `compli-\ncated` → `complicated`
4. Remove null bytes and control characters

**Why it matters:** PDF text extraction is messy. PyPDF2 often produces text with random newlines in the middle of sentences, multiple spaces between words, and encoding artifacts. If we don't clean the text, chunks will contain garbage that degrades embedding quality and search results.

### Function 2: `count_tokens(text: str) -> int`

**Purpose:** Estimate the number of tokens in a text string.

**What it does:**
- Split on whitespace and count the resulting words
- This is an approximation — real tokenizers (like Mistral's) split differently, but whitespace splitting is close enough for chunking purposes and avoids adding a tokenizer dependency

**Why it matters:** The chunker needs to know how many tokens a piece of text contains to split at the right boundaries. We configured `chunk_size: 800` tokens in settings — this function is how we measure that.

### Function 3: `split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]`

**Purpose:** Split a cleaned text string into overlapping chunks of approximately `chunk_size` tokens.

**What it does:**
1. Split the text into sentences (by `.`, `!`, `?` followed by whitespace)
2. Accumulate sentences into a chunk until adding the next sentence would exceed `chunk_size` tokens
3. When the chunk is full, save it and start a new chunk
4. The new chunk starts by including the last few sentences from the previous chunk to create an overlap of approximately `chunk_overlap` tokens
5. If a single sentence exceeds `chunk_size`, split it by whitespace into smaller pieces

**Why overlap matters:**
Imagine a PDF says: "The company was founded in 2015. It reported $5M revenue in 2023."
If the chunk boundary falls between these two sentences, neither chunk alone has the full picture. With 150 tokens of overlap, the second chunk re-includes the end of the first chunk, so important information at boundaries is preserved.

**Why sentence-aware splitting:**
Naive splitting (split every 800 words regardless of sentence boundaries) produces chunks that start or end mid-sentence. This confuses the embedding model because the text fragment is incomplete. By respecting sentence boundaries, each chunk is a coherent piece of text.

### Function 4: `tokenize(text: str) -> list[str]`

**Purpose:** Break text into individual lowercase tokens for keyword scoring.

**What it does:**
1. Convert to lowercase
2. Split on non-alphanumeric characters (whitespace, punctuation)
3. Remove common English stopwords ("the", "is", "at", "a", "an", "and", "or", etc.)
4. Remove empty strings

**Why this is separate from `count_tokens`:** `count_tokens` is a fast approximation for measuring chunk size — it just counts whitespace-separated words. `tokenize` is more thorough — it normalizes case, removes stopwords, and splits on punctuation — because it is used for keyword matching where precision matters.

**File:** `app/utils/text_utils.py`

---

## Step 3 — PII Detector (`app/utils/pii_detector.py`)

**What:** Regex-based detection of Personally Identifiable Information in query text.

**What we will create:**

### Function 1: `contains_pii(text: str) -> bool`

**Purpose:** Returns `True` if the text contains any recognized PII pattern.

**Patterns detected:**
| PII Type | Pattern | Example |
|---|---|---|
| Social Security Number | `\d{3}-\d{2}-\d{4}` | `123-45-6789` |
| Credit Card Number | `\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}` | `4111-1111-1111-1111` |
| Email Address | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | `user@example.com` |
| Phone Number | `(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}` | `+1 (555) 123-4567` |

### Function 2: `get_pii_types(text: str) -> list[str]`

**Purpose:** Returns a list of which PII categories were detected (e.g., `["ssn", "email"]`). Used to construct specific refusal messages.

**Why this exists:** The README specifies "Query Refusal Policies — PII detection: Refuse to process queries containing SSN, credit card numbers, etc." In Phase 2 we build the detector; in a later phase the query endpoint will call `contains_pii()` before processing.

**File:** `app/utils/pii_detector.py`

---

## Step 4 — PDF Processor (`app/core/pdf_processor.py`)

**What:** Extract text from PDF files and split it into chunks with metadata. Pure logic — no FastAPI imports, no request/response objects. Every function is testable in isolation.

**What we will create:**

### Function 1: `extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]`

**Purpose:** Read a PDF file and return a list of `(page_number, raw_text)` tuples.

**How it works:**
1. Try PyPDF2 first:
   ```
   reader = PyPDF2.PdfReader(file_path)
   for i, page in enumerate(reader.pages):
       text = page.extract_text()
   ```
2. If PyPDF2 returns empty/None for a page, fall back to pdfplumber for that page:
   ```
   with pdfplumber.open(file_path) as pdf:
       text = pdf.pages[i].extract_text()
   ```
3. If both fail for a page, skip that page and log a warning
4. Page numbers are 1-indexed (page 1, page 2, ...) to match what users see in their PDF viewer

**Why two libraries:**
- PyPDF2 is fast and handles most PDFs well
- pdfplumber uses a different extraction engine (pdfminer) and handles complex layouts (multi-column, tables) better
- Using PyPDF2 first with pdfplumber fallback gives us the best coverage without always paying pdfplumber's slower processing time

**Error handling:**
- Password-protected PDFs: catch the exception, return an error message
- Corrupted PDFs: catch the exception, return an error message
- Empty PDFs (0 pages): return empty list

### Function 2: `process_pdf(file_path: str, filename: str) -> list[Chunk]`

**Purpose:** The full pipeline — from file path to a list of `Chunk` objects ready for embedding. This is a pipeline of small functions, not one giant function.

**Pipeline pattern:**
```
PDF → pages → cleaned text → chunks → store
Each step = one function.
```

**Flow:**
```
1. extract_text_from_pdf(file_path)
   → list of (page_number, raw_text)

2. Fail fast checks:
   - Empty PDF (0 pages) → raise error
   - No extractable text on any page → raise error

3. For each page:
   a. clean_text(raw_text)
   b. split_into_chunks(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
   c. For each chunk_text, create a Chunk object with deterministic ID:
      Chunk(
          chunk_id=f"{filename}_p{page_number}_c{chunk_index}",
          text=chunk_text,
          source=filename,
          page=page_number,
      )

4. Fail fast: if total chunk_count == 0 → raise error

5. Return list[Chunk]
```

**Deterministic IDs in action:**
For a file called `report.pdf` with 3 pages, chunks might look like:
```
report.pdf_p1_c0    (page 1, first chunk)
report.pdf_p1_c1    (page 1, second chunk)
report.pdf_p2_c0    (page 2, first chunk)
report.pdf_p3_c0    (page 3, first chunk)
report.pdf_p3_c1    (page 3, second chunk)
```

**File:** `app/core/pdf_processor.py`

---

## Step 5 — Embeddings Client (`app/core/embeddings.py`)

**What:** Synchronous client to call the Mistral AI embeddings API and convert text into numerical vectors. Pure logic — no FastAPI imports. Testable in isolation.

**What we will create:**

### Function 1: `get_embeddings_batch(texts: list[str]) -> np.ndarray`

**Purpose:** Send a batch of text strings to the Mistral API and get back a matrix of embedding vectors.

**How it works:**
1. Create an `httpx.Client` (synchronous — no async magic yet)
2. POST to `https://api.mistral.ai/v1/embeddings` with:
   ```json
   {
       "model": "mistral-embed",
       "input": ["chunk text 1", "chunk text 2", ...]
   }
   ```
3. Parse the response:
   ```json
   {
       "data": [
           {"embedding": [0.123, -0.456, ...], "index": 0},
           {"embedding": [0.789, -0.012, ...], "index": 1}
       ]
   }
   ```
4. Extract the embedding arrays, stack them into a NumPy matrix of shape `(n_texts, embedding_dim)`
5. Return the matrix

**Batch handling:**
The Mistral API has a limit on how many texts can be embedded in a single request. We will split large batches into sub-batches (e.g., 16 texts per request) and concatenate the results.

**Fail fast:**
- API key missing/empty → raise immediately with clear message, don't even make the request
- API key invalid (HTTP 401) → raise a clear error
- Rate limiting (HTTP 429) → wait and retry with exponential backoff
- Network errors → retry up to 3 times

### Function 2: `get_embedding(text: str) -> np.ndarray`

**Purpose:** Convenience wrapper — embed a single text string. Calls `get_embeddings_batch([text])` and returns the first (and only) vector.

**Why this exists:** The ingestion pipeline embeds chunks in batches (many texts at once). The query pipeline embeds a single query. Having both functions avoids the caller needing to wrap/unwrap lists.

**File:** `app/core/embeddings.py`

---

## Step 6 — Vector Store (`app/storage/vector_store.py`)

**What:** Dumb storage layer. Stores embeddings as a NumPy matrix and chunk metadata as a Python list. Provides `add`, `get`, `delete`, `save`, `load`, and `clear` operations.

**Design rule: dumb by design.** No business logic. No scoring. No ranking. If you feel tempted to add logic here — stop. This layer just stores and retrieves data.

**This is the component that replaces FAISS. No external search libraries.**

**What we will create:**

### Class: `VectorStore`

```
Attributes:
    embeddings: np.ndarray | None    # Shape (n, embedding_dim). None when empty.
    chunks: list[Chunk]              # Parallel list — chunks[i] corresponds to embeddings[i]
    documents: dict[str, DocumentInfo]  # Tracks ingested documents by document_id
```

### Method 1: `add(self, chunks: list[Chunk], embeddings: np.ndarray)`

**Purpose:** Add new chunks and their embeddings to the store.

**How it works:**
1. If the store is empty (`self.embeddings is None`):
   - Set `self.embeddings = embeddings`
   - Set `self.chunks = list(chunks)`
2. If the store already has data:
   - Stack: `self.embeddings = np.vstack([self.embeddings, embeddings])`
   - Extend: `self.chunks.extend(chunks)`
3. The result is that `self.embeddings[i]` is always the embedding for `self.chunks[i]`

No validation, no scoring — just append.

### Method 2: `get_all(self) -> tuple[list[Chunk], np.ndarray | None]`

**Purpose:** Return all chunks and the embeddings matrix. Used by the search layer (Phase 3) to compute similarity externally.

**Why this instead of a `search()` method:** The storage layer is dumb. Search logic (cosine similarity, keyword scoring, hybrid combination) belongs in `core/`, not here. The store just hands over the data.

### Method 3: `delete_by_source(self, filename: str)`

**Purpose:** Remove all chunks from a specific PDF file.

**How it works:**
1. Build a boolean mask: `keep = [c.source != filename for c in self.chunks]`
2. `self.embeddings = self.embeddings[keep]`
3. `self.chunks = [c for c, k in zip(self.chunks, keep) if k]`
4. Remove the document from `self.documents`

### Method 4: `save(self, path: str)`

**Purpose:** Persist the entire store to disk so it survives server restarts.

**How it works:**
1. `np.save(path + "/embeddings.npy", self.embeddings)` — save the NumPy matrix
2. Save chunk list as JSON: `json.dump(chunks_data, f)` — each chunk serialized via `.model_dump()`
3. Save documents dict as JSON

**Why not pickle:** JSON is human-readable, version-control friendly, and doesn't have pickle's security risks (arbitrary code execution on load). NumPy's `.npy` format is used for the embeddings because JSON would be too slow and large for big float arrays.

### Method 5: `load(self, path: str)`

**Purpose:** Load a previously saved store from disk.

**How it works:**
1. `self.embeddings = np.load(path + "/embeddings.npy")`
2. Load chunks from JSON, reconstruct `Chunk` objects
3. Load documents dict from JSON

### Method 6: `clear(self)`

**Purpose:** Remove everything. Reset to empty state.

### Method 7: `__len__(self) -> int`

**Purpose:** Return the total number of chunks in the store. Used by the health endpoint.

**File:** `app/storage/vector_store.py`

---

## Step 7 — Ingestion API Endpoint (`app/api/ingestion.py`)

**What:** Thin HTTP layer. No business logic, no PDF parsing, no chunking. Just: receive request → call core/storage → return response.

**Design rule:** The API layer is a pass-through. All logic lives in `core/`. All storage lives in `storage/`. The route handler is ~10 lines.

**What we will create:**

### Endpoint 1: `POST /api/ingest`

**Purpose:** Upload a PDF file, process it, embed it, store it.

**Request:** Multipart file upload (`Content-Type: multipart/form-data`)

**Route handler (thin):**
```python
@router.post("/api/ingest")
def ingest(file: UploadFile):
    return ingest_service.run(file)
```

**The actual logic lives in `app/core/ingest_service.py`:**
```
1. Validate file is a PDF (check extension)     → 400 if not
2. Generate a document_id (UUID4)
3. Save the file to disk: data/{document_id}.pdf
4. Process: pdf_processor.process_pdf(file_path, filename)
   → list[Chunk]                                → 400 if empty/corrupted
5. Embed: embeddings.get_embeddings_batch(...)
   → np.ndarray                                 → 500 if API fails
6. Store: vector_store.add(chunks, embeddings)
7. Persist: vector_store.save(settings.index_dir)
8. Record document in vector_store.documents
9. Return IngestResponse
```

**Fail fast, fail loud:**
- File is not a PDF → 400 with "Only PDF files are accepted"
- File is corrupted / no extractable text → 400 with "Could not extract text from PDF"
- Chunk count = 0 after processing → 400 with "PDF produced no text chunks"
- Mistral API key not configured → 500 with "Mistral API key not configured"
- Mistral API fails → 500 with error details

**Response:**
```json
{
    "document_id": "a1b2c3d4-...",
    "filename": "report.pdf",
    "page_count": 12,
    "chunk_count": 34,
    "status": "completed",
    "message": "Successfully processed 12 pages into 34 chunks"
}
```

### Endpoint 2: `GET /api/documents`

**Purpose:** List all ingested documents. Thin wrapper around `vector_store.documents`.

**Response:**
```json
[
    {
        "document_id": "a1b2c3d4-...",
        "filename": "report.pdf",
        "page_count": 12,
        "chunk_count": 34,
        "ingested_at": "2026-02-07T10:30:00Z"
    }
]
```

### Endpoint 3: `DELETE /api/documents/{document_id}`

**Purpose:** Remove a document and all its chunks from the vector store. Thin wrapper.

**Flow:**
1. `vector_store.delete_by_source(filename)`
2. Delete the PDF file from `data/`
3. Re-save the vector store to disk
4. Return 200 with confirmation

---

## Step 8 — Wire Into `app/main.py`

**What:** Register the new ingestion router and add a startup event to load the vector store from disk.

**Changes to `app/main.py`:**
1. Import the ingestion router
2. `app.include_router(ingestion.router)`
3. Add a `@app.on_event("startup")` handler:
   - Create `data/` and `indexes/` directories if they don't exist
   - If `indexes/embeddings.npy` exists, call `vector_store.load(settings.index_dir)` to restore previous state
4. Update the health endpoint to include `documents_count` and `chunks_count` from the vector store

---

## Files Created / Modified in Phase 2

| File | Action | Layer | Purpose |
|---|---|---|---|
| `app/models/schemas.py` | **Create** | models | Locked Chunk schema + all Pydantic data models |
| `app/utils/text_utils.py` | **Create** | utils | clean_text, count_tokens, split_into_chunks, tokenize |
| `app/utils/pii_detector.py` | **Create** | utils | contains_pii, get_pii_types |
| `app/core/__init__.py` | **Create** | core | Package init |
| `app/core/pdf_processor.py` | **Create** | core | extract_text_from_pdf, process_pdf |
| `app/core/embeddings.py` | **Create** | core | get_embedding, get_embeddings_batch |
| `app/core/ingest_service.py` | **Create** | core | Orchestrates the ingestion pipeline |
| `app/storage/__init__.py` | **Create** | storage | Package init |
| `app/storage/vector_store.py` | **Create** | storage | VectorStore class (add, get_all, delete, save, load, clear) |
| `app/api/ingestion.py` | **Create** | api | Thin routes: POST /api/ingest, GET /api/documents, DELETE |
| `app/api/health.py` | **Modify** | api | Add documents_count and chunks_count |
| `app/main.py` | **Modify** | — | Register ingestion router, startup event |

---

## Folder Responsibility Rules

### `api/` — Thin HTTP layer
- No business logic
- No PDF parsing
- No chunking
- Just: request → call core/storage → response

### `core/` — Pure logic
- No FastAPI imports
- No request/response objects
- Functions are testable in isolation
- This is where PDF parsing, chunking, validation, and embeddings live

### `storage/` — Dumb data layer
- No logic
- Just `add()` / `get_all()` / `delete_by_source()` / `save()` / `load()` / `clear()`
- If you feel tempted to add logic here — stop

### `models/` — Data contracts
- Pydantic schemas only
- The Chunk schema is locked — do not change it

### `utils/` — Shared helpers
- Pure functions, no state
- Used by both `core/` and `api/`

---

## Coding Order

We will code the files in dependency order — each file only imports from files created before it:

```
Step 1:  app/models/schemas.py           ← depends on nothing (defines locked Chunk schema)
Step 2:  app/utils/text_utils.py         ← depends on nothing
Step 3:  app/utils/pii_detector.py       ← depends on nothing
Step 4:  app/core/pdf_processor.py       ← depends on schemas, text_utils
Step 5:  app/core/embeddings.py          ← depends on config
Step 6:  app/storage/vector_store.py     ← depends on schemas
Step 7:  app/core/ingest_service.py      ← depends on pdf_processor, embeddings, vector_store
Step 8:  app/api/ingestion.py            ← thin wrapper around ingest_service
Step 9:  app/main.py (modify)            ← wire it all together
Step 10: app/api/health.py (modify)      ← add store stats
```

---

## What We Can Test After Phase 2

Once all 9 steps are coded, we will be able to:

1. **Start the server:** `uvicorn app.main:app --reload --port 8080`
2. **Upload a PDF:** `curl -X POST -F "file=@sample.pdf" http://localhost:8080/api/ingest`
3. **See it processed:** Response shows page count, chunk count, document ID
4. **List documents:** `curl http://localhost:8080/api/documents`
5. **Check health:** `curl http://localhost:8080/api/health` — now shows document and chunk counts
6. **Delete a document:** `curl -X DELETE http://localhost:8080/api/documents/{id}`
7. **Restart the server** and verify the vector store is reloaded from disk (persistence works)

**What we cannot do yet (that's for Phase 3):**
- Query the knowledge base
- Get answers from the LLM
- Hybrid search
- Re-ranking

---

## Configuration — No Magic Numbers

All tunable values live in one place: `app/config.py`. They are never scattered across files.

```python
# app/config.py (already exists from Phase 1)
chunk_size: int = 800          # used by text_utils.split_into_chunks
chunk_overlap: int = 150       # used by text_utils.split_into_chunks
```

When `core/pdf_processor.py` needs the chunk size, it imports `settings` from `config.py`. It does not hardcode `800`.

---

## Constraint Compliance

| Constraint | Phase 2 compliance |
|---|---|
| No FAISS | `vector_store.py` uses `np.ndarray` + manual cosine similarity |
| No rank-bm25 | Not used — keyword search is Phase 3 |
| No sentence-transformers | Not used — no cross-encoder |
| No scikit-learn | Not used — cosine similarity via NumPy |
| Embeddings via Mistral API | `embeddings.py` calls Mistral's `/v1/embeddings` endpoint |
