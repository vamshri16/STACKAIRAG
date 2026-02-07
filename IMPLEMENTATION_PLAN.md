# Implementation Plan — Step-by-Step

Build order follows a bottom-up dependency chain: foundational modules first, then services that depend on them, then API routes, then the app entry point.

```
Phase 1  config, schemas                  (no deps)
Phase 2  utils (text, pii)               (no deps)
Phase 3  pdf_processor                   (PyPDF2, pdfplumber, text_utils)
Phase 4  embeddings client               (httpx, Mistral API)
Phase 5  vector_store                    (NumPy only)
Phase 6  search (hybrid)                 (vector_store, embeddings)
Phase 7  reranker                        (pure Python sorting)
Phase 8  query_processor                 (LLM client, intent rules)
Phase 9  llm_client                      (httpx, Mistral API)
Phase 10 hallucination_filter            (llm_client)
Phase 11 API routes                      (all services)
Phase 12 main.py                         (FastAPI app)
```

---

## Phase 1 — Configuration & Schemas

### `app/config.py`
Pydantic `BaseSettings` for all configurable values.

```python
class Settings(BaseSettings):
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"

    chunk_size: int = 800          # tokens
    chunk_overlap: int = 150       # tokens
    top_k_retrieval: int = 20      # initial candidates
    top_k_final: int = 5           # after re-ranking
    similarity_threshold: float = 0.7
    semantic_weight: float = 0.7   # α  (keyword weight = 1 - α)

    data_dir: str = "./data"
    index_dir: str = "./indexes"

    class Config:
        env_file = ".env"
```

### `app/models/schemas.py`
All Pydantic request/response models.

```
IngestRequest          — file upload metadata
IngestResponse         — job_id, status
DocumentInfo           — id, filename, page_count, ingested_at
QueryRequest           — query, top_k, include_sources, session_id
QueryResponse          — answer, sources[], confidence, intent, processing_time_ms
Source                 — document_id, filename, page_number, chunk_text, score
ChunkMetadata          — document_id, filename, page_number, chunk_index
Chunk                  — text, embedding (optional), metadata: ChunkMetadata
HealthResponse         — status, documents_count, chunks_count
```

**Deliverables:** `app/__init__.py`, `app/config.py`, `app/models/__init__.py`, `app/models/schemas.py`

---

## Phase 2 — Utilities

### `app/utils/text_utils.py`

| Function | Purpose |
|---|---|
| `tokenize(text: str) -> list[str]` | Lowercase, split on whitespace/punctuation, remove stopwords. Used by keyword scorer. |
| `count_tokens(text: str) -> int` | Approximate token count (whitespace split or tiktoken-compatible). Used by chunker. |
| `split_into_chunks(text, chunk_size, overlap) -> list[str]` | Fixed-size chunking with overlap. Respects sentence boundaries when possible. |
| `clean_text(text: str) -> str` | Strip excessive whitespace, fix encoding artifacts. |

### `app/utils/pii_detector.py`

| Function | Purpose |
|---|---|
| `contains_pii(text: str) -> bool` | Regex-based detection: SSN, credit card, email, phone patterns. |
| `get_pii_types(text: str) -> list[str]` | Returns which PII categories were found. |

**Deliverables:** `app/utils/__init__.py`, `app/utils/text_utils.py`, `app/utils/pii_detector.py`

---

## Phase 3 — PDF Processor

### `app/services/pdf_processor.py`

| Function | Purpose |
|---|---|
| `extract_text_from_pdf(file_path) -> list[PageText]` | PyPDF2 first, fallback to pdfplumber. Returns list of `(page_number, text)`. |
| `process_pdf(file_path, settings) -> list[Chunk]` | Full pipeline: extract → clean → chunk → attach metadata. |

Flow:
```
PDF file
  → PyPDF2.PdfReader  (try)
  → pdfplumber.open   (fallback)
  → per-page text
  → clean_text()
  → split_into_chunks(chunk_size, overlap)
  → list[Chunk] with metadata (doc_id, filename, page_number, chunk_index)
```

**Deliverables:** `app/services/__init__.py`, `app/services/pdf_processor.py`

---

## Phase 4 — Embeddings Client

### `app/services/embeddings.py`

| Function | Purpose |
|---|---|
| `async get_embedding(text: str) -> np.ndarray` | Single text → embedding vector via Mistral API. |
| `async get_embeddings_batch(texts: list[str]) -> np.ndarray` | Batch of texts → matrix of embeddings. Handles Mistral batch limits. |

Implementation:
- Use `httpx.AsyncClient` to call `POST /v1/embeddings`
- Request body: `{"model": "mistral-embed", "input": texts}`
- Parse response → extract embedding arrays → return as `np.ndarray`
- Retry logic for rate limits (simple exponential backoff)

**Deliverables:** `app/services/embeddings.py`

---

## Phase 5 — Vector Store (NumPy only)

### `app/services/vector_store.py`

This is the core custom component. No FAISS, no scikit-learn.

```python
class VectorStore:
    embeddings: np.ndarray | None     # shape (n, dim)
    chunks: list[Chunk]               # parallel list of chunk metadata + text

    def add(self, chunks: list[Chunk], embeddings: np.ndarray)
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]
    def delete_document(self, document_id: str)
    def save(self, path: str)
    def load(self, path: str)
    def __len__(self) -> int
```

#### Key method — `search`:
```python
def search(self, query_embedding, top_k):
    # 1. Cosine similarity from first principles
    dot_products = self.embeddings @ query_embedding          # (n,)
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(self.embeddings, axis=1)     # (n,)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)

    # 2. Top-k via argsort
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 3. Return (chunk, score) pairs
    return [(self.chunks[i], similarities[i]) for i in top_indices]
```

#### Persistence:
- `np.save(path + "/embeddings.npy", self.embeddings)`
- `pickle` or `json` for chunk metadata list
- `np.load` / json load on startup

**Deliverables:** `app/services/vector_store.py`

---

## Phase 6 — Hybrid Search

### `app/services/search.py`

| Function | Purpose |
|---|---|
| `keyword_score(query_tokens, chunk_text) -> float` | Custom term-overlap scorer (0–1). |
| `async hybrid_search(query, vector_store, embeddings_client, settings) -> list[ScoredChunk]` | Full hybrid pipeline. |

#### `keyword_score` — from scratch:
```python
def keyword_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    return len(overlap) / len(query_set)
```

#### `hybrid_search` flow:
```
1. Embed query            → query_embedding
2. Semantic search         → vector_store.search(query_embedding, top_k=20)
3. Keyword scoring         → for each chunk in store, compute keyword_score
                             sort, take top_k=20
4. Merge candidates        → union of semantic + keyword candidate sets
5. Weighted combination    → final = α * semantic + (1-α) * keyword
6. Sort by final score     → return top results
```

**Deliverables:** `app/services/search.py`

---

## Phase 7 — Re-ranker

### `app/services/reranker.py`

No cross-encoder. Pure score-based sorting.

| Function | Purpose |
|---|---|
| `rerank(scored_chunks, top_k, threshold) -> list[ScoredChunk]` | Sort by score, drop below threshold, deduplicate, return top-k. |

```python
def rerank(scored_chunks, top_k, threshold):
    # 1. Filter below threshold
    filtered = [(chunk, score) for chunk, score in scored_chunks if score >= threshold]

    # 2. Sort descending by score
    filtered.sort(key=lambda x: x[1], reverse=True)

    # 3. Deduplicate (same document + overlapping text)
    seen = set()
    deduped = []
    for chunk, score in filtered:
        key = (chunk.metadata.document_id, chunk.metadata.page_number)
        if key not in seen:
            seen.add(key)
            deduped.append((chunk, score))

    # 4. Return top-k
    return deduped[:top_k]
```

**Deliverables:** `app/services/reranker.py`

---

## Phase 8 — Query Processor

### `app/services/query_processor.py`

| Function | Purpose |
|---|---|
| `detect_intent(query) -> Intent` | Rule-based + optional LLM classification. |
| `transform_query(query, intent) -> str` | Light query cleanup / expansion. |

Intent detection rules:
```
CHITCHAT     — matches greeting patterns ("hi", "hello", "how are you")
OFF_TOPIC    — no knowledge-base terms detected + LLM confirms
CLARIFICATION — references previous answer ("what about", "can you explain")
KNOWLEDGE_SEARCH — default / everything else
```

**Deliverables:** `app/services/query_processor.py`

---

## Phase 9 — LLM Client

### `app/services/llm_client.py`

| Function | Purpose |
|---|---|
| `async generate(prompt, system_prompt) -> str` | Call Mistral chat completions API. |
| `async classify_intent(query) -> str` | Lightweight intent classification call. |
| `format_context(chunks) -> str` | Build context string from retrieved chunks with citations. |

Implementation:
- `httpx.AsyncClient` → `POST /v1/chat/completions`
- Request body: `{"model": "mistral-large-latest", "messages": [...]}`
- Parse `response["choices"][0]["message"]["content"]`
- Prompt templates from README (QA template, list template)

**Deliverables:** `app/services/llm_client.py`

---

## Phase 10 — Hallucination Filter

### `app/services/hallucination_filter.py`

| Function | Purpose |
|---|---|
| `verify_answer(answer, chunks) -> VerifiedAnswer` | Check claims against source chunks. |
| `compute_confidence(answer, chunks) -> float` | Overall confidence score. |

Approach:
- Split answer into sentences
- For each sentence, check if any source chunk contains supporting text (substring / keyword overlap)
- Flag unsupported sentences
- Confidence = supported_sentences / total_sentences

**Deliverables:** `app/services/hallucination_filter.py`

---

## Phase 11 — API Routes

### `app/api/ingestion.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/ingest` | POST | Upload PDF, trigger processing pipeline |
| `/api/ingest/status/{job_id}` | GET | Check ingestion job status |
| `/api/documents` | GET | List all ingested documents |
| `/api/documents/{document_id}` | DELETE | Remove document + its chunks from store |

### `app/api/query.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/query` | POST | Full RAG pipeline: intent → search → generate → verify |

Flow inside `/api/query`:
```
1. PII check           → refuse if detected
2. Detect intent       → CHITCHAT / KNOWLEDGE_SEARCH / etc.
3. If KNOWLEDGE_SEARCH:
   a. hybrid_search()
   b. rerank()
   c. format_context()
   d. generate()
   e. verify_answer()
4. Return QueryResponse
```

### `app/api/health.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Service status, document count, chunk count |

**Deliverables:** `app/api/__init__.py`, `app/api/ingestion.py`, `app/api/query.py`, `app/api/health.py`

---

## Phase 12 — App Entry Point & Dependencies

### `app/main.py`
```python
app = FastAPI(title="RAG Pipeline")

# Startup: load vector store from disk if exists
# Register routers: ingestion, query, health
# CORS middleware for frontend
```

### `requirements.txt`
```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-dotenv
httpx
numpy
PyPDF2
pdfplumber
pytesseract
spacy
python-multipart
pytest
```

No FAISS. No rank-bm25. No sentence-transformers. No scikit-learn.

### `.env.example`
```
MISTRAL_API_KEY=your_key_here
DATA_DIR=./data
INDEX_DIR=./indexes
```

**Deliverables:** `app/main.py`, `requirements.txt`, `.env.example`

---

## File Creation Order (suggested coding sequence)

| Step | File | Why this order |
|---|---|---|
| 1 | `requirements.txt` | Pin dependencies first |
| 2 | `.env.example` | Environment template |
| 3 | `app/__init__.py` | Package init |
| 4 | `app/config.py` | Everything reads config |
| 5 | `app/models/__init__.py` | Package init |
| 6 | `app/models/schemas.py` | Data contracts used everywhere |
| 7 | `app/utils/__init__.py` | Package init |
| 8 | `app/utils/text_utils.py` | Used by pdf_processor and search |
| 9 | `app/utils/pii_detector.py` | Used by query route |
| 10 | `app/services/__init__.py` | Package init |
| 11 | `app/services/pdf_processor.py` | Ingestion pipeline |
| 12 | `app/services/embeddings.py` | Needed by vector_store and search |
| 13 | `app/services/vector_store.py` | Core storage — NumPy only |
| 14 | `app/services/search.py` | Hybrid search — cosine + keyword overlap |
| 15 | `app/services/reranker.py` | Score sorting + threshold filtering |
| 16 | `app/services/llm_client.py` | Mistral chat/generation |
| 17 | `app/services/query_processor.py` | Intent detection |
| 18 | `app/services/hallucination_filter.py` | Answer verification |
| 19 | `app/api/__init__.py` | Package init |
| 20 | `app/api/health.py` | Simplest route — test the app boots |
| 21 | `app/api/ingestion.py` | PDF upload + processing |
| 22 | `app/api/query.py` | Full RAG query pipeline |
| 23 | `app/main.py` | Wire everything together |

---

## Constraint Checklist

| Constraint | How we comply |
|---|---|
| No FAISS | `vector_store.py` uses `np.ndarray` + manual cosine similarity |
| No rank-bm25 | `search.py` uses custom `keyword_score()` with term overlap |
| No sentence-transformers | `reranker.py` sorts by combined score, no cross-encoder |
| No scikit-learn | Cosine similarity via `np.dot` + `np.linalg.norm`, no TF-IDF |
| Embeddings | Mistral API via httpx (external embedding model is allowed) |
| All retrieval logic | Implemented from first principles in Python + NumPy |
