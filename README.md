# RAG Pipeline - PDF Knowledge Base System

A Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using semantic search and Large Language Models. All retrieval, similarity scoring, and ranking logic is implemented from first principles in Python without using external search, RAG, or vector database libraries.

## System Architecture

### High-Level Overview

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   PDF Files │─────▶│  Data Ingestion  │─────▶│  Vector Store   │
└─────────────┘      └──────────────────┘      └─────────────────┘
                              │                          │
                              │                          │
                              ▼                          ▼
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ User Query  │─────▶│ Query Processing │─────▶│ Semantic Search │
└─────────────┘      └──────────────────┘      └─────────────────┘
                              │                          │
                              │                          │
                              ▼                          ▼
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Response  │◀─────│   LLM Generation │◀─────│ Post-processing │
└─────────────┘      └──────────────────┘      └─────────────────┘
```

## Core Components

### 1. Data Ingestion Pipeline

**Endpoint:** `POST /api/ingest`

**Text Extraction:**
- **Primary:** `PyPDF2` for text extraction
- **Fallback:** `pdfplumber` for pages where PyPDF2 yields poor results
- **Metadata:** Page numbers and source filenames are preserved for citation
- **Error Handling:** Graceful degradation for corrupted or empty pages

**Chunking Strategy:**
- **Fixed-size chunking with overlap:**
  - Chunk size: 800 tokens (configurable)
  - Overlap: 150 tokens (configurable)
  - Sentence boundary awareness: avoids splitting mid-sentence where possible
- **Considerations:**
  - Small chunks: Better precision, more relevant results
  - Large chunks: More context, but may dilute relevance
  - Overlap: Ensures important information at boundaries isn't lost

**Implementation:**
```
PDF → Text Extraction → Chunk Split → Embedding → Vector Store
                    ↓
              Metadata (page, source)
```

### 2. Query Processing

**Intent Detection:**
- **Regex-based classification** to detect greetings and chitchat
- **Categories:**
  - `KNOWLEDGE_SEARCH`: Requires RAG pipeline (retrieval + generation)
  - `CHITCHAT`: Greetings and casual conversation (handled without retrieval)

**PII Detection:**
- Queries containing PII patterns (SSN, credit card numbers, emails, phone numbers) are refused before any retrieval occurs

### 3. Semantic Search

**Vector Store (Custom Implementation):**
- **No third-party vector DB or search library** (assignment constraint met)
- **In-memory NumPy matrix:** Embeddings stored as a NumPy array; retrieval via manual cosine similarity
- **Cosine similarity:** Computed from first principles: `cos(a, b) = (a · b) / (‖a‖ × ‖b‖)` using NumPy dot products and norms
- **Top-k retrieval:** `np.argsort` on similarity scores, selecting the highest-k indices
- **Persistence:** `np.save`/`np.load` for embeddings, JSON for chunk metadata and document records
- **Embedding model:** Mistral AI embeddings API (`mistral-embed`)

**Hybrid Search Strategy:**
- **Semantic search:** Manual cosine similarity over the NumPy embeddings matrix (no FAISS, no scikit-learn)
- **Keyword search:** Custom term-overlap scoring — tokenize query and chunk, compute fraction of query terms found in the chunk, normalize to 0–1
- **Combination — Weighted sum:**
  ```
  final_score = α × semantic_score + (1-α) × keyword_score
  ```
  where α=0.7 (configurable). No external libraries — just math.

**Search Flow:**
```
Query → [Embedding]    → Cosine Similarity over NumPy matrix (top-k=20)
        [Tokenization] → Term-Overlap Keyword Scoring (top-k=20)
                              ↓
                    Weighted Score Combination
                              ↓
                    Sorted Results (top-k=5)
```

### 4. Post-processing & Re-ranking

**Score-based re-ranking:**
- Sort all candidates by their weighted hybrid score (semantic + keyword)
- No external re-ranking models — sorting by combined score only

**Deduplication:**
- Removes duplicate chunks from the same (source, page) pair

**Similarity threshold filtering:**
- Minimum similarity threshold: 0.7 (configurable)
- Chunks below threshold are dropped
- If no chunks pass the threshold, returns "insufficient evidence" response

### 5. LLM Generation

**Mistral AI Integration:**
- Model: `mistral-large-latest` for generation
- Embeddings: `mistral-embed` for vectorization

**Prompt Design:**
- System prompt instructs the LLM to answer only from provided context
- Sources formatted as `[Source: filename, Page N]` with score
- Chitchat handled with a separate lightweight prompt (no retrieval)

**Retry Logic:**
- Exponential backoff on rate limits (HTTP 429)
- Fail-fast on authentication errors (HTTP 401)
- Up to 3 retries on network failures

### 6. Hallucination Filter

**Token-overlap confidence scoring:**
- Split the answer into sentences
- For each sentence, compute keyword overlap against source chunk tokens
- A sentence is "supported" if ≥50% of its tokens appear in at least one source chunk
- Confidence = supported sentences / total scorable sentences
- Meta-statements and citations are excluded from scoring

**Known limitation:** This is a keyword-overlap heuristic, not semantic entailment. It catches obvious hallucinations but can miss subtle contradictions or paraphrased claims.

## API Endpoints

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest` | Upload a single PDF file for processing |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/{document_id}` | Remove a document and its chunks |

### Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Submit a question to the RAG system |

**Request body:**
```json
{
  "query": "What is...?",
  "top_k": 5,
  "include_sources": true,
  "session_id": null
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [{"chunk_id": "...", "source": "file.pdf", "page": 1, "text": "...", "score": 0.85}],
  "confidence": 0.85,
  "intent": "KNOWLEDGE_SEARCH",
  "processing_time_ms": 234
}
```

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service status, API key config, document and chunk counts |

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Backend | FastAPI | Web framework with auto-generated OpenAPI docs |
| Validation | Pydantic | Data validation and settings management |
| PDF Extraction | PyPDF2, pdfplumber | Text extraction with fallback |
| Vector Operations | NumPy | Embeddings storage, cosine similarity, top-k retrieval |
| LLM & Embeddings | Mistral AI API | Text generation and embedding via `httpx` |
| UI | Streamlit | Interactive chat interface |
| Testing | pytest | Unit and integration tests |

## Installation & Setup

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone repository
git clone <repo-url>
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MISTRAL_API_KEY="your-api-key-here"
export DATA_DIR="./data"
export INDEX_DIR="./indexes"
```

### Running the Application

**API server:**
```bash
python -m uvicorn app.main:app --reload --port 8000
```

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

### Access
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## Project Structure

```
rag-pipeline/
├── app/
│   ├── main.py                      # FastAPI entry point, startup, CORS
│   ├── config.py                    # Pydantic settings (.env support)
│   ├── api/
│   │   ├── ingestion.py             # POST /api/ingest, GET/DELETE /api/documents
│   │   ├── query.py                 # POST /api/query
│   │   └── health.py               # GET /api/health
│   ├── core/
│   │   ├── ingest_service.py        # Ingestion orchestrator
│   │   ├── pdf_processor.py         # PDF text extraction and chunking
│   │   ├── embeddings.py            # Mistral embeddings API client
│   │   ├── search.py                # Hybrid search (cosine + keyword overlap)
│   │   ├── reranker.py              # Score-based re-ranking and filtering
│   │   ├── llm_client.py            # Mistral chat completions client
│   │   ├── query_processor.py       # Intent detection and query pipeline
│   │   └── hallucination_filter.py  # Token-overlap confidence scoring
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   ├── storage/
│   │   └── vector_store.py          # NumPy-backed in-memory vector store
│   └── utils/
│       ├── text_utils.py            # Text cleaning, tokenization, chunking
│       └── pii_detector.py          # Regex-based PII detection
├── tests/
│   ├── conftest.py                  # Shared fixtures, mock PDF builder
│   ├── test_text_utils.py
│   ├── test_vector_store.py
│   ├── test_query_processor.py
│   ├── test_llm_client.py
│   ├── test_api.py
│   ├── test_search.py
│   ├── test_reranker.py
│   ├── test_embeddings.py
│   ├── test_hallucination_filter.py
│   ├── test_pdf_processor.py
│   ├── test_pii_detector.py
│   ├── test_config.py
│   └── test_integration.py          # Live API tests (requires MISTRAL_API_KEY)
├── streamlit_app.py                 # Streamlit chat UI
├── data/                            # Uploaded PDFs
├── indexes/                         # Persisted embeddings and metadata
├── requirements.txt
└── README.md
```

## Design Decisions & Trade-offs

### 1. Chunking Strategy
**Decision:** Fixed-size chunking (800 tokens) with 150-token overlap and sentence boundary awareness.

**Reasoning:**
- Predictable chunk sizes for consistent embedding quality
- Overlap preserves context across boundaries
- Sentence boundaries reduce mid-thought splits

**Trade-off:** Does not respect document structure (headings, sections). A semantic chunking approach would produce more meaningful units but at the cost of variable chunk sizes.

### 2. Hybrid Search
**Decision:** Combine semantic (70%) + keyword (30%) search with weighted sum.

**Reasoning:**
- Semantic search handles conceptual matches and paraphrases
- Keyword search catches exact terms, acronyms, and named entities
- Weighted sum is simple, transparent, and requires no external libraries

**Trade-off:** The keyword scorer uses raw term overlap rather than TF-IDF. All terms are weighted equally, so common words contribute as much as rare, informative ones.

### 3. Custom Vector Store
**Decision:** In-memory NumPy matrix with disk persistence.

**Reasoning:**
- Meets the assignment constraint: no external search or vector DB libraries
- Cosine similarity computed manually — simple and transparent
- Persistence via `np.save`/`np.load` and JSON

**Trade-off:** Linear scan over all embeddings (O(n)) for each query. Suitable for thousands of chunks; would need approximate nearest neighbor indexing for larger corpora.

### 4. Re-ranking
**Decision:** Score-based sorting with threshold filtering and deduplication.

**Reasoning:**
- Simple and explainable — no black-box re-ranker
- Threshold filtering removes low-quality results
- Deduplication avoids redundant chunks from the same page

**Trade-off:** No cross-encoder or learned re-ranker. The combined score may not optimally rank results for all query types.

### 5. Hallucination Filter
**Decision:** Post-hoc token-overlap confidence scoring.

**Reasoning:**
- No additional API call required — runs locally
- Provides a coarse signal for answer grounding
- Sentences with <50% token overlap against sources are flagged as unsupported

**Trade-off:** Token overlap is a shallow heuristic. It doesn't verify semantic entailment, so paraphrased claims may be incorrectly flagged and subtle contradictions may be missed.

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_search.py -v

# Run with coverage
pytest --cov=app tests/

# Run integration tests (requires MISTRAL_API_KEY)
pytest tests/test_integration.py -v -m integration
```

The test suite includes 13 modules covering all core components with mocked external dependencies. Integration tests against the live Mistral API are marked with `@pytest.mark.integration` and skipped unless an API key is configured.

## Future Enhancements

1. **Query expansion:** HyDE or multi-query paraphrasing to improve recall
2. **TF-IDF keyword scoring:** Corpus-aware term weighting instead of raw overlap
3. **Stronger hallucination checks:** Claim-level alignment or lightweight entailment
4. **Embedding validation:** Store model name and dimensions in index metadata; fail fast on mismatch
5. **OCR support:** Fallback to `pytesseract` for scanned PDF pages with no extractable text
6. **MMR diversification:** Avoid near-duplicate chunks in retrieval results
7. **Streaming responses:** SSE for real-time answer generation

## References

- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [NumPy Documentation](https://numpy.org/doc/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

MIT
