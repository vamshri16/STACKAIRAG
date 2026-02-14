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

**Chunking Strategy — Fixed-size with overlap:**
- Chunk size: 800 tokens, overlap: 150 tokens (both configurable)
- Sentence boundary awareness: splits on `[.!?]` rather than cutting mid-sentence
- 800 tokens balances embedding quality (too short = no context, too long = diluted relevance)
- 150-token overlap (~19%) ensures boundary content isn't lost

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed-size + overlap (chosen)** | Predictable sizes, consistent embedding quality | Doesn't respect document structure |
| Recursive/semantic splitting | Meaningful units aligned to sections | Variable sizes; complex without heuristics |
| Paragraph-based | Natural boundaries | Wildly variable length |
| Sliding window (no sentence awareness) | Simplest | Mid-sentence splits embed poorly |

**Known limitation:** Sentence detector uses a simple regex — can misfire on abbreviations ("Dr.") and decimals ("3.14"). The strategy also doesn't respect document structure (headings, sections).

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

**Sub-intent Detection (Answer Shaping):**
- Knowledge queries are further classified by answer format: `LIST`, `COMPARISON`, `SUMMARY`, `FACTUAL` (default)
- Each sub-intent selects an intent-specific prompt template to shape the LLM's response format

**Query Transformation:**
- LLM-based query rewriting to improve retrieval (more specific, search-friendly phrasing)
- Dual search: both original and rewritten queries are searched, results merged with best-score deduplication

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

| Approach | Pros | Cons |
|----------|------|------|
| **Weighted fusion (chosen)** | Simple, tunable via one parameter (α) | Assumes comparable score scales |
| Reciprocal Rank Fusion (RRF) | Scale-invariant | Discards score magnitude |
| Interleaving | Easy to implement | No unified ranking |

**Why 0.7/0.3:** Semantic handles paraphrases and conceptual matches; keyword is a safety net for acronyms, product codes, and jargon. Both output [0,1] so combination is meaningful without normalization. The keyword scorer uses raw term overlap rather than TF-IDF — all terms weighted equally.

**Trade-off:** Linear scan O(n) over all embeddings. Suitable for thousands of chunks; would need ANN indexing for larger corpora.

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

No external re-ranking models — pure score-based pipeline:

1. **Threshold filter:** Drop chunks below 0.7 similarity (configurable). If none pass, return "insufficient evidence".
2. **Deduplication:** Keep only the highest-scoring chunk per (source, page) pair.
3. **MMR Diversification:** Iteratively select chunks maximizing `λ * relevance - (1-λ) * max_similarity_to_already_selected` (λ=0.7). Balances relevance with diversity.
4. **Conditional Context Expansion:** When retrieval quality is weak, automatically pull in chunks from adjacent pages (±1) of the same document and re-rerank.

**Context Expansion — How it works:**
- Triggers when *either* the best reranked score is below `expansion_score_threshold` (0.4) or fewer than `expansion_coverage_ratio` (60%) of requested results survived reranking.
- Looks up neighboring chunks (±1 page, same source document) from the vector store.
- Neighbor scores are discounted (`parent_score × 0.8`) so they rank below the chunk that pulled them in.
- The expanded candidate set is re-reranked through the full pipeline (threshold, dedup, MMR) to maintain quality gates.
- When retrieval is already strong, expansion does not fire — no added noise or latency.

| Approach | Pros | Cons |
|----------|------|------|
| **Conditional expansion (chosen)** | Only expands when needed; preserves MMR diversity | Requires threshold tuning |
| Always expand ±1 page | Maximum recall | Floods candidates with same-document chunks; adds noise |
| No expansion | Simplest | Misses cross-page context (tables, multi-page explanations) |

**Trade-off:** No cross-encoder or learned re-ranker. Simple and explainable but may not optimally rank for all query types.

### 5. LLM Generation

**Mistral AI Integration:**
- Model: `mistral-large-latest` for generation
- Embeddings: `mistral-embed` for vectorization

**Prompt Design:**
- System prompt instructs the LLM to answer only from provided context
- Sources formatted as `[Source: filename, Page N]` in the LLM context (scores excluded from prompts to prevent anchoring)
- Chitchat handled with a separate lightweight prompt (no retrieval)

**Retry Logic:**
- Exponential backoff on rate limits (HTTP 429)
- Fail-fast on authentication errors (HTTP 401)
- Up to 3 retries on network failures

### 6. Hallucination Filter

**Two-tier confidence scoring:**

1. **Semantic (primary):** Embed each answer sentence and compute cosine similarity against source chunk embeddings. A sentence is "supported" if its best chunk similarity exceeds 0.6.
2. **Token-overlap (fallback):** If embedding fails, fall back to keyword overlap — a sentence is "supported" if ≥50% of its tokens appear in at least one source chunk.

- Confidence = supported sentences / total scorable sentences
- Meta-statements and citations are excluded from scoring

**Known limitation:** Both tiers are heuristic. Semantic similarity is stronger than token overlap but still does not verify logical entailment — subtle contradictions or paraphrased claims may be missed.

## API Endpoints

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest` | Upload a single PDF file for processing |
| `POST` | `/api/ingest/batch` | Upload multiple PDF files for processing |
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
│   │   ├── context_expander.py      # Conditional context expansion (±1 page)
│   │   ├── llm_client.py            # Mistral chat completions client
│   │   ├── query_processor.py       # Intent detection and query pipeline
│   │   └── hallucination_filter.py  # Two-tier confidence scoring
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
│   ├── test_context_expander.py
│   ├── test_config.py
│   └── test_integration.py          # Live API tests (requires MISTRAL_API_KEY)
├── streamlit_app.py                 # Streamlit chat UI
├── data/                            # Uploaded PDFs
├── indexes/                         # Persisted embeddings and metadata
├── requirements.txt
└── README.md
```

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

The test suite includes 14 modules covering all core components with mocked external dependencies. Integration tests against the live Mistral API are marked with `@pytest.mark.integration` and skipped unless an API key is configured.

## Future Enhancements

1. **TF-IDF keyword scoring:** Corpus-aware term weighting instead of raw overlap
2. **OCR support:** Fallback to `pytesseract` for scanned PDF pages with no extractable text
3. **Streaming responses:** SSE for real-time answer generation

## References

- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [NumPy Documentation](https://numpy.org/doc/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

MIT
