# RAG Pipeline - PDF Knowledge Base System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using semantic search and Large Language Models. To comply with the assignment constraints, all retrieval, similarity scoring, and ranking logic is implemented from first principles in Python without using external search, RAG, or vector database libraries.

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

**Text Extraction Considerations:**
- **Library Choice:** Using `PyPDF2` with fallback to `pdfplumber` for complex layouts
- **Handling Scanned PDFs:** OCR support via `pytesseract` for image-based PDFs
- **Metadata Preservation:** Extract page numbers, file names, creation dates for citation
- **Error Handling:** Graceful degradation for corrupted or password-protected PDFs

**Chunking Strategy:**
- **Fixed-size chunking with overlap:**
  - Chunk size: 500-1000 tokens (configurable)
  - Overlap: 100-200 tokens to preserve context across boundaries
- **Semantic chunking:** Respect paragraph/section boundaries when possible
- **Considerations:**
  - Small chunks: Better precision, more relevant results
  - Large chunks: More context, but may dilute relevance
  - Overlap: Ensures important information at boundaries isn't lost
  - Trade-off: Balancing between context preservation and retrieval precision

**Implementation:**
```
PDF → Text Extraction → Chunk Split → Embedding → Vector Store
                    ↓
              Metadata Extraction
```

### 2. Query Processing

**Intent Detection:**
- **Rule-based filters:** Detect greetings, chitchat, off-topic queries
- **Classification:** Use lightweight Mistral model to classify query intent
- **Categories:**
  - `KNOWLEDGE_SEARCH`: Requires RAG pipeline
  - `CHITCHAT`: Simple greeting/casual conversation
  - `CLARIFICATION`: Follow-up questions
  - `OFF_TOPIC`: Outside knowledge base scope

**Query Transformation:**
- **Query expansion:** Add synonyms, related terms
- **Query rewriting:** Rephrase for better semantic matching
- **Contextual enhancement:** Incorporate chat history for follow-ups
- **Techniques:**
  - HyDE (Hypothetical Document Embeddings): Generate hypothetical answer, use for search
  - Multi-query generation: Generate 2-3 variations of the query
  - Question decomposition: Break complex questions into sub-queries

### 3. Semantic Search

**Vector Store (Custom Implementation):**
- **No third-party vector DB or search library** (bonus requirement met)
- **In-memory NumPy matrix:** Embeddings are stored as a NumPy matrix; retrieval is performed using manual cosine similarity without external search libraries
- **Cosine similarity:** Computed from first principles: `cos(a, b) = (a · b) / (‖a‖ × ‖b‖)` using NumPy dot products and norms
- **Top-k retrieval:** `np.argsort` on the similarity scores, selecting the highest-k indices
- **Persistence:** Save/load embeddings matrix to disk with `np.save`/`np.load` for durability
- **Embedding model:** Mistral AI embeddings API

**Hybrid Search Strategy:**
- **Semantic search:** Manual cosine similarity over the NumPy embeddings matrix (no FAISS, no scikit-learn)
- **Keyword search:** Custom term-overlap scoring function implemented in Python (no rank-bm25). Tokenize query and chunk text, compute keyword overlap as the fraction of query terms found in the chunk, normalize score to 0–1
- **Combination method — Weighted sum:**
  ```
  final_score = α × semantic_score + (1-α) × keyword_score
  ```
  where α=0.7 (configurable). Final scores are computed as a weighted sum of semantic similarity and keyword overlap scores. No RRF library or external helpers — just math.

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
- Re-ranking is performed by sorting combined relevance scores; no external models are used
- Sort all candidates by their weighted hybrid score (semantic + keyword)
- Return the top-k results (default k=5)

**Result merging:**
- Deduplicate chunks from same document
- Apply diversity penalty to avoid redundant results

**Similarity threshold filtering:**
- Minimum similarity threshold: 0.7 (configurable)
- Drop chunks below the threshold
- If top result < threshold → return "insufficient evidence"

### 5. LLM Generation

**Mistral AI Integration:**
- Model: `mistral-large-latest` for generation
- API endpoint: `https://api.mistral.ai/v1/chat/completions`

**Prompt Templates by Intent:**

**Standard QA Template:**
```
You are a helpful AI assistant. Answer the user's question based solely on the provided context.

Context:
{retrieved_chunks}

Question: {user_query}

Rules:
- Only use information from the context above
- If the context doesn't contain enough information, say "I don't have sufficient information to answer this question"
- Cite the source (e.g., [Page X]) when making claims
- Be concise and accurate

Answer:
```

**List/Table Template:**
```
Based on the context below, provide your answer in a structured format.

Context:
{retrieved_chunks}

Question: {user_query}

Provide a structured response (list, table, or bullet points as appropriate).

Answer:
```

**Hallucination Prevention:**
- Evidence checking: Post-process answer to verify each claim against retrieved chunks
- Sentence-level verification: Flag sentences without supporting evidence
- Confidence scoring: Return confidence level with each answer

### 6. Bonus Features Implementation

**Citations Required:**
- Minimum similarity threshold: 0.7
- If best match < 0.7 → "I don't have sufficient evidence to answer this question"
- Include page numbers and source document in citations

**Answer Shaping:**
- Intent-based template selection
- Structured output parsing for lists/tables
- JSON mode for extracting structured data

**Hallucination Filters:**
- **Claim extraction:** Parse answer into atomic claims
- **Evidence verification:** Check each claim against retrieved chunks using entailment model
- **Flagging:** Mark unsupported claims or remove them
- **Confidence score:** Return per-claim confidence

**Query Refusal Policies:**
- **PII detection:** Refuse to process queries containing SSN, credit card numbers, etc.
- **Legal/Medical disclaimers:** Add disclaimers for queries in sensitive domains
- **Out-of-scope:** Detect and refuse queries about topics outside knowledge base

## API Endpoints

### Ingestion Endpoints

**POST /api/ingest**
- Upload one or more PDF files
- Returns: Job ID and processing status

**GET /api/ingest/status/{job_id}**
- Check ingestion status
- Returns: Progress, chunks created, errors

**GET /api/documents**
- List all ingested documents
- Returns: Document metadata, page count, ingestion date

**DELETE /api/documents/{document_id}**
- Remove document from knowledge base

### Query Endpoints

**POST /api/query**
- Submit a question to the RAG system
- Request body:
  ```json
  {
    "query": "What is...?",
    "top_k": 5,
    "include_sources": true,
    "session_id": "optional-session-id"
  }
  ```
- Returns:
  ```json
  {
    "answer": "...",
    "sources": [...],
    "confidence": 0.85,
    "intent": "KNOWLEDGE_SEARCH",
    "processing_time_ms": 234
  }
  ```

**GET /api/health**
- System health check
- Returns: Service status, model availability, index stats

## Technology Stack

### Backend Framework
- **FastAPI**: Modern, fast web framework with async support
- **Pydantic**: Data validation and settings management

### PDF Processing
- **PyPDF2**: PDF text extraction
- **pdfplumber**: Fallback for complex layouts
- **pytesseract**: OCR for scanned PDFs

### Search & Retrieval
- **NumPy**: In-memory embeddings matrix with manual cosine similarity (no FAISS, no third-party vector DB)
- **Custom keyword scorer**: Term-overlap scoring implemented from scratch (no rank-bm25)
- **Score-based re-ranking**: Sorting by combined score (no cross-encoder, no sentence-transformers)

### LLM Integration
- **Mistral AI API**: Embeddings and text generation
- **httpx**: Async HTTP client for API calls

### UI
- **React + Vite**: Modern frontend build
- **TailwindCSS**: Styling
- **Markdown rendering**: For formatted responses

### Other Libraries
- **NumPy**: Vector operations, cosine similarity, embeddings storage
- **spaCy**: Text preprocessing, NER for PII detection
- **asyncio**: Async processing for better performance

## Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 16+ (for UI)
- Tesseract OCR (optional, for scanned PDFs)

### Backend Setup

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
export MISTRAL_API_KEY="CF2DvjIoshzasO0mtBkPj44fo2nXDwPk"
export DATA_DIR="./data"
export INDEX_DIR="./indexes"

# Run the application
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Access the Application
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend UI: http://localhost:5173

## Project Structure

```
rag-pipeline/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and settings
│   ├── api/
│   │   ├── ingestion.py        # Ingestion endpoints
│   │   ├── query.py            # Query endpoints
│   │   └── health.py           # Health check endpoints
│   ├── services/
│   │   ├── pdf_processor.py    # PDF extraction and chunking
│   │   ├── embeddings.py       # Mistral embeddings integration
│   │   ├── vector_store.py     # Custom NumPy-based vector store
│   │   ├── search.py           # Hybrid search (cosine similarity + keyword overlap)
│   │   ├── reranker.py         # Score-based re-ranking and filtering
│   │   ├── llm_client.py       # Mistral AI LLM client
│   │   ├── query_processor.py  # Intent detection and query transformation
│   │   └── hallucination_filter.py  # Evidence verification
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── utils/
│       ├── text_utils.py       # Text processing utilities
│       └── pii_detector.py     # PII detection
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main React component
│   │   ├── components/         # UI components
│   │   └── api/                # API client
│   └── package.json
├── tests/
│   ├── test_ingestion.py
│   ├── test_search.py
│   └── test_generation.py
├── data/                       # Uploaded PDFs
├── indexes/                    # Persisted vector indexes
├── requirements.txt
├── .env.example
└── README.md
```

## Design Decisions & Trade-offs

### 1. Chunking Strategy
**Decision:** Fixed-size chunking (800 tokens) with 150 token overlap
**Reasoning:**
- Predictable chunk sizes for consistent embedding quality
- Overlap preserves context across boundaries
- Balance between precision and context

**Alternative considered:** Semantic chunking (paragraph/section-based)
- Pros: Respects natural document structure
- Cons: Variable chunk sizes can affect retrieval consistency

### 2. Hybrid Search
**Decision:** Combine semantic (70%) + keyword (30%) search with weighted sum
**Reasoning:**
- Semantic search: Better for conceptual matches
- Keyword search: Better for exact terms, acronyms, entities
- Weighted sum: Simple, transparent, and requires no external libraries — just math
- Keyword relevance is computed using a custom term-overlap scoring function implemented in Python

**Alternative considered:** Pure semantic search
- Simpler but misses exact keyword matches

### 3. Custom Vector Store
**Decision:** In-memory NumPy matrix with disk persistence
**Reasoning:**
- No external search or vector database libraries (meets bonus requirement and assignment constraints)
- Embeddings stored as a NumPy matrix; cosine similarity computed manually
- Top-k retrieval via `np.argsort` — simple and transparent
- Persistence with `np.save`/`np.load`

**Alternative considered:** FAISS, Chromadb, Pinecone
- More features but adds forbidden or unnecessary dependencies

### 4. Re-ranking Strategy
**Decision:** Score-based sorting of hybrid results
**Reasoning:**
- Re-ranking is performed by sorting combined relevance scores; no external models are used
- Chunks below the similarity threshold are dropped
- Simple, transparent, and requires no additional libraries

### 5. Hallucination Prevention
**Decision:** Post-hoc evidence verification
**Reasoning:**
- LLM generates answer first (fluent, coherent)
- Then verify claims against evidence
- Balance between answer quality and factuality

**Alternative considered:** RAG-fusion, constrained decoding
- More complex to implement

## Performance Considerations

- **Async endpoints:** Non-blocking I/O for file uploads and LLM calls
- **Batch processing:** Process multiple PDFs in parallel
- **Caching:** Cache embeddings and frequent queries
- **Lazy loading:** Load vector index on first query, not at startup
- **Connection pooling:** Reuse HTTP connections to Mistral AI API

## Testing Strategy

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ingestion.py -v

# Run with coverage
pytest --cov=app tests/
```

## Future Enhancements

1. **Multi-modal support:** Images, tables from PDFs
2. **Conversational memory:** Session-based context tracking
3. **Active learning:** User feedback loop for improving retrieval
4. **Streaming responses:** SSE for real-time answer generation
5. **Advanced chunking:** Recursive character splitting, proposition-based chunking
6. **Query routing:** Route different query types to specialized pipelines

## References & Resources

- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [NumPy Documentation](https://numpy.org/doc/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Hybrid Search Strategies](https://www.pinecone.io/learn/hybrid-search-intro/)

## License

MIT

## Contributing

Pull requests are welcome! Please ensure tests pass and code follows PEP 8 style guidelines.
