# Phase 3: Query & Retrieval Pipeline

**Goal:** Build the complete path from "user asks a question" to "answer with citations grounded in the ingested PDFs." After this phase, you can POST a query to the API and get back an LLM-generated answer with source references, confidence score, and intent classification.

**This phase is NOT about ingestion.** Ingestion is done (Phase 2). Phase 3 is exclusively about retrieval, generation, and verification.

**Execution model:** Synchronous, same as Phase 2. No async magic.

---

## What Phase 3 Covers

```
                     ┌──────────┐
                     │  api/    │  Thin layer — receives query, returns answer
                     └────┬─────┘
                          │
                          ▼
                     ┌──────────┐
                     │  core/   │  All logic lives here
                     └────┬─────┘
                          │
    ┌──────────┬──────────┼──────────┬──────────────┐
    │          │          │          │              │
    ▼          ▼          ▼          ▼              ▼
┌────────┐┌────────┐┌─────────┐┌─────────┐┌──────────────┐
│ query  ││ search ││  llm    ││reranker ││hallucination │
│processor││(hybrid)││ client  ││         ││   filter     │
│(core/) ││(core/) ││(core/)  ││(core/)  ││  (core/)     │
└────────┘└────────┘└─────────┘└─────────┘└──────────────┘
                          │
                          ▼
                     ┌──────────┐
                     │ storage/ │  get_all() — hands over data
                     └──────────┘


Pipeline (each step = one function):

User sends query (api/)
      │
      ▼
┌─────────────────┐
│  PII check      │   utils/pii_detector — refuse if PII detected
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Detect intent  │   core/query_processor — CHITCHAT / OFF_TOPIC /
│                 │   CLARIFICATION / KNOWLEDGE_SEARCH
└────────┬────────┘
         │
         ▼ (if KNOWLEDGE_SEARCH)
┌─────────────────┐
│  Embed query    │   core/embeddings — Mistral API (reuses Phase 2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid search  │   core/search — cosine similarity + keyword scoring
│                 │   Returns top_k_retrieval candidates (20)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rerank         │   core/reranker — score sort, threshold filter,
│                 │   deduplicate, return top_k_final (5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build context  │   core/llm_client — format chunks into prompt
│  + Generate     │   Call Mistral chat completions API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Verify answer  │   core/hallucination_filter — check claims against
│                 │   source chunks, compute confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return         │   api/ returns answer, sources, confidence, intent
│  response       │
└─────────────────┘
```

---

## Precise Requirements

### Functional Requirements

1. **`POST /api/query`** accepts a JSON body matching `QueryRequest` (already defined in schemas.py):
   - `query: str` — the user's question
   - `top_k: int = 5` — how many source chunks to return
   - `include_sources: bool = True` — whether to include source citations
   - `session_id: str | None = None` — reserved for future conversation memory

2. **PII Refusal:** If the query contains PII (SSN, credit card, email, phone), refuse immediately with a clear error message. Do not process the query.

3. **Intent Detection:** Classify every query into one of:
   - `CHITCHAT` — greetings, small talk ("hi", "how are you")
   - `OFF_TOPIC` — questions unrelated to the knowledge base
   - `KNOWLEDGE_SEARCH` — questions answerable from the ingested PDFs (default)

4. **Hybrid Search:** Combine semantic similarity (cosine) and keyword overlap into a single ranked list:
   - Semantic: embed the query, compute cosine similarity against all stored embeddings
   - Keyword: tokenize query and chunks, compute term overlap score
   - Combined: `final_score = α * semantic_score + (1 - α) * keyword_score` where `α = settings.semantic_weight` (0.7)

5. **Reranking:** From the merged candidate set:
   - Filter out chunks below `settings.similarity_threshold` (0.7)
   - Sort descending by combined score
   - Deduplicate (same source + same page)
   - Return top `top_k_final` (5 by default, overridable via request `top_k`)

6. **Answer Generation:** Send the top chunks as context to Mistral's chat model with a structured prompt that:
   - Instructs the LLM to answer ONLY from the provided context
   - Requires inline citations in `[Source: filename, Page N]` format
   - Instructs the LLM to say "I don't have enough information" if context is insufficient

7. **Hallucination Check:** Verify the generated answer against source chunks:
   - Split answer into sentences
   - For each sentence, check if any source chunk provides supporting evidence (keyword overlap)
   - Compute `confidence = supported_sentences / total_sentences`
   - Flag unsupported claims

8. **Response:** Return `QueryResponse` (already defined in schemas.py):
   - `answer: str` — the LLM-generated answer with citations
   - `sources: list[Source]` — the retrieved chunks with scores
   - `confidence: float` — hallucination check confidence (0.0–1.0)
   - `intent: str` — detected intent category
   - `processing_time_ms: int` — total time from request to response

### Non-Functional Requirements

- All search logic implemented from scratch with NumPy — no FAISS, no scikit-learn
- Keyword scoring from scratch — no rank-bm25, no TF-IDF libraries
- No cross-encoder reranking — pure score-based sorting
- LLM calls via httpx to Mistral API — same pattern as embeddings client
- All logic in `core/` — pure Python, no FastAPI imports, testable in isolation
- API layer in `api/` — thin, no business logic

---

## Architecture

### New Files to Create

```
app/
├── core/
│   ├── search.py               ← NEW  Hybrid search (cosine + keyword)
│   ├── reranker.py             ← NEW  Score sorting, threshold, dedup
│   ├── llm_client.py           ← NEW  Mistral chat completions client
│   ├── query_processor.py      ← NEW  Intent detection + query orchestrator
│   └── hallucination_filter.py ← NEW  Answer verification
├── api/
│   └── query.py                ← NEW  POST /api/query endpoint
└── main.py                     ← MODIFY  Register query router
```

### Files Unchanged

```
app/config.py                   ← Already has all needed settings
app/models/schemas.py           ← Already has QueryRequest, QueryResponse, Source
app/core/embeddings.py          ← Reuse get_embedding() for query embedding
app/core/pdf_processor.py       ← Not touched (ingestion only)
app/core/ingest_service.py      ← Not touched (ingestion only)
app/storage/vector_store.py     ← Reuse get_all() to retrieve data
app/utils/text_utils.py         ← Reuse tokenize() for keyword scoring
app/utils/pii_detector.py       ← Reuse contains_pii() for query refusal
app/api/health.py               ← Not touched
app/api/ingestion.py            ← Not touched
```

### Dependency Graph (what imports what)

```
api/query.py
  └── core/query_processor.py
        ├── core/search.py
        │     ├── core/embeddings.py        (get_embedding — reuse)
        │     ├── storage/vector_store.py    (get_all — reuse)
        │     └── utils/text_utils.py        (tokenize — reuse)
        ├── core/reranker.py
        ├── core/llm_client.py
        │     └── config.py                 (settings — reuse)
        ├── core/hallucination_filter.py
        │     └── utils/text_utils.py       (tokenize — reuse)
        └── utils/pii_detector.py           (contains_pii — reuse)
```

---

## Step 1 — Hybrid Search (`app/core/search.py`)

**What:** Combine semantic (cosine) and keyword (term overlap) scoring to find the most relevant chunks for a query.

**Why hybrid:** Semantic search (embeddings) is great at understanding meaning ("What are the financial results?" matches "The company reported $5M revenue"). But it can miss exact keyword matches ("What is EBITDA?" — the word "EBITDA" should boost chunks containing it literally). Combining both gives better retrieval than either alone.

### Function 1: `cosine_similarity(query_embedding, embeddings_matrix) -> np.ndarray`

**Purpose:** Compute cosine similarity between a single query vector and all stored embedding vectors.

**Implementation (from first principles, no scikit-learn):**
```python
def cosine_similarity(query_embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
    dot_products = embeddings_matrix @ query_embedding            # (n,)
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(embeddings_matrix, axis=1)       # (n,)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)
    return similarities
```

**Why `+ 1e-10`:** Prevents division by zero if any vector has zero norm (degenerate edge case).

**Returns:** A 1-D NumPy array of shape `(n_chunks,)` with similarity scores in [-1, 1]. Higher = more similar.

### Function 2: `keyword_score(query_tokens, chunk_tokens) -> float`

**Purpose:** Compute term-overlap score between tokenized query and tokenized chunk.

**Implementation (from scratch, no BM25):**
```python
def keyword_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    return len(overlap) / len(query_set)
```

**Why this formula:** Measures what fraction of query terms appear in the chunk. Score of 1.0 means every query keyword is found. Simple, interpretable, and effective for boosting exact matches.

### Function 3: `hybrid_search(query, top_k) -> list[tuple[Chunk, float]]`

**Purpose:** Full hybrid search pipeline. Returns `(chunk, combined_score)` pairs sorted by score.

**Flow:**
```
1. Embed query             → query_embedding (np.ndarray)
2. Get all chunks + embs   → vector_store.get_all()
3. Cosine similarity       → semantic_scores (np.ndarray)
4. Tokenize query          → query_tokens (list[str])
5. Keyword score each chunk → keyword_scores (list[float])
6. Combine scores          → final = α * semantic + (1-α) * keyword
7. Sort descending         → top_k candidates
8. Return list[(Chunk, float)]
```

**Score combination:**
```python
α = settings.semantic_weight  # 0.7
final_score = α * semantic_score + (1 - α) * keyword_score
```

With `α = 0.7`, semantic similarity contributes 70% and keyword overlap contributes 30%.

**Edge cases:**
- Empty vector store (no documents ingested) → return empty list
- Query embedding fails → propagate EmbeddingError
- All scores below threshold → return empty list (handled by reranker)

**File:** `app/core/search.py`

---

## Step 2 — Reranker (`app/core/reranker.py`)

**What:** Take the raw scored candidates from hybrid search, filter, deduplicate, and return the final top-k results.

**Why separate from search:** Search produces candidates. Reranking applies business rules (threshold, dedup, truncation). Keeping them separate makes each testable and modifiable independently.

### Function: `rerank(scored_chunks, top_k, threshold) -> list[tuple[Chunk, float]]`

**Purpose:** Post-process search results into the final ranked list.

**Steps:**
```
1. Filter       → remove chunks with score < threshold
2. Sort          → descending by score
3. Deduplicate   → keep only the highest-scoring chunk per (source, page) pair
4. Truncate      → return at most top_k results
```

**Deduplication logic:**
```python
seen = set()
deduped = []
for chunk, score in sorted_candidates:
    key = (chunk.source, chunk.page)
    if key not in seen:
        seen.add(key)
        deduped.append((chunk, score))
```

**Why deduplicate on (source, page):** Due to chunk overlap, consecutive chunks from the same page often contain similar text. Returning multiple overlapping chunks from the same page wastes context window space and provides no additional information to the LLM.

**Parameters:**
- `top_k` — from `QueryRequest.top_k` (default 5) or `settings.top_k_final`
- `threshold` — from `settings.similarity_threshold` (0.7)

**Edge cases:**
- All chunks below threshold → return empty list
- Fewer chunks than top_k after dedup → return all that remain
- Empty input → return empty list

**File:** `app/core/reranker.py`

---

## Step 3 — LLM Client (`app/core/llm_client.py`)

**What:** Call Mistral's chat completions API to generate answers from retrieved context. Also formats chunks into the prompt.

**Same HTTP pattern as `embeddings.py`:** httpx, retry with backoff, fail fast on auth errors.

### Function 1: `format_context(chunks_with_scores) -> str`

**Purpose:** Convert a list of `(Chunk, score)` pairs into a formatted context string for the LLM prompt.

**Format:**
```
[Source: report.pdf, Page 3] (Score: 0.92)
The company reported $5M revenue in 2023, representing a 20% increase
over the previous year...

[Source: report.pdf, Page 7] (Score: 0.85)
EBITDA margin improved to 15%, driven by operational efficiency
and cost reduction measures...
```

**Why this format:**
- Each chunk is clearly labeled with source and page — the LLM can reference these in its citations
- Score is included so the LLM knows which sources are most relevant
- Clear visual separation between chunks

### Function 2: `build_qa_prompt(query, context) -> list[dict]`

**Purpose:** Construct the messages array for the Mistral chat API.

**System prompt:**
```
You are a precise, helpful assistant that answers questions based ONLY on the
provided context from PDF documents.

Rules:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. Cite your sources inline using [Source: filename, Page N] format.
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided documents to answer this question."
4. Be concise and direct. Do not speculate or extrapolate beyond what the sources state.
5. If multiple sources support the answer, cite all of them.
```

**User message:**
```
Context:
{formatted_context}

Question: {query}

Answer based only on the context above, with inline citations:
```

### Function 3: `generate(messages) -> str`

**Purpose:** Call the Mistral chat completions API and return the generated text.

**Implementation:**
- POST to `{settings.mistral_base_url}/chat/completions`
- Body: `{"model": settings.mistral_chat_model, "messages": messages}`
- Parse `response["choices"][0]["message"]["content"]`
- Retry logic: same pattern as embeddings (3 attempts, exponential backoff on 429/network errors)
- Fail fast on 401

**Error handling:**
- Missing API key → `LLMError`
- Auth failure (401) → `LLMError`, no retry
- Rate limit (429) → retry with backoff
- Network error → retry with backoff
- Unexpected response format → `LLMError`

### Function 4: `generate_chitchat_response(query) -> str`

**Purpose:** Handle chitchat queries without retrieving context.

**Implementation:** Call Mistral with a simple system prompt:
```
You are a friendly assistant for a PDF knowledge base.
Respond briefly to the greeting or small talk, then mention
that you can help answer questions about the uploaded documents.
```

**Why separate:** Chitchat queries don't need retrieval. Sending them through the full RAG pipeline wastes API calls and produces nonsensical citations.

**File:** `app/core/llm_client.py`

---

## Step 4 — Query Processor (`app/core/query_processor.py`)

**What:** Intent detection and orchestration of the full query pipeline. This is the Phase 3 equivalent of `ingest_service.py` — it coordinates all the pieces.

### Function 1: `detect_intent(query) -> str`

**Purpose:** Classify the query into an intent category.

**Rule-based detection (no LLM call needed):**

```python
CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings)\b",
    r"^how are you",
    r"^good (morning|afternoon|evening)",
    r"^(thanks|thank you|thx)",
    r"^(bye|goodbye|see you)",
    r"^what'?s up",
]

def detect_intent(query: str) -> str:
    lowered = query.strip().lower()

    # Check chitchat patterns.
    for pattern in CHITCHAT_PATTERNS:
        if re.search(pattern, lowered):
            return "CHITCHAT"

    # Default: assume knowledge search.
    return "KNOWLEDGE_SEARCH"
```

**Why rule-based, not LLM-based:** An LLM call adds ~1-2 seconds of latency per query. Chitchat patterns are simple and deterministic — regex handles them in microseconds. If a query doesn't match any chitchat pattern, it's treated as a knowledge search (the safe default).

**Why no OFF_TOPIC or CLARIFICATION yet:** These require either conversation history (CLARIFICATION) or knowledge of what documents are ingested (OFF_TOPIC). Both are better handled in Phase 4 when we add conversation memory. For now, any non-chitchat query goes through the full RAG pipeline, and if no relevant chunks are found, the LLM will respond with "I don't have enough information."

### Function 2: `process_query(request: QueryRequest) -> QueryResponse`

**Purpose:** Full query pipeline orchestrator. The single entry point called by the API layer.

**Flow:**
```python
def process_query(request: QueryRequest) -> QueryResponse:
    start_time = time.time()

    # Step 1 — PII check.
    if contains_pii(request.query):
        pii_types = get_pii_types(request.query)
        raise QueryRefusalError(
            f"Query contains PII ({', '.join(pii_types)}). "
            "Please remove personal information and try again."
        )

    # Step 2 — Intent detection.
    intent = detect_intent(request.query)

    # Step 3 — Handle chitchat (no retrieval needed).
    if intent == "CHITCHAT":
        answer = generate_chitchat_response(request.query)
        elapsed = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
            intent=intent,
            processing_time_ms=elapsed,
        )

    # Step 4 — Hybrid search.
    candidates = hybrid_search(request.query, top_k=settings.top_k_retrieval)

    # Step 5 — Rerank.
    top_k = request.top_k or settings.top_k_final
    ranked = rerank(candidates, top_k=top_k, threshold=settings.similarity_threshold)

    # Step 6 — No results found.
    if not ranked:
        elapsed = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer="I don't have enough information in the provided documents to answer this question.",
            sources=[],
            confidence=0.0,
            intent=intent,
            processing_time_ms=elapsed,
        )

    # Step 7 — Generate answer.
    context = format_context(ranked)
    messages = build_qa_prompt(request.query, context)
    answer = generate(messages)

    # Step 8 — Hallucination check.
    confidence = compute_confidence(answer, [chunk for chunk, _ in ranked])

    # Step 9 — Build sources list.
    sources = []
    if request.include_sources:
        for chunk, score in ranked:
            sources.append(Source(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text[:500],    # Truncate for response size
                score=round(score, 4),
            ))

    elapsed = int((time.time() - start_time) * 1000)

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 4),
        intent=intent,
        processing_time_ms=elapsed,
    )
```

**Error handling:**
- PII detected → `QueryRefusalError` (caught by API layer → 400)
- Embedding fails → `EmbeddingError` propagates → 500
- LLM call fails → `LLMError` propagates → 500
- No chunks in store → empty search results → graceful "no info" response

**File:** `app/core/query_processor.py`

---

## Step 5 — Hallucination Filter (`app/core/hallucination_filter.py`)

**What:** Verify the LLM's answer against the source chunks. Compute a confidence score based on how much of the answer is supported by the sources.

### Function 1: `compute_confidence(answer, chunks) -> float`

**Purpose:** Score how well the answer is grounded in the source chunks.

**Algorithm:**
```
1. Split answer into sentences (reuse _SENTENCE_SPLIT_RE from text_utils)
2. For each sentence:
   a. Tokenize the sentence (lowercase, remove stopwords)
   b. For each source chunk, tokenize it
   c. Compute keyword overlap: |sentence_tokens ∩ chunk_tokens| / |sentence_tokens|
   d. If the best overlap score >= 0.5 for any chunk → sentence is "supported"
3. confidence = supported_count / total_sentences
```

**Why keyword overlap, not embedding similarity:**
- Faster — no API call needed
- More interpretable — we can explain exactly which words matched
- Good enough — if half the non-stopword tokens in a sentence appear in a source chunk, that sentence is likely grounded

**Edge cases:**
- Empty answer → return 0.0
- Answer with no sentences (e.g., single word) → treat as 1 sentence
- Citation-only sentences like "[Source: file.pdf, Page 3]" → skip (they're references, not claims)
- Sentences starting with "I don't have" or "Based on the" → skip (meta-statements, not claims)

**Confidence thresholds (informational, not enforced):**
- `>= 0.8` — high confidence, answer is well-grounded
- `0.5 - 0.8` — moderate confidence, some claims may not be directly supported
- `< 0.5` — low confidence, answer may contain hallucinations

**File:** `app/core/hallucination_filter.py`

---

## Step 6 — Query API Endpoint (`app/api/query.py`)

**What:** Thin HTTP layer for the query endpoint. Same design pattern as `api/ingestion.py` — no business logic, just receive request, call core, return response.

### Endpoint: `POST /api/query`

**Request body:** `QueryRequest` (already defined)
```json
{
    "query": "What were the company's financial results?",
    "top_k": 5,
    "include_sources": true
}
```

**Success response:** `QueryResponse` (already defined)
```json
{
    "answer": "According to the annual report, the company reported $5M revenue in 2023 [Source: report.pdf, Page 3], representing a 20% increase over the previous year. The EBITDA margin improved to 15% [Source: report.pdf, Page 7].",
    "sources": [
        {
            "chunk_id": "report.pdf_p3_c2",
            "source": "report.pdf",
            "page": 3,
            "text": "The company reported $5M revenue...",
            "score": 0.92
        }
    ],
    "confidence": 0.85,
    "intent": "KNOWLEDGE_SEARCH",
    "processing_time_ms": 2340
}
```

**Error responses:**
- PII detected → 400 `{"detail": "Query contains PII (ssn, email). Please remove personal information."}`
- Empty query → 422 (FastAPI validation)
- Embedding/LLM failure → 500 `{"detail": "..."}`

**Route handler (thin):**
```python
@router.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return process_query(request)
    except QueryRefusalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {exc}")
```

**File:** `app/api/query.py`

---

## Step 7 — Wire Into `app/main.py`

**Changes:**
1. Import the query router
2. `app.include_router(query.router)`

That's it. One import, one line. Everything else stays the same.

---

## Files Created / Modified in Phase 3

| File | Action | Layer | Purpose |
|---|---|---|---|
| `app/core/search.py` | **Create** | core | cosine_similarity, keyword_score, hybrid_search |
| `app/core/reranker.py` | **Create** | core | rerank (filter, sort, dedup, truncate) |
| `app/core/llm_client.py` | **Create** | core | format_context, build_qa_prompt, generate |
| `app/core/query_processor.py` | **Create** | core | detect_intent, process_query (orchestrator) |
| `app/core/hallucination_filter.py` | **Create** | core | compute_confidence |
| `app/api/query.py` | **Create** | api | POST /api/query (thin route) |
| `app/main.py` | **Modify** | — | Register query router |

---

## Coding Order

We will code in dependency order — each file only imports from files that already exist:

```
Step 1:  app/core/search.py                ← depends on embeddings, vector_store, text_utils
Step 2:  app/core/reranker.py              ← depends on schemas (Chunk) only
Step 3:  app/core/llm_client.py            ← depends on config only
Step 4:  app/core/hallucination_filter.py   ← depends on text_utils, schemas
Step 5:  app/core/query_processor.py        ← depends on search, reranker, llm_client,
                                              hallucination_filter, pii_detector, schemas
Step 6:  app/api/query.py                   ← thin wrapper around query_processor
Step 7:  app/main.py (modify)               ← register query router
```

---

## Folder Responsibility Rules (unchanged from Phase 2)

### `api/` — Thin HTTP layer
- No business logic
- No search logic, no LLM calls, no scoring
- Just: request → call core → response

### `core/` — Pure logic
- No FastAPI imports
- No request/response HTTP objects
- Functions are testable in isolation
- This is where search, ranking, generation, and verification live

### `storage/` — Dumb data layer (unchanged)
- `get_all()` hands over chunks + embeddings
- No new methods needed — search logic stays in `core/`

### `models/` — Data contracts (unchanged)
- `QueryRequest`, `QueryResponse`, `Source` already defined
- No new schemas needed

### `utils/` — Shared helpers (unchanged)
- `tokenize()` reused by search and hallucination filter
- `contains_pii()` reused by query processor

---

## Configuration Used (all already in `app/config.py`)

| Setting | Value | Used by |
|---|---|---|
| `mistral_api_key` | (from .env) | llm_client, embeddings |
| `mistral_chat_model` | `"mistral-large-latest"` | llm_client |
| `mistral_embed_model` | `"mistral-embed"` | embeddings (reuse) |
| `mistral_base_url` | `"https://api.mistral.ai/v1"` | llm_client, embeddings |
| `top_k_retrieval` | `20` | search (broad retrieval pass) |
| `top_k_final` | `5` | reranker (final results) |
| `similarity_threshold` | `0.7` | reranker (minimum score cutoff) |
| `semantic_weight` | `0.7` | search (α in hybrid formula) |

No new configuration values needed. Everything was defined in Phase 1.

---

## What We Can Test After Phase 3

1. **Query with ingested documents:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'
   ```

2. **PII refusal:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "My SSN is 123-45-6789, what is ML?"}'
   ```
   Expected: 400 with PII error message.

3. **Chitchat handling:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello!"}'
   ```
   Expected: Friendly response, no sources, intent = "CHITCHAT".

4. **No documents ingested:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the revenue?"}'
   ```
   Expected: "I don't have enough information" response, empty sources.

5. **Custom top_k:**
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain the findings", "top_k": 3}'
   ```
   Expected: At most 3 sources returned.

6. **Swagger docs:** `http://localhost:8000/docs` — new `/api/query` endpoint visible.

---

## Constraint Compliance

| Constraint | Phase 3 compliance |
|---|---|
| **No FAISS** | `search.py` uses `np.dot` + `np.linalg.norm` for cosine similarity |
| **No rank-bm25** | `search.py` uses custom `keyword_score()` with set intersection |
| **No sentence-transformers** | `reranker.py` sorts by numeric score, no cross-encoder model |
| **No scikit-learn** | No `sklearn.metrics.pairwise`, no TF-IDF. Pure NumPy. |
| **Embeddings via Mistral API** | Reuses `get_embedding()` from Phase 2 |
| **LLM via Mistral API** | `llm_client.py` calls `/v1/chat/completions` via httpx |
| **All retrieval from scratch** | Cosine sim, keyword scoring, hybrid merge, reranking — all custom |
