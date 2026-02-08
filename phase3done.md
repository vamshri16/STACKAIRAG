# Phase 3 — Complete Implementation Record

Everything that was coded, every design decision, every pattern, every edge case. Nothing omitted.

---

## Table of Contents

1. [Folder Structure After Phase 3](#1-folder-structure-after-phase-3)
2. [File 1: `app/core/search.py`](#2-file-1-appcoresearchpy)
3. [File 2: `app/core/reranker.py`](#3-file-2-appcorererankerpy)
4. [File 3: `app/core/llm_client.py`](#4-file-3-appcorellm_clientpy)
5. [File 4: `app/core/hallucination_filter.py`](#5-file-4-appcorehallucination_filterpy)
6. [File 5: `app/core/query_processor.py`](#6-file-5-appcorequery_processorpy)
7. [File 6: `app/api/query.py`](#7-file-6-appapiquerypy)
8. [File 7: `app/main.py` (modified)](#8-file-7-appmainpy-modified)
9. [Architectural Patterns Used](#9-architectural-patterns-used)
10. [Edge Cases Covered](#10-edge-cases-covered)
11. [Constraint Compliance](#11-constraint-compliance)
12. [What Changed From Phase 2](#12-what-changed-from-phase-2)

---

## 1. Folder Structure After Phase 3

Before Phase 3, the `app/` directory looked like this:

```
app/
├── __init__.py
├── config.py
├── main.py
├── api/
│   ├── __init__.py
│   ├── health.py
│   └── ingestion.py
├── core/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── ingest_service.py
│   └── pdf_processor.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── services/
│   └── __init__.py          ← leftover from Phase 1 (unused, harmless)
├── storage/
│   ├── __init__.py
│   └── vector_store.py
└── utils/
    ├── __init__.py
    ├── pii_detector.py
    └── text_utils.py
```

After Phase 3, it looks like this:

```
app/
├── __init__.py
├── config.py                     ← unchanged
├── main.py                       ← MODIFIED (added query router)
├── api/
│   ├── __init__.py
│   ├── health.py                 ← unchanged
│   ├── ingestion.py              ← unchanged
│   └── query.py                  ← NEW (thin HTTP layer for /api/query)
├── core/
│   ├── __init__.py
│   ├── embeddings.py             ← unchanged (reused by search.py)
│   ├── hallucination_filter.py   ← NEW (answer verification)
│   ├── ingest_service.py         ← unchanged
│   ├── llm_client.py             ← NEW (Mistral chat completions client)
│   ├── pdf_processor.py          ← unchanged
│   ├── query_processor.py        ← NEW (pipeline orchestrator)
│   ├── reranker.py               ← NEW (score sort, dedup, threshold)
│   └── search.py                 ← NEW (hybrid cosine + keyword search)
├── models/
│   ├── __init__.py
│   └── schemas.py                ← unchanged (QueryRequest/Response already defined)
├── services/
│   └── __init__.py               ← leftover from Phase 1
├── storage/
│   ├── __init__.py
│   └── vector_store.py           ← unchanged (get_all() reused by search.py)
└── utils/
    ├── __init__.py
    ├── pii_detector.py            ← unchanged (reused by query_processor.py)
    └── text_utils.py              ← unchanged (tokenize() reused by search.py)
```

**New files created: 5 in `core/`, 1 in `api/`**
**Files modified: 1 (`main.py`)**
**Files unchanged: everything else**

---

## 2. File 1: `app/core/search.py`

**Created from scratch. 93 lines.**

**Purpose:** Hybrid search combining cosine similarity (semantic) and keyword term overlap. All retrieval logic from first principles — NumPy for cosine, Python set intersection for keywords. No FAISS, no scikit-learn, no rank-bm25.

### Full source code:

```python
"""Hybrid search — cosine similarity + keyword overlap.

Pure logic — no FastAPI imports.
All retrieval from first principles: NumPy for cosine, set intersection for keywords.
No FAISS, no scikit-learn, no rank-bm25.
"""

import logging

import numpy as np

from app.config import settings
from app.core.embeddings import get_embedding
from app.models.schemas import Chunk
from app.storage.vector_store import vector_store
from app.utils.text_utils import tokenize

logger = logging.getLogger(__name__)


def cosine_similarity(
    query_embedding: np.ndarray,
    embeddings_matrix: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query vector and all stored vectors.

    Returns a 1-D array of shape ``(n_chunks,)`` with scores in [-1, 1].
    """
    dot_products = embeddings_matrix @ query_embedding  # (n,)
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(embeddings_matrix, axis=1)  # (n,)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)
    return similarities


def keyword_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    """Compute term-overlap score between tokenized query and chunk.

    Returns the fraction of query tokens found in the chunk (0.0–1.0).
    """
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    return len(overlap) / len(query_set)


def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Full hybrid search: semantic + keyword scoring.

    1. Embed the query.
    2. Cosine similarity against all stored embeddings.
    3. Keyword overlap score for each chunk.
    4. Combine: final = α * semantic + (1-α) * keyword.
    5. Return top-k ``(Chunk, score)`` pairs sorted descending.

    Returns an empty list if the store is empty.
    """
    if top_k is None:
        top_k = settings.top_k_retrieval

    chunks, embeddings = vector_store.get_all()

    if not chunks or embeddings is None:
        return []

    # Step 1 — Embed the query.
    query_embedding = get_embedding(query)

    # Step 2 — Semantic scores.
    semantic_scores = cosine_similarity(query_embedding, embeddings)

    # Step 3 — Keyword scores.
    query_tokens = tokenize(query)
    keyword_scores = np.array(
        [keyword_score(query_tokens, tokenize(c.text)) for c in chunks],
        dtype=np.float64,
    )

    # Step 4 — Combine.
    alpha = settings.semantic_weight
    combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

    # Step 5 — Top-k by argsort.
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = [(chunks[i], float(combined_scores[i])) for i in top_indices]
    return results
```

### Design decisions:

**1. Cosine similarity from first principles (no scikit-learn):**

```python
dot_products = embeddings_matrix @ query_embedding    # (n,)
query_norm = np.linalg.norm(query_embedding)
chunk_norms = np.linalg.norm(embeddings_matrix, axis=1)  # (n,)
similarities = dot_products / (chunk_norms * query_norm + 1e-10)
```

This is the textbook cosine similarity formula: `cos(θ) = (A · B) / (||A|| × ||B||)`. The `+ 1e-10` prevents division by zero if any vector has zero norm (a degenerate edge case that shouldn't happen with real embeddings but is mathematically possible).

**Why matrix multiplication instead of a loop:**
`embeddings_matrix @ query_embedding` computes all dot products in a single vectorized operation. NumPy delegates this to BLAS (Basic Linear Algebra Subprograms), which runs in optimized C/Fortran. A Python loop over thousands of chunks would be orders of magnitude slower.

**2. Keyword scoring from scratch (no BM25):**

```python
def keyword_score(query_tokens, chunk_tokens):
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    return len(overlap) / len(query_set)
```

This measures the fraction of query keywords that appear in the chunk. A score of `1.0` means every query keyword was found. A score of `0.0` means no overlap.

**Why not BM25:** BM25 (Best Matching 25) is a TF-IDF-based ranking function that considers term frequency, document length, and inverse document frequency. It's more sophisticated but requires maintaining term frequency statistics across the corpus. Our simpler set-intersection approach is:
- Easier to understand and debug
- No corpus-wide statistics to maintain (avoids stale IDF when documents are added/deleted)
- Combined with semantic search (which already captures meaning), the keyword score only needs to boost exact matches — set overlap is sufficient for this

**3. Hybrid combination formula:**

```python
alpha = settings.semantic_weight  # 0.7
combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
```

With `α = 0.7`:
- Semantic similarity contributes **70%** of the final score
- Keyword overlap contributes **30%** of the final score

**Why 70/30:**
- Semantic search is the primary retrieval mechanism — it understands meaning ("financial results" matches "revenue report")
- Keyword scoring is a booster — it lifts chunks containing exact query terms ("EBITDA" literally appearing in the chunk)
- The 70/30 split ensures semantic understanding dominates while exact matches get a meaningful boost
- This ratio is configurable via `settings.semantic_weight` — no code change needed to adjust

**4. Reuses existing Phase 2 components:**
- `get_embedding()` from `core/embeddings.py` — embeds the query string
- `tokenize()` from `utils/text_utils.py` — tokenizes query and chunks for keyword matching
- `vector_store.get_all()` from `storage/vector_store.py` — retrieves all chunks and embeddings

No new dependencies. Pure composition of existing modules.

**5. Top-k via `np.argsort`:**

```python
top_indices = np.argsort(combined_scores)[::-1][:top_k]
```

`argsort` returns indices that would sort the array. Reversing (`[::-1]`) gives descending order. Slicing (`[:top_k]`) takes the top results. This is an O(n log n) operation — for a few thousand chunks, it completes in microseconds.

**6. Returns `list[tuple[Chunk, float]]` not a custom type:**
The return type is a simple list of `(chunk, score)` pairs. No new Pydantic model needed. The reranker, LLM client, and query processor all consume this format directly.

---

## 3. File 2: `app/core/reranker.py`

**Created from scratch. 48 lines.**

**Purpose:** Post-process search results — filter by threshold, sort by score, deduplicate, truncate. No cross-encoder. No external ranking library. Pure Python sorting on numeric scores.

### Full source code:

```python
"""Reranker — score-based sorting, threshold filtering, deduplication.

No cross-encoder.  No external ranking library.
Pure Python sorting on numeric scores produced by hybrid search.
"""

from app.models.schemas import Chunk


def rerank(
    scored_chunks: list[tuple[Chunk, float]],
    top_k: int,
    threshold: float,
) -> list[tuple[Chunk, float]]:
    """Post-process search results into the final ranked list.

    Steps:
    1. Filter out chunks with score below *threshold*.
    2. Sort descending by score.
    3. Deduplicate — keep only the highest-scoring chunk per (source, page).
    4. Truncate to *top_k* results.

    Returns a list of ``(Chunk, score)`` pairs.
    """
    if not scored_chunks:
        return []

    # Step 1 — Filter.
    filtered = [(chunk, score) for chunk, score in scored_chunks if score >= threshold]

    if not filtered:
        return []

    # Step 2 — Sort descending by score.
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Step 3 — Deduplicate on (source, page).
    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[Chunk, float]] = []
    for chunk, score in filtered:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            deduped.append((chunk, score))

    # Step 4 — Truncate.
    return deduped[:top_k]
```

### Design decisions:

**1. Four-step pipeline:**

```
Filter → Sort → Deduplicate → Truncate
```

Each step is simple and independent. The order matters:
- **Filter first** — removes noise before sorting (fewer items to sort)
- **Sort second** — needed for dedup to keep the *highest*-scoring chunk per key
- **Dedup third** — removes redundant overlapping chunks from the same page
- **Truncate last** — returns exactly `top_k` results

**2. Deduplication on `(source, page)` not just `(source)`:**

```python
key = (chunk.source, chunk.page)
```

**Why (source, page) and not just (source):**
A single PDF can have 50+ pages covering different topics. Deduplicating on filename alone would return at most 1 chunk per document, which is too aggressive. Deduplicating on (filename, page) prevents returning multiple overlapping chunks from the *same page* (caused by chunk overlap during ingestion) while still allowing chunks from different pages of the same document.

**Why this prevents overlap redundancy:**
During ingestion, chunks are created with 150-token overlap. This means chunk_c5 and chunk_c6 from the same page share ~150 tokens of text. Returning both provides almost no additional information to the LLM but wastes context window space. By keeping only the highest-scoring chunk per page, we maximize information density.

**3. Threshold filtering at `settings.similarity_threshold` (0.7):**

Chunks below 0.7 combined score are removed entirely. This prevents low-relevance chunks from being sent to the LLM, which would add noise and potentially mislead the generation.

**Why 0.7:** This is a moderately strict threshold. Cosine similarity for Mistral embeddings typically ranges:
- `> 0.85` — very similar (near-paraphrase)
- `0.7 – 0.85` — semantically related (relevant to the query)
- `0.5 – 0.7` — loosely related (may be tangential)
- `< 0.5` — unrelated

Combined with keyword scoring (which adds up to 0.3), a threshold of 0.7 ensures chunks are at least moderately relevant.

**4. No external ranking library:**

The original IMPLEMENTATION_PLAN.md specified a reranker that uses "pure score-based sorting" with "no cross-encoder." This implementation follows that exactly — it's a Python `sort()` on numeric values. No sentence-transformers, no learned ranking models.

**5. Returns the same type as search:**

Input and output are both `list[tuple[Chunk, float]]`. This means the reranker is a pure filter — it can be composed or bypassed without changing the pipeline's type signatures.

---

## 4. File 3: `app/core/llm_client.py`

**Created from scratch. 176 lines.**

**Purpose:** Mistral AI chat completions client. Formats context, builds prompts, calls the API, parses responses. Same HTTP pattern as `embeddings.py` — httpx with retry and exponential backoff.

### Full source code:

```python
"""Mistral AI chat completions client.

Synchronous — same pattern as embeddings.py.
Pure logic — no FastAPI imports.  Testable in isolation.
"""

import logging
import time

import httpx

from app.config import settings
from app.models.schemas import Chunk

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0

_QA_SYSTEM_PROMPT = """\
You are a precise, helpful assistant that answers questions based ONLY on the \
provided context from PDF documents.

Rules:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. Cite your sources inline using [Source: filename, Page N] format.
3. If the context does not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer this question."
4. Be concise and direct. Do not speculate or extrapolate beyond what the sources state.
5. If multiple sources support the answer, cite all of them."""

_CHITCHAT_SYSTEM_PROMPT = """\
You are a friendly assistant for a PDF knowledge base. \
Respond briefly to the greeting or small talk, then mention \
that you can help answer questions about the uploaded documents."""


class LLMError(Exception):
    """Raised when the chat completions API call fails."""


def _validate_api_key() -> None:
    if not settings.mistral_api_key:
        raise LLMError(
            "Mistral API key is not configured. "
            "Set MISTRAL_API_KEY in your .env file."
        )


# ------------------------------------------------------------------
# Context formatting
# ------------------------------------------------------------------


def format_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    """Convert retrieved chunks into a formatted context string for the prompt."""
    parts: list[str] = []
    for chunk, score in chunks_with_scores:
        header = f"[Source: {chunk.source}, Page {chunk.page}] (Score: {score:.2f})"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_qa_prompt(query: str, context: str) -> list[dict]:
    """Construct the messages array for the Mistral chat API."""
    return [
        {"role": "system", "content": _QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer based only on the context above, with inline citations:"
            ),
        },
    ]


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------


def generate(messages: list[dict]) -> str:
    """Call the Mistral chat completions API and return the generated text.

    Retries on rate limits (429) and network errors.
    Fails fast on auth errors (401).
    """
    _validate_api_key()

    url = f"{settings.mistral_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mistral_chat_model,
        "messages": messages,
    }

    last_error: Exception | None = None
    backoff = _INITIAL_BACKOFF_SECONDS

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code == 401:
                raise LLMError(
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
                raise LLMError("Mistral API rate limit exceeded after retries.")

            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMError(
                    f"Mistral API error (HTTP {response.status_code}): {error_body}"
                )

            return _parse_chat_response(response.json())

        except LLMError:
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

    raise LLMError(
        f"Mistral API request failed after {_MAX_RETRIES} attempts: {last_error}"
    )


def generate_chitchat_response(query: str) -> str:
    """Handle chitchat queries without retrieval."""
    messages = [
        {"role": "system", "content": _CHITCHAT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return generate(messages)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_chat_response(body: dict) -> str:
    """Extract the assistant message from the Mistral chat API response."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Unexpected Mistral chat API response format: {exc}"
        ) from exc
```

### Design decisions:

**1. Two system prompts — QA vs. Chitchat:**

**QA system prompt (`_QA_SYSTEM_PROMPT`):**
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

**Why these specific rules:**
- Rule 1 prevents hallucination — the LLM must not make things up
- Rule 2 enforces citations — every claim must reference a source document and page
- Rule 3 provides a safe fallback — "I don't know" is better than a fabricated answer
- Rule 4 prevents speculation — the LLM should not extrapolate beyond what the documents say
- Rule 5 ensures completeness — if multiple chunks support a point, cite all of them

**Chitchat system prompt (`_CHITCHAT_SYSTEM_PROMPT`):**
```
You are a friendly assistant for a PDF knowledge base.
Respond briefly to the greeting or small talk, then mention
that you can help answer questions about the uploaded documents.
```

**Why a separate prompt:** Chitchat queries ("hi", "how are you") don't need retrieved context. Sending them through the QA prompt would result in "I don't have enough information" which is a poor user experience for a greeting. The chitchat prompt generates a friendly response and redirects the user to ask about their documents.

**2. Context formatting with citation headers:**

```python
def format_context(chunks_with_scores):
    for chunk, score in chunks_with_scores:
        header = f"[Source: {chunk.source}, Page {chunk.page}] (Score: {score:.2f})"
        parts.append(f"{header}\n{chunk.text}")
```

Each chunk in the context is labeled with:
- Source filename — so the LLM can cite it as `[Source: report.pdf, Page 3]`
- Page number — for precise page-level citations
- Score — so the LLM can prioritize higher-confidence sources

**Example formatted context:**
```
[Source: report.pdf, Page 3] (Score: 0.92)
The company reported $5M revenue in 2023, representing a 20% increase
over the previous year...

[Source: report.pdf, Page 7] (Score: 0.85)
EBITDA margin improved to 15%, driven by operational efficiency
and cost reduction measures...
```

**3. User message includes explicit instructions:**

```python
f"Context:\n{context}\n\n"
f"Question: {query}\n\n"
"Answer based only on the context above, with inline citations:"
```

The trailing instruction "Answer based only on the context above, with inline citations:" reinforces the system prompt's rules at the point of response generation. This dual instruction (system + user) significantly improves citation compliance.

**4. Same retry pattern as `embeddings.py`:**

| Scenario | Behavior |
|---|---|
| HTTP 401 (auth failure) | Fail immediately — retrying won't help |
| HTTP 429 (rate limit) | Retry up to 3 times with exponential backoff (1s, 2s, 4s) |
| HTTP 4xx/5xx (other) | Fail immediately — server error |
| Network error (timeout, DNS) | Retry up to 3 times with exponential backoff |
| `LLMError` (our own) | Fail immediately — parse/validation error |

**Why 120 seconds timeout (vs. 60 for embeddings):**
```python
with httpx.Client(timeout=120.0) as client:
```

Chat completions can take significantly longer than embeddings. The LLM needs to generate a multi-sentence answer with citations, which may require 5-15 seconds for complex queries. 120 seconds provides headroom for slow responses without timing out prematurely.

**5. Custom exception class `LLMError`:**

Follows the same pattern as `EmbeddingError` and `PDFProcessingError`. Each layer has its own domain-specific exception:
```
LLMError              ← raised by llm_client.py
EmbeddingError        ← raised by embeddings.py
PDFProcessingError    ← raised by pdf_processor.py
IngestError           ← raised by ingest_service.py
QueryRefusalError     ← raised by query_processor.py
```

**6. `generate_chitchat_response()` is a thin wrapper:**

```python
def generate_chitchat_response(query: str) -> str:
    messages = [
        {"role": "system", "content": _CHITCHAT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return generate(messages)
```

It reuses the `generate()` function with a different system prompt. No code duplication — all HTTP/retry logic lives in `generate()`.

**7. Response parsing is defensive:**

```python
def _parse_chat_response(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Mistral chat API response format: {exc}") from exc
```

Catches all three possible failure modes:
- `KeyError` — missing `"choices"`, `"message"`, or `"content"` key
- `IndexError` — empty `"choices"` array
- `TypeError` — `None` value where a dict/list was expected

---

## 5. File 4: `app/core/hallucination_filter.py`

**Created from scratch. 89 lines.**

**Purpose:** Verify the LLM's answer against the source chunks. Compute a confidence score based on how much of the answer is grounded in the provided sources. No extra API call — uses keyword overlap.

### Full source code:

```python
"""Hallucination filter — verify LLM answers against source chunks.

Computes a confidence score based on keyword overlap between
answer sentences and source chunks.  No extra API call needed.
"""

import re

from app.models.schemas import Chunk
from app.utils.text_utils import tokenize

# Reuse the sentence-splitting regex from text_utils.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Sentences that are meta-statements or citations, not factual claims.
_SKIP_PREFIXES = (
    "i don't have",
    "i do not have",
    "based on the",
    "according to the",
    "[source:",
)

# Minimum keyword overlap ratio to consider a sentence "supported".
_SUPPORT_THRESHOLD = 0.5


def compute_confidence(answer: str, chunks: list[Chunk]) -> float:
    """Score how well *answer* is grounded in the source *chunks*.

    Algorithm:
    1. Split the answer into sentences.
    2. For each sentence, tokenize it and check keyword overlap against
       each source chunk's tokens.
    3. If the best overlap >= 0.5 for any chunk, the sentence is "supported".
    4. confidence = supported_count / total_sentences.

    Returns a float in [0.0, 1.0].
    """
    if not answer or not answer.strip():
        return 0.0

    if not chunks:
        return 0.0

    sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Pre-tokenize all chunks once.
    chunk_token_sets = [set(tokenize(c.text)) for c in chunks]

    scorable: list[str] = []
    supported = 0

    for sentence in sentences:
        # Skip citation-only or meta-statement sentences.
        lowered = sentence.lower().strip()
        if any(lowered.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        # Skip very short fragments (e.g., bare citations like "[Source: ...]").
        sentence_tokens = tokenize(sentence)
        if len(sentence_tokens) < 3:
            continue

        scorable.append(sentence)

        # Check overlap against each chunk.
        sentence_set = set(sentence_tokens)
        best_overlap = 0.0
        for chunk_tokens in chunk_token_sets:
            if not sentence_set:
                break
            overlap = len(sentence_set & chunk_tokens) / len(sentence_set)
            if overlap > best_overlap:
                best_overlap = overlap

        if best_overlap >= _SUPPORT_THRESHOLD:
            supported += 1

    if not scorable:
        # All sentences were skipped (meta-statements / citations only).
        return 1.0

    return supported / len(scorable)
```

### Design decisions:

**1. Keyword overlap, not embedding similarity:**

The hallucination filter uses `tokenize()` + set intersection, not an additional embedding API call.

**Why not embeddings:**
- **Speed:** Embedding the answer + comparing against chunks would require another Mistral API call (1-2 seconds latency). Keyword overlap runs in microseconds.
- **Cost:** No additional API calls means no additional cost per query.
- **Sufficient accuracy:** If at least half the non-stopword tokens in an answer sentence appear in a source chunk, that sentence is very likely grounded in the source. This is a strong signal.

**2. Sentence-level granularity:**

The answer is split into sentences, and each sentence is scored independently. This is more precise than scoring the entire answer as a whole:
- A 5-sentence answer where 4 sentences are grounded and 1 is hallucinated gets `confidence = 0.8` (4/5)
- Whole-answer scoring would give a misleading single score that doesn't distinguish between partial and complete grounding

**3. Skip prefixes — meta-statements are not claims:**

```python
_SKIP_PREFIXES = (
    "i don't have",
    "i do not have",
    "based on the",
    "according to the",
    "[source:",
)
```

Sentences like "Based on the provided documents..." or "According to the context..." are structural framing, not factual claims. Scoring them against source chunks would penalize the answer unfairly (the source chunks don't contain the words "based on the provided documents"). Similarly, citation-only fragments like "[Source: report.pdf, Page 3]" are references, not claims.

**4. Minimum sentence length of 3 tokens:**

```python
sentence_tokens = tokenize(sentence)
if len(sentence_tokens) < 3:
    continue
```

Very short fragments (1-2 non-stopword tokens) are too ambiguous to score meaningfully. A 2-word fragment could trivially match any chunk. By requiring at least 3 tokens, we ensure the overlap score is meaningful.

**5. Support threshold of 0.5:**

```python
_SUPPORT_THRESHOLD = 0.5
```

A sentence is "supported" if at least 50% of its non-stopword tokens appear in any source chunk. This is deliberately moderate:
- Too high (e.g., 0.9) — would penalize the LLM for paraphrasing source material
- Too low (e.g., 0.2) — would consider almost anything as "supported"
- 0.5 is the sweet spot — the LLM can rephrase the source, but the core terms must still be present

**6. All-skipped sentences → confidence 1.0:**

```python
if not scorable:
    return 1.0
```

If the answer consists entirely of meta-statements and citations (e.g., "Based on the documents, I don't have enough information"), there are no factual claims to verify. This is treated as fully grounded (confidence 1.0) because the LLM isn't making any unsupported claims.

**7. Pre-tokenizes chunks once:**

```python
chunk_token_sets = [set(tokenize(c.text)) for c in chunks]
```

Chunk tokenization is done once and cached as sets. Each sentence is then compared against these pre-computed sets. Without this optimization, `tokenize()` would be called `n_sentences × n_chunks` times instead of `n_chunks + n_sentences` times.

---

## 6. File 5: `app/core/query_processor.py`

**Created from scratch. 161 lines.**

**Purpose:** Intent detection and orchestration of the full query pipeline. This is the Phase 3 equivalent of `ingest_service.py` — it coordinates all the pieces. Pure logic — no FastAPI imports.

### Full source code:

```python
"""Query pipeline orchestrator.

Pure logic — no FastAPI imports.  Coordinates:
  pii_detector → intent → search → rerank → llm → hallucination filter

Fail fast, fail loud on every step.
"""

import logging
import re
import time

from app.config import settings
from app.core.hallucination_filter import compute_confidence
from app.core.llm_client import format_context, build_qa_prompt, generate, generate_chitchat_response
from app.core.reranker import rerank
from app.core.search import hybrid_search
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.utils.pii_detector import contains_pii, get_pii_types

logger = logging.getLogger(__name__)


class QueryRefusalError(Exception):
    """Raised when a query is refused (e.g., contains PII)."""


# ------------------------------------------------------------------
# Intent detection
# ------------------------------------------------------------------

_CHITCHAT_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|greetings)\b", re.IGNORECASE),
    re.compile(r"^how are you", re.IGNORECASE),
    re.compile(r"^good (morning|afternoon|evening)", re.IGNORECASE),
    re.compile(r"^(thanks|thank you|thx)\b", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see you)\b", re.IGNORECASE),
    re.compile(r"^what'?s up", re.IGNORECASE),
]


def detect_intent(query: str) -> str:
    """Classify the query into an intent category.

    Returns one of: ``"CHITCHAT"``, ``"KNOWLEDGE_SEARCH"``.
    """
    stripped = query.strip()
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(stripped):
            return "CHITCHAT"
    return "KNOWLEDGE_SEARCH"


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------


def process_query(request: QueryRequest) -> QueryResponse:
    """Execute the full query pipeline.

    Steps:
    1. PII check — refuse if detected.
    2. Intent detection.
    3. Chitchat handling (no retrieval).
    4. Hybrid search.
    5. Rerank.
    6. Handle no results.
    7. Generate answer via LLM.
    8. Hallucination check.
    9. Build response.

    Raises ``QueryRefusalError`` on PII detection.
    """
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

    # Step 3 — Chitchat (no retrieval needed).
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
    ranked = rerank(
        candidates,
        top_k=top_k,
        threshold=settings.similarity_threshold,
    )

    # Step 6 — No results.
    if not ranked:
        elapsed = int((time.time() - start_time) * 1000)
        return QueryResponse(
            answer=(
                "I don't have enough information in the provided "
                "documents to answer this question."
            ),
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
    sources: list[Source] = []
    if request.include_sources:
        for chunk, score in ranked:
            sources.append(
                Source(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text[:500],
                    score=round(score, 4),
                )
            )

    elapsed = int((time.time() - start_time) * 1000)

    logger.info(
        "Query processed: intent=%s, sources=%d, confidence=%.2f, time=%dms",
        intent, len(sources), confidence, elapsed,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 4),
        intent=intent,
        processing_time_ms=elapsed,
    )
```

### Design decisions:

**1. Nine-step pipeline with clear responsibilities:**

```
Step 1: PII check          → refuse or continue
Step 2: Intent detection   → classify query type
Step 3: Chitchat handling  → short-circuit if greeting
Step 4: Hybrid search      → find candidate chunks
Step 5: Rerank             → filter, sort, dedup
Step 6: No results         → graceful "I don't know" response
Step 7: Generate answer    → LLM call with context
Step 8: Hallucination check → verify answer against sources
Step 9: Build response     → assemble QueryResponse
```

Each step is a separate operation. Early returns at steps 1 (PII), 3 (chitchat), and 6 (no results) prevent unnecessary downstream processing.

**2. Intent detection is regex-based, not LLM-based:**

```python
_CHITCHAT_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|greetings)\b", re.IGNORECASE),
    re.compile(r"^how are you", re.IGNORECASE),
    re.compile(r"^good (morning|afternoon|evening)", re.IGNORECASE),
    re.compile(r"^(thanks|thank you|thx)\b", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see you)\b", re.IGNORECASE),
    re.compile(r"^what'?s up", re.IGNORECASE),
]
```

**Why regex, not LLM:**
- An LLM call for intent classification would add 1-2 seconds of latency on every query
- Chitchat patterns are simple and deterministic — "hello", "how are you", "thanks"
- Regex runs in microseconds vs. seconds for an API call
- False negative is safe — if a chitchat query isn't caught, it goes through RAG and gets a "I don't have enough information" response, which is acceptable
- All patterns are anchored to the start of the string (`^`) to avoid false positives (e.g., "How are you going to fix this bug?" would match `how are you` without the `^` anchor, but with it, only queries that *start* with a greeting are caught)

**3. All patterns use `re.IGNORECASE` flag:**

```python
re.compile(r"^(hi|hello|hey|howdy|greetings)\b", re.IGNORECASE)
```

This ensures "Hi", "HI", "hi", "Hello", "HELLO" all match. The `\b` word boundary prevents partial matches (e.g., "history" matching `^hi`).

**4. PII check happens first (before everything else):**

```python
if contains_pii(request.query):
    pii_types = get_pii_types(request.query)
    raise QueryRefusalError(...)
```

PII detection runs before intent detection, before search, before any API calls. This ensures:
- No PII is sent to the Mistral API (even for embedding)
- No PII is logged or stored
- The refusal is immediate — the user sees the error in milliseconds

**5. Source text is truncated to 500 characters:**

```python
text=chunk.text[:500],
```

The full chunk text can be up to 800 tokens (~3200 characters). Including the full text for 5 sources would make the API response very large. Truncating to 500 characters provides enough context for the user to identify the source without bloating the response.

**6. Scores are rounded to 4 decimal places:**

```python
score=round(score, 4),
confidence=round(confidence, 4),
```

Floating-point scores like `0.8723456789123456` are noisy and hard to read. Rounding to 4 places (`0.8723`) is precise enough for comparison while keeping the JSON response clean.

**7. Processing time is measured end-to-end:**

```python
start_time = time.time()
# ... entire pipeline ...
elapsed = int((time.time() - start_time) * 1000)
```

`processing_time_ms` includes everything — PII check, intent detection, embedding API call, search, reranking, LLM generation, and hallucination check. This gives the user a true picture of how long the query took. The value is in milliseconds as an integer (e.g., `2340`), not a float.

**8. `QueryRefusalError` is separate from other errors:**

The API layer catches `QueryRefusalError` specifically and returns HTTP 400. All other exceptions bubble up as HTTP 500. This distinction is important:
- 400 = the *user* did something wrong (included PII) — fixable by the user
- 500 = the *system* failed (API down, parsing error) — not the user's fault

**9. Chitchat gets `confidence: 1.0` and empty sources:**

```python
if intent == "CHITCHAT":
    return QueryResponse(
        answer=answer,
        sources=[],
        confidence=1.0,
        intent=intent,
        ...
    )
```

Chitchat responses are not factual claims that need verification. A greeting response like "Hello! I can help you with questions about your documents." is inherently "confident" — there's nothing to hallucinate.

**10. No-results response is not an error:**

```python
if not ranked:
    return QueryResponse(
        answer="I don't have enough information...",
        sources=[],
        confidence=0.0,
        ...
    )
```

When no chunks pass the reranking threshold, the response is HTTP 200 (not 404 or 500). It's a valid answer — "I don't know" — with `confidence: 0.0` to indicate no supporting evidence was found. This lets the client handle it gracefully (e.g., suggest uploading relevant documents).

---

## 7. File 6: `app/api/query.py`

**Created from scratch. 32 lines.**

**Purpose:** Thin HTTP layer for the query endpoint. Same design pattern as `api/ingestion.py` — no business logic, just receive request, call core, return response.

### Full source code:

```python
"""Query API endpoint.

Thin HTTP layer — no business logic, no search, no LLM calls.
Just: receive request → call core → return response.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.query_processor import QueryRefusalError, process_query
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Full RAG pipeline: intent → search → generate → verify."""
    try:
        return process_query(request)
    except QueryRefusalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )
```

### Design decisions:

**1. The thinnest possible route handler — 10 lines of logic:**

The handler does exactly three things:
1. Call `process_query(request)` — the core orchestrator does all the work
2. Catch `QueryRefusalError` → return HTTP 400 with the error message
3. Catch any other exception → return HTTP 500 with the error message

No search logic, no LLM calls, no scoring, no formatting.

**2. Two exception paths:**

| Exception | HTTP Status | Meaning |
|---|---|---|
| `QueryRefusalError` | 400 | User's fault (PII in query). Fixable by the user. |
| Any other `Exception` | 500 | System failure (API down, parsing error). Not the user's fault. |

**3. `logger.exception()` on 500 errors:**

```python
except Exception as exc:
    logger.exception("Query failed")
```

`logger.exception()` (not `logger.error()`) automatically includes the full stack trace in the log. This is critical for debugging production issues — the HTTP response only shows a summary, but the server log has the complete traceback.

**4. `response_model=QueryResponse` for OpenAPI documentation:**

```python
@router.post("/api/query", response_model=QueryResponse)
```

FastAPI uses this to:
- Auto-generate the response schema in Swagger/OpenAPI docs
- Validate the response before sending it
- Document the expected response format at `/docs`

**5. Request body is auto-validated by FastAPI:**

The `request: QueryRequest` parameter is a Pydantic model. FastAPI automatically:
- Parses the JSON request body
- Validates all fields against the schema (e.g., `query` must be a string, `top_k` must be an int)
- Returns HTTP 422 with detailed validation errors if the body is malformed

No manual validation code needed in the handler.

---

## 8. File 7: `app/main.py` (modified)

**Modified from Phase 2. Now 44 lines (was 43).**

**Changes made:**

| What | Before (Phase 2) | After (Phase 3) |
|---|---|---|
| Imports | `health`, `ingestion` | `health`, `ingestion`, `query` |
| Routers | 2 routers | 3 routers (added `query.router`) |

### Diff:

```diff
-from app.api import health, ingestion
+from app.api import health, ingestion, query

 app.include_router(health.router)
 app.include_router(ingestion.router)
+app.include_router(query.router)
```

Two lines changed. Everything else is identical to Phase 2.

### Full source code:

```python
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, ingestion, query
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
app.include_router(query.router)


@app.on_event("startup")
def startup() -> None:
    """Create data directories and restore vector store from disk."""
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)

    index_path = os.path.join(settings.index_dir, "chunks.json")
    if os.path.exists(index_path):
        vector_store.load(settings.index_dir)
```

### All registered routes after Phase 3:

```
GET    /api/health                    ← Phase 1
POST   /api/ingest                   ← Phase 2
GET    /api/documents                ← Phase 2
DELETE /api/documents/{document_id}  ← Phase 2
POST   /api/query                    ← Phase 3 (NEW)
```

---

## 9. Architectural Patterns Used

### Pattern 1: Pipeline-style query processing

Each step is a separate function. The orchestrator (`process_query`) calls them in sequence with early returns:

```
PII check      → refuse or continue
Intent detect  → CHITCHAT or KNOWLEDGE_SEARCH
  └── CHITCHAT   → generate_chitchat_response() → return (early exit)
Hybrid search  → list[(Chunk, float)]
Rerank         → filtered, sorted, deduped list
  └── empty      → "I don't know" → return (early exit)
Generate       → LLM answer string
Verify         → confidence score
Build response → QueryResponse
```

**Three early-exit points** prevent unnecessary computation:
1. PII → exception (no search, no LLM call)
2. Chitchat → response (no search needed)
3. No results → response (no LLM call needed)

### Pattern 2: Separation of search and ranking

**Search** (`search.py`) produces raw candidates — as many as `top_k_retrieval` (20).
**Reranking** (`reranker.py`) applies business rules — threshold, dedup, truncation to `top_k_final` (5).

This separation means:
- Search parameters can be tuned independently of ranking parameters
- The reranker is testable in isolation (pass in fake scored chunks)
- The search module doesn't need to know about deduplication or thresholds

### Pattern 3: Composition over duplication

Phase 3 reuses 4 modules from Phase 2 without modification:

| Reused module | Function | Used by |
|---|---|---|
| `core/embeddings.py` | `get_embedding()` | `search.py` — embeds the query |
| `storage/vector_store.py` | `get_all()` | `search.py` — retrieves chunks + embeddings |
| `utils/text_utils.py` | `tokenize()` | `search.py` — keyword scoring; `hallucination_filter.py` — answer verification |
| `utils/pii_detector.py` | `contains_pii()`, `get_pii_types()` | `query_processor.py` — query refusal |

Zero code was duplicated. Zero Phase 2 files were modified (except `main.py` for router registration).

### Pattern 4: Custom exception hierarchy (complete)

```
QueryRefusalError          ← query_processor.py (PII detected)
LLMError                   ← llm_client.py (chat API failure)
EmbeddingError             ← embeddings.py (embedding API failure)
IngestError                ← ingest_service.py (ingestion failure)
  ├── PDFProcessingError   ← pdf_processor.py (PDF extraction failure)
  └── EmbeddingError       ← embeddings.py (wrapped during ingestion)
```

Each layer catches exceptions from the layer below:
- `query_processor.py` lets `EmbeddingError` and `LLMError` propagate up
- `api/query.py` catches `QueryRefusalError` → 400, everything else → 500

### Pattern 5: No new configuration values

All Phase 3 settings were already defined in `app/config.py` during Phase 1:

```python
top_k_retrieval: int = 20         # used by hybrid_search()
top_k_final: int = 5              # used by rerank()
similarity_threshold: float = 0.7 # used by rerank()
semantic_weight: float = 0.7      # used by hybrid_search()
mistral_chat_model: str = "..."   # used by generate()
```

This was intentional — all configurable values were planned upfront. Phase 3 simply reads what Phase 1 defined.

---

## 10. Edge Cases Covered

### search.py edge cases:
- Empty vector store (no documents ingested) → returns `[]` (no error)
- Query embedding fails (API error) → `EmbeddingError` propagates to caller
- All keyword scores are 0.0 (no term overlap) → semantic scores still produce results
- All semantic scores are 0.0 (unrelated query) → keyword scores can still rank results
- `cosine_similarity` with zero-norm vector → `+ 1e-10` prevents division by zero
- `keyword_score` with empty query tokens → returns `0.0`
- Single chunk in store → returns that chunk with its score

### reranker.py edge cases:
- Empty input → returns `[]`
- All chunks below threshold → returns `[]`
- Fewer chunks than `top_k` after dedup → returns all that remain
- Multiple chunks from same page → only highest-scoring is kept
- Multiple chunks from same file, different pages → all kept (different keys)
- All chunks have identical scores → dedup still works (first occurrence kept)

### llm_client.py edge cases:
- Missing API key → `LLMError` immediately (no HTTP request made)
- Invalid API key (401) → `LLMError` immediately (no retry)
- Rate limited (429) → retry up to 3 times with 1s, 2s, 4s backoff
- Network timeout → retry up to 3 times with backoff
- Empty `choices` array in response → `IndexError` caught → `LLMError`
- Missing `content` key in response → `KeyError` caught → `LLMError`
- Response body is `None` → `TypeError` caught → `LLMError`
- `format_context` with empty list → returns `""` (empty string)

### hallucination_filter.py edge cases:
- Empty answer → returns `0.0`
- Empty chunks list → returns `0.0`
- Answer with no sentences (single word) → if < 3 tokens, skipped → returns `1.0`
- Answer is entirely meta-statements ("Based on the documents...") → all skipped → returns `1.0`
- Answer is entirely citations ("[Source: ...]") → skipped (starts with `[source:`) → returns `1.0`
- Single-sentence answer with perfect overlap → returns `1.0`
- No overlap between answer and any chunk → returns `0.0`
- Mixed grounded and ungrounded sentences → proportional score (e.g., 3/5 = 0.6)

### query_processor.py edge cases:
- PII in query (SSN, email, etc.) → `QueryRefusalError` before any processing
- Chitchat query ("Hello!") → LLM chitchat response, no search, `confidence: 1.0`
- No documents ingested + knowledge query → empty search results → "I don't know" response
- All search results below threshold → empty after reranking → "I don't know" response
- `top_k=0` in request → `request.top_k or settings.top_k_final` → falls back to 5
- `include_sources=False` → sources list is empty in response
- LLM generation fails → `LLMError` propagates → HTTP 500

### api/query.py edge cases:
- Missing `query` field in request body → FastAPI 422 validation error
- `query` is empty string → processed normally (Pydantic allows it; will likely find no results)
- Invalid JSON body → FastAPI 422 error
- Wrong Content-Type → FastAPI 422 error
- `QueryRefusalError` → HTTP 400 with PII details
- Any other exception → HTTP 500 with logged stack trace

---

## 11. Constraint Compliance

| Constraint | How we comply in Phase 3 |
|---|---|
| **No FAISS** | `search.py` uses `np.dot` + `np.linalg.norm` for cosine similarity. No FAISS import anywhere. |
| **No rank-bm25** | `search.py` uses custom `keyword_score()` with Python `set` intersection. No BM25, no TF-IDF. |
| **No sentence-transformers** | `reranker.py` sorts by numeric scores. No cross-encoder model, no learned re-ranking. |
| **No scikit-learn** | No `sklearn.metrics.pairwise.cosine_similarity`. No `TfidfVectorizer`. Pure NumPy. |
| **Embeddings via Mistral API** | Reuses `get_embedding()` from Phase 2's `embeddings.py`. Allowed. |
| **LLM via Mistral API** | `llm_client.py` calls `POST /v1/chat/completions` via httpx. Allowed. |
| **All retrieval from scratch** | Cosine similarity, keyword scoring, hybrid combination, reranking — all custom Python + NumPy. |
| **No external search libraries** | No Elasticsearch, no Whoosh, no Typesense. In-memory search only. |

---

## 12. What Changed From Phase 2

### Files unchanged:
- `app/__init__.py` — empty, unchanged
- `app/config.py` — unchanged (already had all needed settings)
- `app/api/__init__.py` — empty, unchanged
- `app/api/health.py` — unchanged
- `app/api/ingestion.py` — unchanged
- `app/core/__init__.py` — empty, unchanged
- `app/core/embeddings.py` — unchanged (reused by search.py)
- `app/core/ingest_service.py` — unchanged
- `app/core/pdf_processor.py` — unchanged
- `app/models/__init__.py` — empty, unchanged
- `app/models/schemas.py` — unchanged (QueryRequest/QueryResponse already defined in Phase 2)
- `app/services/__init__.py` — empty, unchanged
- `app/storage/__init__.py` — empty, unchanged
- `app/storage/vector_store.py` — unchanged (get_all() reused by search.py)
- `app/utils/__init__.py` — empty, unchanged
- `app/utils/pii_detector.py` — unchanged (reused by query_processor.py)
- `app/utils/text_utils.py` — unchanged (tokenize() reused by search.py)
- `requirements.txt` — unchanged (no new dependencies)

### Files modified:
- `app/main.py` — added 2 lines (query import + router registration)

### Files created:
- `app/core/search.py` — 93 lines
- `app/core/reranker.py` — 48 lines
- `app/core/llm_client.py` — 176 lines
- `app/core/hallucination_filter.py` — 89 lines
- `app/core/query_processor.py` — 161 lines
- `app/api/query.py` — 32 lines

### Total new code: 599 lines across 6 files.
### Total modified code: 2 lines in 1 file.
### Total unchanged: 17 files.

---

## Complete Query Flow — End to End

```
User sends:  POST /api/query  {"query": "What is the revenue?"}
                │
                ▼
        api/query.py                         ← receives HTTP request
                │
                ▼
        core/query_processor.py              ← orchestrates the pipeline
                │
        ┌───────┼───────────────────────────────────────────┐
        │       │                                           │
        │  Step 1: PII check                               │
        │       │  utils/pii_detector.py                   │
        │       │  contains_pii("What is the revenue?")    │
        │       │  → False (no PII)                        │
        │       │                                           │
        │  Step 2: Intent detection                        │
        │       │  detect_intent("What is the revenue?")   │
        │       │  → "KNOWLEDGE_SEARCH"                    │
        │       │                                           │
        │  Step 4: Hybrid search                           │
        │       │  core/search.py                          │
        │       │  ├── get_embedding("What is the revenue?")
        │       │  │   └── core/embeddings.py → Mistral API │
        │       │  ├── cosine_similarity(query_emb, all_embs)
        │       │  │   └── NumPy dot product + norm         │
        │       │  ├── keyword_score for each chunk         │
        │       │  │   └── tokenize + set intersection      │
        │       │  └── combine: 0.7*semantic + 0.3*keyword  │
        │       │  → 20 candidates [(Chunk, score), ...]   │
        │       │                                           │
        │  Step 5: Rerank                                  │
        │       │  core/reranker.py                        │
        │       │  ├── filter: score >= 0.7                │
        │       │  ├── sort descending                     │
        │       │  ├── dedup on (source, page)             │
        │       │  └── truncate to 5                       │
        │       │  → 5 ranked [(Chunk, score), ...]        │
        │       │                                           │
        │  Step 7: Generate answer                         │
        │       │  core/llm_client.py                      │
        │       │  ├── format_context(ranked_chunks)       │
        │       │  ├── build_qa_prompt(query, context)     │
        │       │  └── generate(messages)                  │
        │       │      └── Mistral chat API → answer text  │
        │       │                                           │
        │  Step 8: Hallucination check                     │
        │       │  core/hallucination_filter.py             │
        │       │  └── compute_confidence(answer, chunks)  │
        │       │      → 0.85                              │
        │       │                                           │
        │  Step 9: Build response                          │
        │       │  → QueryResponse(answer, sources,        │
        │       │     confidence, intent, time_ms)         │
        │       │                                           │
        └───────┼───────────────────────────────────────────┘
                │
                ▼
        api/query.py                         ← returns HTTP response
                │
                ▼
        User receives:
        {
            "answer": "The company reported $5M revenue... [Source: report.pdf, Page 3]",
            "sources": [...],
            "confidence": 0.85,
            "intent": "KNOWLEDGE_SEARCH",
            "processing_time_ms": 2340
        }
```
