# Phase 4: Testing

**Goal:** Prove that every module works correctly — in isolation and end-to-end. After this phase, you can run `pytest` and see green across the board, giving confidence that the ingestion pipeline, search, reranking, LLM generation, and hallucination filter all behave as designed.

**This phase writes no production code.** It only creates test files. If tests reveal bugs, we fix them, but the goal is verification, not new features.

**Testing model:** Unit tests for pure logic modules (no API calls), integration tests for the full pipeline (requires Mistral API key).

---

## What Phase 4 Covers

```
tests/
├── __init__.py
├── conftest.py                 ← Shared fixtures (sample chunks, fake embeddings)
├── test_text_utils.py          ← Unit tests for clean_text, split_into_chunks, tokenize
├── test_pii_detector.py        ← Unit tests for PII regex patterns
├── test_vector_store.py        ← Unit tests for add, delete, save, load, get_all
├── test_search.py              ← Unit tests for cosine_similarity, keyword_score
├── test_reranker.py            ← Unit tests for filter, sort, dedup, truncate
├── test_hallucination_filter.py← Unit tests for confidence scoring
├── test_query_processor.py     ← Unit tests for intent detection
├── test_pdf_processor.py       ← Unit tests for PDF extraction (uses a real small PDF)
├── test_integration.py         ← End-to-end: ingest a PDF → query → get answer
│                                  (requires MISTRAL_API_KEY, skipped if missing)
└── test_api.py                 ← FastAPI TestClient: all endpoints
```

---

## Precise Requirements

### What We Test

| Module | Test type | Needs API key? | What we verify |
|---|---|---|---|
| `utils/text_utils.py` | Unit | No | Cleaning, chunking, tokenization edge cases |
| `utils/pii_detector.py` | Unit | No | All 4 PII patterns + negatives |
| `storage/vector_store.py` | Unit | No | Add, delete, save/load, consistency check |
| `core/search.py` | Unit | No | Cosine similarity math, keyword scoring, hybrid combination |
| `core/reranker.py` | Unit | No | Threshold filter, sort order, dedup logic, truncation |
| `core/hallucination_filter.py` | Unit | No | Confidence scoring, skip logic, edge cases |
| `core/query_processor.py` | Unit | No | Intent detection patterns |
| `core/pdf_processor.py` | Unit | No | Text extraction from a small test PDF |
| `api/*.py` | Integration | Partial | FastAPI TestClient, endpoint validation |
| Full pipeline | Integration | Yes | Ingest PDF → query → verify answer has citations |

### Testing Principles

1. **No mocking the system under test.** Mock external dependencies (Mistral API), not internal logic.
2. **Each test file mirrors one source file.** `test_search.py` tests `core/search.py`. No cross-contamination.
3. **Tests that need the Mistral API are marked and skippable.** Run offline tests with `pytest -m "not integration"`. Run all tests with `pytest`.
4. **Fixtures provide reusable test data.** Sample chunks, fake embeddings, and a tiny test PDF are shared via `conftest.py`.
5. **Tests are fast.** Unit tests complete in < 1 second total. Only integration tests hit the network.

---

## Architecture

### Test Fixtures (`conftest.py`)

Shared test data used across multiple test files:

```python
# Sample chunks — reusable across search, reranker, hallucination tests
@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="doc.pdf_p1_c0", text="Machine learning is a subset of AI...", source="doc.pdf", page=1),
        Chunk(chunk_id="doc.pdf_p2_c1", text="Neural networks consist of layers...", source="doc.pdf", page=2),
        Chunk(chunk_id="other.pdf_p1_c0", text="Revenue increased by 20% in 2023...", source="other.pdf", page=1),
    ]

# Fake embeddings — deterministic, no API call needed
@pytest.fixture
def fake_embeddings() -> np.ndarray:
    np.random.seed(42)
    return np.random.randn(3, 8)  # 3 chunks, 8-dim embeddings (small for tests)

# Tiny test PDF — created programmatically, no external file needed
@pytest.fixture
def test_pdf_path(tmp_path) -> str:
    # Creates a minimal valid PDF using reportlab or raw bytes
```

### Skip Marker for API-Dependent Tests

```python
requires_api_key = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set — skipping integration test",
)
```

Tests marked `@requires_api_key` are skipped when running without the API key. This means:
- CI/CD can run unit tests without a key
- Developers can run the full suite locally with their key
- `pytest -m "not integration"` runs only offline tests

---

## Step 1 — Test Setup (`tests/__init__.py`, `tests/conftest.py`)

**What:** Create the test package and shared fixtures.

### `tests/__init__.py`
Empty file — makes `tests/` a Python package.

### `tests/conftest.py`

**Fixtures provided:**

| Fixture | Type | Purpose |
|---|---|---|
| `sample_chunks` | `list[Chunk]` | 3 chunks from 2 different PDFs |
| `fake_embeddings` | `np.ndarray` | Deterministic 3x8 matrix (seeded random) |
| `fresh_vector_store` | `VectorStore` | Empty store instance (not the singleton) |
| `loaded_vector_store` | `VectorStore` | Store pre-loaded with sample chunks + embeddings |
| `test_pdf_path` | `str` | Path to a tiny valid PDF in `tmp_path` |
| `requires_api_key` | marker | Skips test if `MISTRAL_API_KEY` is not set |

**Why a fresh store instead of the singleton:**
The module-level `vector_store` singleton is shared across the app. Tests should not mutate it. Each test gets its own `VectorStore()` instance to avoid test pollution.

---

## Step 2 — Text Utils Tests (`tests/test_text_utils.py`)

**Tests for:** `app/utils/text_utils.py`

| Test | Input | Expected |
|---|---|---|
| `test_clean_text_whitespace` | `"  hello   world  \n\t "` | `"hello world"` |
| `test_clean_text_ligatures` | `"The ﬁrst ﬂoor oﬃce"` | `"The first floor office"` |
| `test_clean_text_hyphenation` | `"compli-\ncated"` | `"complicated"` |
| `test_clean_text_control_chars` | `"a\x00b\x01c"` | `"a b c"` |
| `test_clean_text_empty` | `""` | `""` |
| `test_clean_text_none_like` | `""` (falsy) | `""` |
| `test_count_tokens_normal` | `"hello world foo bar"` | `4` |
| `test_count_tokens_empty` | `""` | `0` |
| `test_count_tokens_whitespace_only` | `"   "` | `0` |
| `test_split_into_chunks_basic` | 50 sentences, chunk_size=20 | Multiple chunks, all ≤ 20 tokens |
| `test_split_into_chunks_overlap` | Long text, overlap=5 | Last sentences of chunk N appear at start of chunk N+1 |
| `test_split_into_chunks_empty` | `""` | `[]` |
| `test_split_into_chunks_short` | `"short text"` | `["short text"]` |
| `test_split_into_chunks_long_sentence` | One 900-word sentence | Force-split into pieces ≤ chunk_size |
| `test_tokenize_removes_stopwords` | `"The quick brown fox"` | `["quick", "brown", "fox"]` |
| `test_tokenize_lowercases` | `"Hello WORLD"` | `["hello", "world"]` |
| `test_tokenize_empty` | `""` | `[]` |
| `test_tokenize_punctuation` | `"hello, world! foo."` | `["hello", "world", "foo"]` |

---

## Step 3 — PII Detector Tests (`tests/test_pii_detector.py`)

**Tests for:** `app/utils/pii_detector.py`

| Test | Input | Expected |
|---|---|---|
| `test_detects_ssn` | `"My SSN is 123-45-6789"` | `contains_pii → True`, types includes `"ssn"` |
| `test_detects_credit_card_dashes` | `"Card: 4111-1111-1111-1111"` | `True`, types includes `"credit_card"` |
| `test_detects_credit_card_spaces` | `"Card: 4111 1111 1111 1111"` | `True` |
| `test_detects_credit_card_no_sep` | `"Card: 4111111111111111"` | `True` |
| `test_detects_email` | `"Contact user@example.com"` | `True`, types includes `"email"` |
| `test_detects_phone` | `"Call +1 (555) 123-4567"` | `True`, types includes `"phone"` |
| `test_detects_phone_no_country` | `"Call 555-123-4567"` | `True` |
| `test_multiple_pii` | `"SSN 123-45-6789 email a@b.com"` | types = `["ssn", "email"]` |
| `test_no_pii_clean_query` | `"What is machine learning?"` | `False`, types = `[]` |
| `test_no_pii_empty` | `""` | `False` |
| `test_no_pii_numbers` | `"The year 2023 saw growth"` | `False` (not enough digits for SSN/CC) |

---

## Step 4 — Vector Store Tests (`tests/test_vector_store.py`)

**Tests for:** `app/storage/vector_store.py`

| Test | What it verifies |
|---|---|
| `test_empty_store` | `len(store) == 0`, `get_all()` returns empty |
| `test_add_chunks` | After `add()`, length matches, chunks accessible |
| `test_add_validates_shape` | Mismatched chunks/embeddings count → `ValueError` |
| `test_get_all_returns_parallel_arrays` | `chunks[i]` corresponds to `embeddings[i]` |
| `test_delete_by_source` | Removes correct chunks, keeps others, returns count |
| `test_delete_nonexistent_source` | Returns 0, store unchanged |
| `test_delete_all_resets_embeddings` | After deleting everything, `embeddings` is `None` |
| `test_save_and_load` | Save → create new store → load → data matches |
| `test_save_empty_store` | No crash, removes stale `embeddings.npy` |
| `test_load_nonexistent_path` | Store remains empty, no error |
| `test_load_consistency_check` | Corrupt data (mismatched lengths) → store resets to empty |
| `test_clear` | After `clear()`, everything is empty |
| `test_multiple_adds` | Two `add()` calls → embeddings are vstacked, chunks extended |
| `test_documents_dict` | Documents can be added/retrieved by ID |

---

## Step 5 — Search Tests (`tests/test_search.py`)

**Tests for:** `app/core/search.py`

| Test | What it verifies |
|---|---|
| `test_cosine_similarity_identical` | Same vector → score ≈ 1.0 |
| `test_cosine_similarity_orthogonal` | Orthogonal vectors → score ≈ 0.0 |
| `test_cosine_similarity_opposite` | Negated vector → score ≈ -1.0 |
| `test_cosine_similarity_shape` | Returns 1-D array of correct length |
| `test_cosine_similarity_zero_vector` | Zero vector → no crash (1e-10 guard) |
| `test_keyword_score_full_overlap` | All query tokens in chunk → 1.0 |
| `test_keyword_score_no_overlap` | No query tokens in chunk → 0.0 |
| `test_keyword_score_partial` | 2 of 4 query tokens found → 0.5 |
| `test_keyword_score_empty_query` | Empty query tokens → 0.0 |
| `test_keyword_score_stopwords_removed` | Stopwords not counted (tokenize handles this) |

**Note:** `hybrid_search()` itself calls the Mistral API (to embed the query), so it's tested in integration tests, not here. The unit tests verify the *math* functions (`cosine_similarity`, `keyword_score`) in isolation.

---

## Step 6 — Reranker Tests (`tests/test_reranker.py`)

**Tests for:** `app/core/reranker.py`

| Test | What it verifies |
|---|---|
| `test_empty_input` | `rerank([]) → []` |
| `test_threshold_filters` | Chunks below threshold removed |
| `test_all_below_threshold` | All below → `[]` |
| `test_sort_order` | Results sorted descending by score |
| `test_dedup_same_page` | Two chunks from (doc.pdf, page 1) → only highest kept |
| `test_dedup_different_pages` | Chunks from (doc.pdf, page 1) and (doc.pdf, page 2) → both kept |
| `test_dedup_different_sources` | Chunks from (a.pdf, page 1) and (b.pdf, page 1) → both kept |
| `test_truncate_to_top_k` | 10 chunks, top_k=3 → exactly 3 returned |
| `test_fewer_than_top_k` | 2 chunks, top_k=5 → returns 2 |
| `test_combined_filter_sort_dedup_truncate` | Full pipeline with mixed data |

---

## Step 7 — Hallucination Filter Tests (`tests/test_hallucination_filter.py`)

**Tests for:** `app/core/hallucination_filter.py`

| Test | What it verifies |
|---|---|
| `test_empty_answer` | `""` → confidence 0.0 |
| `test_empty_chunks` | No chunks → confidence 0.0 |
| `test_fully_grounded` | All answer tokens appear in chunks → confidence ≈ 1.0 |
| `test_fully_ungrounded` | No overlap between answer and chunks → confidence 0.0 |
| `test_partial_grounding` | 2 of 4 sentences grounded → confidence 0.5 |
| `test_skips_meta_statements` | "Based on the documents..." → skipped, not penalized |
| `test_skips_citations` | "[Source: file.pdf, Page 3]" → skipped |
| `test_skips_i_dont_have` | "I don't have enough information" → skipped → confidence 1.0 |
| `test_skips_short_fragments` | Fragments < 3 tokens → skipped |
| `test_all_skipped_returns_one` | Only meta-statements → confidence 1.0 |

---

## Step 8 — Query Processor Tests (`tests/test_query_processor.py`)

**Tests for:** `app/core/query_processor.py` (intent detection only — `process_query` needs API)

| Test | What it verifies |
|---|---|
| `test_intent_hello` | `"Hello!"` → `"CHITCHAT"` |
| `test_intent_hi` | `"hi"` → `"CHITCHAT"` |
| `test_intent_hey` | `"hey there"` → `"CHITCHAT"` |
| `test_intent_good_morning` | `"Good morning"` → `"CHITCHAT"` |
| `test_intent_thanks` | `"Thank you!"` → `"CHITCHAT"` |
| `test_intent_bye` | `"Goodbye"` → `"CHITCHAT"` |
| `test_intent_whats_up` | `"What's up"` → `"CHITCHAT"` |
| `test_intent_case_insensitive` | `"HELLO"` → `"CHITCHAT"` |
| `test_intent_knowledge_query` | `"What is machine learning?"` → `"KNOWLEDGE_SEARCH"` |
| `test_intent_default` | `"Explain revenue trends"` → `"KNOWLEDGE_SEARCH"` |
| `test_intent_hello_in_middle` | `"Can you say hello?"` → `"KNOWLEDGE_SEARCH"` (not anchored at start) |

---

## Step 9 — PDF Processor Tests (`tests/test_pdf_processor.py`)

**Tests for:** `app/core/pdf_processor.py`

| Test | What it verifies |
|---|---|
| `test_process_valid_pdf` | Tiny PDF → returns list of Chunks with correct IDs |
| `test_chunk_ids_deterministic` | Same PDF → same chunk_id pattern (`filename_p{page}_c{index}`) |
| `test_page_numbers_one_indexed` | First page → `page=1`, not `page=0` |
| `test_chunks_have_text` | Every chunk has non-empty `text` |
| `test_chunks_have_source` | Every chunk's `source` matches the filename |
| `test_corrupted_pdf_raises` | Random bytes as PDF → `PDFProcessingError` |
| `test_empty_file_raises` | 0-byte file → error |
| `test_no_text_pdf_raises` | PDF with only images → `PDFProcessingError` |

**Test PDF creation:** The `test_pdf_path` fixture creates a minimal PDF programmatically using `reportlab` (if available) or raw PDF bytes. No external test files needed.

---

## Step 10 — API Tests (`tests/test_api.py`)

**Tests for:** All API endpoints using FastAPI's `TestClient`

| Test | Endpoint | What it verifies |
|---|---|---|
| `test_health_check` | `GET /api/health` | Returns 200, has all fields |
| `test_health_shows_counts` | `GET /api/health` | `documents_count` and `chunks_count` present |
| `test_documents_empty` | `GET /api/documents` | Returns 200, empty list |
| `test_ingest_rejects_non_pdf` | `POST /api/ingest` | `.txt` file → 400 |
| `test_ingest_rejects_empty` | `POST /api/ingest` | Empty file → 400 |
| `test_query_rejects_pii` | `POST /api/query` | SSN in query → 400 |
| `test_query_missing_body` | `POST /api/query` | No body → 422 |
| `test_query_empty_store` | `POST /api/query` | No docs ingested → 200, "I don't have enough information" |
| `test_openapi_docs` | `GET /docs` | Returns 200 |
| `test_all_routes_registered` | OpenAPI spec | All 5 endpoints present |

**Note:** API tests use `TestClient` (synchronous, no real server needed). Tests that would trigger Mistral API calls are marked `@requires_api_key`.

---

## Step 11 — Integration Tests (`tests/test_integration.py`)

**Tests for:** End-to-end pipeline (requires `MISTRAL_API_KEY`)

All tests in this file are marked `@requires_api_key` and skipped without the key.

| Test | What it verifies |
|---|---|
| `test_ingest_and_query` | Upload PDF → query about its content → answer mentions content → has sources |
| `test_query_has_citations` | Answer contains `[Source:` citation format |
| `test_query_confidence_above_zero` | Confidence > 0.0 for a relevant query |
| `test_chitchat_no_sources` | `"Hello!"` → response with empty sources, intent = CHITCHAT |
| `test_query_respects_top_k` | `top_k=2` → at most 2 sources |
| `test_delete_then_query` | Ingest → delete → query → no results |

---

## Files Created in Phase 4

| File | Purpose |
|---|---|
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Shared fixtures (chunks, embeddings, PDF, markers) |
| `tests/test_text_utils.py` | 18 unit tests for text processing |
| `tests/test_pii_detector.py` | 11 unit tests for PII detection |
| `tests/test_vector_store.py` | 14 unit tests for vector store |
| `tests/test_search.py` | 10 unit tests for cosine similarity + keyword scoring |
| `tests/test_reranker.py` | 10 unit tests for reranking logic |
| `tests/test_hallucination_filter.py` | 10 unit tests for confidence scoring |
| `tests/test_query_processor.py` | 11 unit tests for intent detection |
| `tests/test_pdf_processor.py` | 8 unit tests for PDF extraction |
| `tests/test_api.py` | 10 integration tests for API endpoints |
| `tests/test_integration.py` | 6 end-to-end tests (API key required) |

**New dependency:** `pytest` added to `requirements.txt`

---

## Coding Order

```
Step 1:  tests/__init__.py + tests/conftest.py      ← fixtures first
Step 2:  tests/test_text_utils.py                    ← no dependencies
Step 3:  tests/test_pii_detector.py                  ← no dependencies
Step 4:  tests/test_vector_store.py                  ← uses fixtures
Step 5:  tests/test_search.py                        ← uses fixtures
Step 6:  tests/test_reranker.py                      ← uses fixtures
Step 7:  tests/test_hallucination_filter.py           ← uses fixtures
Step 8:  tests/test_query_processor.py                ← no dependencies
Step 9:  tests/test_pdf_processor.py                  ← uses test PDF fixture
Step 10: tests/test_api.py                            ← uses TestClient
Step 11: tests/test_integration.py                    ← uses API key
```

---

## How to Run Tests

**All offline tests (no API key needed):**
```bash
pytest tests/ -m "not integration" -v
```

**All tests (requires MISTRAL_API_KEY):**
```bash
pytest tests/ -v
```

**Single test file:**
```bash
pytest tests/test_search.py -v
```

**With coverage report:**
```bash
pytest tests/ --cov=app --cov-report=term-missing
```

---

## What We Can Verify After Phase 4

1. **Every pure logic module works:** text utils, PII detector, vector store, cosine similarity, keyword scoring, reranker, hallucination filter, intent detection
2. **Every API endpoint responds correctly:** health, ingest, documents, delete, query
3. **Edge cases are covered:** empty inputs, corrupt PDFs, PII refusal, no results, mismatched data
4. **End-to-end pipeline works:** ingest a PDF → ask a question → get a cited answer (with API key)
5. **Tests run fast:** Unit tests < 1 second. Integration tests < 30 seconds.
6. **CI-friendly:** Can run without API key (unit tests pass, integration tests skip gracefully)
