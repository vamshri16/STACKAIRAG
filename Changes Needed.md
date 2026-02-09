# Changes Needed

Implementation plan for improvements to the RAG pipeline. Organized by priority — items at the top should be done first.

Items 13 and 14 (README fixes) have been completed.

---

## Quick Wins (do these first)

### 1. Remove score leakage from LLM context
- `format_context()` injects `(Score: 0.92)` into the prompt sent to the LLM.
- The LLM may parrot or anchor on these numbers. Scores belong in response metadata only.
- **Change:** Strip the score from the header in `format_context()`, keep it only in the `Source` objects returned to the caller.
- **Location:** `app/core/llm_client.py:59`

### 2. Add embedding dimension / model validation
- The index stores raw embeddings but no record of which model produced them.
- If the user switches from `mistral-embed` to a different model, the stored embeddings silently become garbage.
- **Change:** Write `{"model": "mistral-embed", "dimension": 1024, "version": 1}` into a `metadata.json` alongside `embeddings.npy`. On load, compare against the current config and fail fast with a clear error if they don't match.
- **Location:** `app/storage/vector_store.py`, `app/core/embeddings.py`, `app/config.py`

### 3. Streamlit UX polish
- When no documents are ingested, the chat input still appears and queries will fail with a confusing error.
- Failure modes (missing API key, empty store) should show clear guidance, not stack traces.
- **Changes:**
  - Disable or hide chat input when `len(vector_store) == 0`, show a prompt to upload PDFs.
  - If `MISTRAL_API_KEY` is missing, show a banner in the main area (not just the sidebar).
  - Add a "Clear chat" button.
- **Location:** `streamlit_app.py`

---

## Feature Additions

### 4. Multi-file ingestion
- The current API and Streamlit UI accept only one PDF at a time.
- **API change:** Add `POST /api/ingest/batch` that accepts `list[UploadFile]`. Process each file sequentially, collect results, return a summary with per-file status (so one bad PDF doesn't fail the whole batch).
- **Streamlit change:** Switch `file_uploader` to `accept_multiple_files=True`. Loop over files on ingest, show per-file progress.
- **Location:** `app/api/ingestion.py`, `app/core/ingest_service.py`, `streamlit_app.py`

### 5. Add query expansion
- Currently a single query embedding is used for search. If the user's phrasing doesn't match the document's wording, recall suffers.
- **Change:** Before searching, generate 2–3 paraphrase variants of the query using the LLM (cheap, short completions). Embed each variant, run hybrid search for each, merge and deduplicate results, then rerank the union.
- **Location:** `app/core/query_processor.py`, `app/core/search.py`

### 6. Improve keyword scoring with TF-IDF
- The keyword scorer treats all terms equally. The word "the" contributes the same as a rare domain term.
- **Change:** On ingestion, compute and persist document-frequency counts across all chunks. At query time, weight each query term by its IDF (inverse document frequency) instead of using raw overlap.
- **Location:** `app/core/search.py`, `app/storage/vector_store.py`

---

## Reliability & Robustness

### 7. OCR fallback for scanned PDFs
- `pytesseract` is in `requirements.txt` but never called.
- **Change:** In `pdf_processor.py`, after PyPDF2 + pdfplumber extraction, if a page yields no text (or fewer than ~20 characters), render the page to an image and run OCR via pytesseract.
- **Location:** `app/core/pdf_processor.py`

### 8. Retrieval diversification (MMR)
- Top-k results can contain near-duplicate chunks from adjacent pages that all say the same thing.
- **Change:** After scoring, apply Maximal Marginal Relevance: iteratively pick the next chunk that maximizes `λ * relevance - (1-λ) * max_similarity_to_already_selected`. This trades some relevance for diversity.
- **Location:** `app/core/reranker.py`

### 9. Strengthen hallucination filter
- The current filter checks token overlap. It misses paraphrased hallucinations and can't detect contradictions.
- **Change:** For each answer sentence, compute cosine similarity of its embedding against each source chunk embedding (reuse the embedding client). A sentence is "supported" only if its best chunk similarity exceeds a threshold. This is still heuristic but much stronger than token overlap.
- **Location:** `app/core/hallucination_filter.py`

### 10. Safer persistence (atomic writes)
- If the process crashes mid-save, `chunks.json` and `embeddings.npy` can become inconsistent.
- **Change:** Write to temp files first, then `os.replace()` atomically. Wrap save/load with a simple version check.
- **Location:** `app/storage/vector_store.py`

---

## Lower Priority

### 11. Chunking quality
- Improve sentence boundary detection (handle abbreviations, decimal numbers).
- Add heading/section detection to avoid splitting structured content.
- **Location:** `app/utils/text_utils.py`

### 12. PII detection improvements
- Current regex patterns are basic. Consider adding configurable strict mode and optional masking (replace PII with `[REDACTED]`) instead of full query refusal.
- **Location:** `app/utils/pii_detector.py`, `app/core/query_processor.py`

### 13. Index metadata in health endpoint
- Expose corpus stats (doc count, chunk count, embedding model, index version) in `/api/health` so operators can verify the index state.
- **Location:** `app/api/health.py`, `app/storage/vector_store.py`

### 14. Configurable batch sizes and timeouts
- Embedding batch size (16) and LLM timeout (120s) are hardcoded.
- Move to `config.py` so they can be tuned via environment variables.
- **Location:** `app/core/embeddings.py`, `app/core/llm_client.py`, `app/config.py`
