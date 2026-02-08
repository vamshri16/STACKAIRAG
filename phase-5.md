# Phase 5: Streamlit Frontend

**Goal:** Build a clean, functional web UI that exposes the entire RAG pipeline to end users — upload PDFs, ask questions, see cited answers, manage documents. After this phase, you can run `streamlit run streamlit_app.py` and interact with the full system through a browser.

**This phase writes one new file** (`streamlit_app.py`) at the project root and adds `streamlit` to `requirements.txt`. No changes to any `app/` code — the frontend talks to the backend through Python imports, not HTTP.

**Architecture:** Single-file Streamlit app that imports core modules directly. No intermediate API calls — Streamlit runs in the same Python process as the backend logic.

---

## What Phase 5 Covers

```
rag-pipeline/
├── streamlit_app.py            ← NEW (entire frontend in one file)
├── requirements.txt            ← MODIFIED (add streamlit)
└── app/                        ← UNCHANGED (all backend code)
```

---

## Why Direct Imports, Not HTTP

Streamlit runs as a Python process. Our backend is a Python library. Two options:

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **HTTP calls** | Run FastAPI on port 8000, Streamlit on port 8501, Streamlit calls `requests.post("http://localhost:8000/api/query")` | Clean separation | Must run 2 processes, network overhead, error handling for connection failures |
| **Direct imports** | Streamlit imports `from app.core.query_processor import process_query` and calls it directly | Single process, faster, simpler | Tighter coupling |

**We choose direct imports** because:
1. This is a portfolio project, not a microservices deployment
2. One command to run (`streamlit run streamlit_app.py`) vs. two terminals
3. All our core logic is already in `app/core/` with no FastAPI dependencies — it's importable
4. The FastAPI server still works independently for API access

---

## Precise Requirements

### UI Layout

The app has a **sidebar** for document management and a **main area** for the chat interface.

```
┌──────────────────────┬────────────────────────────────────────────┐
│                      │                                            │
│   SIDEBAR            │   MAIN AREA                               │
│                      │                                            │
│   ┌──────────────┐   │   ┌────────────────────────────────────┐   │
│   │ System Status │   │   │  RAG Pipeline                      │   │
│   │ • Healthy     │   │   │  Ask questions about your PDFs     │   │
│   │ • 3 docs      │   │   └────────────────────────────────────┘   │
│   │ • 47 chunks   │   │                                            │
│   └──────────────┘   │   ┌────────────────────────────────────┐   │
│                      │   │  Chat history                       │   │
│   ┌──────────────┐   │   │                                    │   │
│   │ Upload PDF   │   │   │  User: What is the revenue?        │   │
│   │ [Browse...]  │   │   │                                    │   │
│   │ [Upload]     │   │   │  Assistant: The company reported    │   │
│   └──────────────┘   │   │  $5M revenue... [Source: ...]       │   │
│                      │   │                                    │   │
│   ┌──────────────┐   │   │  Confidence: 85%                   │   │
│   │ Documents    │   │   │  Sources: report.pdf p3, p7        │   │
│   │ • report.pdf │   │   │  Time: 2340ms                      │   │
│   │   [Delete]   │   │   └────────────────────────────────────┘   │
│   │ • guide.pdf  │   │                                            │
│   │   [Delete]   │   │   ┌────────────────────────────────────┐   │
│   └──────────────┘   │   │  [Ask a question...]          [Send]│   │
│                      │   └────────────────────────────────────┘   │
│                      │                                            │
└──────────────────────┴────────────────────────────────────────────┘
```

### Functional Requirements

| Feature | What it does | Backend module used |
|---|---|---|
| **System status** | Shows health: API configured, doc count, chunk count | `app.storage.vector_store.vector_store` |
| **PDF upload** | File uploader → save to disk → ingest → embed → store | `app.core.ingest_service.run` |
| **Document list** | Shows all ingested documents with metadata | `app.core.ingest_service.list_documents` |
| **Document delete** | Remove a document and all its chunks | `app.core.ingest_service.delete_document` |
| **Chat input** | Text input for user queries | — |
| **Query processing** | PII check → intent → search → rerank → LLM → verify | `app.core.query_processor.process_query` |
| **Answer display** | Shows the LLM answer with formatting | — |
| **Source citations** | Expandable section showing source chunks with scores | `QueryResponse.sources` |
| **Confidence score** | Visual indicator of hallucination filter result | `QueryResponse.confidence` |
| **Processing time** | Shows how long the query took | `QueryResponse.processing_time_ms` |
| **Chat history** | Maintains conversation in session state | `st.session_state` |
| **Error handling** | PII refusal → warning, API errors → error message | `QueryRefusalError` |

### Design Principles

1. **Single file.** The entire UI lives in `streamlit_app.py`. No `pages/` directory, no components directory. One file is easy to read, modify, and understand for a portfolio project.
2. **Session state for chat history.** Streamlit reruns the entire script on every interaction. `st.session_state` persists the conversation across reruns.
3. **No custom CSS.** Streamlit's default styling is clean enough. No `st.markdown` with raw HTML/CSS hacks.
4. **Graceful degradation.** If the API key is missing, the app loads but shows a warning. Upload and document management still work (they don't need the API key until embedding).
5. **Error messages are user-friendly.** PII refusal shows a clear warning, not a stack trace. API errors show a brief message, not the full exception.

---

## Step 1 — Add Streamlit Dependency

**What:** Add `streamlit` to `requirements.txt`.

```
streamlit==1.41.0
```

---

## Step 2 — Sidebar: System Status

**What:** Display system health in the sidebar.

```python
import streamlit as st
from app.config import settings
from app.storage.vector_store import vector_store

st.sidebar.title("RAG Pipeline")

# System status
with st.sidebar.container():
    st.sidebar.subheader("System Status")
    api_configured = bool(settings.mistral_api_key)
    st.sidebar.markdown(f"**Mistral API:** {'Configured' if api_configured else 'Not configured'}")
    st.sidebar.markdown(f"**Documents:** {len(vector_store.documents)}")
    st.sidebar.markdown(f"**Chunks:** {len(vector_store)}")
```

If the API key is not configured, show a warning:
```python
if not api_configured:
    st.sidebar.warning("Set MISTRAL_API_KEY in .env to enable queries.")
```

---

## Step 3 — Sidebar: PDF Upload

**What:** File uploader that saves the PDF and runs the ingestion pipeline.

```python
from app.core.ingest_service import run as run_ingest, IngestError

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Ingest"):
    # Save to temp file
    # Call run_ingest(temp_path, uploaded_file.name)
    # Show success or error
```

**Behavior:**
- Only accepts `.pdf` files (Streamlit's `type` filter)
- Saves the uploaded bytes to `settings.data_dir` with a UUID temp name
- Calls `run_ingest(temp_path, filename)` — same function the FastAPI endpoint uses
- On success: shows green success message with page/chunk counts
- On failure: shows red error message (e.g., "Corrupted PDF", "No text extracted")
- Renames temp file to `{document_id}.pdf` after successful ingestion

---

## Step 4 — Sidebar: Document Management

**What:** List all ingested documents with delete buttons.

```python
from app.core.ingest_service import list_documents, delete_document

documents = list_documents()
if documents:
    st.sidebar.subheader("Documents")
    for doc in documents:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.markdown(f"**{doc.filename}**\n{doc.page_count} pages, {doc.chunk_count} chunks")
        if col2.button("Delete", key=f"del_{doc.document_id}"):
            delete_document(doc.document_id)
            st.rerun()
```

**Behavior:**
- Shows filename, page count, chunk count for each document
- Delete button removes the document and all its chunks from the vector store
- `st.rerun()` refreshes the page after deletion to update counts

---

## Step 5 — Main Area: Chat Interface

**What:** Chat input, conversation history, answer display with metadata.

### Session state initialization:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

### Display chat history:

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            # Show confidence, sources, processing time
```

### Handle new query:

```python
from app.core.query_processor import process_query, QueryRefusalError
from app.models.schemas import QueryRequest

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        request = QueryRequest(query=prompt)
        response = process_query(request)

        # Build assistant message with metadata
        # Add to history
        # Display answer, confidence, sources, time

    except QueryRefusalError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")
```

### Answer display:

For each assistant message, show:
1. **The answer text** — rendered as markdown
2. **Confidence indicator** — color-coded: green (≥0.7), yellow (0.4–0.7), red (<0.4)
3. **Processing time** — e.g., "2.3s"
4. **Sources** — in an expandable `st.expander`:
   - Source filename and page number
   - Relevance score
   - Truncated chunk text (first 300 chars)

---

## Step 6 — Vector Store Persistence

**What:** Load the vector store on startup and save after ingestion/deletion.

```python
import os
from app.storage.vector_store import vector_store

# Load on startup (same as FastAPI's startup event)
index_path = os.path.join(settings.index_dir, "chunks.json")
if os.path.exists(index_path) and len(vector_store) == 0:
    vector_store.load(settings.index_dir)
```

After ingestion and deletion, save:
```python
vector_store.save(settings.index_dir)
```

This ensures data persists across Streamlit reruns and restarts.

---

## Files Created in Phase 5

| File | Purpose |
|---|---|
| `streamlit_app.py` | Complete Streamlit frontend (~200 lines) |

## Files Modified in Phase 5

| File | Change |
|---|---|
| `requirements.txt` | Add `streamlit==1.41.0` |

---

## Coding Order

```
Step 1: Add streamlit to requirements.txt    ← dependency first
Step 2: Sidebar — system status              ← read-only, no side effects
Step 3: Sidebar — PDF upload + ingestion     ← uses ingest_service.run()
Step 4: Sidebar — document list + delete     ← uses ingest_service functions
Step 5: Main area — chat interface           ← uses query_processor.process_query()
Step 6: Persistence — load/save vector store ← startup + after mutations
```

All steps are in one file (`streamlit_app.py`), built incrementally.

---

## How to Run

```bash
# Install streamlit
.venv/bin/pip install streamlit

# Run the app
.venv/bin/streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

The FastAPI server is **not needed** — Streamlit imports the backend directly. But you can still run both:
```bash
# Terminal 1: API server (optional)
.venv/bin/uvicorn app.main:app --reload

# Terminal 2: Streamlit UI
.venv/bin/streamlit run streamlit_app.py
```

---

## What We Can Verify After Phase 5

1. **App loads without errors** — `streamlit run streamlit_app.py` shows the UI
2. **System status displays correctly** — API key status, document/chunk counts
3. **PDF upload works** — upload a PDF → success message with page/chunk counts
4. **Document list updates** — uploaded document appears in sidebar
5. **Delete works** — click delete → document removed → counts update
6. **Chat works** — ask a question → get a cited answer with confidence and sources
7. **PII refusal works** — enter SSN in query → warning message (not a crash)
8. **Chitchat works** — say "Hello" → friendly response, no sources
9. **No results handled** — ask about something not in documents → "I don't have enough information"
10. **Persistence works** — restart Streamlit → documents and chunks are still there
