"""Streamlit frontend for the RAG pipeline.

Single-file UI that imports backend modules directly — no HTTP calls needed.
Run with: streamlit run streamlit_app.py
"""

import os
import uuid

import streamlit as st

from app.config import settings
from app.core.ingest_service import IngestError, delete_document, list_documents
from app.core.ingest_service import run as run_ingest
from app.core.query_processor import QueryRefusalError, process_query
from app.models.schemas import QueryRequest
from app.storage.vector_store import vector_store

# ---------------------------------------------------------------------------
# Startup — load persisted vector store
# ---------------------------------------------------------------------------
index_path = os.path.join(settings.index_dir, "chunks.json")
if os.path.exists(index_path) and len(vector_store) == 0:
    vector_store.load(settings.index_dir)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar — system status
# ---------------------------------------------------------------------------
st.sidebar.title("RAG Pipeline")

st.sidebar.subheader("System Status")
api_configured = bool(settings.mistral_api_key)
st.sidebar.markdown(
    f"**Mistral API:** {'Configured' if api_configured else 'Not configured'}"
)
st.sidebar.markdown(f"**Documents:** {len(vector_store.documents)}")
st.sidebar.markdown(f"**Chunks:** {len(vector_store)}")

if not api_configured:
    st.sidebar.warning("Set MISTRAL_API_KEY in .env to enable queries.")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar — PDF upload
# ---------------------------------------------------------------------------
st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file and st.sidebar.button("Ingest"):
    os.makedirs(settings.data_dir, exist_ok=True)
    temp_name = f"_tmp_{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(settings.data_dir, temp_name)

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.sidebar.status("Ingesting...", expanded=True) as status:
            st.write("Processing PDF...")
            response = run_ingest(temp_path, uploaded_file.name)

            # Rename temp file to {document_id}.pdf
            final_path = os.path.join(
                settings.data_dir, f"{response.document_id}.pdf"
            )
            os.rename(temp_path, final_path)

            status.update(label="Done!", state="complete")

        st.sidebar.success(
            f"**{response.filename}**: {response.page_count} pages, "
            f"{response.chunk_count} chunks"
        )
        st.rerun()

    except IngestError as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.sidebar.error(f"Ingestion failed: {exc}")

    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.sidebar.error(f"Unexpected error: {exc}")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
documents = list_documents()
if documents:
    st.sidebar.subheader("Documents")
    for doc in documents:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.markdown(
            f"**{doc.filename}**  \n"
            f"{doc.page_count} pages, {doc.chunk_count} chunks"
        )
        if col2.button("Delete", key=f"del_{doc.document_id}"):
            delete_document(doc.document_id)
            st.rerun()
else:
    st.sidebar.info("No documents ingested yet.")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.title("RAG Pipeline")
st.caption("Ask questions about your uploaded PDFs")

# ---------------------------------------------------------------------------
# Main area — chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        metadata = msg.get("metadata")
        if metadata:
            # Confidence indicator
            confidence = metadata["confidence"]
            if confidence >= 0.7:
                color = "green"
            elif confidence >= 0.4:
                color = "orange"
            else:
                color = "red"

            time_s = metadata["processing_time_ms"] / 1000
            st.caption(
                f"Confidence: :{color}[{confidence:.0%}] "
                f"| Time: {time_s:.1f}s "
                f"| Intent: {metadata['intent']}"
            )

            # Sources
            sources = metadata.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(
                            f"**{src['source']}** — Page {src['page']} "
                            f"(score: {src['score']:.2f})"
                        )
                        st.text(src["text"][:300])
                        st.divider()

# ---------------------------------------------------------------------------
# Main area — chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        try:
            request = QueryRequest(query=prompt)
            with st.spinner("Thinking..."):
                response = process_query(request)

            st.markdown(response.answer)

            # Metadata display
            confidence = response.confidence
            if confidence >= 0.7:
                color = "green"
            elif confidence >= 0.4:
                color = "orange"
            else:
                color = "red"

            time_s = response.processing_time_ms / 1000
            st.caption(
                f"Confidence: :{color}[{confidence:.0%}] "
                f"| Time: {time_s:.1f}s "
                f"| Intent: {response.intent}"
            )

            # Sources
            if response.sources:
                with st.expander(f"Sources ({len(response.sources)})"):
                    for src in response.sources:
                        st.markdown(
                            f"**{src.source}** — Page {src.page} "
                            f"(score: {src.score:.2f})"
                        )
                        st.text(src.text[:300])
                        st.divider()

            # Save to session state
            sources_data = [
                {
                    "chunk_id": s.chunk_id,
                    "source": s.source,
                    "page": s.page,
                    "text": s.text,
                    "score": s.score,
                }
                for s in response.sources
            ]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": {
                        "confidence": response.confidence,
                        "processing_time_ms": response.processing_time_ms,
                        "intent": response.intent,
                        "sources": sources_data,
                    },
                }
            )

        except QueryRefusalError as exc:
            msg = str(exc)
            st.warning(msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )

        except Exception as exc:
            msg = f"Something went wrong: {exc}"
            st.error(msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )
