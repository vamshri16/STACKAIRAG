"""Streamlit frontend for the RAG pipeline.

Single-file UI that imports backend modules directly — no HTTP calls needed.
Run with: streamlit run streamlit_app.py
"""

import os
import uuid

import streamlit as st

from app.config import settings
from app.core.ingest_service import delete_document, list_documents, run_batch
from app.core.query_processor import QueryRefusalError, process_query
from app.models.schemas import QueryRequest
from app.storage.vector_store import vector_store

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _confidence_color(confidence: float) -> str:
    """Return a Streamlit color token based on confidence level."""
    if confidence >= 0.7:
        return "green"
    if confidence >= 0.4:
        return "orange"
    return "red"


def _confidence_label(confidence: float) -> str:
    """Return a human-readable confidence label."""
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low"


def _display_metadata(confidence: float, time_ms: int, intent: str) -> None:
    """Render confidence, time, and intent as metric columns."""
    col1, col2, col3 = st.columns(3)
    color = _confidence_color(confidence)
    label = _confidence_label(confidence)
    col1.markdown(f"**Confidence:** :{color}[{confidence:.0%} ({label})]")
    col2.markdown(f"**Response time:** {time_ms / 1000:.1f}s")
    col3.markdown(f"**Intent:** {intent}")


def _display_sources(sources: list[dict]) -> None:
    """Render a collapsible sources section with tabs per source."""
    if not sources:
        return
    with st.expander(f"📄 View Sources ({len(sources)})"):
        for i, src in enumerate(sources):
            name = src.get("source", "")
            page = src.get("page", 0)
            score = src.get("score", 0.0)
            text = src.get("text", "")
            score_color = _confidence_color(score)
            st.markdown(
                f"**{i + 1}. {name}** — Page {page} "
                f"(relevance: :{score_color}[{score:.2f}])"
            )
            st.info(text[:300])


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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 RAG Pipeline")
    st.markdown("---")

    # --- System status ---
    st.subheader("System Status")
    api_configured = bool(settings.mistral_api_key)

    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Documents", len(vector_store.documents))
    status_col2.metric("Chunks", len(vector_store))

    if api_configured:
        st.success("Mistral API connected", icon="✅")
    else:
        st.error("Mistral API key missing", icon="❌")
        st.caption("Set `MISTRAL_API_KEY` in your `.env` file.")

    st.markdown("---")

    # --- PDF upload ---
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("📥 Ingest Documents", use_container_width=True):
        os.makedirs(settings.data_dir, exist_ok=True)

        file_entries: list[tuple[str, str]] = []
        for uf in uploaded_files:
            temp_name = f"_upload_{uuid.uuid4().hex}.pdf"
            temp_path = os.path.join(settings.data_dir, temp_name)
            with open(temp_path, "wb") as f:
                f.write(uf.getvalue())
            file_entries.append((temp_path, uf.name))

        try:
            with st.status(
                f"Ingesting {len(file_entries)} file(s)...", expanded=True
            ) as status:
                results = run_batch(file_entries)

                for (temp_path, _), result in zip(file_entries, results):
                    if result.status == "completed" and result.document_id:
                        final_path = os.path.join(
                            settings.data_dir, f"{result.document_id}.pdf"
                        )
                        try:
                            os.rename(temp_path, final_path)
                        except OSError:
                            pass
                    else:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                status.update(label="Done!", state="complete")

            any_succeeded = False
            for result in results:
                if result.status == "completed":
                    any_succeeded = True
                    st.success(
                        f"**{result.filename}**: {result.page_count} pages, "
                        f"{result.chunk_count} chunks"
                    )
                else:
                    st.error(f"**{result.filename}**: {result.message}")

            if any_succeeded:
                st.rerun()

        except Exception as exc:
            for temp_path, _ in file_entries:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            st.error(f"Unexpected error: {exc}")

    st.markdown("---")

    # --- Document management ---
    documents = list_documents()
    if documents:
        st.subheader("Ingested Documents")
        for doc in documents:
            with st.container():
                col1, col2 = st.columns([3, 1])
                col1.markdown(
                    f"**{doc.filename}**  \n"
                    f"📄 {doc.page_count} pages · 🧩 {doc.chunk_count} chunks"
                )
                if col2.button("🗑️", key=f"del_{doc.document_id}", help="Delete document"):
                    delete_document(doc.document_id)
                    st.rerun()
    else:
        st.info("No documents ingested yet. Upload PDFs above to get started.")

    st.markdown("---")

    # --- Clear chat ---
    if st.session_state.messages:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.title("🔍 Ask your Documents")

if not vector_store.documents:
    st.markdown(
        "> Upload PDF documents using the sidebar, then ask questions about them here."
    )
elif not st.session_state.messages:
    doc_names = [d.filename for d in list_documents()]
    st.markdown(
        f"> **{len(doc_names)} document(s) loaded:** {', '.join(doc_names)}  \n"
        "> Ask a question below to get started."
    )

# ---------------------------------------------------------------------------
# Main area — chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        metadata = msg.get("metadata")
        if metadata:
            _display_metadata(
                metadata["confidence"],
                metadata["processing_time_ms"],
                metadata["intent"],
            )
            _display_sources(metadata.get("sources", []))

# ---------------------------------------------------------------------------
# Main area — chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            request = QueryRequest(query=prompt)
            with st.spinner("Searching documents and generating answer..."):
                response = process_query(request)

            st.markdown(response.answer)
            _display_metadata(
                response.confidence,
                response.processing_time_ms,
                response.intent,
            )

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
            _display_sources(sources_data)
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
            st.warning(msg, icon="🛡️")
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )

        except Exception as exc:
            msg = f"Something went wrong: {exc}"
            st.error(msg, icon="⚠️")
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )
