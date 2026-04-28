"""Streamlit chat UI for the PDF RAG Assistant."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import Settings
from rag.ingest import ingest_file
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_settings() -> Settings | None:
    try:
        return Settings()
    except Exception as exc:
        st.error(
            f"**Configuration error:** {exc}\n\n"
            "Create a `.env` file in your working directory (see `.env.example`)."
        )
        return None


@st.cache_resource(show_spinner=False)
def _load_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(settings)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "messages": [],    # [{"role": str, "content": str}]
        "sources": {},     # {msg_index: list[dict]}
        "upload_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": role, "content": content})
    if sources is not None:
        st.session_state.sources[idx] = sources


# ---------------------------------------------------------------------------
# Source citation component
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            col_name, col_rel = st.columns([4, 1])
            col_name.markdown(f"**[{i}] {src['source']}** — page {src['page']}")
            col_rel.caption(f"{max(0.0, 1.0 - src['score']):.0%} match")
            st.caption(src["text"][:400])
            if i < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(settings: Settings, pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.title("📚 PDF RAG Assistant")
        st.caption("Upload documents, then ask questions.")
        st.divider()

        # --- Upload ---
        st.subheader("📂 Knowledge Base")
        uploaded = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_key}",
            help="PDF, TXT, Markdown, DOCX",
        )
        if uploaded:
            if st.button("⚡ Process documents", type="primary", use_container_width=True):
                _ingest_uploads(uploaded, settings, pipeline)

        st.divider()

        # --- Stats ---
        doc_count = pipeline.document_count()
        sources = pipeline.list_sources()
        c1, c2 = st.columns(2)
        c1.metric("Chunks indexed", doc_count)
        c2.metric("Documents", len(sources))

        if sources:
            with st.expander("Indexed documents"):
                for src in sources:
                    cn, cd = st.columns([4, 1])
                    cn.caption(src)
                    if cd.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                        deleted = pipeline.vectorstore.delete_source(src)
                        st.toast(f"Removed {deleted} chunks for '{src}'")
                        st.rerun()

        st.divider()

        # --- Settings ---
        with st.expander("⚙️ Settings"):
            model_label = settings.model_filename.replace(".gguf", "")
            st.caption(f"**Model:** `{model_label}`")
            st.caption(f"**Embedding:** `{settings.embedding_model.split('/')[-1]}`")
            st.caption(f"**Context:** {settings.n_ctx:,} tokens")

            new_k = st.slider("Chunks per query", 1, 10, settings.retrieval_k)
            if new_k != settings.retrieval_k:
                settings.retrieval_k = new_k
                pipeline.retriever.settings = settings

            new_thr = st.slider(
                "Distance threshold", 0.5, 2.0, settings.max_distance, step=0.05,
                help="Lower = stricter. Tune if answers seem off-topic.",
            )
            if new_thr != settings.max_distance:
                settings.max_distance = new_thr
                pipeline.retriever.settings = settings

        st.divider()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.rerun()

        if st.button("🔄 Reset knowledge base", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                pipeline.vectorstore.reset()
                st.session_state.update(messages=[], sources={}, confirm_reset=False)
                st.toast("Knowledge base cleared.")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("Click again to confirm.")


def _ingest_uploads(uploaded_files, settings: Settings, pipeline: RAGPipeline) -> None:
    n = len(uploaded_files)
    progress = st.sidebar.progress(0, text="Processing…")
    total_chunks = 0

    for i, f in enumerate(uploaded_files):
        progress.progress(i / n, text=f"Loading {f.name}…")
        suffix = Path(f.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = Path(tmp.name)
        try:
            result = ingest_file(tmp_path, settings)
            if result.errors:
                for fname, err in result.errors.items():
                    st.sidebar.error(f"Error in {fname}: {err}")
            elif result.chunks:
                pipeline.vectorstore.add_documents(result.chunks)
                total_chunks += result.chunk_count
                st.sidebar.success(f"✅ {f.name} — {result.chunk_count} chunks")
        finally:
            tmp_path.unlink(missing_ok=True)

    progress.empty()
    if total_chunks:
        st.session_state.upload_key += 1
        st.toast(f"Added {total_chunks} chunks to the knowledge base.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

def _render_chat(pipeline: RAGPipeline) -> None:
    st.title("Ask your documents")

    if not pipeline.is_ready():
        st.info(
            "👈 Upload one or more PDFs in the sidebar to get started. "
            "Once processed, ask questions here."
        )
        return

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i in st.session_state.sources:
                _render_sources(st.session_state.sources[i])

    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    if question := st.chat_input("Ask a question about your documents…"):
        with st.chat_message("user"):
            st.markdown(question)
        _append_message("user", question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context…"):
                result = pipeline.stream(question, chat_history=chat_history[:-1])

            if result.found:
                response_text = st.write_stream(result.stream)
            else:
                response_text = "".join(result.stream)
                st.warning(response_text)

            _render_sources(result.sources)

        _append_message("assistant", response_text, sources=result.sources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    settings = _load_settings()
    if settings is None:
        st.stop()

    with st.spinner("Loading model (first run may take a minute)…"):
        pipeline = _load_pipeline(settings)

    _render_sidebar(settings, pipeline)
    _render_chat(pipeline)


if __name__ == "__main__":
    main()
