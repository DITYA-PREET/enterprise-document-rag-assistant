import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv()

from src.config import settings
from src.embeddings import Embedder
from src.retriever import HybridRetriever
from src.rag_pipeline import RAGPipeline
from src.indexer import build_index

st.set_page_config(
    page_title="Enterprise AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def load_engine():
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(embedder, settings.vectorstore_dir)
    retriever.load()
    return RAGPipeline(retriever, settings.llm_model, settings.min_confidence)

def ensure_dirs():
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

ensure_dirs()

st.title("📚 Enterprise AI Document Assistant")
st.caption("Upload documents here → automatic indexing → hybrid RAG → grounded answers with sources")

with st.sidebar:
    st.header("Knowledge Base")
    uploads = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads:
        # Files selected in the website uploader are the CURRENT knowledge base.
        # Old local/uploaded documents are removed before rebuilding, so an old
        # PDF can never remain in the retrieval index by accident.
        import hashlib

        def file_signature(uploaded_file):
            data = uploaded_file.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            return f"{Path(uploaded_file.name).name}:{digest}"

        upload_signature = "|".join(
            sorted(file_signature(file) for file in uploads)
        )

        if st.session_state.get("last_upload_signature") != upload_signature:
            # Remove old document files.
            for old_file in settings.documents_dir.iterdir():
                if old_file.is_file() and old_file.name != ".gitkeep":
                    old_file.unlink()

            # Remove the previous vector index.
            for old_file in settings.vectorstore_dir.iterdir():
                if old_file.is_file() and old_file.name != ".gitkeep":
                    old_file.unlink()

            # Save ONLY the files selected through the website.
            for uploaded_file in uploads:
                target = settings.documents_dir / Path(uploaded_file.name).name
                target.write_bytes(uploaded_file.getvalue())

            with st.spinner("Processing your uploaded document(s) and building the knowledge base..."):
                try:
                    manifest = build_index()
                    load_engine.clear()
                    st.session_state.last_upload_signature = upload_signature
                    st.session_state.messages = []

                    st.success(
                        f"Knowledge base ready: {manifest['documents']} document(s), "
                        f"{manifest['pages']} page(s), {manifest['chunks']} chunk(s)."
                    )
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
        else:
            st.success("Your currently uploaded document(s) are the active knowledge base.")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Extracting, chunking, embedding and indexing..."):
            try:
                manifest = build_index()
                load_engine.clear()
                st.success(
                    f"Indexed {manifest['documents']} documents → "
                    f"{manifest['chunks']} chunks."
                )
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.write(f"**Embedding:** `{settings.embedding_model}`")
    st.write(f"**LLM:** `{settings.llm_model}`")
    st.write(f"**Chunk:** {settings.chunk_size} words / {settings.chunk_overlap} overlap")
    st.write(f"**Top-K:** {settings.top_k}")
    st.write(f"**Min confidence:** {settings.min_confidence}")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.write(
                        f"**{s['source']}** — page {s['page']} "
                        f"— score `{s['score']}`"
                    )
                    st.caption(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        engine = load_engine()
        history = st.session_state.messages[:-1]
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                result = engine.answer(question, history=history, top_k=settings.top_k)
            st.markdown(result["answer"])
            st.caption(f"Retrieval confidence: {result['confidence']:.3f}")

            with st.expander("Sources & retrieved context"):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['rank']}. {s['source']} — page {s['page']}** "
                        f"(score `{s['score']}`)"
                    )
                    st.write(s["text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
    except Exception as exc:
        st.error(
            "The knowledge base is not ready. Upload documents and click "
            "'Build / Rebuild Index'. Details: " + str(exc)
        )
