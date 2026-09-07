from pathlib import Path
import json
from .config import settings
from .document_loader import load_documents
from .chunking import chunk_pages
from .embeddings import Embedder
from .retriever import HybridRetriever

def build_index():
    pages = load_documents(settings.documents_dir)
    if not pages:
        raise RuntimeError(
            f"No supported documents found in {settings.documents_dir}. "
            "Add PDF, DOCX, TXT or MD files."
        )
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(embedder, settings.vectorstore_dir)
    retriever.build(chunks)
    retriever.save()

    manifest = {
        "documents": len({p.source for p in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    (settings.processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(build_index())
