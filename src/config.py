from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    documents_dir: Path = project_root / "data" / "documents"
    processed_dir: Path = project_root / "data" / "processed"
    vectorstore_dir: Path = project_root / "vectorstore"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.28"))

settings = Settings()
