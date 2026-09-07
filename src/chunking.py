from __future__ import annotations
from typing import List, Dict
import re

def _words(text: str):
    return re.findall(r"\S+", text)

def chunk_pages(pages, chunk_size=900, overlap=150):
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        words = _words(page.text)
        start = 0
        chunk_no = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append({
                    "text": text,
                    "source": page.source,
                    "page": page.page,
                    "section": page.section,
                    "file_type": page.file_type,
                    "document_id": page.document_id,
                    "chunk_id": f"{page.document_id}-{page.page}-{chunk_no}",
                    "start_word": start,
                    "end_word": end,
                })
            if end == len(words):
                break
            start = end - overlap
            chunk_no += 1
    return chunks
