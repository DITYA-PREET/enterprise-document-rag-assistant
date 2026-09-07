# Enterprise Document Intelligence & RAG Assistant

An end-to-end enterprise knowledge assistant that ingests business documents, creates semantic embeddings, stores them in FAISS, combines vector retrieval with BM25 lexical search, and generates source-grounded answers with Gemini.

## Architecture

Documents
→ extraction
→ cleaning
→ chunking + metadata
→ embeddings
→ FAISS vector index + BM25
→ hybrid retrieval
→ confidence threshold
→ grounded Gemini generation
→ source attribution
→ Streamlit UI

## Features

- PDF, DOCX, TXT and Markdown ingestion
- Page/document/chunk metadata
- Configurable chunk size and overlap
- Sentence-Transformers embeddings
- FAISS dense retrieval
- BM25 keyword retrieval
- Hybrid score fusion
- MMR-style result diversification
- Top-K retrieval
- Retrieval confidence threshold
- Grounded generation prompt
- Hallucination refusal behavior
- Source/page attribution
- Conversation context
- Upload + re-index workflow
- Streamlit interface
- Basic retrieval evaluation scaffold

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Gemini API key in `.env`.

### Add documents

Place permitted business documents in:

`data/documents/`

Or upload them through the Streamlit sidebar.

### Build the index

```powershell
python -m src.indexer
```

If running that command directly, add this block to `src/indexer.py` or use the Streamlit button:

```python
if __name__ == "__main__":
    print(build_index())
```

### Run

```powershell
streamlit run app.py
```

## What to explain in the final demonstration

1. Business problem: employees/business users waste time searching many documents.
2. Why RAG: the model receives retrieved evidence instead of relying only on its pretrained knowledge.
3. Embeddings: numerical vectors representing semantic meaning.
4. Vector search: finds semantically similar chunks.
5. BM25: catches exact terminology and keywords.
6. Hybrid retrieval: combines semantic and lexical evidence.
7. Chunking: balances context size and retrieval precision.
8. Metadata: source, page, section and chunk ID provide traceability.
9. Confidence threshold: blocks generation when evidence is weak.
10. Source attribution: every answer exposes retrieved documents/pages.
11. Evaluation: measure whether the expected source was retrieved, then assess answer correctness and groundedness.

## Security

Never commit `.env`, API keys, passwords, confidential documents, or private company material.

## Project requirements mapping

The implementation covers the minimum requirements in the capstone specification: ingestion, extraction, chunking, metadata, embeddings, vector database, semantic retrieval, LLM integration, RAG, prompt engineering, source attribution, hallucination handling, evaluation scaffolding, Streamlit deployment, GitHub-ready structure and README.
