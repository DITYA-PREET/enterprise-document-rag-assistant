from __future__ import annotations
from pathlib import Path
import json
import pickle
import re
import numpy as np
import faiss

class HybridRetriever:
    """
    Dense semantic retrieval + BM25 lexical retrieval.
    Scores are normalized and fused, then diversified with MMR.
    """

    def __init__(self, embedder, store_dir: Path):
        self.embedder = embedder
        self.store_dir = Path(store_dir)
        self.index = None
        self.chunks = []
        self.bm25 = None

    def build(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([self._tokenize(t) for t in texts])

    def save(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.store_dir / "index.faiss"))
        (self.store_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.store_dir / "meta.json").write_text(
            json.dumps({"count": len(self.chunks)}, indent=2),
            encoding="utf-8",
        )

    def load(self):
        from rank_bm25 import BM25Okapi
        self.index = faiss.read_index(str(self.store_dir / "index.faiss"))
        self.chunks = json.loads((self.store_dir / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([self._tokenize(c["text"]) for c in self.chunks])

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    @staticmethod
    def _minmax(x):
        x = np.asarray(x, dtype="float32")
        if len(x) == 0 or float(x.max() - x.min()) < 1e-8:
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    def search(self, query, top_k=8, candidate_k=30, alpha=0.70, mmr_lambda=0.72):
        if self.index is None:
            raise RuntimeError("Retriever is not loaded or built.")

        qvec = self.embedder.encode([query])
        dense_scores, dense_ids = self.index.search(qvec, min(candidate_k, len(self.chunks)))
        dense_scores = dense_scores[0]
        dense_ids = dense_ids[0]

        bm_scores = self.bm25.get_scores(self._tokenize(query))
        bm_order = np.argsort(bm_scores)[::-1][:min(candidate_k, len(self.chunks))]

        candidate_ids = list(dict.fromkeys(
            [int(i) for i in dense_ids if i >= 0] + [int(i) for i in bm_order]
        ))

        dense_map = {int(i): float(s) for i, s in zip(dense_ids, dense_scores) if i >= 0}
        lexical_map = {int(i): float(bm_scores[i]) for i in candidate_ids}

        d = self._minmax([dense_map.get(i, 0.0) for i in candidate_ids])
        b = self._minmax([lexical_map.get(i, 0.0) for i in candidate_ids])
        fused = {i: alpha * float(ds) + (1-alpha) * float(bs)
                 for i, ds, bs in zip(candidate_ids, d, b)}

        selected = []
        remaining = set(candidate_ids)
        while remaining and len(selected) < top_k:
            best_id, best_score = None, -1e9
            for idx in remaining:
                relevance = fused[idx]
                if not selected:
                    mmr = relevance
                else:
                    candidate_vec = self.embedder.encode([self.chunks[idx]["text"]])[0]
                    diversity = max(
                        float(np.dot(candidate_vec, self.embedder.encode([self.chunks[j]["text"]])[0]))
                        for j in selected
                    )
                    mmr = mmr_lambda * relevance - (1-mmr_lambda) * diversity
                if mmr > best_score:
                    best_id, best_score = idx, mmr
            selected.append(best_id)
            remaining.remove(best_id)

        results = []
        for rank, idx in enumerate(selected, start=1):
            item = dict(self.chunks[idx])
            item["score"] = round(float(fused[idx]), 4)
            item["rank"] = rank
            results.append(item)
        return results
