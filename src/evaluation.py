from __future__ import annotations
import json
from pathlib import Path

def evaluate_retrieval(retriever, dataset_path: Path, k=5):
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = []
    hits = 0
    for item in data:
        results = retriever.search(item["question"], top_k=k)
        expected = {s.lower() for s in item.get("expected_sources", [])}
        returned = {r["source"].lower() for r in results}
        hit = bool(expected & returned) if expected else False
        hits += int(hit)
        rows.append({
            "question": item["question"],
            "hit": hit,
            "expected_sources": list(expected),
            "returned_sources": list(returned),
        })
    return {
        "questions": len(data),
        "retrieval_hit_rate": hits / len(data) if data else 0,
        "details": rows,
    }
