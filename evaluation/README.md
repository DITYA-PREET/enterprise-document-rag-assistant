# Evaluation

Put a small JSON dataset in `questions.json`.

Example:

```json
[
  {
    "question": "What is the leave policy?",
    "expected_sources": ["Leave Policy.pdf"]
  }
]
```

The evaluation module reports retrieval hit rate. For the final internship submission, expand this dataset and record:
- retrieval relevance
- context relevance
- answer correctness
- groundedness
- completeness
- hallucination rate
