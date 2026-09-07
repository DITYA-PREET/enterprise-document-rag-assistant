from __future__ import annotations
import os
from typing import List, Dict

SYSTEM_INSTRUCTION = """
You are an enterprise document assistant.
Answer ONLY from the supplied CONTEXT. Do not use outside knowledge.
If the context does not contain enough evidence, say:
"I could not find sufficient information in the provided documents to answer this question."

Rules:
- Never invent facts, numbers, policies, dates, names, or procedures.
- Prefer precise answers over speculation.
- When useful, mention the source document and page.
- If multiple sources disagree, explicitly say so.
- Keep the answer concise but complete.
"""

class RAGPipeline:
    def __init__(self, retriever, model_name: str, min_confidence: float = 0.28):
        self.retriever = retriever
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.client = None

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Streamlit Community Cloud stores secrets in st.secrets.
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None

        if api_key:
            from google import genai
            self.client = genai.Client(api_key=api_key)

        # Gemini can temporarily return 503 when a model is overloaded.
        # Keep multiple currently supported Flash models available so a
        # temporary capacity problem does not take down the whole app.
        configured_fallbacks = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
        )
        self.fallback_models = []
        for model in [self.model_name] + [m.strip() for m in configured_fallbacks.split(",")]:
            if model and model not in self.fallback_models:
                self.fallback_models.append(model)

    def _context(self, results):
        blocks = []
        for r in results:
            blocks.append(
                f"[SOURCE {r['rank']}]\n"
                f"Document: {r['source']}\n"
                f"Page: {r['page']}\n"
                f"Section: {r.get('section','')}\n"
                f"Retrieval score: {r['score']}\n"
                f"Content:\n{r['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, history=None, top_k=8):
        results = self.retriever.search(question, top_k=top_k)
        best = max((r["score"] for r in results), default=0.0)

        if best < self.min_confidence:
            return {
                "answer": "I could not find sufficient information in the provided documents to answer this question.",
                "sources": results[:3],
                "confidence": best,
            }

        history_text = ""
        if history:
            history_text = "\nPrevious conversation:\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in history[-6:]
            )

        prompt = (
            SYSTEM_INSTRUCTION
            + history_text
            + "\n\nCONTEXT:\n"
            + self._context(results)
            + "\n\nUSER QUESTION:\n"
            + question
        )

        if not self.client:
            return {
                "answer": "GEMINI_API_KEY is not configured. Retrieval succeeded, but generation is disabled.",
                "sources": results,
                "confidence": best,
            }

        errors = []
        for model in self.fallback_models:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                text = (response.text or "").strip()
                if text:
                    return {
                        "answer": text,
                        "sources": results,
                        "confidence": best,
                        "model": model,
                    }
            except Exception as exc:
                errors.append(f"{model}: {exc}")

        return {
            "answer": (
                "The document retrieval worked, but the Gemini generation service is "
                "temporarily unavailable. Please try the question again in a moment. "
                "The app automatically tried multiple Gemini Flash models."
            ),
            "sources": results,
            "confidence": best,
            "error": " | ".join(errors),
        }
