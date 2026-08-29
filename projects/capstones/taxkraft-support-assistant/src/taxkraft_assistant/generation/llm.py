from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import build_messages
from ..retrieval.retriever import tokenize


@dataclass
class LLMClient:
    """OpenAI-compatible client (works with Groq's /v1 endpoint)."""

    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.2
    max_tokens: int = 300

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[dict]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class ExtractiveGenerator:
    """Offline answer builder: stitches the most on-topic sentences from the
    retrieved TaxKraft context. Deterministic, $0, and faithful by construction."""

    def __init__(self, max_sentences: int = 4, min_sentence_len: int = 20):
        self.max_sentences = max_sentences
        self.min_sentence_len = min_sentence_len

    def generate(
        self, query: str, chunks: list[dict], top_context: str | None = None
    ) -> str:
        # Deduplicate identical snippets, keep the best-hit order.
        seen: set[str] = set()
        blocks: list[str] = []
        for c in chunks:
            text = c["text"]
            if text in seen:
                continue
            seen.add(text)
            blocks.append(text)
        if not blocks:
            return (
                "I'm sorry, I couldn't find that information in my TaxKraft knowledge base. "
                "For a precise answer please reach us at +91-8608601620 or info@taxkraft.com "
                "— a TaxKraft expert will help."
            )
        joined = "\n".join(blocks)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", joined)
            if len(s.strip()) >= self.min_sentence_len
        ]
        q_terms = set(tokenize(query))
        ranked = sorted(
            sentences,
            key=lambda s: _overlap(s, q_terms),
            reverse=True,
        )
        selected = _dedupe_window(ranked[: self.max_sentences * 2], max_sentences=self.max_sentences)
        answer = " ".join(selected)
        if not answer:
            answer = blocks[0][:600].strip()
        return answer


def _overlap(sentence: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    hits = sum(1 for t in tokenize(sentence) if t in query_terms)
    return hits / len(query_terms)


def _dedupe_window(sentences: list[str], max_sentences: int) -> list[str]:
    """Drop near-duplicate or overlapping windows so the answer reads cleanly."""
    selected: list[str] = []
    for s in sentences:
        if len(selected) >= max_sentences:
            break
        if any(_token_jaccard(s, old) > 0.85 for old in selected):
            continue
        selected.append(s)
    return selected


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / (len(ta | tb) or 1)


def build_answer(client: LLMClient | None, query: str, chunks: list[dict]) -> str:
    if client is not None and client.available():
        context = "\n\n".join(c["text"] for c in chunks[:5])
        citations = "\n".join(
            f"- {c['source_title']}: {c['source_url']}" for c in chunks[:5]
        )
        try:
            return client.complete(build_messages(query, context, citations))
        except Exception as exc:  # degrade gracefully on provider errors
            print(f"[generation] LLM call failed ({exc}); using extractive fallback.")
    return ExtractiveGenerator().generate(query, chunks)