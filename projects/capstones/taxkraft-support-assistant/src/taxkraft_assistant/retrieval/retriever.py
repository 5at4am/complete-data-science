from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ..schemas import RetrievedChunk


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "by", "from", "as", "it", "its", "this",
    "that", "these", "those", "my", "our", "your", "you", "i", "we", "they", "he", "she",
    "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "have",
    "has", "had", "what", "which", "who", "whom", "how", "when", "where", "why", "about",
    "than", "so", "such", "not", "no", "nor", "if", "then", "else", "too", "very", "just",
    "also", "more", "most", "some", "any", "all", "each", "every", "both", "few", "other",
    "another", "please", "tell", "need", "want", "like",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


class BM25:
    """Small, dependency-free BM25 (Okapi variant) over a static corpus."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_terms = [Counter(tokenize(doc)) for doc in corpus]
        self.doc_len = [sum(tf.values()) or 1 for tf in self.doc_terms]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        n = len(corpus)
        df: Counter[str] = Counter()
        for tf in self.doc_terms:
            df.update(tf.keys())
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_terms: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for term in set(query_terms):
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.doc_terms):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


@dataclass
class Retriever:
    store: object
    settings: object
    embedder: object = field(default=None)
    _bm25: BM25 = field(default=None, repr=False)

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = self.store._get_embedder()
        return self.embedder

    def _ensure_bm25(self):
        if self._bm25 is None:
            if self.store.collection is None:
                self.store.connect_collection()
            docs = self.store.collection.get(include=["documents"])["documents"]
            self._bm25 = BM25(docs or [""])
        return self._bm25

    def _fuse(
        self,
        dense: list[dict],
        bm25_scores: list[float] | None,
        query_terms: list[str],
        top_k: int,
    ) -> list[dict]:
        """Normalize each score stream to [0,1] and interpolate."""
        if not dense:
            return []

        def norm(values: list[float]) -> list[float]:
            lo, hi = min(values), max(values)
            if hi - lo < 1e-9:
                return [0.5] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        if bm25_scores is not None:
            # realign bm25 to dense result order: both are over same corpus order
            bm25_norm = norm(bm25_scores)
            for rank, hit in enumerate(dense):
                idx = _corpus_rank(hit, self._bm25.corpus)
                kw = bm25_norm[idx] if idx is not None else 0.0
                sw = hit.get("semantic", 0.0)
                hit["keyword"] = kw
                hit["score"] = (
                    self.settings.semantic_weight * sw + self.settings.bm25_weight * kw
                )
        else:
            for hit in dense:
                hit["score"] = hit.get("semantic", 0.0)
        dense.sort(key=lambda h: h["score"], reverse=True)
        return dense[:top_k]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Hybrid retrieval: dense (Chroma, cosine) fused with BM25 keyword scores."""
        top_k = top_k or self.settings.top_k
        embedder = self._get_embedder()
        q_vec = embedder.embed([query])[0].tolist()
        rerank_k = max(top_k, self.settings.rerank_k)
        dense = self.store.query(
            q_vec, top_k=min(rerank_k, max(self.store.chunk_count, 1))
        )
        dense = sorted(dense, key=lambda h: h["score"], reverse=True)

        bm25 = self._ensure_bm25()
        bm25_scores = bm25.score(tokenize(query))

        merged = self._fuse(dense, bm25_scores, tokenize(query), top_k)
        return merged

    def as_citations(self, hits: list[dict]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text=h["text"][:400],
                source_url=h["source_url"],
                source_title=h["source_title"],
                topic=h["topic"],
                score=round(float(h["score"]), 4),
                char_count=len(h["text"]),
            )
            for h in hits
        ]


def _corpus_rank(hit: dict, corpus: list[str]) -> int | None:
    """BM25 scores align with corpus order; map a dense hit back to that order."""
    try:
        return corpus.index(hit["text"])
    except ValueError:
        return None