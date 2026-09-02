"""Simple, dependency-light retrievers for the RAG notebooks.

Two flavours: a TfidfRetriever (pure scikit-learn, fast to get started) and an
EmbeddingRetriever that lazily loads sentence-transformers only when used.
"""

from __future__ import annotations

import numpy as np


class TfidfRetriever:
    """TF-IDF based dense-retrieval-free retriever over a chunk corpus."""

    def __init__(self, stop_words: str = "english"):
        self.stop_words = stop_words
        self._vectorizer = None
        self._matrix = None
        self._chunks: list[str] = []

    def index(self, chunks: list[str]) -> "TfidfRetriever":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self._matrix = self._vectorizer.fit_transform(chunks)
        self._chunks = list(chunks)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k (chunk_index, similarity) pairs for `query`."""
        q_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ q_vec.T).toarray().ravel()
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    @property
    def chunks(self) -> list[str]:
        return self._chunks


class EmbeddingRetriever:
    """Sentence-embedding retriever (lazy sentence-transformers / numpy)."""

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._chunks: list[str] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, chunks: list[str], batch_size: int = 32) -> "EmbeddingRetriever":
        self._embeddings = self._get_model().encode(chunks, batch_size=batch_size, show_progress_bar=False)
        self._chunks = list(chunks)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q_vec = self._get_model().encode([query])[0]
        scores = self._embeddings @ q_vec
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    @property
    def chunks(self) -> list[str]:
        return self._chunks