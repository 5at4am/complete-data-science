from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    model_id: str

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


@dataclass
class MiniLMEembder:
    """Local CPU embedder wrapping sentence-transformers.

    Loads lazily so light tests never trigger the model download.
    """

    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model = None
    dim: int = 384

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


@dataclass
class DummyEmbedder:
    """Deterministic hash-based embedder for tests / offline CI (no model download)."""

    model_id: str = "dummy-hash-128"
    dim: int = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = _stable_hash(t)
            rng = np.random.default_rng(h)
            v = rng.normal(size=self.dim).astype(np.float32)
            v[len(v) // 2 :] *= 0.5
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


def _stable_hash(text: str) -> int:
    # Python's hash() is salted per process; use a deterministic digest instead.
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def make_embedder(model_id: str, offline_ok: bool = False) -> Embedder:
    if offline_ok or "dummy" in model_id:
        return DummyEmbedder()
    return MiniLMEembder(model_id=model_id)