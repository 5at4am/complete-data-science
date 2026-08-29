from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from .chunker import Chunk, chunk_document
from .embeddings import Embedder


@dataclass
class VectorStore:
    settings: object
    collection = None
    embedder: Embedder = field(default=None, repr=False)
    chunk_count: int = 0

    def _client(self):
        import chromadb

        return chromadb.PersistentClient(path=str(self.settings.vector_dir))

    def connect_collection(self) -> None:
        client = self._client()
        self.collection = client.get_or_create_collection(name=self.settings.collection_name)
        self.chunk_count = self.collection.count()

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            from .embeddings import make_embedder

            self.embedder = make_embedder(self.settings.embedding_model)
        return self.embedder

    def build(self, docs: list[dict], reset: bool = False) -> int:
        """Embed every chunk and (re)populate the collection. Returns chunk count."""
        import chromadb

        embedder = self._get_embedder()
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap))
        if not chunks:
            raise ValueError("No chunks produced from the corpus — nothing to ingest.")

        client = self._client()
        if reset:
            try:
                client.delete_collection(self.settings.collection_name)
            except Exception:
                pass
        self.collection = client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Ingesting {len(chunks)} chunks ...")
        t0 = time.perf_counter()
        batch = 64
        for start in range(0, len(chunks), batch):
            group = chunks[start : start + batch]
            vecs = embedder.embed([c.text for c in group])
            self.collection.add(
                ids=[f"{c.source_url}::c{c.index}::{start + i}" for i, c in enumerate(group)],
                embeddings=vecs.tolist(),
                documents=[c.text for c in group],
                metadatas=[
                    {
                        "source_url": c.source_url,
                        "source_title": c.source_title,
                        "topic": c.topic,
                        "index": c.index,
                    }
                    for c in group
                ],
            )
        self.chunk_count = self.collection.count()
        print(f"Stored {self.chunk_count} chunks in {time.perf_counter() - t0:.2f}s")
        return self.chunk_count

    def query(self, query_embedding: list[float], top_k: int) -> dict:
        """Dense-only query. Embeddings are unit-normalized, so cosine
        similarity reduces to (1 - l2_distance**2 / 2)."""
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.chunk_count, 1)),
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for text, meta, dist in zip(docs, metas, dists):
            d = float(dist)
            # cosine = 1 - d^2/2 for unit vectors; clamp to [0,1]
            score = max(0.0, min(1.0, 1.0 - (d ** 2) / 2.0))
            out.append(
                {
                    "text": text,
                    "source_url": meta.get("source_url", ""),
                    "source_title": meta.get("source_title", ""),
                    "topic": meta.get("topic", ""),
                    "index": meta.get("index", 0),
                    "score": score,
                    "semantic": score,
                }
            )
        return out

    def size(self) -> int:
        return self.chunk_count

    @property
    def ready(self) -> bool:
        return self.collection is not None and self.chunk_count > 0