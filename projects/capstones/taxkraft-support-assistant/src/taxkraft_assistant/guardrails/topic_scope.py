from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import GuardResult, GuardStatus, GuardType
from .anchors import SCOPE_TERMS, OFF_TOPIC_TERMS, EMBEDDING_ANCHORS


@dataclass
class TopicScopeGuard:
    """Two-stage input guardrail enforcing the TaxKraft-only boundary.

    Stage 1 (deterministic): weighted keyword/vocabulary scoring.
    Stage 2 (optional, requires embedder): cosine similarity of the query against
    embeddings of on-scope anchor statements built from the knowledge base.
    """

    threshold: float = 0.42
    embedder: object = None
    _anchors: list[dict] = field(default_factory=list)

    # -- deterministic stage -------------------------------------------------
    def keyword_score(self, text: str) -> float:
        text = text.lower()
        matched = 0.0
        for term, weight in SCOPE_TERMS.items():
            if re.search(term, text):
                matched += weight
        penalty = 0.0
        for term, weight in OFF_TOPIC_TERMS.items():
            if re.search(term, text):
                penalty += weight
        score = (matched - 2.0 * penalty) / 3.0
        return max(0.0, min(1.0, score))

    def _anchor_embeddings(self):
        if self.embedder is None:
            return {}
        if not self._anchors:
            self._anchors = [
                self.embedder.embed([a])[0] for a in EMBEDDING_ANCHORS
            ]
        return self._anchors

    def embedding_score(self, text: str) -> float | None:
        anchors = self._anchor_embeddings()
        if not anchors:
            return None
        import numpy as np

        q = self.embedder.embed([text])[0]
        sims = np.asarray(anchors) @ q
        return float(sims.max())

    # -- public check --------------------------------------------------------
    def check(self, text: str) -> GuardResult:
        kw = self.keyword_score(text)
        emb = self.embedding_score(text)
        score = kw
        if emb is not None:
            score = 0.55 * emb + 0.45 * kw
        passed = score >= self.threshold
        return GuardResult(
            name=GuardType.TOPIC_SCOPE,
            status=GuardStatus.PASSED if passed else GuardStatus.FAILED,
            passed=passed,
            score=round(score, 4),
            threshold=self.threshold,
            reason="",
            detail=f"keyword={kw:.3f} embedding={emb if emb is None else round(emb, 3)}",
        )