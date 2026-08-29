from __future__ import annotations

import re

import numpy as np

from ..schemas import GuardResult, GuardStatus, GuardType
from ..retrieval.retriever import tokenize


class FaithfulnessGuard:
    """Output guardrail: is the answer grounded in the retrieved TaxKraft context?

    Off-LLM scoring (deterministic, GPU/network-free):
      1. lexical coverage — fraction of answer's content tokens present in context
      2. sentence coverage — mean per-sentence content-token coverage
      3. embedding similarity (optional) — cosine of answer vs context centre

    An optional LLM-as-judge pass (when an API key exists) is layered on top in the engine.
    """

    def __init__(self, threshold: float = 0.30, embedder=None):
        self.threshold = threshold
        self.embedder = embedder

    def score(self, answer: str, context: str) -> float:
        ctx_tokens: set[str] = set(tokenize(context))
        answer_tokens = tokenize(answer)
        if not answer_tokens:
            return 1.0  # an empty answer trivially contradicts nothing
        lexical = sum(1 for t in answer_tokens if t in ctx_tokens) / len(answer_tokens)

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        sentence_scores = []
        for sent in sentences:
            toks = tokenize(sent)
            if not toks:
                continue
            covered = sum(1 for t in toks if t in ctx_tokens) / len(toks)
            sentence_scores.append(covered)
        sentence = float(np.mean(sentence_scores)) if sentence_scores else lexical

        emb = None
        if self.embedder is not None and context.strip():
            try:
                a = self.embedder.embed([answer])[0]
                c = self.embedder.embed([context])[0]
                cnorm = np.linalg.norm(c)
                emb = float(a @ c / (np.linalg.norm(a) * cnorm + 1e-9)) if cnorm else 0.0
            except Exception:
                emb = None

        if emb is None:  # no embedder → rely on lexical evidence
            return 0.6 * lexical + 0.4 * sentence
        return 0.45 * lexical + 0.35 * sentence + 0.20 * max(0.0, emb)

    def check(self, answer: str, context: str) -> GuardResult:
        if not answer.strip():
            return GuardResult(
                name=GuardType.FAITHFULNESS,
                status=GuardStatus.FAILED,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                reason="empty answer",
            )
        s = self.score(answer, context)
        passed = s >= self.threshold
        return GuardResult(
            name=GuardType.FAITHFULNESS,
            status=GuardStatus.PASSED if passed else GuardStatus.FAILED,
            passed=passed,
            score=round(s, 4),
            threshold=self.threshold,
            reason="",
        )