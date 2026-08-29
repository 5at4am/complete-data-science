from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config import load_settings
from .generation.llm import build_answer
from .guardrails.engine import (
    DEFLECT_GENERIC,
    DEFLECT_FAITHFULNESS,
    DEFLECT_PII,
    DEFLECT_RETRIEVAL,
    GuardEngine,
)
from .ingestion.embeddings import make_embedder
from .ingestion.chunker import load_documents
from .ingestion.store import VectorStore
from .retrieval.retriever import Retriever
from .schemas import ChatResponse, GuardResult, GuardType


@dataclass
class Pipeline:
    settings: object
    store: VectorStore = field(default=None, repr=False)
    retriever: Retriever = field(default=None, repr=False)
    engine: GuardEngine = field(default=None, repr=False)
    embedder: object = field(default=None, repr=False)
    llm_client: object = field(default=None, repr=False)
    _llm_builder = None

    @classmethod
    def build(cls, settings=None, embedder=None, llm_client=None, connect: bool = True):
        settings = settings or load_settings()
        store = VectorStore(settings=settings, embedder=embedder)
        if connect:
            store.connect_collection()
        retriever = Retriever(store=store, settings=settings, embedder=embedder)
        engine = GuardEngine(settings=settings, embedder=embedder or store.embedder)
        return cls(
            settings=settings,
            store=store,
            retriever=retriever,
            engine=engine,
            embedder=embedder,
            llm_client=llm_client,
        )

    # ---- ingest ------------------------------------------------------------
    def ingest(self, reset: bool = False) -> int:
        docs = load_documents(self.settings.kb_dir)
        return self.store.build(docs, reset=reset)

    # ---- chat --------------------------------------------------------------
    def chat(self, message: str) -> str:
        return self.chat_full(message)["answer"]

    def chat_full(self, message: str) -> dict:
        t0 = time.perf_counter()
        answers: list[str] = []
        guard_results: list[GuardResult] = []

        # 1) input guardrails
        input_results = self.engine.input_checks(message)
        guard_results.extend(input_results)
        deflection = self.engine.deflection_for(input_results)
        if deflection:
            return self._out(
                message, deflection, guard_results, citations=[], refused=True, t0=t0
            )

        # 2) retrieval
        hits = self.retriever.retrieve(message)
        citations = self.retriever.as_citations(hits)

        # 3) output guardrail A — retrieval confidence
        top_score = hits[0]["score"] if hits else 0.0
        conf = self.engine.retrieval_check(top_score)
        guard_results.append(conf)
        if not conf.passed:
            return self._out(
                message, DEFLECT_RETRIEVAL, guard_results, citations=citations, refused=True, t0=t0
            )

        # 4) generate
        context = "\n\n".join(h["text"] for h in hits[:5])
        answer = build_answer(self.llm_client, message, hits)
        answers.append(answer)

        # 5) output guardrail B — faithfulness vs retrieved context
        fb = self.engine.faithfulness_check(answer, context)
        guard_results.append(fb)
        if not fb.passed:
            return self._out(
                message, DEFLECT_FAITHFULNESS, guard_results, citations, refused=True, t0=t0
            )

        return self._out(
            message,
            answer,
            guard_results,
            citations,
            refused=False,
            t0=t0,
            generator=self._generator_name() + ("+llm" if self.llm_available() else "+extractive"),
        )

    # ---- helpers -----------------------------------------------------------
    def llm_available(self) -> bool:
        return self.llm_client is not None and bool(getattr(self.llm_client, "api_key", ""))

    def _generator_name(self) -> str:
        return f"{self.settings.llm_provider}:{self.settings.llm_model}"

    def _out(
        self,
        message: str,
        answer: str,
        guards: list[GuardResult],
        citations,
        refused: bool,
        t0: float,
        generator: str | None = None,
    ) -> dict:
        if self.settings.log_requests:
            d = {
                "refused": refused,
                "guards": {g.name.value: {"passed": g.passed, "score": g.score} for g in guards},
                "top_score": citations[0].score if citations else None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            print(f"[chat] {message!r} -> {d}")
        return {
            "id": uuid.uuid4().hex[:12],
            "message": message,
            "answer": answer,
            "refused": refused,
            "guard_results": guards,
            "citations": citations,
            "generator": generator or "extractive",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def default_llm_client(settings):
    from .generation.llm import LLMClient

    key = settings.llm_key
    if not key:
        return None
    return LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=key,
        base_url=settings.llm_base_url or "https://api.groq.com/openai/v1",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )