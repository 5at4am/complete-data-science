from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import WEB_DIR, load_settings
from .evaluation.run_eval import run_full_eval
from .pipeline import Pipeline, default_llm_client
from .schemas import (
    ChatRequest,
    ChatResponse,
    EvalResponse,
    GuardDiagnostics,
    HealthResponse,
)


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    settings = load_settings()
    pipeline = pipeline or Pipeline.build(
        settings=settings, llm_client=default_llm_client(settings)
    )

    app = FastAPI(
        title="TaxKraft Support Assistant",
        description="Company-scoped, guardrailed RAG chatbot for TaxKraft.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production (set to the deployed UI origin)
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.pipeline = pipeline

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok" if pipeline.store.ready else "not_ingested",
            collection=pipeline.settings.collection_name,
            chunks=pipeline.store.chunk_count or None,
            embedding_model=pipeline.settings.embedding_model,
            llm_backend="llm" if pipeline.llm_available() else "extractive(offline)",
        )

    @app.get("/guardrails/status", response_model=GuardDiagnostics)
    def guard_status():
        return GuardDiagnostics(
            topic_scope_centroid_ready=pipeline.engine.topic_scope.embedder is not None,
            pii_enabled=pipeline.engine.pii.enabled,
            injection_enabled=pipeline.engine.injection.enabled,
            retrieval_threshold=pipeline.settings.retrieval_threshold,
            faithfulness_threshold=pipeline.settings.faithfulness_threshold,
            collection_size=pipeline.store.chunk_count or None,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        out = pipeline.chat_full(req.message)
        return ChatResponse(
            id=out["id"],
            answer=out["answer"],
            refused=out["refused"],
            guard_results=out["guard_results"],
            citations=out["citations"],
            generator=out["generator"],
            latency_ms=out["latency_ms"],
        )

    @app.post("/eval", response_model=EvalResponse)
    def run_eval():
        report_path, summary = run_full_eval(pipeline)
        return EvalResponse(report_path=str(report_path), summary=summary)

    _mount_web(app)
    return app


def _mount_web(app: FastAPI) -> None:
    """Serve the zero-build chat UI from /web when present."""
    if (WEB_DIR / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")


_app: FastAPI | None = None


def get_app() -> FastAPI:
    global _app
    if _app is None:
        _app = create_app()
    return _app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("taxkraft_assistant.api:get_app", host="127.0.0.1", port=8000, reload=False, factory=True)