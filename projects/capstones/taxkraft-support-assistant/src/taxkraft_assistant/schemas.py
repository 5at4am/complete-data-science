from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class GuardType(str, Enum):
    TOPIC_SCOPE = "topic_scope"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    RETRIEVAL_CONFIDENCE = "retrieval_confidence"
    FAITHFULNESS = "faithfulness"


class GuardStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class GuardResult(BaseModel):
    name: GuardType
    status: GuardStatus
    passed: bool
    score: float = 0.0
    threshold: float | None = None
    reason: str = ""
    detail: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: str | None = Field(
        default=None, max_length=64, description="Optional hashed session id (never raw PII)"
    )


class RetrievedChunk(BaseModel):
    text: str
    source_url: str
    source_title: str
    topic: str
    score: float
    char_count: int


class ChatResponse(BaseModel):
    id: str
    answer: str
    refused: bool
    guard_results: list[GuardResult]
    citations: list[RetrievedChunk]
    generator: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    collection: str
    chunks: int | None
    embedding_model: str
    llm_backend: str


class EvalResponse(BaseModel):
    report_path: str
    summary: dict


class GuardDiagnostics(BaseModel):
    topic_scope_centroid_ready: bool
    pii_enabled: bool
    injection_enabled: bool
    retrieval_threshold: float
    faithfulness_threshold: float
    collection_size: int | None