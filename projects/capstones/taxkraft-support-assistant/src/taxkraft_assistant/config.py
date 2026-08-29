from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[2]
KB_DIR = PROJECT_DIR / "knowledge_base"
VECTOR_DIR = PROJECT_DIR / "vectors"
EVAL_DIR = PROJECT_DIR / "evaluation"
REPORTS_DIR = EVAL_DIR / "reports"
DATASETS_DIR = EVAL_DIR / "datasets"
WEB_DIR = PROJECT_DIR / "web"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Typed, env-driven configuration. All knobs are optional with safe defaults."""

    # Corpus + store
    kb_dir: Path = field(default_factory=lambda: KB_DIR)
    vector_dir: Path = field(default_factory=lambda: VECTOR_DIR)
    collection_name: str = "taxkraft_corpus"

    # Embeddings + retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = int(os.getenv("TAXKRAFT_CHUNK_SIZE", "350"))
    chunk_overlap: int = int(os.getenv("TAXKRAFT_CHUNK_OVERLAP", "60"))
    top_k: int = int(os.getenv("TAXKRAFT_TOP_K", "5"))
    bm25_weight: float = float(os.getenv("TAXKRAFT_BM25_WEIGHT", "0.30"))
    semantic_weight: float = float(os.getenv("TAXKRAFT_SEMANTIC_WEIGHT", "0.70"))
    rerank_k: int = int(os.getenv("TAXKRAFT_RERANK_K", "8"))

    # Guardrail thresholds
    scope_threshold: float = float(os.getenv("TAXKRAFT_SCOPE_THRESHOLD", "0.42"))
    retrieval_threshold: float = float(os.getenv("TAXKRAFT_RETRIEVAL_THRESHOLD", "0.30"))
    faithfulness_threshold: float = float(os.getenv("TAXKRAFT_FAITHFULNESS_THRESHOLD", "0.30"))
    pii_enabled: bool = field(default_factory=lambda: _bool("TAXKRAFT_PII_ENABLED", True))
    injection_enabled: bool = field(default_factory=lambda: _bool("TAXKRAFT_INJECTION_ENABLED", True))
    scope_enabled: bool = field(default_factory=lambda: _bool("TAXKRAFT_SCOPE_ENABLED", True))

    # Generation
    llm_provider: str = os.getenv("TAXKRAFT_LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("TAXKRAFT_LLM_MODEL", "llama-3.1-8b-instant")
    llm_temperature: float = float(os.getenv("TAXKRAFT_LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("TAXKRAFT_LLM_MAX_TOKENS", "300"))
    max_context_chars: int = int(os.getenv("TAXKRAFT_MAX_CONTEXT_CHARS", "6000"))

    # Logging
    log_requests: bool = field(default_factory=lambda: _bool("TAXKRAFT_LOG_REQUESTS", True))

    def resolved(self) -> "Settings":
        return self

    @property
    def llm_key(self) -> str | None:
        key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        return key or None

    @property
    def llm_base_url(self) -> str | None:
        if self.llm_provider == "groq":
            return os.getenv("TAXKRAFT_GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv(ROOT / ".env")
    load_dotenv()
    return Settings()