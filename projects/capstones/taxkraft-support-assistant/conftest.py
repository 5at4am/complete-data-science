from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from taxkraft_assistant.config import Settings  # noqa: E402
from taxkraft_assistant.ingestion.embeddings import DummyEmbedder  # noqa: E402

MINI_KB = {
    "company_overview.md": (
        "<!--\nsource-title: TaxKraft Overview\nsource-url: https://taxkraft.com/\n"
        "-->\n# TaxKraft\n\nTaxKraft is a CA firm in India. "
        "It helps businesses with GST registration, income tax filing and compliance."
    ),
    "services_gst.md": (
        "<!--\nsource-title: GST Services\nsource-url: https://taxkraft.com/service/gst-registration\n"
        "-->\n# GST Services\n\nTaxKraft files monthly, quarterly and annual GST returns. "
        "It also handles GST notices and LUT registration."
    ),
    "contact_support.md": (
        "<!--\nsource-title: Contact\nsource-url: https://taxkraft.com/contact-us\n"
        "-->\n# Contact\n\nCall TaxKraft at +91-8608601620 or email info@taxkraft.com."
    ),
}


def write_mini_kb(root: Path) -> Path:
    kb = root / "kb"
    kb.mkdir(parents=True, exist_ok=True)
    for name, content in MINI_KB.items():
        (kb / name).write_text(content, encoding="utf-8")
    return kb


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        kb_dir=write_mini_kb(tmp_path),
        vector_dir=tmp_path / "vectors",
        collection_name="test_corpus",
        embedding_model="dummy-hash-128",
        chunk_size=120,
        chunk_overlap=30,
        scope_threshold=0.15,
        retrieval_threshold=0.15,
        faithfulness_threshold=0.15,
    )


@pytest.fixture
def dummy_embedder() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture
def pipeline(settings: Settings, dummy_embedder):
    from taxkraft_assistant.pipeline import Pipeline

    p = Pipeline.build(settings=settings, embedder=dummy_embedder, connect=True)
    p.ingest(reset=True)
    return p