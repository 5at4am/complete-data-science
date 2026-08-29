from __future__ import annotations

from taxkraft_assistant.ingestion.chunker import load_documents, chunk_document
from taxkraft_assistant.retrieval.retriever import BM25, Retriever, tokenize


def test_tokenize_strips_stopwords():
    tokens = tokenize("Please tell me about TaxKraft GST filing in India")
    assert "tell" not in tokens and "gst" in tokens and "taxkraft" in tokens


def test_bm25_ranks_relevant_first():
    corpus = [
        "TaxKraft GST registration for new businesses in India",
        "Weather forecast for Delhi tomorrow",
        "Income tax return filing ITR by TaxKraft chartered accountants",
    ]
    bm = BM25(corpus)
    scores = bm.score(["gst"])
    assert max(scores) == scores[0]


def test_retriever_returns_citations(pipeline):
    out = pipeline.chat_full("How do I contact TaxKraft?")
    assert out["citations"], "expected citations from retrieval"
    top = out["citations"][0]
    assert top.source_url.startswith("https://taxkraft.com")


def test_retriever_topic_specificity(pipeline):
    hits = pipeline.retriever.retrieve("file monthly and quarterly GST returns", top_k=3)
    topics = {h["topic"] for h in hits}
    assert "services_gst" in topics


def test_extractive_generation_grounded(pipeline):
    out = pipeline.chat_full("Does TaxKraft help with GST notices?")
    assert out["answer"]
    assert not out["refused"] or "GST" in out["answer"]


def test_ingest_roundtrip(pipeline, settings):
    docs = load_documents(settings.kb_dir)
    assert len(docs) >= 3
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, 200, 40))
    assert len(chunks) >= 3
    assert pipeline.store.chunk_count == pipeline.store.size()