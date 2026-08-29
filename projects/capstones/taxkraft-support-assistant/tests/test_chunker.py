from __future__ import annotations

from taxkraft_assistant.ingestion.chunker import (
    chunk_document,
    chunk_markdown,
    parse_provenance,
    strip_provenance,
)


def test_parse_provenance():
    md = "<!--\nsource-title: GST\nsource-url: https://x\n-->\nbody"
    meta = parse_provenance(md)
    assert meta["source_title"] == "GST"
    assert meta["source_url"] == "https://x"


def test_strip_provenance_removes_comment():
    md = "<!--\nsource-url: x\n-->\n# Hello\nWorld"
    out = strip_provenance(md)
    assert "source-url" not in out
    assert "Hello" in out


def test_chunk_markdown_respects_size():
    body = "word " * 500  # ~2500 chars
    chunks = chunk_markdown(body, chunk_size=200, overlap=40)
    assert chunks
    assert all(len(c) <= 200 + 40 for c in chunks)
    assert len(chunks) > 1


def test_chunk_document_metadata():
    doc = {
        "text": "Paragraph one with enough length to survive the minimum. " * 20,
        "source_url": "https://taxkraft.com/x",
        "source_title": "X",
        "topic": "services_gst",
    }
    chunks = chunk_document(doc, chunk_size=150, overlap=30)
    assert chunks
    assert all(c.source_url == "https://taxkraft.com/x" for c in chunks)
    assert all(c.source_title == "X" and c.topic == "services_gst" for c in chunks)


def test_heading_sections_kept_intact():
    md = "# A\n\nshort\n\n## B\n\n" + ("body " * 60)
    chunks = chunk_markdown(md, chunk_size=200, overlap=40)
    assert any("A" in c for c in chunks)
    assert any("B" in c for c in chunks)