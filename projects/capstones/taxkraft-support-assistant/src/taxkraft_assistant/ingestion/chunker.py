from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str
    topic: str
    char_count: int
    index: int


def parse_provenance(markdown: str) -> dict[str, str]:
    """Extract the provenance HTML comment block, if present."""
    m = re.search(r"<!--(.*?)-->", markdown, flags=re.DOTALL)
    meta: dict[str, str] = {}
    if not m:
        return meta
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_").replace("-", "_")
            meta[key] = value.strip()
    return meta


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def topic_from_path(path: Path) -> str:
    """Derive a coarse topic from the markdown filename."""
    return _slugify(path.stem.replace(".md", ""))


def strip_provenance(markdown: str) -> str:
    """Remove the provenance comment block so it does not pollute chunks."""
    return re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL).strip()


def load_documents(kb_dir: Path, excluded: set[str] | None = None) -> list[dict]:
    """Read every markdown file under kb_dir as a document dict.

    A document is: {text, source_url, source_title, topic, path}.
    """
    excluded = excluded or set()
    docs: list[dict] = []
    files = sorted(kb_dir.rglob("*.md"))
    for f in files:
        relative = f.relative_to(kb_dir).as_posix()
        if relative in excluded or "crawled" in f.parts and f.name == "README.md":
            continue
        raw = f.read_text(encoding="utf-8")
        meta = parse_provenance(raw)
        docs.append(
            {
                "text": strip_provenance(raw),
                "source_url": meta.get("source_url", "https://taxkraft.com"),
                "source_title": meta.get("source_title", f.stem.replace("_", " ").title()),
                "topic": topic_from_path(f),
                "path": relative,
            }
        )
    return docs


def split_sections(markdown: str, min_len: int = 140) -> list[str]:
    """Split markdown on headings first (best chunks), keeping heading context."""
    sections: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    merged: list[str] = []
    buffer = ""
    for sec in sections:
        if len(sec.strip()) < min_len and buffer:
            buffer += "\n" + sec
            continue
        if buffer:
            merged.append(buffer.strip())
        buffer = sec.strip()
    if buffer.strip():
        merged.append(buffer.strip())
    return [m for m in merged if m]


def _split_long(paragraph: str, size: int, overlap: int) -> list[str]:
    """Split a long block into overlapping windows on sentence boundaries."""
    if len(paragraph) <= size:
        return [paragraph.strip()] if paragraph.strip() else []
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent).strip()
            continue
        if current:
            chunks.append(current)
        # sentences longer than size get hard-cut
        while len(sent) > size:
            chunks.append(sent[:size].strip())
            sent = sent[size - overlap :]
        current = sent
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def chunk_markdown(
    markdown: str, chunk_size: int = 350, overlap: int = 60, min_len: int = 60
) -> list[str]:
    sections = split_sections(markdown, min_len=140)
    # A short section (e.g. a bare heading that opens a page) is glued to the
    # following section so heading context is never lost.
    merged: list[str] = []
    buffer = ""
    for sec in sections:
        if buffer and len(buffer) >= min_len and len(sec) >= min_len:
            merged.append(buffer)
            buffer = sec
        else:
            buffer = (buffer + "\n" + sec).strip()
    if buffer.strip():
        merged.append(buffer.strip())

    chunks: list[str] = []
    for section in merged:
        for part in _split_long(section, chunk_size, overlap):
            text = part.strip()
            if len(text) >= min_len:
                chunks.append(text)
    return chunks


def chunk_document(
    doc: dict, chunk_size: int = 350, overlap: int = 60
) -> list[Chunk]:
    pieces = chunk_markdown(doc["text"], chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            text=p,
            source_url=doc["source_url"],
            source_title=doc["source_title"],
            topic=doc["topic"],
            char_count=len(p),
            index=i,
        )
        for i, p in enumerate(pieces)
    ]