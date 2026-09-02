"""Text chunking helpers used by the RAG notebooks (Phase 11)."""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split `text` into chunks, keeping sentences intact where possible.

    Sentences are grouped until a chunk would exceed ``chunk_size`` characters.
    Over-long sentences are hard-split on word boundaries. Every chunk after
    the first is prefixed with the last ``overlap`` characters of the previous
    chunk (snapped to a word boundary) so meaning is never lost across chunks.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Requires chunk_size > 0 and 0 <= overlap < chunk_size")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    groups: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
            continue
        if current:
            groups.append(current)
            current = ""
        if len(sentence) > chunk_size:
            groups.extend(_hard_split(sentence, chunk_size))
        else:
            current = sentence
    if current:
        groups.append(current)

    return _apply_overlap(groups, overlap)


def _hard_split(text: str, size: int) -> list[str]:
    """Split a too-long single sentence into word-preserving pieces of ~size chars."""
    words = text.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= size:
            current = (current + " " + word).strip()
        else:
            if current:
                pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _apply_overlap(groups: list[str], overlap: int) -> list[str]:
    """Prefix each chunk (after the first) with trailing text of the previous one."""
    if overlap <= 0 or len(groups) < 2:
        return groups
    out = [groups[0]]
    for chunk in groups[1:]:
        prefix = _overlap_prefix(out[-1], overlap)
        out.append((prefix + " " + chunk).strip() if prefix else chunk)
    return out


def _overlap_prefix(previous: str, overlap: int) -> str:
    """Return up to ``overlap`` trailing chars of ``previous``, snapped to a word start."""
    if len(previous) > overlap:
        tail = previous[-overlap:]
        space_at = tail.find(" ")
        return tail[space_at + 1 :].strip() if space_at != -1 else tail.strip()
    return previous.strip()


def chunk_documents(documents: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Chunk many documents, returning a flat list of chunks."""
    chunks: list[str] = []
    for doc in documents:
        chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
    return chunks