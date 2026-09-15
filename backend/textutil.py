"""Text cleaning and chunking helpers (pure functions, no I/O)."""

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in _PARAGRAPH_SPLIT.split(text)
    ]
    return "\n\n".join(p for p in paragraphs if p)


def split_chunks(
    text: str, chunk_chars: int = 4000, overlap: int = 200
) -> list[str]:
    """Split *text* into word-boundary chunks with character overlap."""
    if not text or chunk_chars <= 0:
        return []
    if overlap < 0:
        overlap = 0
    if chunk_chars > 1:
        overlap = min(overlap, chunk_chars - 1)
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_chars, total)
        if end < total:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return chunks
