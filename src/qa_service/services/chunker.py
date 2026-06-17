"""Token-aware recursive text chunker for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk of text with its position in the source document."""

    index: int
    text: str
    token_count: int


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Return a cached tiktoken encoder."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` using the ``cl100k_base`` encoding."""
    return len(_encoder().encode(text))


_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _split_recursive(text: str, max_tokens: int, separators: tuple[str, ...]) -> list[str]:
    """Split ``text`` into pieces no larger than ``max_tokens`` tokens.

    The function tries each separator in order, falling back to a hard
    character split for any piece that still exceeds the budget.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        encoder = _encoder()
        token_ids = encoder.encode(text)
        return [
            encoder.decode(token_ids[i : i + max_tokens])
            for i in range(0, len(token_ids), max_tokens)
        ]

    sep, *rest = separators
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, max_tokens, tuple(rest))

    pieces: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for part in parts:
        candidate = part + (sep if part is not parts[-1] else "")
        candidate_tokens = count_tokens(candidate)
        if candidate_tokens > max_tokens:
            if buffer:
                pieces.append("".join(buffer))
                buffer, buffer_tokens = [], 0
            pieces.extend(_split_recursive(candidate, max_tokens, tuple(rest)))
            continue
        if buffer_tokens + candidate_tokens > max_tokens:
            pieces.append("".join(buffer))
            buffer, buffer_tokens = [candidate], candidate_tokens
        else:
            buffer.append(candidate)
            buffer_tokens += candidate_tokens
    if buffer:
        pieces.append("".join(buffer))
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split ``text`` into overlapping token-bounded chunks.

    Args:
        text: Source text.
        chunk_size: Target maximum token count per chunk.
        overlap: Number of tokens of overlap between consecutive chunks.

    Returns:
        Ordered list of :class:`Chunk` objects.

    Raises:
        ValueError: If ``chunk_size`` is non-positive or ``overlap`` is
            greater than or equal to ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    base_pieces = _split_recursive(cleaned, chunk_size, _SEPARATORS)
    if overlap == 0 or len(base_pieces) <= 1:
        return [
            Chunk(index=i, text=piece, token_count=count_tokens(piece))
            for i, piece in enumerate(base_pieces)
        ]

    encoder = _encoder()
    chunks: list[Chunk] = []
    previous_tail: list[int] = []
    for i, piece in enumerate(base_pieces):
        token_ids = encoder.encode(piece)
        if previous_tail:
            token_ids = previous_tail + token_ids
            if len(token_ids) > chunk_size:
                token_ids = token_ids[-chunk_size:]
        chunk_text_str = encoder.decode(token_ids).strip()
        chunks.append(Chunk(index=i, text=chunk_text_str, token_count=len(token_ids)))
        previous_tail = token_ids[-overlap:] if overlap > 0 else []
    return chunks
