"""Split extracted text into overlapping, metadata-bearing chunks.

The default strategy is sentence/paragraph-aware: text is split at sentence
boundaries first, then packed into token-bounded chunks so a sentence is not
cut in half just because it crossed a fixed window boundary. Paragraph breaks
are preserved between chunks when possible. Text without any sentence or
paragraph structure falls back to the original fixed token windows.

Token counting uses ``tiktoken`` when available. If the encoding cannot be
loaded (for example, offline on first run), a whitespace tokenizer is used as a
documented approximation so indexing still works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, Sequence

from ingestion.models import LoadedResource
from ingestion.normalization import normalize_text
from ingestion.schemas import DocumentChunk
from utils.errors import IngestionError

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", re.UNICODE
)
_PARAGRAPH_BREAK = re.compile(r"\n{2,}")
_HORIZONTAL_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _TextUnit:
    """A sentence-sized piece that should stay intact when possible."""

    text: str
    paragraph_index: int


class Tokenizer(Protocol):
    """Minimal interface needed for token-window chunking."""

    def encode(self, text: str) -> Sequence[Any]: ...

    def decode(self, tokens: Sequence[Any]) -> str: ...


class TiktokenTokenizer:
    """Approximate tokenizer shared with the LLM (cl100k_base)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> Sequence[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))


class WhitespaceTokenizer:
    """Offline fallback: one whitespace-delimited word counts as one token."""

    def encode(self, text: str) -> Sequence[str]:
        return text.split()

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(str(token) for token in tokens)


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    try:
        return TiktokenTokenizer()
    except Exception:  # missing package or no cached encoding / offline
        return WhitespaceTokenizer()


def _paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK.split(text)
        if paragraph.strip()
    ]


def _sentences(paragraph: str) -> list[str]:
    flattened = _HORIZONTAL_SPACE.sub(" ", paragraph).strip()
    if not flattened:
        return []
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(flattened)
        if sentence.strip()
    ]


def _build_units(
    text: str,
    chunk_size_tokens: int,
    tokenizer: Tokenizer,
) -> list[_TextUnit]:
    """Split text into sentence units, breaking only oversized sentences."""
    units: list[_TextUnit] = []
    for paragraph_index, paragraph in enumerate(_paragraphs(text)):
        for sentence in _sentences(paragraph):
            tokens = list(tokenizer.encode(sentence))
            if not tokens:
                continue
            if len(tokens) <= chunk_size_tokens:
                units.append(_TextUnit(sentence, paragraph_index))
                continue

            # A single sentence larger than the target chunk cannot be kept
            # intact; fall back to token windows for that sentence only.
            for start in range(0, len(tokens), chunk_size_tokens):
                piece = normalize_text(
                    tokenizer.decode(tokens[start : start + chunk_size_tokens])
                )
                if piece:
                    units.append(_TextUnit(piece, paragraph_index))
    return units


def _join_units(units: Sequence[_TextUnit]) -> str:
    if not units:
        return ""
    parts = [units[0].text]
    for previous, current in zip(units, units[1:]):
        separator = "\n\n" if current.paragraph_index != previous.paragraph_index else " "
        parts.extend((separator, current.text))
    return "".join(parts)


def _token_window_chunks(
    clean: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> list[str]:
    """Original fixed-window strategy, used for unstructured text."""
    tokens = list(tokenizer.encode(clean))
    if not tokens:
        return []

    step = chunk_size_tokens - overlap_tokens
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size_tokens]
        piece = normalize_text(tokenizer.decode(window))
        if piece:
            chunks.append(piece)
        if start + chunk_size_tokens >= len(tokens):
            break
    return chunks


def _sentence_aware_chunks(
    clean: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> list[str]:
    """Pack sentence units into chunks with sentence-level overlap."""
    units = _build_units(clean, chunk_size_tokens, tokenizer)
    if not units:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units):
            candidate = _join_units(units[start : end + 1])
            if count_tokens(candidate, tokenizer) > chunk_size_tokens:
                break
            end += 1

        piece = normalize_text(_join_units(units[start:end]))
        if piece:
            chunks.append(piece)
        if end >= len(units):
            break

        # Start the next chunk with as many trailing sentence units as fit in
        # the overlap budget. If no whole sentence fits, prefer no overlap over
        # cutting a sentence in half.
        new_start = end
        for candidate_start in range(end - 1, start - 1, -1):
            suffix = _join_units(units[candidate_start:end])
            if count_tokens(suffix, tokenizer) <= overlap_tokens:
                new_start = candidate_start
            else:
                break

        # Always leave room for at least one new unit; otherwise the next
        # chunk would contain overlap only and make no forward progress.
        while new_start < end:
            candidate = _join_units(units[new_start : end + 1])
            if count_tokens(candidate, tokenizer) <= chunk_size_tokens:
                break
            new_start += 1

        start = new_start if new_start < end else end

    return chunks


def split_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Split text into overlapping, sentence-aware chunks."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_size_tokens")

    clean = normalize_text(text)
    if not clean:
        return []

    active_tokenizer = tokenizer or default_tokenizer()

    # Preserve the exact old behavior for unstructured word lists. This keeps
    # fallback text deterministic and avoids inventing sentence boundaries.
    if not _SENTENCE_BOUNDARY.search(clean) and "\n\n" not in clean:
        return _token_window_chunks(
            clean,
            chunk_size_tokens,
            overlap_tokens,
            active_tokenizer,
        )

    chunks = _sentence_aware_chunks(
        clean,
        chunk_size_tokens,
        overlap_tokens,
        active_tokenizer,
    )
    return chunks or _token_window_chunks(
        clean,
        chunk_size_tokens,
        overlap_tokens,
        active_tokenizer,
    )


def chunk_resource(
    resource: LoadedResource,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer | None = None,
) -> list[DocumentChunk]:
    """Chunk every page of a resource, preserving page/timestamp provenance."""
    chunks: list[DocumentChunk] = []
    index = 0

    for page in resource.pages:
        for piece in split_text(
            page.text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer=tokenizer,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{resource.resource_id}:{index:04d}",
                    resource_id=resource.resource_id,
                    text=piece,
                    chunk_index=index,
                    source_type=resource.source_type,
                    source_name=resource.source_name,
                    page_number=page.page_number,
                    start_timestamp=page.start_timestamp,
                )
            )
            index += 1

    if not chunks:
        raise IngestionError(
            f"No text chunks could be produced from {resource.source_name!r}."
        )

    return chunks


def count_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    """Count tokens with the configured tokenizer."""
    if not text:
        return 0
    active_tokenizer = tokenizer or default_tokenizer()
    return len(list(active_tokenizer.encode(text)))


def truncate_text(
    text: str,
    max_tokens: int,
    tokenizer: Tokenizer | None = None,
) -> str:
    """Return at most ``max_tokens`` tokens of normalized text."""
    if max_tokens <= 0:
        return ""
    active_tokenizer = tokenizer or default_tokenizer()
    tokens = list(active_tokenizer.encode(normalize_text(text)))
    return normalize_text(active_tokenizer.decode(tokens[:max_tokens]))
