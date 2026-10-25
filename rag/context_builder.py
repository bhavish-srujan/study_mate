"""Turn ranked chunks into a deduplicated, token-budgeted LLM context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ingestion.chunker import (
    Tokenizer,
    count_tokens,
    default_tokenizer,
    truncate_text,
)
from rag.schemas import ContextPassage
from utils.errors import RetrievalError


@dataclass(frozen=True)
class BuiltContext:
    """Formatted context plus the chunks that actually fit."""

    text: str
    chunks: tuple[ContextPassage, ...]
    token_count: int
    truncated: bool


def _normalize_for_compare(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe(hits: Sequence[ContextPassage]) -> list[ContextPassage]:
    """Drop repeated IDs and near-duplicate/contained passages."""
    kept: list[ContextPassage] = []
    seen_ids: set[str] = set()
    normalized_texts: list[str] = []

    for hit in hits:
        if hit.chunk_id in seen_ids:
            continue

        text = _normalize_for_compare(hit.text)
        is_duplicate = False
        for other in normalized_texts:
            if text == other:
                is_duplicate = True
                break
            shorter, longer = (
                (text, other) if len(text) <= len(other) else (other, text)
            )
            # Overlapping chunks are often contained in a neighbouring chunk.
            if len(shorter) >= 80 and shorter in longer:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        seen_ids.add(hit.chunk_id)
        normalized_texts.append(text)
        kept.append(hit)

    return kept


def build_context(
    hits: Sequence[ContextPassage],
    max_context_tokens: int,
    tokenizer: Tokenizer | None = None,
) -> BuiltContext:
    """Format hits with source labels, respecting the token budget.

    Chunks are considered in ranked order. The first chunk that would exceed
    the budget is truncated; later chunks are dropped.
    """
    if max_context_tokens <= 0:
        raise RetrievalError(
            "max_context_tokens must be greater than 0.",
            user_message="The context limit must be greater than zero.",
        )

    active_tokenizer = tokenizer or default_tokenizer()
    unique = _dedupe(hits)
    truncated = len(unique) < len(hits)

    blocks: list[str] = []
    included: list[ContextPassage] = []
    used = 0

    for hit in unique:
        remaining = max_context_tokens - used
        label = f"[Source: {hit.source_label()}]"
        label_tokens = count_tokens(label, active_tokenizer)

        if remaining <= label_tokens + 1:
            truncated = True
            break

        body_budget = remaining - label_tokens
        body = hit.text
        if count_tokens(body, active_tokenizer) > body_budget:
            body = truncate_text(body, body_budget, active_tokenizer)
            truncated = True

        if not body:
            truncated = True
            break

        block = f"{label}\n{body}"
        block_tokens = count_tokens(block, active_tokenizer)
        if block_tokens > remaining:
            # Tokenizer overhead around the newline/label; trim a little more.
            body = truncate_text(body, max(0, body_budget - 2), active_tokenizer)
            block = f"{label}\n{body}"
            block_tokens = count_tokens(block, active_tokenizer)
            truncated = True

        if not body or block_tokens > remaining:
            break

        blocks.append(block)
        included.append(hit)
        used += block_tokens

        if used >= max_context_tokens:
            if len(included) < len(unique):
                truncated = True
            break

    text = "\n\n".join(blocks)
    # Per-block counts can undercount the separators / BPE merges at block
    # boundaries, so enforce the budget once more on the assembled context.
    encoded = list(active_tokenizer.encode(text)) if text else []
    if len(encoded) > max_context_tokens:
        text = active_tokenizer.decode(encoded[:max_context_tokens])
        truncated = True

    return BuiltContext(
        text=text,
        chunks=tuple(included),
        token_count=count_tokens(text, active_tokenizer),
        truncated=truncated,
    )
