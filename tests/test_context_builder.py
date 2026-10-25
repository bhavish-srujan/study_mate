"""Context builder tests: dedupe, labels, ordering, and token budgets."""

from __future__ import annotations

import pytest

from ingestion.chunker import WhitespaceTokenizer, count_tokens
from rag.context_builder import build_context
from rag.schemas import RetrievedChunk
from utils.errors import RetrievalError

TOKENIZER = WhitespaceTokenizer()


def _hit(
    chunk_id: str,
    text: str,
    *,
    chunk_index: int = 0,
    source_name: str = "notes",
    page_number: int | None = 1,
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        resource_id="r1",
        text=text,
        chunk_index=chunk_index,
        source_name=source_name,
        source_type="pasted_text",
        page_number=page_number,
        distance=1.0 - score,
        score=score,
    )


def test_empty_hits_return_empty_context() -> None:
    context = build_context([], 100, TOKENIZER)

    assert context.text == ""
    assert context.chunks == ()
    assert context.token_count == 0
    assert context.truncated is False


def test_identical_text_and_repeated_ids_are_deduped() -> None:
    hits = [
        _hit("c1", "alpha beta gamma"),
        _hit("c2", "alpha beta gamma"),
        _hit("c1", "different text with the same id"),
    ]

    context = build_context(hits, 1000, TOKENIZER)

    assert [hit.chunk_id for hit in context.chunks] == ["c1"]
    assert context.truncated is True


def test_contained_overlapping_chunk_is_deduped() -> None:
    shorter = " ".join(f"word{i}" for i in range(20))
    longer = shorter + " " + " ".join(f"extra{i}" for i in range(10))
    hits = [_hit("c1", shorter), _hit("c2", longer)]

    context = build_context(hits, 1000, TOKENIZER)

    assert [hit.chunk_id for hit in context.chunks] == ["c1"]
    assert context.truncated is True


def test_rank_order_and_source_labels_are_preserved() -> None:
    hits = [
        _hit(
            "c1",
            "first passage",
            source_name="lecture.pdf",
            page_number=2,
        ),
        _hit(
            "c2",
            "second passage",
            chunk_index=1,
            source_name="notes",
            page_number=None,
        ),
    ]

    context = build_context(hits, 1000, TOKENIZER)

    first = context.text.index("[Source: lecture.pdf · p.2 · chunk 0]")
    second = context.text.index("[Source: notes · chunk 1]")
    assert first < second
    assert "first passage" in context.text
    assert "second passage" in context.text
    assert context.truncated is False


def test_long_chunk_is_truncated_to_budget() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    hits = [_hit("c1", text)]

    context = build_context(hits, 30, TOKENIZER)

    assert context.token_count <= 30
    assert count_tokens(context.text, TOKENIZER) == context.token_count
    assert "w0" in context.text
    assert "w99" not in context.text
    assert context.truncated is True
    assert len(context.chunks) == 1


def test_later_chunks_are_dropped_when_budget_is_full() -> None:
    hits = [
        _hit("c1", "one two three four five six"),
        _hit("c2", "seven eight nine ten"),
    ]

    context = build_context(hits, 15, TOKENIZER)

    assert [hit.chunk_id for hit in context.chunks] == ["c1"]
    assert "seven" not in context.text
    assert context.truncated is True


def test_budget_smaller_than_label_returns_empty_context() -> None:
    hits = [_hit("c1", "some useful text")]

    context = build_context(hits, 5, TOKENIZER)

    assert context.text == ""
    assert context.chunks == ()
    assert context.truncated is True


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_non_positive_budget_is_rejected(max_tokens: int) -> None:
    with pytest.raises(RetrievalError):
        build_context([_hit("c1", "text")], max_tokens, TOKENIZER)
