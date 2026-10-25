"""Chunking tests: window size, overlap, provenance, and edge cases."""

from __future__ import annotations

import pytest

from ingestion.chunker import WhitespaceTokenizer, chunk_resource, split_text
from ingestion.models import LoadedPage, LoadedResource
from utils.errors import IngestionError

TOKENIZER = WhitespaceTokenizer()


def _resource(pages: list[LoadedPage]) -> LoadedResource:
    return LoadedResource(
        resource_id="res1",
        source_name="notes.txt",
        source_type="pasted_text",
        pages=tuple(pages),
    )


def test_short_text_produces_one_chunk() -> None:
    chunks = split_text("one two three", 10, 2, TOKENIZER)
    assert chunks == ["one two three"]


def test_overlap_produces_expected_windows() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    chunks = split_text(text, chunk_size_tokens=10, overlap_tokens=3, tokenizer=TOKENIZER)

    assert len(chunks) == 4
    assert chunks[0].split() == [f"w{i}" for i in range(10)]
    assert chunks[1].split()[0] == "w7"
    # Three words of overlap between consecutive full windows.
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w24"


def test_no_overlap_keeps_all_content() -> None:
    text = " ".join(f"w{i}" for i in range(11))
    chunks = split_text(text, 10, 0, TOKENIZER)
    assert len(chunks) == 2
    assert chunks[1] == "w10"


def test_sentence_aware_chunks_keep_sentences_intact() -> None:
    text = (
        "Alpha beta gamma. Delta epsilon zeta. "
        "Eta theta iota. Kappa lambda mu."
    )
    chunks = split_text(text, chunk_size_tokens=8, overlap_tokens=3, tokenizer=TOKENIZER)

    assert chunks == [
        "Alpha beta gamma. Delta epsilon zeta.",
        "Delta epsilon zeta. Eta theta iota.",
        "Eta theta iota. Kappa lambda mu.",
    ]
    assert all(chunk.endswith(".") for chunk in chunks)
    assert all(not chunk.startswith("eta") for chunk in chunks)


def test_sentence_aware_overlap_reuses_whole_sentences() -> None:
    text = "One two three. Four five six. Seven eight nine."
    chunks = split_text(text, chunk_size_tokens=6, overlap_tokens=3, tokenizer=TOKENIZER)

    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[0] == "One two three. Four five six."
    assert chunks[1] == "Four five six. Seven eight nine."


def test_paragraph_breaks_survive_when_a_chunk_spans_paragraphs() -> None:
    text = "First sentence. Second sentence.\n\nThird sentence. Fourth sentence."
    chunks = split_text(text, chunk_size_tokens=10, overlap_tokens=2, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    assert "\n\n" in chunks[0]
    assert chunks[0] == text


def test_blank_text_returns_no_chunks() -> None:
    assert split_text("   \n\t  ", 10, 2, TOKENIZER) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_window_settings_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("some text", chunk_size, overlap, TOKENIZER)


def test_chunk_resource_preserves_page_numbers_and_order() -> None:
    resource = _resource(
        [
            LoadedPage(text="alpha beta gamma", page_number=1),
            LoadedPage(text="delta epsilon", page_number=2),
        ]
    )
    chunks = chunk_resource(resource, 10, 2, TOKENIZER)

    assert [chunk.chunk_id for chunk in chunks] == ["res1:0000", "res1:0001"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert all(chunk.source_name == "notes.txt" for chunk in chunks)
    assert all(chunk.source_type == "pasted_text" for chunk in chunks)


def test_chunk_resource_keeps_timestamp_metadata() -> None:
    resource = LoadedResource(
        resource_id="video1",
        source_name="Lecture 1",
        source_type="youtube",
        pages=(LoadedPage(text="operating systems", start_timestamp="00:01:05"),),
    )
    chunks = chunk_resource(resource, 10, 2, TOKENIZER)

    assert chunks[0].start_timestamp == "00:01:05"
    assert "start_timestamp" in chunks[0].to_metadata()


def test_chunk_metadata_omits_absent_provenance() -> None:
    resource = _resource([LoadedPage(text="pasted notes")])
    chunk = chunk_resource(resource, 10, 2, TOKENIZER)[0]
    metadata = chunk.to_metadata()

    assert "page_number" not in metadata
    assert "start_timestamp" not in metadata
    assert metadata["resource_id"] == "res1"
    assert metadata["chunk_index"] == 0


def test_resource_without_text_raises() -> None:
    with pytest.raises(IngestionError):
        chunk_resource(_resource([LoadedPage(text="   ")]), 10, 2, TOKENIZER)
