"""Indexer tests using an in-memory store and fake embedder."""

from __future__ import annotations

from typing import Sequence

import pytest

from ingestion.chunker import WhitespaceTokenizer
from ingestion.indexer import Indexer
from ingestion.models import LoadedPage, LoadedResource
from ingestion.schemas import DocumentChunk, ResourceInfo
from utils.errors import IndexingError, IngestionError

TOKENIZER = WhitespaceTokenizer()


class FakeStore:
    def __init__(self) -> None:
        self.chunks: dict[str, DocumentChunk] = {}
        self.upsert_calls = 0
        self.deleted: list[str] = []
        self.fail_upsert = False

    def get_resource_info(self, resource_id: str) -> ResourceInfo | None:
        matching = [
            chunk
            for chunk in self.chunks.values()
            if chunk.resource_id == resource_id
        ]
        if not matching:
            return None
        first = matching[0]
        return ResourceInfo(
            resource_id=resource_id,
            source_name=first.source_name,
            source_type=first.source_type,
            chunk_count=len(matching),
        )

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if self.fail_upsert:
            raise RuntimeError("simulated storage failure")
        self.upsert_calls += 1
        for chunk, _vector in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk

    def delete_resource(self, resource_id: str) -> None:
        self.deleted.append(resource_id)
        for key in [
            key
            for key, chunk in self.chunks.items()
            if chunk.resource_id == resource_id
        ]:
            del self.chunks[key]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


def _resource(text: str = "alpha beta gamma delta epsilon zeta") -> LoadedResource:
    return LoadedResource(
        resource_id="res1",
        source_name="notes.txt",
        source_type="pasted_text",
        pages=(LoadedPage(text=text),),
    )


def test_index_stores_chunks_and_reports_new_resource() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    indexer = Indexer(store, embedder, chunk_size_tokens=4, overlap_tokens=0, tokenizer=TOKENIZER)

    result = indexer.index(_resource())

    assert result.already_indexed is False
    assert result.resource.chunk_count == 2
    assert len(store.chunks) == 2
    assert embedder.calls == 1


def test_reindexing_same_content_is_idempotent() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    indexer = Indexer(store, embedder, chunk_size_tokens=4, overlap_tokens=0, tokenizer=TOKENIZER)

    first = indexer.index(_resource())
    second = indexer.index(_resource())

    assert first.already_indexed is False
    assert second.already_indexed is True
    assert second.resource.chunk_count == 2
    assert embedder.calls == 1  # no re-embedding


def test_storage_failure_cleans_up_partial_chunks() -> None:
    store = FakeStore()
    store.fail_upsert = True
    indexer = Indexer(store, FakeEmbedder(), 4, 0, TOKENIZER)

    with pytest.raises(IndexingError):
        indexer.index(_resource())

    assert "res1" in store.deleted
    assert store.chunks == {}


def test_empty_resource_raises_ingestion_error() -> None:
    indexer = Indexer(FakeStore(), FakeEmbedder(), 4, 0, TOKENIZER)

    with pytest.raises(IngestionError):
        indexer.index(_resource("   "))


def test_delete_delegates_to_store() -> None:
    store = FakeStore()
    indexer = Indexer(store, FakeEmbedder(), 4, 0, TOKENIZER)

    indexer.delete("res1")

    assert store.deleted == ["res1"]
