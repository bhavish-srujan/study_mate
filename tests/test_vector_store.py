"""ChromaDB integration tests using an ephemeral client."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ingestion.schemas import DocumentChunk
from rag.vector_store import ChromaVectorStore
from utils.errors import VectorStoreError

chromadb = pytest.importorskip("chromadb")


def _new_collection_name() -> str:
    return f"test_{uuid4().hex[:8]}"


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="r1:0000",
            resource_id="r1",
            text="alpha",
            chunk_index=0,
            source_type="pdf",
            source_name="a.pdf",
            page_number=1,
        ),
        DocumentChunk(
            chunk_id="r1:0001",
            resource_id="r1",
            text="beta",
            chunk_index=1,
            source_type="pdf",
            source_name="a.pdf",
            page_number=2,
        ),
        DocumentChunk(
            chunk_id="r2:0000",
            resource_id="r2",
            text="gamma",
            chunk_index=0,
            source_type="pasted_text",
            source_name="notes",
        ),
    ]


def _store(
    client,
    model: str = "all-MiniLM-L6-v2",
    collection: str | None = None,
) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=Path("unused"),
        collection_name=collection or _new_collection_name(),
        embedding_model=model,
        client=client,
    )


def test_upsert_list_and_filter_by_resource() -> None:
    client = chromadb.EphemeralClient()
    store = _store(client)
    chunks = _chunks()

    store.upsert_chunks(chunks, [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    assert store.count_all() == 3
    resources = store.list_resources()
    assert [item.resource_id for item in resources] == ["r1", "r2"]
    assert resources[0].chunk_count == 2

    info = store.get_resource_info("r1")
    assert info is not None
    assert info.chunk_count == 2
    assert info.source_name == "a.pdf"
    assert store.get_resource_info("missing") is None


def test_delete_resource_removes_only_that_resource() -> None:
    client = chromadb.EphemeralClient()
    store = _store(client)
    store.upsert_chunks(_chunks(), [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    store.delete_resource("r1")

    assert store.count_all() == 1
    assert store.get_resource_info("r1") is None
    assert store.get_resource_info("r2") is not None


def test_chunk_embedding_count_mismatch_is_rejected() -> None:
    store = _store(chromadb.EphemeralClient())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks(_chunks(), [[1.0, 0.0]])


def test_embedding_model_mismatch_is_rejected() -> None:
    client = chromadb.EphemeralClient()
    collection = _new_collection_name()
    _store(client, model="model-a", collection=collection)

    with pytest.raises(VectorStoreError):
        _store(client, model="model-b", collection=collection)


def test_sample_chunks_spreads_across_resources() -> None:
    client = chromadb.EphemeralClient()
    store = _store(client)
    store.upsert_chunks(_chunks(), [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    sampled = store.sample_chunks(["r1", "r2"], limit=2)

    assert [(item.resource_id, item.chunk_index) for item in sampled] == [
        ("r1", 1),
        ("r2", 0),
    ]


def test_sample_chunks_returns_all_when_limit_exceeds_count() -> None:
    client = chromadb.EphemeralClient()
    store = _store(client)
    store.upsert_chunks(_chunks(), [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    sampled = store.sample_chunks(["r1", "r2"], limit=10)

    assert len(sampled) == 3
    assert [item.chunk_id for item in sampled] == ["r1:0000", "r1:0001", "r2:0000"]


def test_sample_chunks_rejects_non_positive_limit() -> None:
    store = _store(chromadb.EphemeralClient())

    with pytest.raises(VectorStoreError):
        store.sample_chunks(["r1"], limit=0)
