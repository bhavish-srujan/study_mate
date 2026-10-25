"""Retrieval tests: resource filtering, ranking, limits, and empty inputs."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ingestion.schemas import DocumentChunk
from rag.retriever import Retriever
from rag.vector_store import ChromaVectorStore
from utils.errors import RetrievalError

chromadb = pytest.importorskip("chromadb")


class FakeEmbedder:
    """Return a fixed query vector so ranking is deterministic."""

    model_name = "fake"

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls = 0

    def embed_documents(self, texts):
        raise AssertionError("Retriever must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector


def _store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=Path("unused"),
        collection_name=f"test_retrieval_{uuid4().hex[:8]}",
        embedding_model="fake",
        client=chromadb.EphemeralClient(),
    )


def _chunk(
    resource_id: str,
    index: int,
    text: str,
    source_name: str,
    page_number: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{resource_id}:{index:04d}",
        resource_id=resource_id,
        text=text,
        chunk_index=index,
        source_type="pdf",
        source_name=source_name,
        page_number=page_number,
    )


def _seeded_store() -> ChromaVectorStore:
    store = _store()
    store.upsert_chunks(
        [
            _chunk("r1", 0, "alpha", "a.pdf", 1),
            _chunk("r1", 1, "beta", "a.pdf", 2),
            _chunk("r2", 0, "gamma", "b.pdf", 1),
        ],
        [[1.0, 0.0], [0.8, 0.6], [1.0, 0.0]],
    )
    return store


def test_retrieval_filters_to_selected_resources_and_ranks() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([1.0, 0.0]))

    hits = retriever.retrieve("alpha", ["r1"])

    assert [hit.chunk_id for hit in hits] == ["r1:0000", "r1:0001"]
    assert {hit.resource_id for hit in hits} == {"r1"}
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score == pytest.approx(0.8, abs=1e-6)
    assert hits[0].source_label() == "a.pdf · p.1 · chunk 0"


def test_top_k_limits_results() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([1.0, 0.0]))

    hits = retriever.retrieve("alpha", ["r1"], top_k=1)

    assert [hit.chunk_id for hit in hits] == ["r1:0000"]


def test_default_top_k_is_used() -> None:
    retriever = Retriever(
        _seeded_store(), FakeEmbedder([1.0, 0.0]), default_top_k=1
    )

    assert len(retriever.retrieve("alpha", ["r1"])) == 1


def test_empty_resource_selection_returns_empty_without_embedding() -> None:
    embedder = FakeEmbedder([1.0, 0.0])
    retriever = Retriever(_seeded_store(), embedder)

    assert retriever.retrieve("alpha", []) == []
    assert embedder.calls == 0


def test_missing_resource_returns_no_hits() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([1.0, 0.0]))

    assert retriever.retrieve("alpha", ["does-not-exist"]) == []


def test_blank_query_is_rejected() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([1.0, 0.0]))

    with pytest.raises(RetrievalError):
        retriever.retrieve("   ", ["r1"])


def test_non_positive_top_k_is_rejected() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([1.0, 0.0]))

    with pytest.raises(RetrievalError):
        retriever.retrieve("alpha", ["r1"], top_k=0)


def test_min_score_filters_weak_matches() -> None:
    retriever = Retriever(_seeded_store(), FakeEmbedder([0.0, 1.0]))

    hits = retriever.retrieve("beta", ["r1"], min_score=0.5)

    assert [hit.chunk_id for hit in hits] == ["r1:0001"]
    assert hits[0].score == pytest.approx(0.6, abs=1e-6)


def test_warm_up_embeds_a_query() -> None:
    embedder = FakeEmbedder([1.0, 0.0])
    retriever = Retriever(_seeded_store(), embedder)

    retriever.warm_up()

    assert embedder.calls == 1
