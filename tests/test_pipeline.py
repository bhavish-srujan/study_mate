"""End-to-end ingestion pipeline test (real Chroma, fake embeddings)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pytest

from ingestion.chunker import WhitespaceTokenizer
from ingestion.indexer import Indexer
from ingestion.text_loader import load_pasted_text
from rag.vector_store import ChromaVectorStore

chromadb = pytest.importorskip("chromadb")


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_paste_notes_through_index_and_delete() -> None:
    store = ChromaVectorStore(
        persist_dir=Path("unused"),
        collection_name=f"pipeline_{uuid4().hex[:8]}",
        embedding_model="fake-embedder",
        client=chromadb.EphemeralClient(),
    )
    indexer = Indexer(
        store=store,
        embedder=FakeEmbedder(),
        chunk_size_tokens=4,
        overlap_tokens=0,
        tokenizer=WhitespaceTokenizer(),
    )
    resource = load_pasted_text(
        "alpha beta gamma delta epsilon zeta",
        "OS notes",
    )

    first = indexer.index(resource)
    assert first.already_indexed is False
    assert first.resource.chunk_count == 2

    info = store.get_resource_info(resource.resource_id)
    assert info is not None
    assert info.source_name == "OS notes"
    assert info.source_type == "pasted_text"
    assert info.chunk_count == 2

    second = indexer.index(resource)
    assert second.already_indexed is True
    assert store.count_all() == 2

    indexer.delete(resource.resource_id)
    assert store.count_all() == 0
    assert store.list_resources() == []
