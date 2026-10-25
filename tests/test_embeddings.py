"""Embedding wrapper tests that do not load a real model."""

from __future__ import annotations

from typing import Sequence

import pytest

from rag.embeddings import LocalSentenceTransformerEmbedder
from utils.errors import ConfigurationError, EmbeddingError


def test_query_embeddings_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = LocalSentenceTransformerEmbedder("fake-model")
    calls: list[list[str]] = []

    def fake_embed_documents(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embedder, "embed_documents", fake_embed_documents)

    first = embedder.embed_query("  CPU scheduling  ")
    second = embedder.embed_query("CPU scheduling")

    assert first == second == [0.1, 0.2, 0.3]
    assert calls == [["CPU scheduling"]]
    assert first is not second


def test_query_cache_respects_lru_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = LocalSentenceTransformerEmbedder("fake-model", query_cache_size=1)
    calls: list[str] = []

    def fake_embed_documents(texts: Sequence[str]) -> list[list[float]]:
        calls.extend(texts)
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embedder, "embed_documents", fake_embed_documents)

    embedder.embed_query("one")
    embedder.embed_query("two")
    embedder.embed_query("one")

    assert calls == ["one", "two", "one"]


def test_empty_query_is_rejected() -> None:
    embedder = LocalSentenceTransformerEmbedder("fake-model")

    with pytest.raises(EmbeddingError):
        embedder.embed_query("   ")


def test_invalid_embedding_options_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        LocalSentenceTransformerEmbedder("fake-model", batch_size=0)
    with pytest.raises(ConfigurationError):
        LocalSentenceTransformerEmbedder("fake-model", max_seq_length=0)
