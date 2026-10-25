"""Orchestrate chunking, embedding, and storage for one resource."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from ingestion.chunker import Tokenizer, chunk_resource
from ingestion.models import LoadedResource
from ingestion.schemas import DocumentChunk, ResourceInfo
from utils.errors import EmbeddingError, IndexingError

logger = logging.getLogger(__name__)


class ChunkStore(Protocol):
    def get_resource_info(self, resource_id: str) -> ResourceInfo | None: ...

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def delete_resource(self, resource_id: str) -> None: ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IndexResult:
    resource: ResourceInfo
    already_indexed: bool


class Indexer:
    """Index a resource exactly once per content hash."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        chunk_size_tokens: int,
        overlap_tokens: int,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunk_size_tokens = chunk_size_tokens
        self._overlap_tokens = overlap_tokens
        self._tokenizer = tokenizer

    def index(self, resource: LoadedResource) -> IndexResult:
        existing = self._store.get_resource_info(resource.resource_id)
        if existing is not None:
            logger.info(
                "Resource %s (%s) is already indexed with %d chunks.",
                resource.resource_id,
                resource.source_name,
                existing.chunk_count,
            )
            return IndexResult(resource=existing, already_indexed=True)

        chunks = chunk_resource(
            resource,
            chunk_size_tokens=self._chunk_size_tokens,
            overlap_tokens=self._overlap_tokens,
            tokenizer=self._tokenizer,
        )

        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingError(
                f"Embedding count mismatch: {len(vectors)} vectors for "
                f"{len(chunks)} chunks."
            )
        if any(not vector for vector in vectors):
            raise EmbeddingError("The embedding model returned an empty vector.")

        try:
            self._store.upsert_chunks(chunks, vectors)
        except Exception:
            logger.exception(
                "Indexing failed for %s; removing partial chunks.",
                resource.source_name,
            )
            try:
                self._store.delete_resource(resource.resource_id)
            except Exception:
                logger.exception(
                    "Could not clean up partial chunks for %s.",
                    resource.resource_id,
                )
            raise IndexingError(
                f"Failed to store chunks for {resource.source_name!r}."
            ) from None

        logger.info(
            "Indexed %s (%s): %d chunks.",
            resource.source_name,
            resource.resource_id,
            len(chunks),
        )
        return IndexResult(
            resource=ResourceInfo(
                resource_id=resource.resource_id,
                source_name=resource.source_name,
                source_type=resource.source_type,
                chunk_count=len(chunks),
            ),
            already_indexed=False,
        )

    def delete(self, resource_id: str) -> None:
        self._store.delete_resource(resource_id)
        logger.info("Deleted resource %s from the index.", resource_id)
