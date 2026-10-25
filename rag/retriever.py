"""Resource-filtered vector retrieval."""

from __future__ import annotations

import logging
from typing import Sequence

from config.settings import Settings
from rag.embeddings import Embedder
from rag.schemas import ContextPassage, RetrievedChunk
from rag.vector_store import ChromaVectorStore
from utils.errors import RetrievalError, StudyMateError

logger = logging.getLogger(__name__)


class Retriever:
    """Embed a query and rank chunks from the selected resources only."""

    def __init__(
        self,
        store: ChromaVectorStore,
        embedder: Embedder,
        default_top_k: int = 10,
    ) -> None:
        if default_top_k <= 0:
            raise RetrievalError("default_top_k must be greater than 0.")
        self._store = store
        self._embedder = embedder
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        resource_ids: Sequence[str],
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks from ``resource_ids`` for ``query``.

        Negative/weak matches are dropped by default (cosine similarity below
        ``min_score``). An empty resource selection returns no results rather
        than searching the whole index.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            raise RetrievalError(
                "Retrieval query is empty.",
                user_message="Enter a topic or question to search for.",
            )

        selected = [str(resource_id) for resource_id in resource_ids if resource_id]
        if not selected:
            return []

        limit = self._default_top_k if top_k is None else top_k
        if limit <= 0:
            raise RetrievalError(
                "top_k must be greater than 0.",
                user_message="The retrieval limit must be greater than zero.",
            )

        try:
            query_vector = self._embedder.embed_query(clean_query)
        except StudyMateError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Could not embed the retrieval query: {exc}"
            ) from exc

        hits = self._store.query(query_vector, selected, limit)

        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            if hit.score < min_score:
                continue
            unique.append(hit)

        logger.info(
            "Retrieved %d chunks for %r from %d resource(s) (top_k=%d).",
            len(unique),
            clean_query,
            len(selected),
            limit,
        )
        return unique

    def warm_up(self) -> None:
        """Load the embedding model and prime its query cache."""
        try:
            self._embedder.embed_query("studymate retrieval warmup")
        except StudyMateError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Could not warm up the embedding model: {exc}"
            ) from exc

    def sample_chunks(
        self,
        resource_ids: Sequence[str],
        limit: int | None = None,
    ) -> list[ContextPassage]:
        """Return representative chunks for broad, no-topic generation."""
        selected = [
            str(resource_id) for resource_id in resource_ids if resource_id
        ]
        if not selected:
            return []

        sample_limit = self._default_top_k * 2 if limit is None else limit
        if sample_limit <= 0:
            raise RetrievalError(
                "limit must be greater than 0.",
                user_message="The coverage limit must be greater than zero.",
            )
        return self._store.sample_chunks(selected, sample_limit)


def build_retriever(
    settings: Settings,
    store: ChromaVectorStore,
    embedder: Embedder,
) -> Retriever:
    return Retriever(
        store=store,
        embedder=embedder,
        default_top_k=settings.retrieval_top_k,
    )
