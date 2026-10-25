"""ChromaDB-backed persistent chunk storage."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from config.settings import Settings
from ingestion.schemas import DocumentChunk, ResourceInfo
from rag.schemas import ContextPassage, RetrievedChunk
from utils.errors import VectorStoreError

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_VALID_SOURCE_TYPES = ("pdf", "pasted_text", "youtube")


def _sample_allocations(
    groups: dict[str, list[ContextPassage]],
    limit: int,
) -> dict[str, int]:
    """Allocate sample slots across resources, biased toward larger ones."""
    if len(groups) >= limit:
        chosen = sorted(
            groups,
            key=lambda resource_id: len(groups[resource_id]),
            reverse=True,
        )[:limit]
        return {resource_id: 1 for resource_id in chosen}

    allocations = {resource_id: 1 for resource_id in groups}
    remaining = limit - len(groups)
    while remaining > 0:
        candidates = [
            resource_id
            for resource_id in groups
            if allocations[resource_id] < len(groups[resource_id])
        ]
        if not candidates:
            break
        resource_id = max(
            candidates,
            key=lambda item: len(groups[item]) - allocations[item],
        )
        allocations[resource_id] += 1
        remaining -= 1
    return allocations


def _evenly_spaced(
    items: Sequence[ContextPassage],
    count: int,
) -> list[ContextPassage]:
    """Pick ``count`` items spread across the sequence, including the ends."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]

    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


class ChromaVectorStore:
    """Thin wrapper around one persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
        client: Any | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed. Run `pip install -r requirements.txt`.",
                user_message=(
                    "The vector database is not installed. Run "
                    "`pip install -r requirements.txt` and restart the app."
                ),
            ) from exc

        try:
            if client is None:
                persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(persist_dir))
            else:
                self._client = client

            existing = self._get_collection()
            if existing is None:
                self._collection = self._client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "embedding_model": embedding_model,
                    },
                )
                logger.info("Created Chroma collection %r.", collection_name)
            else:
                stored_model = (existing.metadata or {}).get("embedding_model")
                if stored_model and stored_model != embedding_model:
                    raise VectorStoreError(
                        f"Collection {collection_name!r} was built with embedding "
                        f"model {stored_model!r}, but the app is configured for "
                        f"{embedding_model!r}. Delete the Chroma data or use a new "
                        "CHROMA_COLLECTION and re-index.",
                        user_message=(
                            "The study index was built with a different embedding "
                            "model. Reset the index (delete data/chroma) or change "
                            "CHROMA_COLLECTION, then re-upload your resources."
                        ),
                    )
                self._collection = existing
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not open ChromaDB: {exc}") from exc

    def _get_collection(self) -> Any | None:
        try:
            return self._client.get_collection(name=self._collection_name)
        except Exception:
            return None

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs "
                f"{len(embeddings)}."
            )

        try:
            for start in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[start : start + _BATCH_SIZE]
                vectors = embeddings[start : start + _BATCH_SIZE]
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.to_metadata() for chunk in batch],
                    embeddings=[list(vector) for vector in vectors],
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not store chunks: {exc}") from exc

    def query(
        self,
        query_embedding: Sequence[float],
        resource_ids: Sequence[str],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rank chunks from the selected resources against a query vector."""
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")

        ids = [str(resource_id) for resource_id in resource_ids if resource_id]
        if not ids:
            return []

        try:
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where={"resource_id": {"$in": ids}},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not query the index: {exc}") from exc

        raw_ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            raw_ids, documents, metadatas, distances
        ):
            passage = self._to_passage(chunk_id, document, metadata)
            if passage is None:
                continue

            distance_value = float(distance)
            hits.append(
                RetrievedChunk(
                    **passage.model_dump(),
                    distance=distance_value,
                    score=1.0 - distance_value,
                )
            )

        return hits

    def sample_chunks(
        self,
        resource_ids: Sequence[str],
        limit: int,
    ) -> list[ContextPassage]:
        """Return evenly spaced chunks across the selected resources.

        Used for broad, no-topic coverage when a similarity query is not
        appropriate. Results are ordered by source name and chunk index.
        """
        if limit <= 0:
            raise VectorStoreError("limit must be greater than 0.")

        ids = [str(resource_id) for resource_id in resource_ids if resource_id]
        if not ids:
            return []

        try:
            result = self._collection.get(
                where={"resource_id": {"$in": ids}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not sample indexed chunks: {exc}"
            ) from exc

        groups: dict[str, list[ContextPassage]] = {}
        for chunk_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            passage = self._to_passage(chunk_id, document, metadata)
            if passage is None:
                continue
            groups.setdefault(passage.resource_id, []).append(passage)

        for group in groups.values():
            group.sort(key=lambda item: item.chunk_index)

        total = sum(len(group) for group in groups.values())
        if total <= limit:
            return sorted(
                (item for group in groups.values() for item in group),
                key=lambda item: (item.source_name.lower(), item.chunk_index),
            )

        allocations = _sample_allocations(groups, limit)
        selected: list[ContextPassage] = []
        for resource_id in sorted(groups):
            selected.extend(_evenly_spaced(groups[resource_id], allocations[resource_id]))

        selected.sort(key=lambda item: (item.source_name.lower(), item.chunk_index))
        return selected

    @staticmethod
    def _to_passage(
        chunk_id: Any,
        document: Any,
        metadata: Any,
    ) -> ContextPassage | None:
        if not chunk_id or not document or not metadata:
            return None

        source_type = metadata.get("source_type")
        if source_type not in _VALID_SOURCE_TYPES:
            source_type = "pasted_text"

        return ContextPassage(
            chunk_id=str(chunk_id),
            resource_id=str(metadata.get("resource_id") or ""),
            text=str(document),
            chunk_index=int(metadata.get("chunk_index") or 0),
            source_name=str(metadata.get("source_name") or "(untitled)"),
            source_type=source_type,
            page_number=metadata.get("page_number"),
            start_timestamp=metadata.get("start_timestamp"),
        )

    def get_resource_info(self, resource_id: str) -> ResourceInfo | None:
        try:
            result = self._collection.get(
                where={"resource_id": resource_id},
                include=["metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not read resource {resource_id!r}: {exc}"
            ) from exc

        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return self._to_resource_info(resource_id, metadatas[0], len(metadatas))

    def list_resources(self) -> list[ResourceInfo]:
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Could not list resources: {exc}") from exc

        metadatas = result.get("metadatas") or []
        counts: Counter[str] = Counter()
        first_seen: dict[str, dict[str, Any]] = {}

        for metadata in metadatas:
            if not metadata:
                continue
            resource_id = metadata.get("resource_id")
            if not resource_id:
                continue
            key = str(resource_id)
            counts[key] += 1
            first_seen.setdefault(key, metadata)

        resources = [
            self._to_resource_info(
                resource_id,
                first_seen[resource_id],
                count,
            )
            for resource_id, count in counts.items()
        ]
        return sorted(resources, key=lambda item: item.source_name.lower())

    @staticmethod
    def _to_resource_info(
        resource_id: str,
        metadata: dict[str, Any],
        chunk_count: int,
    ) -> ResourceInfo:
        source_type = metadata.get("source_type")
        if source_type not in _VALID_SOURCE_TYPES:
            source_type = "pasted_text"
        return ResourceInfo(
            resource_id=resource_id,
            source_name=str(metadata.get("source_name") or "(untitled)"),
            source_type=source_type,
            chunk_count=chunk_count,
        )

    def delete_resource(self, resource_id: str) -> None:
        try:
            self._collection.delete(where={"resource_id": resource_id})
        except Exception as exc:
            raise VectorStoreError(
                f"Could not delete resource {resource_id!r}: {exc}"
            ) from exc

    def count_all(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Could not count chunks: {exc}") from exc


def build_vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )
