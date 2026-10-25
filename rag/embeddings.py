"""Local sentence-transformers embeddings.

The model is loaded lazily so the Streamlit app can start quickly and tests
that do not touch embeddings never download a model.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Protocol, Sequence

from config.settings import Settings
from utils.errors import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_QUERY_CACHE_SIZE = 128


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalSentenceTransformerEmbedder:
    """Encode text with one local model, used for both documents and queries."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        device: str | None = None,
        max_seq_length: int = 1024,
        query_cache_size: int = _QUERY_CACHE_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ConfigurationError("Embedding batch size must be greater than 0.")
        if max_seq_length <= 0:
            raise ConfigurationError("Embedding max sequence length must be greater than 0.")
        if query_cache_size < 0:
            raise ConfigurationError("Query cache size cannot be negative.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or None
        self.max_seq_length = max_seq_length
        self.query_cache_size = query_cache_size
        self._model = None
        self._dimension = 0
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._cache_lock = Lock()

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "sentence-transformers is not installed. "
                "Run `pip install -r requirements.txt`."
            ) from exc

        try:
            logger.info("Loading embedding model %s ...", self.model_name)
            try:
                # Prefer the local Hugging Face cache. This avoids a remote
                # metadata check on every process start once the model exists.
                model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception:
                logger.info(
                    "Embedding model %s is not cached locally; downloading ...",
                    self.model_name,
                )
                model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model {self.model_name!r}: {exc}",
                user_message=(
                    "The local embedding model could not be loaded. Check your "
                    "installation and your internet connection for the first "
                    "model download."
                ),
            ) from exc

        model.max_seq_length = self.max_seq_length
        self._model = model
        if hasattr(model, "get_embedding_dimension"):
            dimension = model.get_embedding_dimension()
        else:
            dimension = model.get_sentence_embedding_dimension()
        self._dimension = int(dimension or 0)
        logger.info(
            "Embedding model ready: %s (dimension=%d, max_seq_length=%d).",
            self.model_name,
            self._dimension,
            self.max_seq_length,
        )
        return model

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._load()
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if any(not text.strip() for text in items):
            raise EmbeddingError("Cannot embed empty text.")

        model = self._load()
        try:
            vectors = model.encode(
                items,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query, reusing recent vectors for repeated searches."""
        key = (text or "").strip()
        if not key:
            raise EmbeddingError("Cannot embed empty text.")

        if self.query_cache_size:
            with self._cache_lock:
                cached = self._query_cache.get(key)
                if cached is not None:
                    self._query_cache.move_to_end(key)
                    return list(cached)

        vectors = self.embed_documents([key])
        if not vectors:
            raise EmbeddingError("The embedding model returned no vector.")
        vector = vectors[0]

        if self.query_cache_size:
            with self._cache_lock:
                self._query_cache[key] = tuple(vector)
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self.query_cache_size:
                    self._query_cache.popitem(last=False)

        return list(vector)


def build_embedder(settings: Settings) -> Embedder:
    """Construct the single embedding provider used in v1."""
    if settings.embedding_provider != "local":
        raise ConfigurationError(
            "Only local embeddings are supported in v1. "
            "Set EMBEDDING_PROVIDER=local."
        )
    return LocalSentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device or None,
        max_seq_length=settings.embedding_max_seq_length,
    )
