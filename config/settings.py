"""Typed application settings loaded from environment variables / ``.env``.

Only local embeddings are supported in v1, per the PRD's "pick one provider"
rule. The LLM is any OpenAI-compatible Chat Completions endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `override=False` keeps real environment variables (CI, Streamlit secrets)
# ahead of the local `.env` file.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc


def _path(name: str, default: str) -> Path:
    path = Path(_str(name, default)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    llm_api_key: str = field(repr=False)
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_max_completion_tokens: int
    llm_reasoning_effort: str

    embedding_provider: str
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    embedding_max_seq_length: int

    chroma_dir: Path
    chroma_collection: str
    sqlite_path: Path

    chunk_size_tokens: int
    chunk_overlap_tokens: int

    retrieval_top_k: int
    max_context_tokens: int
    max_questions: int
    default_question_count: int
    max_pdfs: int

    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            llm_api_key=_str("LLM_API_KEY", ""),
            llm_base_url=_str("LLM_BASE_URL", ""),
            llm_model=_str("LLM_MODEL", ""),
            llm_temperature=_float("LLM_TEMPERATURE", 0.2),
            llm_max_completion_tokens=_int("LLM_MAX_COMPLETION_TOKENS", 2048),
            llm_reasoning_effort=_str("LLM_REASONING_EFFORT", "medium").lower(),
            embedding_provider=_str("EMBEDDING_PROVIDER", "local").lower(),
            embedding_model=_str("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=_str("EMBEDDING_DEVICE", ""),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 8),
            embedding_max_seq_length=_int("EMBEDDING_MAX_SEQ_LENGTH", 1024),
            chroma_dir=_path("CHROMA_DIR", "data/chroma"),
            chroma_collection=_str("CHROMA_COLLECTION", "studymate_bge_m3"),
            sqlite_path=_path("SQLITE_PATH", "data/studymate.db"),
            chunk_size_tokens=_int("CHUNK_SIZE_TOKENS", 512),
            chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 96),
            retrieval_top_k=_int("RETRIEVAL_TOP_K", 10),
            max_context_tokens=_int("MAX_CONTEXT_TOKENS", 6000),
            max_questions=_int("MAX_QUESTIONS", 30),
            default_question_count=_int("DEFAULT_QUESTION_COUNT", 10),
            max_pdfs=_int("MAX_PDFS", 5),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.embedding_provider != "local":
            raise ConfigurationError(
                "Only EMBEDDING_PROVIDER=local is supported in v1. "
                "Set EMBEDDING_PROVIDER=local."
            )
        if self.embedding_batch_size <= 0:
            raise ConfigurationError("EMBEDDING_BATCH_SIZE must be greater than 0.")
        if self.embedding_max_seq_length <= 0:
            raise ConfigurationError(
                "EMBEDDING_MAX_SEQ_LENGTH must be greater than 0."
            )
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")
        if self.llm_max_completion_tokens <= 0:
            raise ConfigurationError(
                "LLM_MAX_COMPLETION_TOKENS must be greater than 0."
            )
        if self.llm_reasoning_effort not in {"low", "medium", "high"}:
            raise ConfigurationError(
                "LLM_REASONING_EFFORT must be low, medium, or high."
            )
        if self.chunk_size_tokens <= 0:
            raise ConfigurationError("CHUNK_SIZE_TOKENS must be greater than 0.")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ConfigurationError(
                "CHUNK_OVERLAP_TOKENS must be >= 0 and smaller than CHUNK_SIZE_TOKENS."
            )
        if self.embedding_max_seq_length < self.chunk_size_tokens:
            raise ConfigurationError(
                "EMBEDDING_MAX_SEQ_LENGTH must be at least CHUNK_SIZE_TOKENS so "
                "the embedding model sees the whole chunk."
            )
        if self.retrieval_top_k <= 0:
            raise ConfigurationError("RETRIEVAL_TOP_K must be greater than 0.")
        if self.max_context_tokens <= 0:
            raise ConfigurationError("MAX_CONTEXT_TOKENS must be greater than 0.")
        if self.max_questions <= 0:
            raise ConfigurationError("MAX_QUESTIONS must be greater than 0.")
        if not 1 <= self.default_question_count <= self.max_questions:
            raise ConfigurationError(
                "DEFAULT_QUESTION_COUNT must be between 1 and MAX_QUESTIONS."
            )
        if self.max_pdfs <= 0:
            raise ConfigurationError("MAX_PDFS must be greater than 0.")

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    def ensure_directories(self) -> None:
        """Create local storage directories if they do not exist."""
        try:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Could not create the data directory: {exc}"
            ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings snapshot."""
    return Settings.from_env()
