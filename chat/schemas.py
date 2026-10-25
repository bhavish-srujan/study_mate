"""Models for grounded chat messages and their retrieved sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ingestion.models import SourceType

ChatRole = Literal["user", "assistant"]


class ChatSource(BaseModel):
    """A retrieved passage shown beneath an assistant answer."""

    model_config = ConfigDict(frozen=True)

    source_label: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: SourceType
    page_number: int | None = Field(default=None, ge=1)
    start_timestamp: str | None = None
    chunk_index: int = Field(ge=0)
    score: float | None = None
    snippet: str = Field(min_length=1)


class ChatMessage(BaseModel):
    """One turn in the conversation, safe to keep in session state."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str = Field(min_length=1)
    sources: list[ChatSource] = Field(default_factory=list)
    grounded: bool = True

    @field_validator("content")
    @classmethod
    def _clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Chat content must not be blank.")
        return cleaned


@dataclass(frozen=True)
class ChatAnswer:
    """A grounded answer plus the passages that supported it."""

    text: str
    sources: tuple[ChatSource, ...]
    context_token_count: int
    truncated: bool
    grounded: bool
