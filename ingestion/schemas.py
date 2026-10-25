"""Pydantic models for chunks and resource summaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ingestion.models import SourceType


class DocumentChunk(BaseModel):
    """One overlapping chunk of a resource, with provenance metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    resource_id: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    source_type: SourceType
    source_name: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    start_timestamp: str | None = None

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chunk text must not be blank")
        return value

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata for ChromaDB (``None`` values are not allowed)."""
        metadata: dict[str, Any] = {
            "resource_id": self.resource_id,
            "chunk_index": self.chunk_index,
            "source_type": self.source_type,
            "source_name": self.source_name,
        }
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        if self.start_timestamp:
            metadata["start_timestamp"] = self.start_timestamp
        return metadata


class ResourceInfo(BaseModel):
    """Summary of an indexed resource, used by the library UI."""

    model_config = ConfigDict(frozen=True)

    resource_id: str
    source_name: str
    source_type: SourceType
    chunk_count: int = Field(ge=0)
