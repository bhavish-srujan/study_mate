"""Models returned by retrieval and representative chunk sampling."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ingestion.models import SourceType


class ContextPassage(BaseModel):
    """A chunk with enough provenance to label it in an LLM prompt."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    resource_id: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    source_name: str = Field(min_length=1)
    source_type: SourceType
    page_number: int | None = Field(default=None, ge=1)
    start_timestamp: str | None = None

    def source_label(self) -> str:
        """Human-readable citation label for prompts and the UI."""
        parts = [self.source_name]
        if self.page_number is not None:
            parts.append(f"p.{self.page_number}")
        if self.start_timestamp:
            parts.append(self.start_timestamp)
        parts.append(f"chunk {self.chunk_index}")
        return " · ".join(parts)


class RetrievedChunk(ContextPassage):
    """A context passage ranked by vector similarity."""

    distance: float
    score: float
