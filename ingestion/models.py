"""Intermediate models shared by loaders, the chunker, and the indexer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["pdf", "pasted_text", "youtube"]


@dataclass(frozen=True)
class LoadedPage:
    """A unit of extracted text with optional provenance.

    PDFs produce one ``LoadedPage`` per page. Pasted text and YouTube
    transcripts produce one or more pages with a timestamp instead of a page
    number.
    """

    text: str
    page_number: int | None = None
    start_timestamp: str | None = None


@dataclass(frozen=True)
class LoadedResource:
    """A fully extracted learning resource, before chunking."""

    resource_id: str
    source_name: str
    source_type: SourceType
    pages: tuple[LoadedPage, ...]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def is_empty(self) -> bool:
        return not self.full_text.strip()
