"""Load pasted notes as a learning resource."""

from __future__ import annotations

from ingestion.models import LoadedPage, LoadedResource
from ingestion.normalization import make_resource_id, normalize_text
from utils.errors import EmptyDocumentError


def load_pasted_text(text: str, title: str = "Pasted notes") -> LoadedResource:
    """Normalize pasted notes and wrap them in the common resource model."""
    clean = normalize_text(text)
    if not clean:
        raise EmptyDocumentError(
            "Pasted text was empty after normalization.",
            user_message="The pasted notes are empty. Paste some study material first.",
        )

    source_name = (title or "").strip() or "Pasted notes"
    return LoadedResource(
        resource_id=make_resource_id(clean, "pasted_text"),
        source_name=source_name,
        source_type="pasted_text",
        pages=(LoadedPage(text=clean),),
    )
