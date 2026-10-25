"""Text normalization and stable resource identifiers.

Normalization removes extraction artefacts without deleting meaningful
content. Resource IDs are content-based so re-uploading the same resource is
idempotent within the index.
"""

from __future__ import annotations

import hashlib
import re

# Zero-width and soft-hyphen characters commonly produced by PDF extraction.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)


def normalize_text(text: str) -> str:
    """Collapse whitespace and remove invisible extraction artefacts."""
    if not text:
        return ""

    text = text.translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Re-join words split across lines with a hyphen, e.g. "embed-\nding".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize horizontal whitespace but keep paragraph breaks.
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_resource_id(text: str, source_type: str) -> str:
    """Return a stable, short ID derived from the normalized content."""
    digest = hashlib.sha256()
    digest.update(source_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(normalize_text(text).encode("utf-8"))
    return digest.hexdigest()[:16]
