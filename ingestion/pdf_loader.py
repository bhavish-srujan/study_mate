"""Extract text from PDFs with PyMuPDF, one ``LoadedPage`` per page."""

from __future__ import annotations

from pathlib import Path

from ingestion.models import LoadedPage, LoadedResource
from ingestion.normalization import make_resource_id, normalize_text
from utils.errors import (
    EmptyDocumentError,
    IngestionError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
)


def load_pdf(data: bytes, filename: str) -> LoadedResource:
    """Extract and normalize a PDF supplied as bytes.

    PyMuPDF is imported lazily so the Streamlit app can start (and report a
    friendly error) even when optional dependencies are missing.
    """
    source_name = Path(filename or "document.pdf").name or "document.pdf"
    if not source_name.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(
            f"Expected a .pdf file, got {source_name!r}."
        )
    if not data:
        raise InvalidDocumentError(
            f"The uploaded PDF {source_name!r} is empty.",
            user_message="That PDF is empty. Please upload a file that contains text.",
        )

    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz  # older PyMuPDF releases
        except ImportError as exc:
            raise IngestionError(
                "PyMuPDF is not installed. Run `pip install -r requirements.txt`."
            ) from exc

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError(
            f"Could not open PDF {source_name!r}: {exc}"
        ) from exc

    pages: list[LoadedPage] = []
    try:
        if getattr(document, "needs_pass", False):
            raise InvalidDocumentError(
                f"PDF {source_name!r} is password-protected."
            )

        for page_number, page in enumerate(document, start=1):
            try:
                raw_text = page.get_text("text")
            except Exception as exc:
                raise IngestionError(
                    f"Could not extract text from page {page_number} of "
                    f"{source_name!r}: {exc}"
                ) from exc

            clean = normalize_text(raw_text)
            if clean:
                pages.append(LoadedPage(text=clean, page_number=page_number))
    finally:
        document.close()

    if not pages:
        raise EmptyDocumentError(
            f"No extractable text in PDF {source_name!r}.",
            user_message=(
                f"'{source_name}' contains no extractable text. It may be a "
                "scanned, image-only PDF, which needs OCR (not supported yet). "
                "Try a text-based PDF or paste your notes instead."
            ),
        )

    full_text = "\n\n".join(page.text for page in pages)
    return LoadedResource(
        resource_id=make_resource_id(full_text, "pdf"),
        source_name=source_name,
        source_type="pdf",
        pages=tuple(pages),
    )
