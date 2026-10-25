"""Loader and normalization tests."""

from __future__ import annotations

import pytest

from ingestion.models import LoadedPage
from ingestion.normalization import make_resource_id, normalize_text
from ingestion.pdf_loader import load_pdf
from ingestion.text_loader import load_pasted_text
from utils.errors import (
    EmptyDocumentError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
)


def test_normalize_collapses_whitespace_and_keeps_paragraphs() -> None:
    raw = "  First   line \t here.\n\n\n\nSecond\tline.  "
    assert normalize_text(raw) == "First line here.\n\nSecond line."


def test_normalize_removes_zero_width_and_joins_hyphenated_lines() -> None:
    raw = "embed\u200bding\nsplit-\nting"
    assert normalize_text(raw) == "embedding\nsplitting"


def test_resource_id_is_stable_across_whitespace_variants() -> None:
    first = make_resource_id("Hello   world", "pasted_text")
    second = make_resource_id("Hello world", "pasted_text")
    assert first == second


def test_resource_id_changes_with_content() -> None:
    first = make_resource_id("Hello world", "pasted_text")
    second = make_resource_id("Hello there", "pasted_text")
    assert first != second


def test_load_pasted_text_builds_a_resource() -> None:
    resource = load_pasted_text("  CPU   scheduling \n notes ", "OS Notes")

    assert resource.source_name == "OS Notes"
    assert resource.source_type == "pasted_text"
    assert resource.pages == (LoadedPage(text="CPU scheduling\nnotes"),)
    assert resource.resource_id == make_resource_id(resource.full_text, "pasted_text")


def test_load_pasted_text_uses_default_title() -> None:
    assert load_pasted_text("some notes", "   ").source_name == "Pasted notes"


def test_load_pasted_text_rejects_blank_input() -> None:
    with pytest.raises(EmptyDocumentError):
        load_pasted_text(" \n\t ")


def test_load_pdf_rejects_non_pdf_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_pdf(b"%PDF-1.4 fake", "notes.txt")


def test_load_pdf_rejects_invalid_bytes() -> None:
    with pytest.raises(InvalidDocumentError):
        load_pdf(b"this is not a pdf", "broken.pdf")


def test_load_pdf_extracts_text_and_page_numbers() -> None:
    fitz = pytest.importorskip("pymupdf")

    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Operating systems manage hardware.")
        data = document.tobytes()
    finally:
        document.close()

    resource = load_pdf(data, "os.pdf")

    assert resource.source_type == "pdf"
    assert resource.source_name == "os.pdf"
    assert len(resource.pages) == 1
    assert resource.pages[0].page_number == 1
    assert "Operating systems" in resource.pages[0].text


def test_load_pdf_reports_image_only_documents() -> None:
    fitz = pytest.importorskip("pymupdf")

    document = fitz.open()
    try:
        document.new_page()  # no text at all
        data = document.tobytes()
    finally:
        document.close()

    with pytest.raises(EmptyDocumentError):
        load_pdf(data, "scanned.pdf")
