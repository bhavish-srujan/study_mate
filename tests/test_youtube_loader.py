"""YouTube URL parsing and transcript loading tests. No network calls."""

from __future__ import annotations

import pytest

from ingestion.chunker import WhitespaceTokenizer, chunk_resource
from ingestion.normalization import make_resource_id
from ingestion.youtube_loader import (
    extract_video_id,
    format_timestamp,
    load_youtube_transcript,
)
from utils.errors import InvalidYouTubeUrlError, TranscriptUnavailableError


class FakeSnippet:
    def __init__(self, text: str, start: float, duration: float = 1.0) -> None:
        self.text = text
        self.start = start
        self.duration = duration


class FakeFetchedTranscript:
    def __init__(self, snippets: list[FakeSnippet]) -> None:
        self.snippets = snippets

    def __iter__(self):
        return iter(self.snippets)


class NewApi:
    def __init__(self, snippets: list[FakeSnippet]) -> None:
        self.snippets = snippets
        self.video_ids: list[str] = []
        self.language_requests: list[tuple[str, ...]] = []

    def fetch(
        self,
        video_id: str,
        languages: tuple[str, ...] = ("en",),
    ) -> FakeFetchedTranscript:
        self.video_ids.append(video_id)
        self.language_requests.append(tuple(languages))
        return FakeFetchedTranscript(self.snippets)


class OldApi:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.video_ids: list[str] = []
        self.language_requests: list[tuple[str, ...]] = []

    def get_transcript(
        self,
        video_id: str,
        languages: tuple[str, ...] = ("en",),
    ) -> list[dict[str, object]]:
        self.video_ids.append(video_id)
        self.language_requests.append(tuple(languages))
        return self.rows


class BrokenApi:
    def fetch(self, video_id: str):
        raise RuntimeError("transcripts are disabled")


class FakeTranscript:
    def __init__(self, language: str, language_code: str) -> None:
        self.language = language
        self.language_code = language_code


class NoTranscriptFound(Exception):
    """Stand-in for the transcript API's missing-language exception."""


class NoEnglishApi:
    def fetch(
        self,
        video_id: str,
        languages: tuple[str, ...] = ("en",),
    ):
        raise NoTranscriptFound("no English transcript")

    def list(self, video_id: str) -> list[FakeTranscript]:
        return [FakeTranscript("Hindi (auto-generated)", "hi")]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc123&t=30", "abc123"),
        ("https://youtube.com/watch?v=abc123", "abc123"),
        ("https://m.youtube.com/watch?v=abc123", "abc123"),
        ("https://music.youtube.com/watch?v=abc123", "abc123"),
        ("https://youtu.be/abc123", "abc123"),
        ("youtu.be/abc123", "abc123"),
        ("https://www.youtube.com/embed/abc123", "abc123"),
        ("https://www.youtube.com/shorts/abc123?feature=share", "abc123"),
        ("https://www.youtube.com/live/abc123", "abc123"),
        ("https://www.youtube.com/v/abc123", "abc123"),
        ("https://youtube-nocookie.com/embed/abc123", "abc123"),
    ],
)
def test_extract_video_id_supports_common_urls(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/watch?v=abc123",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/playlist?list=abc123",
        "https://youtu.be/",
    ],
)
def test_extract_video_id_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValueError):
        extract_video_id(url)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (5, "00:05"),
        (65, "01:05"),
        (3661, "1:01:01"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_load_new_style_transcript_groups_segments_and_keeps_timestamps() -> None:
    api = NewApi(
        [
            FakeSnippet("Operating systems manage hardware.", 0),
            FakeSnippet("They schedule processes and memory.", 10),
            FakeSnippet("Scheduling decides which process runs.", 70),
            FakeSnippet("Context switching saves process state.", 75),
            FakeSnippet("Deadlock happens when processes wait forever.", 140),
        ]
    )

    resource = load_youtube_transcript(
        "https://www.youtube.com/watch?v=abc123",
        title="OS Lecture 4",
        api=api,
    )

    assert api.video_ids == ["abc123"]
    assert "en-US" in api.language_requests[0]
    assert "en-GB" in api.language_requests[0]
    assert resource.source_type == "youtube"
    assert resource.source_name == "OS Lecture 4"
    assert [page.start_timestamp for page in resource.pages] == [
        "00:00",
        "01:10",
        "02:20",
    ]
    assert "Operating systems" in resource.pages[0].text
    assert "Context switching" in resource.pages[1].text
    assert "Deadlock" in resource.pages[2].text
    assert resource.resource_id == make_resource_id(resource.full_text, "youtube")


def test_youtube_pages_preserve_timestamps_when_chunked() -> None:
    resource = load_youtube_transcript(
        "https://youtu.be/abc123",
        api=NewApi(
            [
                FakeSnippet("alpha beta gamma delta epsilon", 0),
                FakeSnippet("zeta eta theta iota kappa", 70),
            ]
        ),
    )

    chunks = chunk_resource(
        resource,
        chunk_size_tokens=3,
        overlap_tokens=0,
        tokenizer=WhitespaceTokenizer(),
    )

    assert chunks
    assert all(chunk.source_type == "youtube" for chunk in chunks)
    assert {chunk.start_timestamp for chunk in chunks} == {"00:00", "01:10"}


def test_load_legacy_transcript_dictionaries() -> None:
    api = OldApi(
        [
            {"text": "First point.", "start": 0.0, "duration": 2.0},
            {"text": "Second point.", "start": 12.0, "duration": 2.0},
        ]
    )

    resource = load_youtube_transcript("https://youtu.be/abc123", api=api)

    assert api.video_ids == ["abc123"]
    assert "en-US" in api.language_requests[0]
    assert resource.source_name == "YouTube video abc123"
    assert len(resource.pages) == 1
    assert resource.pages[0].start_timestamp == "00:00"
    assert "First point. Second point." in resource.pages[0].text


def test_load_rejects_invalid_url() -> None:
    with pytest.raises(InvalidYouTubeUrlError):
        load_youtube_transcript(
            "https://example.com/watch?v=abc123",
            api=NewApi([FakeSnippet("text", 0)]),
        )


def test_load_rejects_an_empty_transcript() -> None:
    with pytest.raises(TranscriptUnavailableError):
        load_youtube_transcript(
            "https://youtu.be/abc123",
            api=NewApi([]),
        )


def test_load_wraps_transcript_api_failures() -> None:
    with pytest.raises(TranscriptUnavailableError):
        load_youtube_transcript("https://youtu.be/abc123", api=BrokenApi())


def test_load_reports_available_languages_when_english_is_missing() -> None:
    with pytest.raises(TranscriptUnavailableError) as info:
        load_youtube_transcript("https://youtu.be/abc123", api=NoEnglishApi())

    assert "Hindi" in info.value.user_message
