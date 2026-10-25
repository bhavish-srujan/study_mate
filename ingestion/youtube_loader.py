"""Load a YouTube lecture transcript as a timestamped learning resource.

The transcript API returns short caption snippets. Indexing each snippet as its
own page would create tiny, noisy chunks, so consecutive snippets are grouped
into roughly one-minute pages. Each page keeps the timestamp of its first
snippet, which the chunker copies onto every chunk produced from that page.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from ingestion.models import LoadedPage, LoadedResource
from ingestion.normalization import make_resource_id, normalize_text
from utils.errors import (
    IngestionError,
    InvalidYouTubeUrlError,
    StudyMateError,
    TranscriptUnavailableError,
)

logger = logging.getLogger(__name__)

_GROUP_SECONDS = 60.0
_GROUP_CHARS = 1200
_SUPPORTED_HOSTS = {
    "youtu.be",
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
}
_VIDEO_PATHS = ("/embed/", "/shorts/", "/live/", "/v/")
# YouTube often labels an English track as en-US, en-GB, etc. rather than
# plain "en". The transcript API matches language codes exactly, so ask for
# the common English variants explicitly instead of relying on the "en"
# default.
_ENGLISH_LANGUAGE_CODES = (
    "en",
    "en-US",
    "en-GB",
    "en-IN",
    "en-AU",
    "en-CA",
    "en-IE",
    "en-NZ",
    "en-ZA",
    "en-SG",
    "en-PH",
    "en-orig",
)


@dataclass(frozen=True)
class _TranscriptSegment:
    text: str
    start: float
    duration: float


def extract_video_id(url: str) -> str:
    """Return the 11-ish-character video ID from a common YouTube URL.

    Both full URLs and bare ``youtu.be/...`` links are accepted. A ``ValueError``
    is raised when the URL has no recognizable video ID.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Invalid YouTube URL")

    # Be forgiving when a student pastes a link without a scheme.
    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host not in _SUPPORTED_HOSTS:
        raise ValueError("Invalid YouTube URL")

    path = parsed.path.rstrip("/")
    if host == "youtu.be":
        video_id = path.strip("/").split("/")[0]
    elif path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
    elif path.startswith(_VIDEO_PATHS):
        parts = [part for part in path.split("/") if part]
        video_id = parts[1] if len(parts) >= 2 else ""
    else:
        video_id = ""

    video_id = video_id.strip()
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    return video_id


def format_timestamp(seconds: float) -> str:
    """Format a transcript offset as ``MM:SS`` or ``H:MM:SS``."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _field(item: object, name: str, default: object) -> object:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_segment(item: object) -> _TranscriptSegment | None:
    """Accept both new-style snippets and old-style transcript dictionaries."""
    text = str(_field(item, "text", "") or "").strip()
    if not text:
        return None
    return _TranscriptSegment(
        text=text,
        start=_as_float(_field(item, "start", 0.0)),
        duration=_as_float(_field(item, "duration", 0.0)),
    )


def _accepts_languages(func: object) -> bool:
    """Return True when a transcript API callable accepts a languages kwarg."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return "languages" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _call_transcript_api(func: object, video_id: str) -> object:
    """Call a new- or old-style transcript method with English variants."""
    if _accepts_languages(func):
        return func(video_id, languages=_ENGLISH_LANGUAGE_CODES)  # type: ignore[operator]
    return func(video_id)  # type: ignore[operator]


def _available_language_labels(api: object, video_id: str) -> list[str]:
    """Best-effort list of caption languages available for an error message."""
    list_transcripts = getattr(api, "list", None)
    if not callable(list_transcripts):
        return []

    try:
        transcripts = list(list_transcripts(video_id))
    except Exception:
        return []

    labels: list[str] = []
    for transcript in transcripts:
        label = (
            getattr(transcript, "language", "")
            or getattr(transcript, "language_code", "")
        )
        label = str(label).strip()
        if label and label not in labels:
            labels.append(label)
    return labels[:6]


def _fetch_segments(api: object, video_id: str) -> list[_TranscriptSegment]:
    """Fetch snippets through the new or legacy youtube-transcript-api API."""
    if hasattr(api, "fetch"):
        fetched = _call_transcript_api(api.fetch, video_id)  # type: ignore[attr-defined]
        raw_items = getattr(fetched, "snippets", fetched)
    else:
        raw_items = _call_transcript_api(api.get_transcript, video_id)  # type: ignore[attr-defined]

    if not isinstance(raw_items, Iterable):
        raise TranscriptUnavailableError(
            f"Transcript API returned an unexpected payload for {video_id!r}."
        )

    segments: list[_TranscriptSegment] = []
    for item in raw_items:
        segment = _coerce_segment(item)
        if segment is not None:
            segments.append(segment)
    return segments


def _group_into_pages(
    segments: Sequence[_TranscriptSegment],
) -> tuple[LoadedPage, ...]:
    """Group short caption snippets into timestamped, chunk-friendly pages."""
    pages: list[LoadedPage] = []
    buffer: list[str] = []
    buffer_start: float | None = None
    buffer_chars = 0

    def flush() -> None:
        nonlocal buffer, buffer_start, buffer_chars
        if not buffer or buffer_start is None:
            return
        text = normalize_text(" ".join(buffer))
        if text:
            pages.append(
                LoadedPage(
                    text=text,
                    start_timestamp=format_timestamp(buffer_start),
                )
            )
        buffer = []
        buffer_start = None
        buffer_chars = 0

    for segment in segments:
        if (
            buffer
            and buffer_start is not None
            and (
                segment.start - buffer_start >= _GROUP_SECONDS
                or buffer_chars >= _GROUP_CHARS
            )
        ):
            flush()

        if buffer_start is None:
            buffer_start = segment.start
        buffer.append(segment.text)
        buffer_chars += len(segment.text) + 1

    flush()
    return tuple(pages)


def load_youtube_transcript(
    url: str,
    title: str = "",
    *,
    api: object | None = None,
) -> LoadedResource:
    """Fetch a YouTube transcript and wrap it in the common resource model.

    ``api`` is injectable for tests; production calls use
    ``YouTubeTranscriptApi().fetch(video_id, languages=...)`` with common
    English language variants (``en``, ``en-US``, ``en-GB``, and so on).
    """
    try:
        video_id = extract_video_id(url)
    except ValueError as exc:
        raise InvalidYouTubeUrlError(
            f"Could not parse a YouTube video ID from {url!r}.",
            user_message=InvalidYouTubeUrlError.user_message,
        ) from exc

    try:
        if api is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
            except ImportError as exc:
                raise IngestionError(
                    "youtube-transcript-api is not installed.",
                    user_message=(
                        "YouTube transcript support is not installed. Run "
                        "`pip install -r requirements.txt` and restart the app."
                    ),
                ) from exc
            api = YouTubeTranscriptApi()

        segments = _fetch_segments(api, video_id)
    except StudyMateError:
        raise
    except Exception as exc:
        logger.warning(
            "Could not fetch YouTube transcript for %s: %s",
            video_id,
            exc,
        )
        user_message = TranscriptUnavailableError.user_message
        available = _available_language_labels(api, video_id)
        if type(exc).__name__ == "NoTranscriptFound" and available:
            user_message = (
                "That video has no English transcript. Available caption "
                f"languages: {', '.join(available)}. Try a video with English "
                "captions, or upload a PDF / paste your notes instead."
            )
        raise TranscriptUnavailableError(
            f"Could not fetch the transcript for {video_id!r}: {exc}",
            user_message=user_message,
        ) from exc

    if not segments:
        raise TranscriptUnavailableError(
            f"The transcript for {video_id!r} contained no text.",
            user_message=(
                "The transcript for that video is empty. Try another video, or "
                "upload a PDF / paste your notes instead."
            ),
        )

    pages = _group_into_pages(segments)
    if not pages:
        raise TranscriptUnavailableError(
            f"The transcript for {video_id!r} was empty after normalization.",
            user_message=(
                "The transcript for that video is empty. Try another video, or "
                "upload a PDF / paste your notes instead."
            ),
        )

    full_text = "\n\n".join(page.text for page in pages)
    source_name = (title or "").strip() or f"YouTube video {video_id}"
    resource = LoadedResource(
        resource_id=make_resource_id(full_text, "youtube"),
        source_name=source_name,
        source_type="youtube",
        pages=pages,
    )
    logger.info(
        "Loaded YouTube transcript %s: %d segment(s), %d timestamped page(s).",
        video_id,
        len(segments),
        len(pages),
    )
    return resource
