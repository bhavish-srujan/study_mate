"""Application errors with safe, student-facing messages.

Technical detail goes to the log; ``user_message`` is what the UI shows.
"""

from __future__ import annotations


class StudyMateError(Exception):
    """Base class for expected application failures."""

    user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str = "", *, user_message: str | None = None) -> None:
        super().__init__(detail or self.user_message)
        self.detail = detail
        if user_message is not None:
            self.user_message = user_message


class ConfigurationError(StudyMateError):
    user_message = (
        "The application is not configured correctly. Check your `.env` file "
        "(start by copying `.env.example`)."
    )


class UnsupportedFileTypeError(StudyMateError):
    user_message = "That file type is not supported. Please upload a PDF."


class InvalidDocumentError(StudyMateError):
    user_message = (
        "That file could not be read as a PDF. It may be corrupted or "
        "password-protected."
    )


class EmptyDocumentError(StudyMateError):
    user_message = (
        "No extractable text was found. Scanned, image-only PDFs need OCR, "
        "which this app does not support yet. Try a text-based PDF or paste "
        "your notes instead."
    )


class IngestionError(StudyMateError):
    user_message = "The document could not be processed. Please try again."


class InvalidYouTubeUrlError(StudyMateError):
    user_message = (
        "That YouTube link is not valid. Paste a youtube.com/watch, youtu.be, "
        "Shorts, or live-video URL."
    )


class TranscriptUnavailableError(StudyMateError):
    user_message = (
        "No transcript is available for that video. Try another video, or "
        "upload a PDF / paste your notes instead."
    )


class EmbeddingError(StudyMateError):
    user_message = (
        "The embedding model could not process this text. Check that the model "
        "is installed and try again."
    )


class VectorStoreError(StudyMateError):
    user_message = (
        "The study index could not be opened or updated. Check the application "
        "logs for details."
    )


class IndexingError(StudyMateError):
    user_message = "This resource could not be indexed. Please try again."


class RetrievalError(StudyMateError):
    user_message = (
        "Relevant study material could not be retrieved. Try a different topic "
        "or re-index your resources."
    )


class GenerationError(StudyMateError):
    user_message = (
        "Practice questions could not be generated from the selected material. "
        "Try a different topic, select more resources, or reduce the question "
        "count."
    )


class LLMError(StudyMateError):
    user_message = (
        "The AI service could not complete the request. Check your API key, "
        "model name, and internet connection, then try again."
    )


class EvaluationError(StudyMateError):
    user_message = (
        "This quiz could not be scored. Check the application logs, then try "
        "again."
    )


class ResourceNotFoundError(StudyMateError):
    user_message = "That resource could not be found. It may have been deleted."
