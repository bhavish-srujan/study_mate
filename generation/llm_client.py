"""Groq chat client used for question generation and short-answer grading.

Groq exposes an OpenAI-compatible API. The Groq SDK appends ``/openai/v1`` to
its base URL, so ``LLM_BASE_URL`` accepts either the host
(``https://api.groq.com``) or the full OpenAI-compatible path
(``https://api.groq.com/openai/v1``) and is normalized here.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

from config.settings import Settings
from utils.errors import ConfigurationError, LLMError

logger = logging.getLogger(__name__)

Message = dict[str, str]
_OPENAI_SUFFIX = "/openai/v1"


def _sdk_base_url(raw: str) -> str | None:
    """Return the base URL expected by the Groq SDK."""
    value = (raw or "").strip().rstrip("/")
    if not value:
        return None
    if value.endswith(_OPENAI_SUFFIX):
        value = value[: -len(_OPENAI_SUFFIX)].rstrip("/")
    return value or None


class GroqChatClient:
    """Thin wrapper around one Groq chat model."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_configured:
            raise ConfigurationError(
                "LLM_API_KEY and LLM_MODEL must be set in .env.",
                user_message=(
                    "The LLM is not configured. Add `LLM_API_KEY` and "
                    "`LLM_MODEL` to `.env` and restart the app."
                ),
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ConfigurationError(
                "groq is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_completion_tokens = settings.llm_max_completion_tokens
        self.reasoning_effort = settings.llm_reasoning_effort

        kwargs: dict[str, str] = {"api_key": settings.llm_api_key}
        base_url = _sdk_base_url(settings.llm_base_url)
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self._client = Groq(**kwargs)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not create the Groq client: {exc}"
            ) from exc

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Return the full response text for a chat request."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=(
                    self.temperature if temperature is None else temperature
                ),
                max_completion_tokens=(
                    self.max_completion_tokens
                    if max_tokens is None
                    else max_tokens
                ),
                top_p=1,
                reasoning_effort=(
                    self.reasoning_effort
                    if reasoning_effort is None
                    else reasoning_effort
                ),
            )
        except Exception as exc:
            raise LLMError(f"Groq completion failed: {exc}") from exc

        if not response.choices:
            raise LLMError("Groq returned no choices.")

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("Groq returned an empty response.")
        return content

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Yield response text incrementally."""
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=(
                    self.temperature if temperature is None else temperature
                ),
                max_completion_tokens=(
                    self.max_completion_tokens
                    if max_tokens is None
                    else max_tokens
                ),
                top_p=1,
                reasoning_effort=(
                    self.reasoning_effort
                    if reasoning_effort is None
                    else reasoning_effort
                ),
                stream=True,
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Groq stream failed: {exc}") from exc

    def check_connection(self) -> str:
        """Small deterministic request used by the UI connection test."""
        return self.complete(
            [
                {
                    "role": "user",
                    "content": "Reply with exactly: StudyMate LLM ready",
                }
            ],
            temperature=0.0,
            max_tokens=128,
            reasoning_effort="low",
        )


def build_llm_client(settings: Settings) -> GroqChatClient:
    return GroqChatClient(settings)
