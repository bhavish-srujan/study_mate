"""Groq client tests. No network calls are made."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from generation.llm_client import GroqChatClient, _sdk_base_url
from utils.errors import ConfigurationError, LLMError


def _settings(**overrides):
    values = {
        "llm_api_key": "test-key",
        "llm_base_url": "https://api.groq.com/openai/v1",
        "llm_model": "openai/gpt-oss-120b",
        "llm_temperature": 0.2,
        "llm_max_completion_tokens": 2048,
        "llm_reasoning_effort": "medium",
    }
    values.update(overrides)
    values["llm_configured"] = bool(
        values["llm_api_key"] and values["llm_model"]
    )
    return SimpleNamespace(**values)


class _FakeCompletions:
    def __init__(self, response=None, chunks=None) -> None:
        self._response = response
        self._chunks = chunks or []

    def create(self, **kwargs):
        if self._response is not None:
            return self._response
        return iter(self._chunks)


def _client_with(fake: _FakeCompletions) -> GroqChatClient:
    client = GroqChatClient(_settings())
    client._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=fake)
    )
    return client


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://api.groq.com/openai/v1", "https://api.groq.com"),
        ("https://api.groq.com/openai/v1/", "https://api.groq.com"),
        ("https://api.groq.com", "https://api.groq.com"),
        ("  https://api.groq.com  ", "https://api.groq.com"),
        ("", None),
        ("   ", None),
    ],
)
def test_sdk_base_url_normalizes(raw: str, expected: str | None) -> None:
    assert _sdk_base_url(raw) == expected


def test_missing_configuration_raises() -> None:
    with pytest.raises(ConfigurationError):
        GroqChatClient(_settings(llm_api_key=""))


def test_complete_returns_trimmed_content() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  hello  "))]
    )
    client = _client_with(_FakeCompletions(response=response))

    assert client.complete([{"role": "user", "content": "hi"}]) == "hello"


def test_complete_rejects_empty_choices() -> None:
    client = _client_with(
        _FakeCompletions(response=SimpleNamespace(choices=[]))
    )

    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "hi"}])


def test_complete_rejects_blank_content() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
    )
    client = _client_with(_FakeCompletions(response=response))

    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "hi"}])


def test_stream_yields_text_deltas() -> None:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Study"))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Mate"))]
        ),
        SimpleNamespace(choices=[]),
    ]
    client = _client_with(_FakeCompletions(chunks=chunks))

    assert "".join(
        client.stream([{"role": "user", "content": "hi"}])
    ) == "StudyMate"
