"""Grounded chat service tests with fake retrieval and a fake LLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chat.schemas import ChatMessage
from chat.service import NO_MATCH_ANSWER, ChatService
from rag.schemas import RetrievedChunk
from utils.errors import LLMError, RetrievalError


def _settings(**overrides):
    values = {
        "max_context_tokens": 2000,
        "llm_temperature": 0.2,
        "llm_max_completion_tokens": 2048,
        "llm_reasoning_effort": "medium",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hit() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="r1:0002",
        resource_id="r1",
        text="A process is a program in execution with its own state.",
        chunk_index=2,
        source_name="OS Lecture 1",
        source_type="youtube",
        start_timestamp="12:34",
        distance=0.2,
        score=0.8,
    )


class FakeRetriever:
    def __init__(self, hits=None) -> None:
        self.hits = list(hits or [])
        self.calls: list[tuple[str, list[str]]] = []

    def retrieve(self, query, resource_ids, top_k=None, min_score=0.0):
        self.calls.append((query, list(resource_ids)))
        return list(self.hits)


class FakeLLM:
    def __init__(self, response: str = "A process is a running program.") -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def complete(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return self.response


def test_answer_retrieves_context_and_returns_sources() -> None:
    retriever = FakeRetriever([_hit()])
    llm = FakeLLM()
    service = ChatService(retriever, llm, _settings())

    answer = service.answer("What is a process?", ["r1"])

    assert answer.text == "A process is a running program."
    assert answer.grounded is True
    assert answer.truncated is False
    assert answer.context_token_count > 0
    assert len(answer.sources) == 1
    assert answer.sources[0].source_label == "OS Lecture 1 · 12:34 · chunk 2"
    assert answer.sources[0].start_timestamp == "12:34"
    assert answer.sources[0].score == pytest.approx(0.8)
    assert retriever.calls == [("What is a process?", ["r1"])]

    messages, kwargs = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "What is a process?" in messages[-1]["content"]
    assert "[Source: OS Lecture 1 · 12:34 · chunk 2]" in messages[-1]["content"]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 1024
    assert kwargs["reasoning_effort"] == "medium"


def test_answer_includes_recent_history_before_the_current_question() -> None:
    llm = FakeLLM()
    service = ChatService(FakeRetriever([_hit()]), llm, _settings())
    history = [
        ChatMessage(role="user", content="Earlier question"),
        ChatMessage(role="assistant", content="Earlier answer"),
    ]

    service.answer("Follow-up?", ["r1"], history=history)

    messages, _ = llm.calls[0]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "Earlier question"
    assert messages[2]["content"] == "Earlier answer"
    assert "Follow-up?" in messages[3]["content"]


def test_no_retrieved_hits_returns_a_grounded_fallback_without_llm_call() -> None:
    llm = FakeLLM()
    service = ChatService(FakeRetriever([]), llm, _settings())

    answer = service.answer("What is a process?", ["r1"])

    assert answer.text == NO_MATCH_ANSWER
    assert answer.grounded is False
    assert answer.sources == ()
    assert llm.calls == []


def test_blank_question_is_rejected() -> None:
    service = ChatService(FakeRetriever([_hit()]), FakeLLM(), _settings())

    with pytest.raises(RetrievalError):
        service.answer("   ", ["r1"])


def test_empty_resource_selection_is_rejected() -> None:
    service = ChatService(FakeRetriever([_hit()]), FakeLLM(), _settings())

    with pytest.raises(RetrievalError):
        service.answer("What is a process?", [])


def test_empty_llm_response_is_an_error() -> None:
    service = ChatService(FakeRetriever([_hit()]), FakeLLM("   "), _settings())

    with pytest.raises(LLMError):
        service.answer("What is a process?", ["r1"])
