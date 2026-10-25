"""Question-generator tests with a fake retriever and fake LLM."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from generation.question_generator import QuestionGenerator
from generation.schemas import GenerationRequest
from rag.schemas import ContextPassage, RetrievedChunk
from utils.errors import GenerationError

SOURCE = "[Source: notes · p.1 · chunk 0]"


def _settings(**overrides):
    values = {
        "max_questions": 30,
        "retrieval_top_k": 10,
        "max_context_tokens": 6000,
        "llm_temperature": 0.2,
        "llm_max_completion_tokens": 2048,
        "llm_reasoning_effort": "medium",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _passage(index: int = 0) -> ContextPassage:
    return ContextPassage(
        chunk_id=f"r1:{index:04d}",
        resource_id="r1",
        text=f"Passage {index} explains Python functions and decorators.",
        chunk_index=index,
        source_name="notes",
        source_type="pasted_text",
        page_number=1,
    )


def _hit(index: int = 0) -> RetrievedChunk:
    passage = _passage(index)
    return RetrievedChunk(
        **passage.model_dump(),
        distance=0.1,
        score=0.9,
    )


class FakeRetriever:
    def __init__(self, *, hits=None, passages=None) -> None:
        self.hits = list(hits or [])
        self.passages = list(passages or [])
        self.retrieve_calls: list[tuple[str, list[str], int | None]] = []
        self.sample_calls: list[tuple[list[str], int | None]] = []

    def retrieve(self, query, resource_ids, top_k=None):
        self.retrieve_calls.append((query, list(resource_ids), top_k))
        return list(self.hits)

    def sample_chunks(self, resource_ids, limit=None):
        self.sample_calls.append((list(resource_ids), limit))
        return list(self.passages)


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def complete(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)


def _mcq(
    question_id: str,
    difficulty: str,
    text: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_type": "mcq",
        "question_text": text,
        "difficulty": difficulty,
        "options": ["def", "class", "return", "import"],
        "correct_answer": "0",
        "explanation": "The context identifies def as the function keyword.",
        "source_references": [SOURCE],
    }


def _fill_blank(question_id: str, difficulty: str, text: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "fill_blank",
        "question_text": text,
        "difficulty": difficulty,
        "correct_answer": "def",
        "accepted_answers": ["def"],
        "explanation": "The context says def introduces a function.",
        "source_references": [SOURCE],
    }


def _payload(*questions: dict) -> str:
    return json.dumps({"questions": list(questions)})


def _request(
    *,
    total: int = 2,
    easy: int = 1,
    medium: int = 1,
    hard: int = 0,
    types=("mcq",),
    topic: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        resource_ids=["r1"],
        question_types=list(types),
        total_questions=total,
        easy=easy,
        medium=medium,
        hard=hard,
        topic=topic,
    )


def test_topic_generation_uses_retrieval_and_returns_valid_quiz() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _mcq("q2", "medium", "What does a decorator return?"),
            )
        ]
    )
    generator = QuestionGenerator(retriever, llm, _settings())

    quiz = generator.generate(_request(topic="functions"))

    assert len(quiz.questions) == 2
    assert [question.question_id for question in quiz.questions] == ["q1", "q2"]
    assert quiz.topic == "functions"
    assert quiz.context_token_count > 0
    assert retriever.retrieve_calls == [("functions", ["r1"], 10)]
    assert retriever.sample_calls == []
    assert len(llm.calls) == 1


def test_invalid_json_is_repaired() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            "this is not json",
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _mcq("q2", "medium", "What does a decorator return?"),
            ),
        ]
    )
    generator = QuestionGenerator(retriever, llm, _settings())

    quiz = generator.generate(_request(topic="functions"))

    assert len(quiz.questions) == 2
    assert len(llm.calls) == 2
    repair_prompt = llm.calls[1][0][1]["content"]
    assert "failed validation" in repair_prompt


def test_wrong_difficulty_is_corrected_on_repair() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _mcq("q2", "easy", "What is a Python list?"),
            ),
            _payload(
                _mcq("q1", "medium", "What does a decorator return?"),
            ),
        ]
    )
    generator = QuestionGenerator(retriever, llm, _settings())

    quiz = generator.generate(_request(topic="functions"))

    assert len(quiz.questions) == 2
    assert quiz.difficulty_distribution == {"easy": 1, "medium": 1, "hard": 0}
    assert [question.difficulty for question in quiz.questions] == ["easy", "medium"]
    assert len(llm.calls) == 2


def test_no_topic_uses_representative_sampling_and_batches() -> None:
    retriever = FakeRetriever(passages=[_passage(index) for index in range(4)])
    llm = FakeLLM(
        [
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _mcq("q2", "medium", "What does a decorator return?"),
            ),
            _payload(
                _mcq("q1", "easy", "What is a closure?"),
                _mcq("q2", "medium", "Why use a generator?"),
            ),
        ]
    )
    generator = QuestionGenerator(
        retriever,
        llm,
        _settings(),
        batch_questions=2,
        batch_chunks=2,
    )

    quiz = generator.generate(_request(total=4, easy=2, medium=2))

    assert len(quiz.questions) == 4
    assert quiz.difficulty_distribution == {"easy": 2, "medium": 2, "hard": 0}
    assert retriever.retrieve_calls == []
    assert retriever.sample_calls == [(["r1"], 10)]
    assert len(llm.calls) == 2


def test_selected_types_are_present_in_final_quiz() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _fill_blank(
                    "q2",
                    "medium",
                    "A function is introduced with the ____ keyword.",
                ),
            )
        ]
    )
    generator = QuestionGenerator(retriever, llm, _settings())

    quiz = generator.generate(
        _request(types=("mcq", "fill_blank"), topic="functions")
    )

    assert {question.question_type for question in quiz.questions} == {
        "mcq",
        "fill_blank",
    }


def test_missing_selected_type_fails_final_validation() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            _payload(
                _mcq("q1", "easy", "Which keyword defines a function?"),
                _mcq("q2", "medium", "What does a decorator return?"),
            )
        ]
    )
    generator = QuestionGenerator(retriever, llm, _settings())

    with pytest.raises(GenerationError):
        generator.generate(
            _request(types=("mcq", "short_answer"), topic="functions")
        )


def test_empty_retrieval_fails_without_calling_the_llm() -> None:
    retriever = FakeRetriever(hits=[])
    llm = FakeLLM([])
    generator = QuestionGenerator(retriever, llm, _settings())

    with pytest.raises(GenerationError):
        generator.generate(_request(topic="functions"))

    assert llm.calls == []


def test_request_above_maximum_is_rejected_before_retrieval() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM([])
    generator = QuestionGenerator(retriever, llm, _settings(max_questions=1))

    with pytest.raises(GenerationError):
        generator.generate(_request(total=2, easy=1, medium=1, topic="functions"))

    assert retriever.retrieve_calls == []
    assert llm.calls == []


def test_shortfall_after_repairs_fails() -> None:
    retriever = FakeRetriever(hits=[_hit()])
    llm = FakeLLM(
        [
            _payload(_mcq("q1", "easy", "Which keyword defines a function?")),
            '{"questions": []}',
        ]
    )
    generator = QuestionGenerator(
        retriever,
        llm,
        _settings(),
        max_repair_attempts=0,
        batch_questions=5,
    )

    with pytest.raises(GenerationError):
        generator.generate(_request(topic="functions"))

    assert len(llm.calls) == 2
