"""Question schema and validator tests. No network calls."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from generation.schemas import GenerationRequest, Question
from generation.validator import (
    is_duplicate,
    parse_questions,
    validate_questions,
)

SOURCE = "[Source: notes · p.1 · chunk 0]"


def _mcq(
    question_id: str = "q1",
    *,
    difficulty: str = "easy",
    text: str = "Which keyword defines a function in Python?",
    correct: str = "0",
    source: str = SOURCE,
) -> Question:
    return Question(
        question_id=question_id,
        question_type="mcq",
        question_text=text,
        difficulty=difficulty,
        options=["def", "class", "return", "import"],
        correct_answer=correct,
        explanation="The def keyword introduces a function definition.",
        source_references=[source],
    )


def _fill_blank(
    question_id: str = "q1",
    *,
    difficulty: str = "medium",
    text: str = "A function is defined with the ____ keyword.",
) -> Question:
    return Question(
        question_id=question_id,
        question_type="fill_blank",
        question_text=text,
        difficulty=difficulty,
        correct_answer="def",
        accepted_answers=["def"],
        explanation="Python uses def to define functions.",
        source_references=[SOURCE],
    )


def _short_answer(
    question_id: str = "q1",
    *,
    difficulty: str = "hard",
    text: str = "Explain why functions improve code reuse.",
) -> Question:
    return Question(
        question_id=question_id,
        question_type="short_answer",
        question_text=text,
        difficulty=difficulty,
        correct_answer="They package reusable logic behind a name.",
        key_points=["reuse", "abstraction"],
        explanation="Functions avoid repeating the same logic.",
        source_references=[SOURCE],
    )


def test_valid_question_shapes_are_accepted() -> None:
    assert _mcq().correct_answer == "0"
    assert _fill_blank().question_type == "fill_blank"
    assert _short_answer().question_type == "short_answer"


def test_mcq_requires_four_options() -> None:
    with pytest.raises(ValidationError):
        Question(
            question_id="q1",
            question_type="mcq",
            question_text="Pick one.",
            difficulty="easy",
            options=["a", "b", "c"],
            correct_answer="0",
            explanation="Because.",
        )


def test_mcq_rejects_invalid_correct_index() -> None:
    with pytest.raises(ValidationError):
        _mcq(correct="4")


def test_fill_blank_requires_a_blank_marker() -> None:
    with pytest.raises(ValidationError):
        _fill_blank(text="A function is defined with the def keyword.")


def test_non_mcq_rejects_options() -> None:
    with pytest.raises(ValidationError):
        Question(
            question_id="q1",
            question_type="short_answer",
            question_text="Explain functions.",
            difficulty="hard",
            options=["a", "b"],
            correct_answer="Reuse.",
            explanation="Because.",
        )


def test_generation_request_requires_matching_distribution() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            resource_ids=["r1"],
            question_types=["mcq"],
            total_questions=3,
            easy=1,
            medium=1,
            hard=0,
        )


def test_generation_request_cleans_topic_and_rejects_duplicates() -> None:
    request = GenerationRequest(
        resource_ids=["r1"],
        question_types=["mcq"],
        total_questions=1,
        easy=1,
        medium=0,
        hard=0,
        topic="  CPU scheduling  ",
    )
    assert request.topic == "CPU scheduling"

    with pytest.raises(ValidationError):
        GenerationRequest(
            resource_ids=["r1", "r1"],
            question_types=["mcq"],
            total_questions=1,
            easy=1,
            medium=0,
            hard=0,
        )


def test_parse_questions_accepts_fenced_json() -> None:
    raw = "```json\n" + json.dumps(
        {"questions": [_mcq().model_dump()]}
    ) + "\n```"

    questions = parse_questions(raw)

    assert len(questions) == 1
    assert questions[0].question_id == "q1"


def test_parse_questions_rejects_missing_questions_key() -> None:
    with pytest.raises(ValueError):
        parse_questions(json.dumps({"items": []}))


def test_duplicate_detection_is_case_and_punctuation_insensitive() -> None:
    first = _mcq(text="Which keyword defines a function in Python?")
    second = _mcq(
        question_id="q2",
        text="which KEYWORD defines a function in python",
    )

    assert is_duplicate(first, second)


def test_validate_questions_checks_counts_and_difficulty() -> None:
    questions = [
        _mcq("q1", difficulty="easy", text="Which keyword defines a function?"),
        _mcq("q2", difficulty="easy", text="What does the return statement do?"),
    ]

    result = validate_questions(
        questions,
        expected_total=2,
        expected_difficulty={"easy": 1, "medium": 1, "hard": 0},
        allowed_types=["mcq"],
        allowed_sources=["notes"],
    )

    assert result.valid is False
    assert any("medium" in error for error in result.errors)
    assert len(result.questions) == 2


def test_validate_questions_requires_known_source() -> None:
    question = _mcq(source="[Source: unrelated.pdf · p.9 · chunk 1]")

    result = validate_questions(
        [question],
        expected_total=1,
        allowed_sources=["notes"],
    )

    assert result.valid is False
    assert result.questions == []
    assert any("source reference" in error for error in result.errors)


def test_validate_questions_rejects_near_duplicates() -> None:
    first = _mcq("q1", text="What is a Python decorator?")
    second = _mcq("q2", text="What is a Python decorator?")

    result = validate_questions(
        [first, second],
        expected_total=2,
        allowed_sources=["notes"],
    )

    assert result.valid is False
    assert len(result.questions) == 1
    assert any("duplicates" in error for error in result.errors)


def test_validate_questions_can_require_every_selected_type() -> None:
    result = validate_questions(
        [
            _mcq("q1", text="Which keyword defines a function?"),
            _mcq("q2", text="What does the return statement do?"),
        ],
        expected_total=2,
        allowed_types=["mcq", "short_answer"],
        allowed_sources=["notes"],
        require_all_types=True,
    )

    assert result.valid is False
    assert any("short_answer" in error for error in result.errors)
