"""Quiz view and answer-vault tests. No Streamlit or network calls."""

from __future__ import annotations

from generation.schemas import GeneratedQuiz, Question
from quiz.quiz_service import (
    QuizVault,
    answered_count,
    new_attempt,
    to_quiz_view,
    unanswered_question_ids,
)
from quiz.schemas import QuizAttempt

SOURCE = "[Source: notes · p.1 · chunk 0]"


def _mcq(question_id: str = "q1") -> Question:
    return Question(
        question_id=question_id,
        question_type="mcq",
        question_text="Which keyword defines a function in Python?",
        difficulty="easy",
        options=["def", "class", "return", "import"],
        correct_answer="0",
        explanation="The def keyword introduces a function definition.",
        source_references=[SOURCE],
    )


def _fill_blank(question_id: str = "q2") -> Question:
    return Question(
        question_id=question_id,
        question_type="fill_blank",
        question_text="A function is defined with the ____ keyword.",
        difficulty="medium",
        correct_answer="def",
        accepted_answers=["def"],
        explanation="Python uses def to define functions.",
        source_references=[SOURCE],
    )


def _short_answer(question_id: str = "q3") -> Question:
    return Question(
        question_id=question_id,
        question_type="short_answer",
        question_text="Explain why functions improve code reuse.",
        difficulty="hard",
        correct_answer="They package reusable logic behind a name.",
        key_points=["reuse", "abstraction"],
        explanation="Functions avoid repeating the same logic.",
        source_references=[SOURCE],
    )


def _generated_quiz() -> GeneratedQuiz:
    return GeneratedQuiz(
        quiz_id="quiz-1",
        resource_ids=["resource-1"],
        question_types=["mcq", "fill_blank", "short_answer"],
        requested_question_count=3,
        difficulty_distribution={"easy": 1, "medium": 1, "hard": 1},
        topic="Python basics",
        questions=[_mcq(), _fill_blank(), _short_answer()],
        context_token_count=42,
    )


def test_to_quiz_view_strips_the_answer_key() -> None:
    view = to_quiz_view(_generated_quiz())

    assert view.quiz_id == "quiz-1"
    assert len(view.questions) == 3
    assert view.questions[0].options == ["def", "class", "return", "import"]
    assert view.questions[1].options is None
    assert view.questions[2].options is None

    serialized = repr(view.model_dump())
    for forbidden in (
        "correct_answer",
        "explanation",
        "accepted_answers",
        "key_points",
        "source_references",
    ):
        assert forbidden not in serialized


def test_vault_stores_and_discards_full_quizzes() -> None:
    vault = QuizVault()
    quiz = _generated_quiz()

    vault.store(quiz)

    assert vault.get(quiz.quiz_id) is quiz
    assert len(vault) == 1
    assert vault.get("missing") is None

    vault.discard(quiz.quiz_id)

    assert vault.get(quiz.quiz_id) is None
    assert len(vault) == 0


def test_attempt_persists_and_clears_answers() -> None:
    attempt = new_attempt(to_quiz_view(_generated_quiz()))

    attempt = attempt.with_answer("q1", "2")
    assert attempt.answers == {"q1": "2"}

    attempt = attempt.with_answer("q1", "   ")
    assert attempt.answers == {}


def test_submit_is_idempotent_and_blocks_further_answers() -> None:
    attempt = QuizAttempt(quiz_id="quiz-1").with_answer("q1", "0")
    submitted = attempt.submit()

    assert submitted.submitted is True
    assert submitted.submitted_at is not None
    assert submitted.with_answer("q2", "def") is submitted
    assert submitted.submit() is submitted


def test_unanswered_and_answered_counts() -> None:
    view = to_quiz_view(_generated_quiz())
    attempt = QuizAttempt(
        quiz_id=view.quiz_id,
        answers={"q1": "0", "q2": "def", "q3": "   "},
    )

    assert unanswered_question_ids(view, attempt) == ["q3"]
    assert answered_count(view, attempt) == 2


def test_attempt_validator_drops_blank_answers() -> None:
    attempt = QuizAttempt(
        quiz_id="quiz-1",
        answers={"q1": "0", "q2": "  ", "q3": "answer"},
    )

    assert attempt.answers == {"q1": "0", "q3": "answer"}
