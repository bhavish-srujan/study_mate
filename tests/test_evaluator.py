"""Evaluator tests. No Streamlit or network calls."""

from __future__ import annotations

import pytest

from generation.schemas import GeneratedQuiz, Question
from quiz.evaluator import QuizEvaluator, build_quiz_evaluator
from quiz.schemas import QuizAttempt
from utils.errors import EvaluationError, LLMError

SOURCE = "[Source: notes · p.1 · chunk 0]"


def _mcq() -> Question:
    return Question(
        question_id="q1",
        question_type="mcq",
        question_text="Which keyword defines a function in Python?",
        difficulty="easy",
        options=["def", "class", "return", "import"],
        correct_answer="0",
        explanation="The def keyword introduces a function definition.",
        source_references=[SOURCE],
    )


def _fill_blank() -> Question:
    return Question(
        question_id="q2",
        question_type="fill_blank",
        question_text="A function is defined with the ____ keyword.",
        difficulty="medium",
        correct_answer="def",
        accepted_answers=["define"],
        explanation="Python uses def before the function name.",
        source_references=[SOURCE],
    )


def _short_answer() -> Question:
    return Question(
        question_id="q3",
        question_type="short_answer",
        question_text="Why are functions useful?",
        difficulty="hard",
        correct_answer="Functions package reusable logic and make programs easier to read.",
        key_points=[
            "Reuse code",
            "Organize programs",
        ],
        explanation="Functions avoid repetition and improve structure.",
        source_references=[SOURCE],
    )


def _quiz() -> GeneratedQuiz:
    return GeneratedQuiz(
        quiz_id="quiz-1",
        resource_ids=["resource-1"],
        question_types=["mcq", "fill_blank", "short_answer"],
        requested_question_count=3,
        difficulty_distribution={"easy": 1, "medium": 1, "hard": 1},
        questions=[_mcq(), _fill_blank(), _short_answer()],
    )


class FakeGrader:
    def __init__(self, response: str = '{"score": 1, "justification": "Partial."}') -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.response


def test_evaluator_scores_exact_questions_and_partial_short_answer() -> None:
    grader = FakeGrader(
        '{"score": 1, "justification": "Covers reuse but not program structure."}'
    )
    evaluator = QuizEvaluator(llm=grader)

    result = evaluator.evaluate(
        _quiz(),
        QuizAttempt(
            quiz_id="quiz-1",
            answers={
                "q1": "0",
                "q2": " DEF. ",
                "q3": "Functions let you reuse code.",
            },
        ),
    )

    assert result.total_questions == 3
    assert result.answered_count == 3
    assert result.correct_count == 2
    assert result.partial_count == 1
    assert result.incorrect_count == 0
    assert result.unanswered_count == 0
    assert result.ungraded_count == 0
    assert result.score_points == 2.5
    assert result.max_points == 3.0
    assert result.percentage == pytest.approx(83.333333, abs=1e-5)

    mcq, blank, short = result.results
    assert mcq.status == "correct"
    assert mcq.exact_match is True
    assert mcq.student_answer_display == "A. def"
    assert mcq.correct_answer_display == "A. def"
    assert blank.status == "correct"
    assert blank.exact_match is True
    assert short.status == "partial"
    assert short.exact_match is False
    assert short.points_awarded == 0.5
    assert short.grading_note == "Covers reuse but not program structure."
    assert short.source_references == [SOURCE]
    assert len(grader.calls) == 1
    assert grader.calls[0]["temperature"] == 0.0
    assert grader.calls[0]["reasoning_effort"] == "low"


def test_unanswered_questions_are_counted_and_not_graded_by_llm() -> None:
    grader = FakeGrader()
    evaluator = QuizEvaluator(llm=grader)

    result = evaluator.evaluate(
        _quiz(),
        QuizAttempt(quiz_id="quiz-1", answers={"q1": "0"}),
    )

    assert result.correct_count == 1
    assert result.unanswered_count == 2
    assert result.score_points == 1.0
    assert result.max_points == 3.0
    assert result.results[1].status == "unanswered"
    assert result.results[1].student_answer_display == "No answer submitted."
    assert result.results[2].status == "unanswered"
    assert grader.calls == []


def test_short_answer_is_ungraded_when_no_llm_is_configured() -> None:
    result = QuizEvaluator(llm=None).evaluate(
        _quiz(),
        QuizAttempt(
            quiz_id="quiz-1",
            answers={
                "q1": "0",
                "q2": "def",
                "q3": "Functions let you reuse code.",
            },
        ),
    )

    assert result.correct_count == 2
    assert result.ungraded_count == 1
    assert result.graded_count == 2
    assert result.score_points == 2.0
    assert result.max_points == 2.0
    assert result.percentage == 100.0
    assert result.results[2].status == "ungraded"
    assert "not configured" in (result.results[2].grading_note or "")


def test_short_answer_grading_failure_does_not_block_exact_scores() -> None:
    class FailingGrader:
        def complete(self, messages, **kwargs):
            raise LLMError("service unavailable")

    result = QuizEvaluator(llm=FailingGrader()).evaluate(
        _quiz(),
        QuizAttempt(
            quiz_id="quiz-1",
            answers={
                "q1": "1",
                "q2": "def",
                "q3": "Functions let you reuse code.",
            },
        ),
    )

    assert result.correct_count == 1
    assert result.incorrect_count == 1
    assert result.ungraded_count == 1
    assert result.results[2].status == "ungraded"
    assert "failed" in (result.results[2].grading_note or "")


def test_invalid_grader_json_marks_only_that_answer_ungraded() -> None:
    result = QuizEvaluator(llm=FakeGrader("not json")).evaluate(
        _quiz(),
        QuizAttempt(
            quiz_id="quiz-1",
            answers={
                "q1": "0",
                "q2": "def",
                "q3": "Functions let you reuse code.",
            },
        ),
    )

    assert result.results[0].status == "correct"
    assert result.results[1].status == "correct"
    assert result.results[2].status == "ungraded"


@pytest.mark.parametrize(
    ("score", "status", "points"),
    [
        (2, "correct", 1.0),
        (1, "partial", 0.5),
        (0, "incorrect", 0.0),
    ],
)
def test_short_answer_score_mapping(
    score: int,
    status: str,
    points: float,
) -> None:
    grader = FakeGrader(
        f'{{"score": {score}, "justification": "Grade {score}."}}'
    )
    result = QuizEvaluator(llm=grader).evaluate(
        _quiz(),
        QuizAttempt(
            quiz_id="quiz-1",
            answers={"q3": "Functions let you reuse code."},
        ),
    )

    short = result.results[2]
    assert short.status == status
    assert short.points_awarded == points


def test_attempt_and_quiz_id_mismatch_is_rejected() -> None:
    with pytest.raises(EvaluationError):
        QuizEvaluator().evaluate(
            _quiz(),
            QuizAttempt(quiz_id="other-quiz", answers={}),
        )


def test_build_quiz_evaluator_returns_an_evaluator() -> None:
    assert isinstance(build_quiz_evaluator(), QuizEvaluator)
