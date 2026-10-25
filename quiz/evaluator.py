"""Score a submitted quiz and build the post-submit review payload."""

from __future__ import annotations

import json
import logging
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from generation.prompts import (
    SHORT_ANSWER_GRADER_SYSTEM_PROMPT,
    build_short_answer_grading_prompt,
)
from generation.schemas import GeneratedQuiz, Question
from generation.validator import extract_json
from quiz.schemas import (
    AnswerStatus,
    QuestionResult,
    QuizAttempt,
    QuizResult,
)
from quiz.scoring import (
    answers_match,
    mcq_is_correct,
    short_answer_points,
)
from utils.errors import EvaluationError

logger = logging.getLogger(__name__)

NO_ANSWER = "No answer submitted."
_GRADER_MAX_TOKENS = 512


class ChatClient(Protocol):
    """Minimal LLM interface needed for short-answer grading."""

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: object,
    ) -> str: ...


class _GradePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int = Field(ge=0, le=2)
    justification: str = Field(min_length=1)

    @field_validator("justification")
    @classmethod
    def _clean_justification(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Justification must not be blank.")
        return cleaned


class QuizEvaluator:
    """Grade exact-match questions locally and short answers with the LLM."""

    def __init__(self, llm: ChatClient | None = None) -> None:
        self._llm = llm

    def evaluate(self, quiz: GeneratedQuiz, attempt: QuizAttempt) -> QuizResult:
        """Return one result per generated question, in quiz order."""
        if attempt.quiz_id != quiz.quiz_id:
            raise EvaluationError(
                "Attempt and quiz IDs do not match.",
                user_message="This attempt does not belong to the active quiz.",
            )

        results = [
            self._evaluate_question(question, attempt.answers.get(question.question_id))
            for question in quiz.questions
        ]
        logger.info(
            "Scored quiz %s: %d question(s), %d ungraded.",
            quiz.quiz_id,
            len(results),
            sum(1 for result in results if result.status == "ungraded"),
        )
        return QuizResult(quiz_id=quiz.quiz_id, results=results)

    def _evaluate_question(
        self,
        question: Question,
        raw_answer: str | None,
    ) -> QuestionResult:
        student_answer = (
            raw_answer.strip() if raw_answer and raw_answer.strip() else None
        )
        exact_match = question.question_type in {"mcq", "fill_blank"}
        common = {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question_text": question.question_text,
            "difficulty": question.difficulty,
            "options": list(question.options) if question.options else None,
            "student_answer": student_answer,
            "student_answer_display": _display_answer(question, student_answer),
            "correct_answer": question.correct_answer,
            "correct_answer_display": _display_correct_answer(question),
            "accepted_answers": list(question.accepted_answers),
            "key_points": list(question.key_points),
            "explanation": question.explanation,
            "source_references": list(question.source_references),
            "max_points": 1.0,
            "exact_match": exact_match,
        }

        if student_answer is None:
            return QuestionResult(
                **common,
                status="unanswered",
                points_awarded=0.0,
            )

        if question.question_type == "mcq":
            correct = mcq_is_correct(student_answer, question.correct_answer)
            return QuestionResult(
                **common,
                status="correct" if correct else "incorrect",
                points_awarded=1.0 if correct else 0.0,
            )

        if question.question_type == "fill_blank":
            correct = answers_match(
                student_answer,
                question.correct_answer,
                question.accepted_answers,
            )
            return QuestionResult(
                **common,
                status="correct" if correct else "incorrect",
                points_awarded=1.0 if correct else 0.0,
            )

        return self._evaluate_short_answer(
            question=question,
            student_answer=student_answer,
            common=common,
        )

    def _evaluate_short_answer(
        self,
        *,
        question: Question,
        student_answer: str,
        common: dict[str, object],
    ) -> QuestionResult:
        if self._llm is None:
            return QuestionResult(
                **common,
                status="ungraded",
                points_awarded=0.0,
                grading_note=(
                    "Automatic short-answer grading is unavailable because the "
                    "LLM is not configured."
                ),
            )

        try:
            grade = self._grade_short_answer(question, student_answer)
        except Exception as exc:
            logger.warning(
                "Short-answer grading failed for %s: %s",
                question.question_id,
                exc,
            )
            return QuestionResult(
                **common,
                status="ungraded",
                points_awarded=0.0,
                grading_note=(
                    "Automatic grading failed for this answer. Review it "
                    "manually or retry scoring."
                ),
            )

        status: AnswerStatus
        if grade.score == 2:
            status = "correct"
        elif grade.score == 1:
            status = "partial"
        else:
            status = "incorrect"

        return QuestionResult(
            **common,
            status=status,
            points_awarded=short_answer_points(grade.score),
            grading_note=grade.justification,
        )

    def _grade_short_answer(
        self,
        question: Question,
        student_answer: str,
    ) -> _GradePayload:
        assert self._llm is not None
        prompt = build_short_answer_grading_prompt(
            question_text=question.question_text,
            reference_answer=question.correct_answer,
            key_points=question.key_points,
            student_answer=student_answer,
        )
        raw = self._llm.complete(
            [
                {"role": "system", "content": SHORT_ANSWER_GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=_GRADER_MAX_TOKENS,
            reasoning_effort="low",
        )

        try:
            data = json.loads(extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"Could not parse the short-answer grade: {exc}"
            ) from exc

        try:
            return _GradePayload.model_validate(data)
        except ValidationError as exc:
            raise EvaluationError(
                f"Short-answer grade failed validation: {exc}"
            ) from exc


def _display_answer(question: Question, answer: str | None) -> str:
    if answer is None or not str(answer).strip():
        return NO_ANSWER
    text = str(answer).strip()
    if question.question_type == "mcq" and question.options:
        try:
            option_index = int(text)
        except ValueError:
            return text
        if 0 <= option_index < len(question.options):
            return f"{chr(65 + option_index)}. {question.options[option_index]}"
    return text


def _display_correct_answer(question: Question) -> str:
    if question.question_type == "mcq" and question.options:
        try:
            option_index = int(question.correct_answer)
        except ValueError:
            return question.correct_answer
        if 0 <= option_index < len(question.options):
            return f"{chr(65 + option_index)}. {question.options[option_index]}"
    return question.correct_answer


def build_quiz_evaluator(llm: ChatClient | None = None) -> QuizEvaluator:
    return QuizEvaluator(llm=llm)
