"""Quiz models for the active session and the post-submit review.

Before submission, ``st.session_state`` holds only ``QuizView`` and
``QuizAttempt``. Those models contain no answer, explanation, accepted
alternatives, key points, or source references; the full key stays in
``QuizVault``. ``QuestionResult`` and ``QuizResult`` contain key data and are
created and stored only after the attempt is submitted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from generation.schemas import Difficulty, QuestionType
from quiz.scoring import percentage

AnswerStatus = Literal["correct", "partial", "incorrect", "unanswered", "ungraded"]


class QuizQuestionView(BaseModel):
    """A question rendered in the quiz UI, with no answer or explanation."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question_type: QuestionType
    question_text: str = Field(min_length=1)
    difficulty: Difficulty
    options: list[str] | None = None


class QuizView(BaseModel):
    """The public quiz payload held in ``st.session_state``."""

    model_config = ConfigDict(frozen=True)

    quiz_id: str = Field(min_length=1)
    resource_ids: list[str] = Field(min_length=1)
    question_types: list[QuestionType] = Field(min_length=1)
    requested_question_count: int = Field(ge=1)
    difficulty_distribution: dict[str, int]
    topic: str | None = None
    questions: list[QuizQuestionView] = Field(min_length=1)
    created_at: datetime
    context_token_count: int = Field(default=0, ge=0)


class QuizAttempt(BaseModel):
    """Student answers and submission state for one quiz."""

    model_config = ConfigDict(frozen=True)

    quiz_id: str = Field(min_length=1)
    answers: dict[str, str] = Field(default_factory=dict)
    submitted: bool = False
    submitted_at: datetime | None = None

    @field_validator("answers")
    @classmethod
    def _clean_answers(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for question_id, answer in value.items():
            if question_id and answer is not None and str(answer).strip():
                cleaned[str(question_id)] = str(answer)
        return cleaned

    def with_answer(self, question_id: str, value: str | None) -> "QuizAttempt":
        """Return a copy with one answer added, replaced, or cleared."""
        if self.submitted:
            return self

        answers = dict(self.answers)
        text = "" if value is None else str(value)
        if text.strip():
            answers[question_id] = text
        else:
            answers.pop(question_id, None)
        return self.model_copy(update={"answers": answers})

    def submit(self) -> "QuizAttempt":
        """Mark the attempt submitted; submitting twice is a no-op."""
        if self.submitted:
            return self
        return self.model_copy(
            update={
                "submitted": True,
                "submitted_at": datetime.now(timezone.utc),
            }
        )


class QuestionResult(BaseModel):
    """One graded question, including the key for post-submit review."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question_type: QuestionType
    question_text: str = Field(min_length=1)
    difficulty: Difficulty
    options: list[str] | None = None
    student_answer: str | None = None
    student_answer_display: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    correct_answer_display: str = Field(min_length=1)
    accepted_answers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    source_references: list[str] = Field(default_factory=list)
    status: AnswerStatus
    points_awarded: float = Field(ge=0)
    max_points: float = Field(default=1.0, ge=0)
    exact_match: bool
    grading_note: str | None = None


class QuizResult(BaseModel):
    """Scored quiz plus the data needed for a post-submit review."""

    model_config = ConfigDict(frozen=True)

    quiz_id: str = Field(min_length=1)
    results: list[QuestionResult] = Field(min_length=1)
    graded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def total_questions(self) -> int:
        return len(self.results)

    @property
    def answered_count(self) -> int:
        return sum(
            1 for result in self.results if result.status != "unanswered"
        )

    @property
    def unanswered_count(self) -> int:
        return sum(1 for result in self.results if result.status == "unanswered")

    @property
    def correct_count(self) -> int:
        return sum(1 for result in self.results if result.status == "correct")

    @property
    def partial_count(self) -> int:
        return sum(1 for result in self.results if result.status == "partial")

    @property
    def incorrect_count(self) -> int:
        return sum(1 for result in self.results if result.status == "incorrect")

    @property
    def ungraded_count(self) -> int:
        return sum(1 for result in self.results if result.status == "ungraded")

    @property
    def graded_count(self) -> int:
        return self.total_questions - self.ungraded_count

    @property
    def score_points(self) -> float:
        return sum(result.points_awarded for result in self.results)

    @property
    def max_points(self) -> float:
        return sum(
            result.max_points
            for result in self.results
            if result.status != "ungraded"
        )

    @property
    def percentage(self) -> float:
        return percentage(self.score_points, self.max_points)
