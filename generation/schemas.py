"""Pydantic models for generation requests, questions, and generated quizzes."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["mcq", "fill_blank", "short_answer"]
Difficulty = Literal["easy", "medium", "hard"]

_BLANK_PATTERN = re.compile(r"_{3,}")


class GenerationRequest(BaseModel):
    """Validated student configuration for one generation run."""

    model_config = ConfigDict(frozen=True)

    resource_ids: list[str] = Field(min_length=1)
    question_types: list[QuestionType] = Field(min_length=1)
    total_questions: int = Field(ge=1)
    easy: int = Field(ge=0)
    medium: int = Field(ge=0)
    hard: int = Field(ge=0)
    topic: str | None = None

    @field_validator("resource_ids")
    @classmethod
    def _clean_resource_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("Select at least one resource.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Resource IDs must be unique.")
        return cleaned

    @field_validator("question_types")
    @classmethod
    def _unique_types(cls, value: list[QuestionType]) -> list[QuestionType]:
        if len(set(value)) != len(value):
            raise ValueError("Question types must be unique.")
        return value

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _check_distribution(self) -> "GenerationRequest":
        if self.easy + self.medium + self.hard != self.total_questions:
            raise ValueError(
                "Easy + medium + hard must equal the total question count."
            )
        return self

    @property
    def difficulty_counts(self) -> dict[str, int]:
        return {"easy": self.easy, "medium": self.medium, "hard": self.hard}


class Question(BaseModel):
    """One generated question with its answer key and source references."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question_type: QuestionType
    question_text: str = Field(min_length=1)
    difficulty: Difficulty
    options: list[str] | None = None
    correct_answer: str = Field(min_length=1)
    accepted_answers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    source_references: list[str] = Field(default_factory=list)

    @field_validator(
        "question_id",
        "question_text",
        "correct_answer",
        "explanation",
    )
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text fields must not be blank.")
        return cleaned

    @field_validator("options")
    @classmethod
    def _clean_options(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [option.strip() for option in value]

    @field_validator("accepted_answers", "key_points", "source_references")
    @classmethod
    def _clean_string_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @model_validator(mode="after")
    def _check_question_shape(self) -> "Question":
        if self.question_type == "mcq":
            if self.options is None or len(self.options) != 4:
                raise ValueError("MCQ questions must have exactly four options.")
            if any(not option for option in self.options):
                raise ValueError("MCQ options must not be blank.")
            if len({option.casefold() for option in self.options}) != 4:
                raise ValueError("MCQ options must be distinct.")
            if self.correct_answer not in {"0", "1", "2", "3"}:
                raise ValueError(
                    "MCQ correct_answer must be an option index from 0 to 3."
                )
        else:
            if self.options:
                raise ValueError("Only MCQ questions may have options.")
            if (
                self.question_type == "fill_blank"
                and not _BLANK_PATTERN.search(self.question_text)
            ):
                raise ValueError(
                    "Fill-in-the-blank questions must contain a blank (___)."
                )
        return self


class QuizPayload(BaseModel):
    """The JSON object the LLM must return."""

    questions: list[Question] = Field(min_length=1)


class GeneratedQuiz(BaseModel):
    """A validated quiz ready for the Phase 4 quiz UI."""

    model_config = ConfigDict(frozen=True)

    quiz_id: str
    resource_ids: list[str]
    question_types: list[QuestionType]
    requested_question_count: int
    difficulty_distribution: dict[str, int]
    topic: str | None = None
    questions: list[Question]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    context_token_count: int = Field(default=0, ge=0)
