"""Parse and validate LLM question output.

Pydantic handles structural rules (MCQ shape, fill-blank marker, required
text). This module adds cross-question checks: requested counts, allowed types,
source traceability, and near-duplicate detection.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Mapping, Sequence

from pydantic import ValidationError

from generation.schemas import Question, QuestionType, QuizPayload


@dataclass(frozen=True)
class ValidationResult:
    """Accepted questions plus human-readable validation errors."""

    questions: list[Question]
    errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def difficulty_counts(self) -> dict[str, int]:
        counts = Counter(question.difficulty for question in self.questions)
        return {
            difficulty: counts.get(difficulty, 0)
            for difficulty in ("easy", "medium", "hard")
        }


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", value).strip()


def extract_json(raw: str) -> str:
    """Return the first JSON object/array found in an LLM response."""
    text = _strip_code_fence(raw)
    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            return text[start : end + 1]
    raise ValueError("The LLM response did not contain a JSON object or array.")


def format_pydantic_error(exc: ValidationError) -> str:
    """Compact, student-safe Pydantic errors for the repair prompt."""
    parts: list[str] = []
    for error in exc.errors()[:10]:
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def parse_questions(raw: str) -> list[Question]:
    """Parse and structurally validate the LLM's JSON response."""
    text = extract_json(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc.msg}.") from exc

    if isinstance(data, list):
        data = {"questions": data}
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object with a 'questions' list.")

    try:
        payload = QuizPayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError(format_pydantic_error(exc)) from exc
    return payload.questions


def _normalize_question_text(text: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", text.casefold())
    return " ".join(value.split())


def is_duplicate(first: Question, second: Question) -> bool:
    """Detect exact and near-duplicate question stems."""
    left = _normalize_question_text(first.question_text)
    right = _normalize_question_text(second.question_text)
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.85


def _normalize_reference(text: str) -> str:
    return " ".join(text.casefold().split())


def _references_source(references: Sequence[str], allowed_sources: Sequence[str]) -> bool:
    normalized_sources = [
        _normalize_reference(source) for source in allowed_sources if source.strip()
    ]
    if not normalized_sources:
        return True

    for reference in references:
        normalized = _normalize_reference(reference)
        if any(
            source in normalized or normalized in source
            for source in normalized_sources
        ):
            return True
    return False


def validate_questions(
    questions: Sequence[Question],
    *,
    expected_total: int | None = None,
    expected_difficulty: Mapping[str, int] | None = None,
    allowed_types: Sequence[QuestionType] | None = None,
    allowed_sources: Sequence[str] = (),
    existing_questions: Sequence[Question] = (),
    require_all_types: bool = False,
) -> ValidationResult:
    """Validate a batch or a complete quiz.

    Questions that pass per-question checks are returned even when count or
    difficulty errors exist, so the generator can keep valid work and request
    only the missing questions in a repair call.
    """
    errors: list[str] = []
    accepted: list[Question] = []

    if expected_total is not None and len(questions) != expected_total:
        errors.append(f"Expected {expected_total} questions, got {len(questions)}.")

    allowed_type_set = set(allowed_types) if allowed_types else None

    for index, question in enumerate(questions, start=1):
        question_errors: list[str] = []

        if allowed_type_set and question.question_type not in allowed_type_set:
            question_errors.append(
                f"Question {index} has type {question.question_type!r}, which was "
                "not selected."
            )

        if allowed_sources and not _references_source(
            question.source_references, allowed_sources
        ):
            question_errors.append(
                f"Question {index} needs at least one source reference copied "
                "from the supplied context."
            )

        if any(
            is_duplicate(question, other)
            for other in (*existing_questions, *accepted)
        ):
            question_errors.append(
                f"Question {index} duplicates or closely paraphrases an earlier "
                "question."
            )

        if question_errors:
            errors.extend(question_errors)
        else:
            accepted.append(question)

    if expected_difficulty is not None:
        counts = Counter(question.difficulty for question in accepted)
        for difficulty in ("easy", "medium", "hard"):
            expected = int(expected_difficulty.get(difficulty, 0))
            actual = counts.get(difficulty, 0)
            if actual != expected:
                errors.append(
                    f"Expected {expected} {difficulty} question(s), got {actual}."
                )

    if require_all_types and allowed_type_set and len(accepted) >= len(allowed_type_set):
        present = {question.question_type for question in accepted}
        missing = sorted(allowed_type_set - present)
        if missing:
            errors.append(
                "The quiz must include every selected question type; missing: "
                + ", ".join(missing)
                + "."
            )

    return ValidationResult(questions=accepted, errors=errors)
