"""Pure scoring helpers for exact-match questions and result totals."""

from __future__ import annotations

import unicodedata
from typing import Iterable

# Only punctuation at the edges is ignored. Internal punctuation can carry
# meaning (for example "C++" or "self-contained"), so it is preserved.
_EDGE_PUNCTUATION = " \t\n\r\f\v.,;:!?\"'`()[]{}<>"
_DASHES = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-"})


def normalize_exact_answer(text: str | None) -> str:
    """Normalize a short text answer for conservative exact comparison.

    Rules:
    - Unicode NFKC normalization, casefold, and whitespace collapsing.
    - Common dash variants become ``-``.
    - Leading/trailing punctuation and wrapping quotes/brackets are removed.
    - Internal punctuation and word order are preserved.
    """
    value = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    value = " ".join(value.casefold().split())

    while value and value[0] in _EDGE_PUNCTUATION:
        value = value[1:].lstrip()
    while value and value[-1] in _EDGE_PUNCTUATION:
        value = value[:-1].rstrip()
    return value


def answers_match(
    student_answer: str | None,
    correct_answer: str,
    accepted_answers: Iterable[str] = (),
) -> bool:
    """Return whether a fill-in-the-blank answer matches the key or an alias."""
    normalized = normalize_exact_answer(student_answer)
    if not normalized:
        return False

    candidates = [correct_answer, *accepted_answers]
    return any(
        normalized == normalize_exact_answer(candidate)
        for candidate in candidates
        if candidate and str(candidate).strip()
    )


def mcq_is_correct(
    student_answer: str | None,
    correct_answer: str,
) -> bool:
    """Compare a stored MCQ option index to the answer key."""
    if student_answer is None or not str(student_answer).strip():
        return False
    try:
        return int(str(student_answer).strip()) == int(str(correct_answer).strip())
    except (TypeError, ValueError):
        return False


def short_answer_points(score: int) -> float:
    """Convert a 0–2 short-answer grade to the 0–1 quiz weight."""
    if score not in (0, 1, 2):
        raise ValueError("Short-answer score must be 0, 1, or 2.")
    return {0: 0.0, 1: 0.5, 2: 1.0}[score]


def percentage(score_points: float, max_points: float) -> float:
    """Return a clamped percentage, or 0 when there is nothing to grade."""
    if max_points <= 0:
        return 0.0
    return max(0.0, min(100.0, (score_points / max_points) * 100.0))
