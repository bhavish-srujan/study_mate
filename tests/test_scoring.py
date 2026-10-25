"""Pure scoring tests: normalization, exact matches, and percentages."""

from __future__ import annotations

import pytest

from quiz.scoring import (
    answers_match,
    mcq_is_correct,
    normalize_exact_answer,
    percentage,
    short_answer_points,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  DEF.  ", "def"),
        ('"A list."', "a list"),
        ("self-contained", "self-contained"),
        ("C++", "c++"),
        ("  ", ""),
        ("word\u2014word", "word-word"),
        ("  multiple   spaces  ", "multiple spaces"),
    ],
)
def test_normalize_exact_answer(raw: str, expected: str) -> None:
    assert normalize_exact_answer(raw) == expected


def test_answers_match_uses_key_and_alternatives() -> None:
    assert answers_match(" DEF. ", "def")
    assert answers_match("define", "def", ["define"])
    assert answers_match("  (def)  ", "def")


def test_answers_match_rejects_blank_and_wrong_answers() -> None:
    assert not answers_match("", "def")
    assert not answers_match("   ", "def")
    assert not answers_match("class", "def", ["define"])


def test_mcq_index_comparison() -> None:
    assert mcq_is_correct("0", "0")
    assert mcq_is_correct(" 2 ", "2")
    assert not mcq_is_correct("1", "0")
    assert not mcq_is_correct(None, "0")
    assert not mcq_is_correct("A", "0")


@pytest.mark.parametrize(
    ("score", "expected"),
    [(2, 1.0), (1, 0.5), (0, 0.0)],
)
def test_short_answer_points(score: int, expected: float) -> None:
    assert short_answer_points(score) == expected


def test_short_answer_points_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        short_answer_points(3)
    with pytest.raises(ValueError):
        short_answer_points(-1)


def test_percentage() -> None:
    assert percentage(2.5, 3.0) == pytest.approx(83.333333, abs=1e-5)
    assert percentage(0, 3.0) == 0.0
    assert percentage(3.0, 3.0) == 100.0
    assert percentage(4.0, 3.0) == 100.0
    assert percentage(1.0, 0.0) == 0.0
