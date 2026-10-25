"""Quiz view conversion, answer tracking, and the server-side answer vault."""

from __future__ import annotations

from threading import Lock

from generation.schemas import GeneratedQuiz
from quiz.schemas import QuizAttempt, QuizQuestionView, QuizView


def to_quiz_view(quiz: GeneratedQuiz) -> QuizView:
    """Strip answers, explanations, and sources from a generated quiz.

    The result is safe to place in ``st.session_state`` and render before
    submission. The full ``GeneratedQuiz`` remains in ``QuizVault``.
    """
    return QuizView(
        quiz_id=quiz.quiz_id,
        resource_ids=list(quiz.resource_ids),
        question_types=list(quiz.question_types),
        requested_question_count=quiz.requested_question_count,
        difficulty_distribution=dict(quiz.difficulty_distribution),
        topic=quiz.topic,
        questions=[
            QuizQuestionView(
                question_id=question.question_id,
                question_type=question.question_type,
                question_text=question.question_text,
                difficulty=question.difficulty,
                options=(
                    list(question.options)
                    if question.options is not None
                    else None
                ),
            )
            for question in quiz.questions
        ],
        created_at=quiz.created_at,
        context_token_count=quiz.context_token_count,
    )


class QuizVault:
    """In-process storage for full quizzes, including answer keys.

    This intentionally lives outside ``st.session_state`` so the key is not
    reachable from the active quiz UI or browser session before submission.
    """

    def __init__(self) -> None:
        self._quizzes: dict[str, GeneratedQuiz] = {}
        self._lock = Lock()

    def store(self, quiz: GeneratedQuiz) -> None:
        with self._lock:
            self._quizzes[quiz.quiz_id] = quiz

    def get(self, quiz_id: str) -> GeneratedQuiz | None:
        with self._lock:
            return self._quizzes.get(quiz_id)

    def discard(self, quiz_id: str) -> None:
        with self._lock:
            self._quizzes.pop(quiz_id, None)

    def clear(self) -> None:
        with self._lock:
            self._quizzes.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._quizzes)


def new_attempt(view: QuizView) -> QuizAttempt:
    return QuizAttempt(quiz_id=view.quiz_id)


def unanswered_question_ids(
    view: QuizView,
    attempt: QuizAttempt,
) -> list[str]:
    """Return question IDs with no non-whitespace answer."""
    return [
        question.question_id
        for question in view.questions
        if not attempt.answers.get(question.question_id, "").strip()
    ]


def answered_count(view: QuizView, attempt: QuizAttempt) -> int:
    return len(view.questions) - len(unanswered_question_ids(view, attempt))
