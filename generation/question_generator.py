"""Grounded question generation with batching, validation, and repair."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence
from uuid import uuid4

from config.settings import Settings
from generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from generation.schemas import GeneratedQuiz, GenerationRequest, Question
from generation.validator import parse_questions, validate_questions
from rag.context_builder import BuiltContext, build_context
from rag.retriever import Retriever
from utils.errors import GenerationError, StudyMateError

logger = logging.getLogger(__name__)

DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_BATCH_QUESTIONS = 5
DEFAULT_BATCH_CHUNKS = 4


class ChatClient(Protocol):
    """Minimal LLM interface used by the generator."""

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: object,
    ) -> str: ...


@dataclass(frozen=True)
class _GenerationOutput:
    questions: list[Question]
    context_token_count: int
    allowed_sources: tuple[str, ...]


class QuestionGenerator:
    """Retrieve context, call the LLM in batches, and validate the result."""

    def __init__(
        self,
        retriever: Retriever,
        llm: ChatClient,
        settings: Settings,
        *,
        max_repair_attempts: int = 2,
        batch_questions: int = DEFAULT_BATCH_QUESTIONS,
        batch_chunks: int = DEFAULT_BATCH_CHUNKS,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        if batch_questions <= 0 or batch_chunks <= 0:
            raise ValueError("batch sizes must be greater than 0")

        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._max_repair_attempts = max_repair_attempts
        self._batch_questions = batch_questions
        self._batch_chunks = batch_chunks

    def generate(self, request: GenerationRequest) -> GeneratedQuiz:
        """Generate and validate a complete quiz for one request."""
        if request.total_questions > self._settings.max_questions:
            raise GenerationError(
                f"Requested {request.total_questions} questions, but the maximum "
                f"is {self._settings.max_questions}.",
                user_message=(
                    f"The maximum is {self._settings.max_questions} questions per "
                    "quiz. Reduce the total and try again."
                ),
            )

        if request.topic:
            output = self._generate_for_topic(request)
        else:
            output = self._generate_for_resources(request)

        final = validate_questions(
            output.questions,
            expected_total=request.total_questions,
            expected_difficulty=request.difficulty_counts,
            allowed_types=request.question_types,
            allowed_sources=output.allowed_sources,
            require_all_types=True,
        )
        if not final.valid:
            detail = "Final validation failed: " + " ".join(final.errors[:8])
            logger.warning(
                "Question generation returned an invalid quiz for %d question(s): %s",
                request.total_questions,
                detail,
            )
            raise GenerationError(
                detail,
                user_message=(
                    "The questions could not be validated against your requested "
                    "configuration. Try again, use a broader topic, or reduce the "
                    "question count."
                ),
            )

        logger.info(
            "Generated a validated quiz with %d question(s) (%s).",
            len(final.questions),
            ", ".join(
                f"{difficulty}={count}"
                for difficulty, count in final.difficulty_counts.items()
            ),
        )
        questions = [
            question.model_copy(update={"question_id": f"q{index}"})
            for index, question in enumerate(final.questions, start=1)
        ]
        return GeneratedQuiz(
            quiz_id=uuid4().hex,
            resource_ids=list(request.resource_ids),
            question_types=list(request.question_types),
            requested_question_count=request.total_questions,
            difficulty_distribution=request.difficulty_counts,
            topic=request.topic,
            questions=questions,
            context_token_count=output.context_token_count,
        )

    def _generate_for_topic(self, request: GenerationRequest) -> _GenerationOutput:
        topic = request.topic or ""
        hits = self._retriever.retrieve(
            topic,
            request.resource_ids,
            top_k=self._settings.retrieval_top_k,
        )
        if not hits:
            raise GenerationError(
                f"No indexed chunks matched topic {topic!r}.",
                user_message=(
                    "No relevant material was found for that topic in the selected "
                    "resources. Try a broader topic or select more resources."
                ),
            )

        built = build_context(hits, self._settings.max_context_tokens)
        if not built.text:
            raise GenerationError(
                "Retrieved context was empty after token budgeting.",
                user_message=(
                    "The retrieved material did not fit in the context window. "
                    "Try a narrower topic or fewer resources."
                ),
            )

        sources = _sources_for(built)
        questions = self._generate_from_contexts(
            request=request,
            contexts=[built],
            allowed_sources=sources,
        )
        return _GenerationOutput(
            questions=questions,
            context_token_count=built.token_count,
            allowed_sources=sources,
        )

    def _generate_for_resources(
        self,
        request: GenerationRequest,
    ) -> _GenerationOutput:
        sample_limit = max(
            self._settings.retrieval_top_k,
            min(
                self._settings.max_questions,
                request.total_questions * 2,
            ),
        )
        passages = self._retriever.sample_chunks(
            request.resource_ids,
            limit=sample_limit,
        )
        if not passages:
            raise GenerationError(
                "No indexed chunks were available for representative sampling.",
                user_message=(
                    "No indexed material was found in the selected resources. "
                    "Upload and index a resource first."
                ),
            )

        contexts: list[BuiltContext] = []
        sources: set[str] = set()
        for start in range(0, len(passages), self._batch_chunks):
            built = build_context(
                passages[start : start + self._batch_chunks],
                self._settings.max_context_tokens,
            )
            if built.text:
                contexts.append(built)
                sources.update(_sources_for(built))

        if not contexts:
            raise GenerationError(
                "Representative chunks produced no usable context.",
                user_message=(
                    "The selected material could not be prepared for generation. "
                    "Try a different resource."
                ),
            )

        questions = self._generate_from_contexts(
            request=request,
            contexts=contexts,
            allowed_sources=tuple(sorted(sources)),
        )
        return _GenerationOutput(
            questions=questions,
            context_token_count=max(built.token_count for built in contexts),
            allowed_sources=tuple(sorted(sources)),
        )

    def _generate_from_contexts(
        self,
        *,
        request: GenerationRequest,
        contexts: Sequence[BuiltContext],
        allowed_sources: Sequence[str],
    ) -> list[Question]:
        accepted: list[Question] = []
        max_batches = (
            math.ceil(request.total_questions / self._batch_questions)
            + len(contexts)
        )

        for batch_number in range(1, max_batches + 1):
            if len(accepted) >= request.total_questions:
                break

            context = contexts[(batch_number - 1) % len(contexts)]
            remaining_total = request.total_questions - len(accepted)
            target_total = min(remaining_total, self._batch_questions)
            target_difficulty = _allocate_difficulty(
                _remaining_difficulty(request.difficulty_counts, accepted),
                target_total,
            )
            if sum(target_difficulty.values()) == 0:
                break

            before = len(accepted)
            batch = self._generate_one_batch(
                context=context.text,
                request=request,
                target_total=target_total,
                target_difficulty=target_difficulty,
                allowed_sources=allowed_sources,
                existing_questions=accepted,
                batch_note=f"Batch {batch_number} of at most {max_batches}.",
            )
            accepted.extend(batch)

            if len(accepted) == before and batch_number >= len(contexts):
                # Every context has had a chance; stop rather than loop forever.
                break

        return accepted

    def _generate_one_batch(
        self,
        *,
        context: str,
        request: GenerationRequest,
        target_total: int,
        target_difficulty: Mapping[str, int],
        allowed_sources: Sequence[str],
        existing_questions: Sequence[Question],
        batch_note: str,
    ) -> list[Question]:
        accepted: list[Question] = []
        errors: list[str] = []

        for attempt in range(self._max_repair_attempts + 1):
            remaining_total = target_total - len(accepted)
            if remaining_total <= 0:
                break

            remaining_difficulty = _remaining_difficulty(
                target_difficulty,
                accepted,
            )
            prompt = build_generation_prompt(
                context=context,
                question_types=request.question_types,
                count=remaining_total,
                difficulty_counts=remaining_difficulty,
                topic=request.topic,
                existing_questions=[
                    question.question_text
                    for question in (*existing_questions, *accepted)
                ],
                previous_errors=errors,
                batch_note=batch_note,
            )

            try:
                raw = self._llm.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._settings.llm_temperature,
                    max_tokens=self._settings.llm_max_completion_tokens,
                    reasoning_effort=self._settings.llm_reasoning_effort,
                )
            except StudyMateError:
                raise
            except Exception as exc:
                raise GenerationError(
                    f"LLM question-generation call failed: {exc}"
                ) from exc

            try:
                batch = parse_questions(raw)
            except ValueError as exc:
                errors = [str(exc)]
                logger.warning(
                    "Question generation parse failed (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                continue

            result = validate_questions(
                batch,
                expected_total=remaining_total,
                expected_difficulty=remaining_difficulty,
                allowed_types=request.question_types,
                allowed_sources=allowed_sources,
                existing_questions=tuple(existing_questions) + tuple(accepted),
            )
            selected = _take_within_quota(result.questions, remaining_difficulty)
            accepted.extend(selected)

            if result.valid and len(selected) == len(batch):
                break

            errors = list(result.errors)
            if len(selected) < len(result.questions):
                errors.append(
                    "Some questions exceeded the requested difficulty counts; "
                    "return only the missing counts."
                )
            logger.warning(
                "Question generation validation failed (attempt %d): %s",
                attempt + 1,
                "; ".join(errors[:6]),
            )

        return accepted


def _sources_for(built: BuiltContext) -> tuple[str, ...]:
    return tuple(sorted({passage.source_name for passage in built.chunks}))


def _remaining_difficulty(
    expected: Mapping[str, int],
    accepted: Sequence[Question],
) -> dict[str, int]:
    counts = Counter(question.difficulty for question in accepted)
    return {
        difficulty: max(0, int(expected.get(difficulty, 0)) - counts.get(difficulty, 0))
        for difficulty in DIFFICULTIES
    }


def _allocate_difficulty(
    remaining: Mapping[str, int],
    count: int,
) -> dict[str, int]:
    """Allocate a batch across difficulties, giving each a chance first."""
    allocation = {difficulty: 0 for difficulty in DIFFICULTIES}
    budget = count

    for difficulty in DIFFICULTIES:
        if budget <= 0:
            break
        if remaining.get(difficulty, 0) > 0:
            allocation[difficulty] = 1
            budget -= 1

    while budget > 0:
        candidates = [
            difficulty
            for difficulty in DIFFICULTIES
            if allocation[difficulty] < remaining.get(difficulty, 0)
        ]
        if not candidates:
            break
        difficulty = max(
            candidates,
            key=lambda item: remaining.get(item, 0) - allocation[item],
        )
        allocation[difficulty] += 1
        budget -= 1

    return allocation


def _take_within_quota(
    questions: Sequence[Question],
    quotas: Mapping[str, int],
) -> list[Question]:
    remaining = {difficulty: int(quotas.get(difficulty, 0)) for difficulty in DIFFICULTIES}
    selected: list[Question] = []

    for question in questions:
        if remaining.get(question.difficulty, 0) <= 0:
            continue
        selected.append(question)
        remaining[question.difficulty] -= 1

    return selected


def build_question_generator(
    retriever: Retriever,
    llm: ChatClient,
    settings: Settings,
) -> QuestionGenerator:
    return QuestionGenerator(retriever=retriever, llm=llm, settings=settings)
