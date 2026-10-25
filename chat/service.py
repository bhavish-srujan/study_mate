"""Grounded RAG chat over the student's selected resources."""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

from chat.schemas import ChatAnswer, ChatMessage, ChatSource
from config.settings import Settings
from generation.prompts import CHAT_SYSTEM_PROMPT, build_chat_prompt
from rag.context_builder import build_context
from rag.retriever import Retriever
from rag.schemas import ContextPassage
from utils.errors import LLMError, RetrievalError

logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 8
_SNIPPET_CHARS = 600
_CHAT_MAX_TOKENS = 1024

NO_MATCH_ANSWER = (
    "I couldn't find anything about that in the selected resources. "
    "Try rephrasing the question, selecting more resources, or adding material "
    "in the Library."
)


class ChatClient(Protocol):
    """Minimal LLM interface needed by the chat service."""

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: object,
    ) -> str: ...


def _to_source(passage: ContextPassage) -> ChatSource:
    snippet = passage.text.strip()
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS].rstrip() + " …"
    return ChatSource(
        source_label=passage.source_label(),
        source_name=passage.source_name,
        source_type=passage.source_type,
        page_number=passage.page_number,
        start_timestamp=passage.start_timestamp,
        chunk_index=passage.chunk_index,
        score=getattr(passage, "score", None),
        snippet=snippet,
    )


class ChatService:
    """Retrieve passages, build grounded context, and ask the LLM one question."""

    def __init__(
        self,
        retriever: Retriever,
        llm: ChatClient,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    def answer(
        self,
        question: str,
        resource_ids: Sequence[str],
        history: Sequence[ChatMessage] = (),
    ) -> ChatAnswer:
        """Answer one question using only retrieved study material."""
        clean_question = (question or "").strip()
        if not clean_question:
            raise RetrievalError(
                "Chat question is empty.",
                user_message="Type a question first.",
            )

        selected = [
            str(resource_id)
            for resource_id in resource_ids
            if resource_id and str(resource_id).strip()
        ]
        if not selected:
            raise RetrievalError(
                "No resources were selected for chat.",
                user_message="Select at least one resource to chat about.",
            )

        hits = self._retriever.retrieve(clean_question, selected)
        if not hits:
            logger.info("Chat found no relevant passages for %r.", clean_question)
            return ChatAnswer(
                text=NO_MATCH_ANSWER,
                sources=(),
                context_token_count=0,
                truncated=False,
                grounded=False,
            )

        built = build_context(hits, self._settings.max_context_tokens)
        if not built.text:
            logger.info(
                "Chat retrieved passages but the context was empty for %r.",
                clean_question,
            )
            return ChatAnswer(
                text=NO_MATCH_ANSWER,
                sources=(),
                context_token_count=0,
                truncated=built.truncated,
                grounded=False,
            )

        messages = self._build_messages(clean_question, built.text, history)
        raw_answer = self._llm.complete(
            messages,
            temperature=self._settings.llm_temperature,
            max_tokens=min(
                self._settings.llm_max_completion_tokens,
                _CHAT_MAX_TOKENS,
            ),
            reasoning_effort=self._settings.llm_reasoning_effort,
        )
        answer = (raw_answer or "").strip()
        if not answer:
            raise LLMError("The chat model returned an empty response.")

        sources = tuple(_to_source(passage) for passage in built.chunks)
        logger.info(
            "Chat answered %r with %d source passage(s) and %d context tokens.",
            clean_question,
            len(sources),
            built.token_count,
        )
        return ChatAnswer(
            text=answer,
            sources=sources,
            context_token_count=built.token_count,
            truncated=built.truncated,
            grounded=True,
        )

    def _build_messages(
        self,
        question: str,
        context: str,
        history: Sequence[ChatMessage],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT}
        ]
        for message in history[-_HISTORY_LIMIT:]:
            content = message.content.strip()
            if content:
                messages.append({"role": message.role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": build_chat_prompt(context=context, question=question),
            }
        )
        return messages


def build_chat_service(
    retriever: Retriever,
    llm: ChatClient,
    settings: Settings,
) -> ChatService:
    """Construct a grounded chat service."""
    return ChatService(retriever, llm, settings)
