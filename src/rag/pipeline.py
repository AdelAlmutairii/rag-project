from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import Settings
from .llm import LocalLLM
from .prompts import (
    NOT_FOUND_REPLY,
    build_contextualize_messages,
    build_history_messages,
    build_rag_messages,
    format_context,
)
from .retriever import RetrievalResult, Retriever
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    retrieval: RetrievalResult
    model: str = ""
    # Full context string injected into the LLM prompt — stored so callers
    # can include it in conversation history for grounded multi-turn replies.
    injected_context: str = ""

    @property
    def sources(self) -> list[dict]:
        return self.retrieval.as_sources()

    @property
    def found(self) -> bool:
        return self.retrieval.has_relevant


@dataclass
class StreamResult:
    stream: Iterator[str]
    retrieval: RetrievalResult
    model: str = ""
    injected_context: str = ""

    @property
    def sources(self) -> list[dict]:
        return self.retrieval.as_sources()

    @property
    def found(self) -> bool:
        return self.retrieval.has_relevant


def _is_not_found(answer: str) -> bool:
    """Return True if the LLM signalled that the answer is not in the sources.

    A plain string equality check is fragile — the model may add punctuation,
    whitespace, or a disclaimer.  Checking for the sentinel as a substring
    (case-insensitive) handles all common variations without false positives
    on legitimate answers that happen to start with "I cannot find…".
    """
    normalized = answer.strip().upper()
    return normalized == "NOT_FOUND" or normalized.startswith("NOT_FOUND")


class RAGPipeline:
    """Orchestrates retrieval + generation for a single-turn or multi-turn QA session."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vectorstore = VectorStore(settings)
        self.retriever = Retriever(self.vectorstore, settings)
        self.llm = LocalLLM(settings)

    # ------------------------------------------------------------------
    # Query contextualization (multi-turn)
    # ------------------------------------------------------------------

    def contextualize_query(
        self,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        """Rewrite a follow-up question as a standalone question for retrieval.

        Without this, pronouns and references in follow-up questions ("What
        about its performance?") are embedded and searched literally, producing
        poor retrieval because the vectorstore has never seen those pronouns.
        This costs one extra LLM call but significantly improves multi-turn recall.
        """
        if not chat_history or not self.settings.use_query_contextualization:
            return question

        messages = build_contextualize_messages(question, chat_history)
        rewritten = self.llm.complete(messages).strip()
        if rewritten and rewritten != question:
            logger.debug("Contextualized query: '%s' → '%s'", question, rewritten)
        return rewritten or question

    # ------------------------------------------------------------------
    # Synchronous query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        source_filter: str | None = None,
    ) -> QueryResult:
        retrieval_query = self.contextualize_query(question, chat_history or [])
        retrieval = self.retriever.retrieve(retrieval_query, source_filter=source_filter)

        if not retrieval.has_relevant:
            return QueryResult(
                answer=NOT_FOUND_REPLY,
                retrieval=retrieval,
                model=self.llm.model_name,
            )

        context = format_context(retrieval.filtered)
        messages = (
            build_history_messages(context, question, chat_history)
            if chat_history
            else build_rag_messages(context, question)
        )
        answer = self.llm.complete(messages)

        if _is_not_found(answer):
            answer = NOT_FOUND_REPLY

        return QueryResult(
            answer=answer,
            retrieval=retrieval,
            model=self.llm.model_name,
            injected_context=context,
        )

    # ------------------------------------------------------------------
    # Streaming query
    # ------------------------------------------------------------------

    def stream(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        source_filter: str | None = None,
    ) -> StreamResult:
        retrieval_query = self.contextualize_query(question, chat_history or [])
        retrieval = self.retriever.retrieve(retrieval_query, source_filter=source_filter)

        if not retrieval.has_relevant:
            def _not_found() -> Iterator[str]:
                yield NOT_FOUND_REPLY

            return StreamResult(
                stream=_not_found(),
                retrieval=retrieval,
                model=self.llm.model_name,
            )

        context = format_context(retrieval.filtered)
        messages = (
            build_history_messages(context, question, chat_history)
            if chat_history
            else build_rag_messages(context, question)
        )

        return StreamResult(
            stream=self.llm.stream(messages),
            retrieval=retrieval,
            model=self.llm.model_name,
            injected_context=context,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.vectorstore.exists()

    def document_count(self) -> int:
        return self.vectorstore.count()

    def list_sources(self) -> list[str]:
        return self.vectorstore.list_sources()
