from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import Settings
from .llm import LocalLLM
from .prompts import (
    NOT_FOUND_REPLY,
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

    @property
    def sources(self) -> list[dict]:
        return self.retrieval.as_sources()

    @property
    def found(self) -> bool:
        return self.retrieval.has_relevant


class RAGPipeline:
    """Orchestrates retrieval + generation for a single-turn or multi-turn QA session."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vectorstore = VectorStore(settings)
        self.retriever = Retriever(self.vectorstore, settings)
        self.llm = LocalLLM(settings)

    # ------------------------------------------------------------------
    # Synchronous query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        source_filter: str | None = None,
    ) -> QueryResult:
        retrieval = self.retriever.retrieve(question, source_filter=source_filter)

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

        if answer.strip() == "NOT_FOUND":
            answer = NOT_FOUND_REPLY

        return QueryResult(
            answer=answer,
            retrieval=retrieval,
            model=self.llm.model_name,
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
        retrieval = self.retriever.retrieve(question, source_filter=source_filter)

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
