"""Tests for the RAGPipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.pipeline import QueryResult, RAGPipeline, StreamResult
from rag.retriever import RetrievalResult


def _make_retrieval(relevant: bool = True) -> RetrievalResult:
    """Build a fake RetrievalResult for testing."""
    from langchain_core.documents import Document

    doc = Document(
        "Machine learning is a branch of AI.",
        metadata={"source": "/docs/ml.pdf", "page": 1, "chunk_id": "abc123"},
    )
    hit = (doc, 0.7)
    filtered = [hit] if relevant else []
    return RetrievalResult(
        hits=[hit],
        filtered=filtered,
        best_score=0.7,
        threshold=1.4,
        query="what is ML?",
    )


class TestQueryResult:
    def test_found_true_when_relevant(self):
        result = QueryResult(answer="Answer here.", retrieval=_make_retrieval(relevant=True))
        assert result.found is True

    def test_found_false_when_not_relevant(self):
        result = QueryResult(answer="Not found.", retrieval=_make_retrieval(relevant=False))
        assert result.found is False

    def test_sources_list(self):
        result = QueryResult(answer="A", retrieval=_make_retrieval(relevant=True))
        sources = result.sources
        assert len(sources) == 1
        assert sources[0]["source"] == "ml.pdf"


class TestRAGPipeline:
    @pytest.fixture()
    def pipeline(self, settings, mock_llm):
        """Pipeline with mocked vectorstore and LLM — no I/O occurs."""
        with (
            patch("rag.pipeline.VectorStore") as MockVS,
            patch("rag.pipeline.LocalLLM", return_value=mock_llm),
        ):
            mock_vs_instance = MagicMock()
            mock_vs_instance.exists.return_value = True
            mock_vs_instance.count.return_value = 42
            mock_vs_instance.list_sources.return_value = ["ml.pdf"]
            mock_vs_instance.similarity_search_with_score.return_value = [
                _make_retrieval().filtered[0]
            ]
            MockVS.return_value = mock_vs_instance

            p = RAGPipeline(settings)
            # Inject the mock vectorstore into the retriever too
            p.retriever.vectorstore = mock_vs_instance
            p.llm = mock_llm
            yield p

    def test_query_returns_query_result(self, pipeline):
        result = pipeline.query("what is ML?")
        assert isinstance(result, QueryResult)

    def test_query_calls_llm_complete(self, pipeline, mock_llm):
        pipeline.query("what is ML?")
        mock_llm.complete.assert_called_once()

    def test_query_not_found_when_no_relevant_docs(self, pipeline, settings):
        pipeline.retriever.vectorstore.similarity_search_with_score.return_value = [
            _make_retrieval(relevant=False).hits[0]  # score will be above threshold
        ]
        settings.max_distance = 0.0  # make everything fail threshold
        pipeline.retriever.settings = settings

        result = pipeline.query("unknown topic?")
        assert not result.found
        assert "cannot find" in result.answer.lower()

    def test_stream_returns_stream_result(self, pipeline):
        result = pipeline.stream("what is ML?")
        assert isinstance(result, StreamResult)
        text = "".join(result.stream)
        assert len(text) > 0

    def test_stream_not_found_yields_message(self, pipeline, settings):
        pipeline.retriever.vectorstore.similarity_search_with_score.return_value = [
            _make_retrieval(relevant=False).hits[0]
        ]
        settings.max_distance = 0.0
        pipeline.retriever.settings = settings

        result = pipeline.stream("totally unknown?")
        assert not result.found
        text = "".join(result.stream)
        assert "cannot find" in text.lower()

    def test_is_ready_delegates_to_vectorstore(self, pipeline):
        assert pipeline.is_ready() is True

    def test_document_count_delegates(self, pipeline):
        assert pipeline.document_count() == 42

    def test_list_sources_delegates(self, pipeline):
        assert pipeline.list_sources() == ["ml.pdf"]

    def test_chat_history_is_passed_to_llm(self, pipeline, mock_llm):
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]
        pipeline.query("And what is ML?", chat_history=history)
        call_args = mock_llm.complete.call_args
        messages = call_args[0][0]
        # History entries should appear in the messages list
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles
