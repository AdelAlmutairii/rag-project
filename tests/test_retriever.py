"""Tests for the Retriever class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.retriever import Retriever, RetrievalResult


def _make_hit(content: str, source: str = "doc.pdf", page: int = 1, score: float = 0.5):
    from langchain_core.documents import Document
    doc = Document(page_content=content, metadata={"source": source, "page": page})
    return (doc, score)


class TestRetrievalResult:
    def test_has_relevant_true_when_filtered(self):
        hit = _make_hit("relevant content", score=0.3)
        result = RetrievalResult(hits=[hit], filtered=[hit], best_score=0.3, threshold=1.0)
        assert result.has_relevant is True

    def test_has_relevant_false_when_empty(self):
        result = RetrievalResult(hits=[], filtered=[], best_score=float("inf"), threshold=1.0)
        assert result.has_relevant is False

    def test_as_sources_returns_correct_keys(self, make_document):
        doc = make_document("Neural networks are powerful.", source="ai.pdf", page=5)
        result = RetrievalResult(
            filtered=[(doc, 0.42)],
            best_score=0.42,
            threshold=1.0,
        )
        sources = result.as_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "ai.pdf"
        assert sources[0]["page"] == 5
        assert sources[0]["score"] == pytest.approx(0.42)
        assert "Neural networks" in sources[0]["text"]


class TestRetriever:
    @pytest.fixture()
    def mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            _make_hit("Supervised learning uses labelled data.", score=0.8),
            _make_hit("Clustering is unsupervised.", score=1.2),
            _make_hit("Random noise content.", score=1.6),  # above threshold
        ]
        return vs

    def test_retrieve_filters_by_threshold(self, mock_vectorstore, settings):
        settings.max_distance = 1.4
        retriever = Retriever(mock_vectorstore, settings)
        result = retriever.retrieve("what is supervised learning?")

        assert len(result.hits) == 3
        assert len(result.filtered) == 2  # third hit is above 1.4

    def test_best_score_is_minimum(self, mock_vectorstore, settings):
        retriever = Retriever(mock_vectorstore, settings)
        result = retriever.retrieve("supervised")
        assert result.best_score == pytest.approx(0.8)

    def test_source_filter(self, settings):
        from langchain_core.documents import Document
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            (Document("Doc A text", metadata={"source": "/path/to/a.pdf", "page": 1}), 0.5),
            (Document("Doc B text", metadata={"source": "/path/to/b.pdf", "page": 2}), 0.6),
        ]
        retriever = Retriever(vs, settings)
        result = retriever.retrieve("query", source_filter="a.pdf")
        assert all(
            Path(doc.metadata["source"]).name == "a.pdf"
            for doc, _ in result.filtered
        )

    def test_no_hits_returns_empty_result(self, settings):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = []
        retriever = Retriever(vs, settings)
        result = retriever.retrieve("query")
        assert not result.has_relevant
        assert result.best_score == float("inf")

    def test_all_above_threshold_is_not_relevant(self, settings):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            _make_hit("Far from query", score=1.9),
        ]
        settings.max_distance = 1.5
        retriever = Retriever(vs, settings)
        result = retriever.retrieve("query")
        assert not result.has_relevant
        assert len(result.hits) == 1
