from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .config import Settings
from .reranker import Reranker
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: list[tuple[Document, float]] = field(default_factory=list)
    filtered: list[tuple[Document, float]] = field(default_factory=list)
    best_score: float = float("inf")
    threshold: float = 0.8
    query: str = ""
    reranked: bool = False

    @property
    def has_relevant(self) -> bool:
        return len(self.filtered) > 0

    def as_sources(self) -> list[dict]:
        return [
            {
                "source": (
                    doc.metadata.get("source_name")
                    or Path(doc.metadata.get("source", "unknown")).name
                ),
                "page": doc.metadata.get("page", "?"),
                "score": score,
                "reranked": self.reranked,
                "text": doc.page_content[:400],
            }
            for doc, score in self.filtered
        ]


class Retriever:
    """Semantic retrieval with distance filtering and optional cross-encoder reranking.

    Chroma returns L2 distances on normalized embeddings (range [0, 2]):
      0.0  = identical vectors
      ~0.8 = cosine similarity ≈ 0.68  (default threshold)
      1.41 = orthogonal (cosine similarity = 0)
      2.0  = opposite vectors

    When reranking is enabled, the retriever fetches `retrieval_k_fetch`
    candidates, applies the cross-encoder, and keeps the top `retrieval_k`
    by cross-encoder score.  The distance threshold is still applied first
    to avoid passing garbage to the reranker.
    """

    def __init__(self, vectorstore: VectorStore, settings: Settings) -> None:
        self.vectorstore = vectorstore
        self.settings = settings
        self._reranker: Reranker | None = (
            Reranker(settings.reranker_model) if settings.use_reranker else None
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        """Run a similarity search, apply the distance threshold, and optionally rerank.

        Args:
            query:         The user question.
            k:             Final number of chunks to return (defaults to settings.retrieval_k).
            source_filter: If set, restrict retrieval to chunks from this filename.
        """
        k = k or self.settings.retrieval_k

        # When reranking, fetch a larger candidate pool so the cross-encoder
        # has enough material to reorder.
        fetch_k = self.settings.retrieval_k_fetch if self._reranker else k

        # Build a ChromaDB where-filter so the vector search itself is scoped
        # to the target document — avoids wasting the retrieval budget on
        # irrelevant files before Python-side filtering.
        chroma_filter: dict | None = None
        if source_filter:
            # source_name stores just the filename (portable); source stores the
            # full path (legacy).  Try source_name first; the $eq operator is
            # exact-match and far more reliable than $contains on full paths.
            chroma_filter = {"source_name": {"$eq": source_filter}}

        raw_hits = self.vectorstore.similarity_search_with_score(
            query, k=fetch_k, filter=chroma_filter
        )

        # Safety fallback: if source_name is absent on legacy chunks (pre-fix
        # ingestion), filter in Python using the full source path.
        if source_filter and raw_hits:
            any_name_match = any(
                doc.metadata.get("source_name") == source_filter
                for doc, _ in raw_hits
            )
            if not any_name_match:
                raw_hits = [
                    (doc, score)
                    for doc, score in raw_hits
                    if Path(doc.metadata.get("source", "")).name == source_filter
                ]

        best = min((s for _, s in raw_hits), default=float("inf"))
        filtered = [
            (doc, score)
            for doc, score in raw_hits
            if score <= self.settings.max_distance
        ]

        reranked = False
        if self._reranker and filtered:
            filtered = self._reranker.rerank(query, filtered, top_k=k)
            reranked = True
        else:
            filtered = filtered[:k]

        logger.debug(
            "Query: '%s' | candidates=%d filtered=%d best=%.4f threshold=%.2f reranked=%s",
            query,
            len(raw_hits),
            len(filtered),
            best,
            self.settings.max_distance,
            reranked,
        )

        return RetrievalResult(
            hits=raw_hits,
            filtered=filtered,
            best_score=best,
            threshold=self.settings.max_distance,
            query=query,
            reranked=reranked,
        )
