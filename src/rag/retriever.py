from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .config import Settings
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: list[tuple[Document, float]] = field(default_factory=list)
    filtered: list[tuple[Document, float]] = field(default_factory=list)
    best_score: float = float("inf")
    threshold: float = 1.4
    query: str = ""

    @property
    def has_relevant(self) -> bool:
        return len(self.filtered) > 0

    def as_sources(self) -> list[dict]:
        return [
            {
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "page": doc.metadata.get("page", "?"),
                "score": score,
                "text": doc.page_content[:400],
            }
            for doc, score in self.filtered
        ]


class Retriever:
    """Semantic retrieval with configurable distance filtering.

    Chroma returns L2 distances (lower = more similar).  After embedding
    normalisation the effective range is [0, 2].  Adjust `max_distance` in
    Settings to tune precision/recall.
    """

    def __init__(self, vectorstore: VectorStore, settings: Settings) -> None:
        self.vectorstore = vectorstore
        self.settings = settings

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        """Run a similarity search and apply the distance threshold.

        Args:
            query:         The user question.
            k:             How many candidates to fetch (defaults to settings.retrieval_k).
            source_filter: If set, only return chunks from this source filename.
        """
        k = k or self.settings.retrieval_k
        raw_hits = self.vectorstore.similarity_search_with_score(query, k=k)

        if source_filter:
            raw_hits = [
                (doc, score)
                for doc, score in raw_hits
                if Path(doc.metadata.get("source", "")).name == source_filter
            ]

        best = min((s for _, s in raw_hits), default=float("inf"))
        filtered = [(doc, score) for doc, score in raw_hits if score <= self.settings.max_distance]

        logger.debug(
            "Query: '%s' | hits=%d filtered=%d best=%.4f threshold=%.2f",
            query,
            len(raw_hits),
            len(filtered),
            best,
            self.settings.max_distance,
        )

        return RetrievalResult(
            hits=raw_hits,
            filtered=filtered,
            best_score=best,
            threshold=self.settings.max_distance,
            query=query,
        )
