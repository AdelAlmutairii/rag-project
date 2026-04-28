from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using sentence-transformers.

    Bi-encoder similarity (ChromaDB) is fast but imprecise — it compares
    embeddings independently.  A cross-encoder sees (query, passage) together
    and produces a calibrated relevance score, significantly improving the
    ordering of retrieved chunks before they reach the LLM.

    The model is lazy-loaded on first use and cached on the instance.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for reranking.\n"
                    "Install it with: pip install sentence-transformers"
                ) from exc
            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        hits: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Rerank *hits* by cross-encoder score and return the top *top_k*.

        The returned list uses the cross-encoder score in the second position
        (higher = more relevant), replacing the original L2 distance (lower =
        more relevant).  Callers must be aware of this sign flip.
        """
        if not hits:
            return hits

        model = self._load()
        pairs = [(query, doc.page_content) for doc, _ in hits]
        scores: list[float] = model.predict(pairs).tolist()

        reranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc_score[0], ce_score) for doc_score, ce_score in reranked[:top_k]]
