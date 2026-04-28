from __future__ import annotations

import logging
import shutil
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Settings
from .embeddings import build_embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a persisted ChromaDB collection.

    Supports upsert-style document addition: re-ingesting the same document
    (with the same chunk IDs) will update existing vectors rather than duplicate them.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: chromadb.PersistentClient | None = None
        self._store: Chroma | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embeddings(self):
        return build_embeddings(self.settings)

    def _open(self) -> Chroma:
        # Resolve to an absolute path — ChromaDB's Rust backend compares the
        # path it was opened with against the WAL/SHM sidecar locations; a
        # relative path can cause a mismatch and trigger SQLITE_READONLY_DBMOVED.
        abs_dir = self.settings.vectorstore_dir.resolve()
        abs_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(abs_dir))
        return Chroma(
            client=self._client,
            embedding_function=self._embeddings(),
            collection_name=self.settings.collection_name,
        )

    def _close(self) -> None:
        """Release the SQLite file lock before touching the files on disk."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._store = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = self._open()
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[Document]) -> int:
        """Upsert *chunks* into the collection.  Returns number of chunks stored."""
        if not chunks:
            return 0

        ids = [c.metadata.get("chunk_id") for c in chunks]
        # Fall back to positional IDs if ingest didn't assign chunk_id
        ids = [uid or f"chunk-{i}" for i, uid in enumerate(ids)]

        self.store.add_documents(chunks, ids=ids)

        logger.info("Upserted %d chunks into '%s'", len(chunks), self.settings.collection_name)
        return len(chunks)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        k = k or self.settings.retrieval_k
        return self.store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int | None = None):
        """Return a LangChain-compatible retriever (for chain composition)."""
        k = k or self.settings.retrieval_k
        return self.store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        """Return unique source file paths stored in metadata."""
        try:
            data = self.store._collection.get(include=["metadatas"])
            sources: set[str] = set()
            for meta in data.get("metadatas") or []:
                if src := (meta or {}).get("source"):
                    sources.add(Path(src).name)
            return sorted(sources)
        except Exception:
            return []

    def exists(self) -> bool:
        """True when the persist directory has data."""
        return (
            self.settings.vectorstore_dir.exists()
            and any(self.settings.vectorstore_dir.iterdir())
        )

    def reset(self) -> None:
        """Wipe the entire collection and start fresh."""
        self._close()
        if self.settings.vectorstore_dir.exists():
            shutil.rmtree(self.settings.vectorstore_dir)
        self.settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Vector store reset — all data deleted")

    def delete_source(self, source_name: str) -> int:
        """Delete all chunks whose metadata 'source' matches *source_name*. Returns deleted count."""
        try:
            results = self.store._collection.get(include=["metadatas"])
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(
                    results.get("ids", []), results.get("metadatas") or []
                )
                if Path((meta or {}).get("source", "")).name == source_name
            ]
            if ids_to_delete:
                self.store._collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d chunks for source '%s'", len(ids_to_delete), source_name)
            self._close()
            return len(ids_to_delete)
        except Exception as exc:
            logger.error("Failed to delete source '%s': %s", source_name, exc)
            return 0
