from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


@dataclass
class IngestResult:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def loaded_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def load_file(file_path: Path) -> list[Document]:
    """Load a single file into LangChain Document objects."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {list(SUPPORTED_EXTENSIONS)}"
        )
    loader_cls = SUPPORTED_EXTENSIONS[suffix]
    loader = loader_cls(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        # Store filename separately so ChromaDB can filter on it without a
        # $contains scan over full paths (which are machine-specific).
        doc.metadata["source_name"] = file_path.name
    return docs


def load_directory(directory: Path) -> tuple[list[Document], list[str], dict[str, str]]:
    """Recursively load all supported files from *directory*.

    Returns:
        documents: flat list of loaded Document objects
        skipped:   filenames with unsupported extensions
        errors:    {filename: error_message} for files that failed to load
    """
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    documents: list[Document] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            skipped.append(file_path.name)
            continue

        try:
            docs = load_file(file_path)
            documents.extend(docs)
            logger.info("Loaded %d sections from %s", len(docs), file_path.name)
        except Exception as exc:
            errors[file_path.name] = str(exc)
            logger.error("Failed to load %s: %s", file_path.name, exc)

    return documents, skipped, errors


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into overlapping chunks with deterministic IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Separators ordered from coarsest to finest structural unit so that
        # section/paragraph boundaries are preferred over mid-sentence splits.
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    _assign_ids(chunks)
    return chunks


def _assign_ids(chunks: list[Document]) -> None:
    """Assign a deterministic ID to each chunk based on content hash (SHA-256).

    Using SHA-256 instead of MD5 eliminates collision risk at scale and avoids
    birthday-paradox duplicates when ingesting thousands of chunks.  Re-ingesting
    the same document produces identical IDs, enabling ChromaDB upsert deduplication.
    """
    seen: dict[str, int] = {}
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        source_hash = hashlib.sha256(
            chunk.metadata.get("source", "").encode()
        ).hexdigest()[:8]
        base_id = f"{source_hash}-{content_hash}"
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        chunk.metadata["chunk_id"] = f"{base_id}-{count}"


def ingest_directory(directory: Path, settings: Settings) -> IngestResult:
    """High-level entry point: load → chunk → return result."""
    result = IngestResult()
    docs, result.skipped, result.errors = load_directory(directory)
    result.documents = docs

    if docs:
        result.chunks = chunk_documents(docs, settings)

    return result


def ingest_file(file_path: Path, settings: Settings) -> IngestResult:
    """High-level entry point for a single file."""
    result = IngestResult()
    try:
        result.documents = load_file(file_path)
    except Exception as exc:
        result.errors[file_path.name] = str(exc)
        return result

    if result.documents:
        result.chunks = chunk_documents(result.documents, settings)

    return result
