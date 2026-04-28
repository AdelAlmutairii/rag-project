"""Tests for the document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingest import (
    IngestResult,
    _assign_ids,
    chunk_documents,
    ingest_directory,
    ingest_file,
    load_file,
)


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_loads_txt(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        assert len(docs) >= 1
        full_text = " ".join(d.page_content for d in docs)
        assert "machine learning" in full_text.lower()

    def test_metadata_source_is_set(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        for doc in docs:
            assert "source" in doc.metadata
            assert str(sample_txt) in doc.metadata["source"]

    def test_unsupported_extension_raises(self, tmp_path: Path):
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_file(bad_file)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(Exception):
            load_file(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_chunks_are_within_size(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        chunks = chunk_documents(docs, settings)
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.chunk_size * 1.2  # 20% slack for splitter

    def test_chunk_count_greater_than_zero(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        chunks = chunk_documents(docs, settings)
        assert len(chunks) >= 1

    def test_chunk_ids_assigned(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        chunks = chunk_documents(docs, settings)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata

    def test_chunk_ids_are_unique(self, sample_txt: Path, settings):
        docs = load_file(sample_txt)
        chunks = chunk_documents(docs, settings)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_deterministic(self, sample_txt: Path, settings):
        docs1 = load_file(sample_txt)
        docs2 = load_file(sample_txt)
        chunks1 = chunk_documents(docs1, settings)
        chunks2 = chunk_documents(docs2, settings)
        ids1 = [c.metadata["chunk_id"] for c in chunks1]
        ids2 = [c.metadata["chunk_id"] for c in chunks2]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# _assign_ids
# ---------------------------------------------------------------------------

class TestAssignIds:
    def test_unique_ids(self, make_document):
        chunks = [
            make_document("Content A", source="doc1.txt"),
            make_document("Content B", source="doc1.txt"),
            make_document("Content C", source="doc2.txt"),
        ]
        _assign_ids(chunks)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_same_content_different_source_gets_different_id(self, make_document):
        c1 = make_document("Same content", source="a.txt")
        c2 = make_document("Same content", source="b.txt")
        _assign_ids([c1, c2])
        assert c1.metadata["chunk_id"] != c2.metadata["chunk_id"]


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_ingest_txt_returns_chunks(self, sample_txt: Path, settings):
        result = ingest_file(sample_txt, settings)
        assert isinstance(result, IngestResult)
        assert result.chunk_count > 0
        assert result.loaded_count > 0
        assert not result.errors

    def test_ingest_missing_file_stores_error(self, tmp_path: Path, settings):
        result = ingest_file(tmp_path / "missing.txt", settings)
        assert result.chunk_count == 0
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_ingest_multiple_files(self, documents_dir: Path, settings):
        result = ingest_directory(documents_dir, settings)
        assert result.chunk_count > 0
        assert result.loaded_count > 0

    def test_skips_unsupported_files(self, tmp_path: Path, settings):
        docs_dir = tmp_path / "mixed"
        docs_dir.mkdir()
        (docs_dir / "file.txt").write_text("Hello world " * 100)
        (docs_dir / "data.csv").write_text("a,b\n1,2\n")
        result = ingest_directory(docs_dir, settings)
        assert "data.csv" in result.skipped
        assert result.chunk_count > 0

    def test_missing_directory_raises(self, tmp_path: Path, settings):
        with pytest.raises(FileNotFoundError):
            ingest_directory(tmp_path / "does_not_exist", settings)

    def test_empty_directory_returns_zero_chunks(self, tmp_path: Path, settings):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = ingest_directory(empty, settings)
        assert result.chunk_count == 0
        assert result.loaded_count == 0
