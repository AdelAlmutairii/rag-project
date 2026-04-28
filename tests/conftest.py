"""Shared pytest fixtures for the RAG test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag.config import Settings


# ---------------------------------------------------------------------------
# Settings fixture — no API key, temp dirs, avoids any I/O
# ---------------------------------------------------------------------------

@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        # Local LLM
        model_repo="bartowski/Ministral-8B-Instruct-2410-GGUF",
        model_filename="Ministral-8B-Instruct-2410-Q4_K_M.gguf",
        model_path=None,
        n_ctx=512,
        n_gpu_layers=0,
        n_threads=2,
        n_batch=8,
        max_tokens=64,
        temperature=0.0,
        # Embeddings
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
        embedding_batch_size=4,
        # Storage (temp dirs so tests are isolated)
        vectorstore_dir=tmp_path / "vectorstore",
        collection_name="test_collection",
        documents_dir=tmp_path / "documents",
        # Chunking
        chunk_size=500,
        chunk_overlap=50,
        # Retrieval
        retrieval_k=3,
        max_distance=1.5,
    )


# ---------------------------------------------------------------------------
# Temporary document files
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    doc = tmp_path / "sample.txt"
    doc.write_text(
        textwrap.dedent("""\
            Introduction to Machine Learning

            Machine learning is a branch of artificial intelligence that focuses
            on building systems that learn from data. It includes supervised
            learning, unsupervised learning, and reinforcement learning.

            Supervised Learning

            In supervised learning, a model is trained on labelled data.
            Common algorithms include linear regression, decision trees,
            and neural networks.

            Unsupervised Learning

            Unsupervised learning discovers hidden patterns in unlabelled data.
            Clustering and dimensionality reduction are key techniques.
        """)
    )
    return doc


@pytest.fixture()
def sample_txt_2(tmp_path: Path) -> Path:
    doc = tmp_path / "deep_learning.txt"
    doc.write_text(
        textwrap.dedent("""\
            Deep Learning Overview

            Deep learning uses multi-layer neural networks to model complex patterns.
            Convolutional Neural Networks (CNNs) excel at image recognition tasks.
            Recurrent Neural Networks (RNNs) handle sequential data like text.
            Transformers have become the dominant architecture for NLP.
        """)
    )
    return doc


@pytest.fixture()
def documents_dir(tmp_path: Path, sample_txt: Path, sample_txt_2: Path) -> Path:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / sample_txt.name).write_text(sample_txt.read_text())
    (docs_dir / sample_txt_2.name).write_text(sample_txt_2.read_text())
    return docs_dir


# ---------------------------------------------------------------------------
# Mock LLM — stands in for LocalLLM without loading any model
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = (
        "Machine learning focuses on building systems that learn from data. "
        "[Source 1]\n\nCitations:\n- \"machine learning\" [Source 1]"
    )
    llm.stream.return_value = iter(["Machine ", "learning ", "is ", "great."])
    return llm


# ---------------------------------------------------------------------------
# LangChain Document factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_document():
    from langchain_core.documents import Document

    def _make(content: str, source: str = "test.txt", page: int = 0) -> Document:
        return Document(
            page_content=content,
            metadata={"source": source, "page": page},
        )

    return _make
