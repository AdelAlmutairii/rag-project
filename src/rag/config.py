from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration — read from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Local LLM (llama-cpp-python / GGUF) ──────────────────────────────
    model_repo: str = Field(
        "bartowski/Ministral-8B-Instruct-2410-GGUF",
        description="HuggingFace repo containing the GGUF model file",
    )
    model_filename: str = Field(
        "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
        description="GGUF filename inside the repo (supports fnmatch wildcards)",
    )
    model_path: Path | None = Field(
        None,
        description="Absolute path to a local GGUF file — skips the HF download when set",
    )

    # llama.cpp inference parameters
    n_ctx: int = Field(8192, ge=512, le=131072, description="Context window (tokens)")
    n_gpu_layers: int = Field(
        -1,
        description="Layers to offload to GPU (-1 = all, 0 = CPU only)",
    )
    n_threads: int = Field(8, ge=1, description="CPU threads for inference")
    n_batch: int = Field(512, ge=1, description="Batch size for prompt processing")

    # Generation parameters
    max_tokens: int = Field(1024, ge=64, le=8192)
    temperature: float = Field(0.1, ge=0.0, le=1.0)

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        "cpu",
        description="Torch device for the embedding model",
    )
    embedding_batch_size: int = Field(32, ge=1, le=512)

    # ── ChromaDB ─────────────────────────────────────────────────────────
    vectorstore_dir: Path = Field(
        Path("data/vectorstore"),
        description="Directory where ChromaDB persists its data",
    )
    collection_name: str = Field("rag_documents")

    # ── Document chunking ─────────────────────────────────────────────────
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=1000)

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_k: int = Field(5, ge=1, le=20)
    max_distance: float = Field(
        1.4,
        ge=0.0,
        le=2.0,
        description="Maximum Chroma L2 distance to consider a chunk relevant",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    documents_dir: Path = Field(
        Path("data/documents"),
        description="Default directory scanned during batch ingestion",
    )

    # ── HuggingFace (optional) ────────────────────────────────────────────
    hf_token: str | None = Field(
        None,
        description="HuggingFace access token — required for gated/private models",
    )

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("vectorstore_dir", "documents_dir", "model_path", mode="before")
    @classmethod
    def _coerce_path(cls, v) -> Path | None:
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def _overlap_lt_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
