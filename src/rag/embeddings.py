from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings


@lru_cache(maxsize=4)
def _get_embeddings(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    """Load and cache an embedding model. Keyed by (model, device, batch_size) so
    different configs can coexist during tests without re-downloading."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        cache_folder=".cache_embeddings",
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return _get_embeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
