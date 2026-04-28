"""PDF RAG Assistant — Ministral running locally via llama-cpp-python."""

__version__ = "0.1.0"

from .config import Settings, get_settings
from .pipeline import RAGPipeline

__all__ = ["Settings", "get_settings", "RAGPipeline", "__version__"]
