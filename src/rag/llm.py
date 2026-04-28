from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """llama-cpp-python wrapper that runs Ministral (or any GGUF model) locally.

    On first construction the model is loaded into RAM/VRAM — this is the slow
    step (10–60 s depending on hardware).  Cache this instance at the app level.

    GPU offload:
        Apple Silicon (Metal) — set n_gpu_layers=-1 (default) and install
            llama-cpp-python with Metal support (see README).
        CUDA — same n_gpu_layers=-1 flag, different build.
        CPU only — set n_gpu_layers=0.
    """

    def __init__(self, settings: Settings) -> None:
        self._max_tokens = settings.max_tokens
        self._temperature = settings.temperature
        self._llm = self._load(settings)

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load(settings: Settings) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Install for your platform:\n"
                "  Apple Silicon: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python\n"
                "  CUDA:          CMAKE_ARGS='-DGGML_CUDA=on'  pip install llama-cpp-python\n"
                "  CPU only:      pip install llama-cpp-python"
            ) from exc

        common = dict(
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
            n_batch=settings.n_batch,
            n_threads=settings.n_threads,
            verbose=False,
        )

        if settings.model_path:
            if not settings.model_path.exists():
                raise FileNotFoundError(
                    f"MODEL_PATH points to a missing file: {settings.model_path}"
                )
            logger.info("Loading model from local path: %s", settings.model_path)
            return Llama(model_path=str(settings.model_path), **common)

        logger.info(
            "Downloading/loading %s from %s",
            settings.model_filename,
            settings.model_repo,
        )
        return Llama.from_pretrained(
            repo_id=settings.model_repo,
            filename=settings.model_filename,
            **common,
        )

    # ------------------------------------------------------------------
    # Synchronous completion
    # ------------------------------------------------------------------

    def complete(self, messages: list[dict[str, str]]) -> str:
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return (result["choices"][0]["message"]["content"] or "").strip()

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield text tokens as they are generated."""
        try:
            chunks = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True,
            )
            for chunk in chunks:
                delta: str = chunk["choices"][0]["delta"].get("content") or ""
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            yield f"\n\n[Error: {exc}]"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        try:
            return self._llm.model_path  # type: ignore[attr-defined]
        except AttributeError:
            return "local-gguf"
