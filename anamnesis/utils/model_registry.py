"""Shared model registry for SentenceTransformer instances.

Avoids duplicate model loads across MemoryIndex and EmbeddingEngine
(each ~80-200MB RAM). Thread-safe singleton cache keyed by model+device.
"""

from __future__ import annotations

import threading
from typing import Any

from anamnesis.utils.logger import logger

_model_cache: dict[str, Any] = {}
_lock = threading.Lock()


def get_shared_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Any:
    """Return a shared SentenceTransformer instance, loading on first call.

    Args:
        model_name: HuggingFace model name or local path.
        device: Target device ("cpu", "cuda", "mps").

    Returns:
        A SentenceTransformer model instance (shared, do NOT mutate).

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    key = f"{model_name}:{device}"

    # Fast path: already cached (no lock needed for dict read of immutable ref)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    with _lock:
        # Double-check under lock
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading shared SentenceTransformer: {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)
        _model_cache[key] = model
        return model


def clear_model_cache() -> None:
    """Drop all cached models (for testing / shutdown)."""
    with _lock:
        _model_cache.clear()


def get_model_cache_stats() -> dict:
    """Return cache size and loaded model keys.

    Returns:
        Dict with ``size`` and ``models`` keys.
    """
    with _lock:
        return {
            "size": len(_model_cache),
            "models": list(_model_cache.keys()),
        }
