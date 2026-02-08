"""Tests for the shared SentenceTransformer model registry."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from anamnesis.utils.model_registry import (
    clear_model_cache,
    get_model_cache_stats,
    get_shared_sentence_transformer,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure a fresh cache for every test."""
    clear_model_cache()
    yield
    clear_model_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_st(name: str = "model", device: str = "cpu"):
    """Return a lightweight mock that quacks like SentenceTransformer."""
    m = MagicMock()
    m.__class__.__name__ = "SentenceTransformer"
    m._name = f"{name}:{device}"
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingletonBehaviour:
    """get_shared_sentence_transformer returns the same object on repeat calls."""

    def test_singleton_returns_same_instance(self):
        fake = _make_fake_st()
        with patch(
            "anamnesis.utils.model_registry.SentenceTransformer",
            return_value=fake,
            create=True,
        ):
            # Patch the import inside the function
            import anamnesis.utils.model_registry as mod

            orig = mod.get_shared_sentence_transformer

            def _patched(model_name="all-MiniLM-L6-v2", device="cpu"):
                # Inject mock into the import path
                import sys
                mock_module = MagicMock()
                mock_module.SentenceTransformer = MagicMock(return_value=fake)
                sys.modules["sentence_transformers"] = mock_module
                try:
                    return orig(model_name, device)
                finally:
                    del sys.modules["sentence_transformers"]

            first = _patched()
            # Second call should hit cache, no new import needed
            second = _patched()
            assert first is second
            assert id(first) == id(second)

    def test_different_configs_get_different_models(self):
        fake_a = _make_fake_st("model-a")
        fake_b = _make_fake_st("model-b")

        call_count = {"n": 0}

        def _factory(model_name, device="cpu"):
            call_count["n"] += 1
            if "model-a" in model_name:
                return fake_a
            return fake_b

        import sys
        mock_module = MagicMock()
        mock_module.SentenceTransformer = _factory
        sys.modules["sentence_transformers"] = mock_module
        try:
            a = get_shared_sentence_transformer(model_name="model-a", device="cpu")
            b = get_shared_sentence_transformer(model_name="model-b", device="cpu")
            assert a is not b
            assert a is fake_a
            assert b is fake_b
            assert call_count["n"] == 2
        finally:
            del sys.modules["sentence_transformers"]


class TestThreadSafety:
    """Concurrent requests for the same model must all get the same instance."""

    def test_thread_safety(self):
        fake = _make_fake_st()
        call_count = {"n": 0}
        load_lock = threading.Lock()

        def _slow_factory(model_name, device="cpu"):
            with load_lock:
                call_count["n"] += 1
            return fake

        import sys
        mock_module = MagicMock()
        mock_module.SentenceTransformer = _slow_factory
        sys.modules["sentence_transformers"] = mock_module

        results: list = [None] * 4
        errors: list = []

        def worker(idx):
            try:
                results[idx] = get_shared_sentence_transformer()
            except Exception as exc:
                errors.append(exc)

        try:
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert not errors, f"Thread errors: {errors}"
            # All threads got the same object
            for r in results:
                assert r is fake
            # Factory called exactly once (singleton)
            assert call_count["n"] == 1
        finally:
            del sys.modules["sentence_transformers"]


class TestClearCache:
    """clear_model_cache drops instances, next load creates a new one."""

    def test_clear_cache(self):
        fake_1 = _make_fake_st("v1")
        fake_2 = _make_fake_st("v2")
        versions = iter([fake_1, fake_2])

        def _factory(model_name, device="cpu"):
            return next(versions)

        import sys
        mock_module = MagicMock()
        mock_module.SentenceTransformer = _factory
        sys.modules["sentence_transformers"] = mock_module
        try:
            first = get_shared_sentence_transformer()
            assert first is fake_1

            clear_model_cache()

            second = get_shared_sentence_transformer()
            assert second is fake_2
            assert first is not second
        finally:
            del sys.modules["sentence_transformers"]


class TestImportError:
    """Graceful ImportError when sentence-transformers is missing."""

    def test_import_error_graceful(self):
        import sys
        # Ensure sentence_transformers is NOT importable
        saved = sys.modules.pop("sentence_transformers", None)
        import importlib
        # Also block the import machinery
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError):
                get_shared_sentence_transformer()
        if saved is not None:
            sys.modules["sentence_transformers"] = saved


class TestCacheStats:
    """get_model_cache_stats reports correct info."""

    def test_empty_cache_stats(self):
        stats = get_model_cache_stats()
        assert stats["size"] == 0
        assert stats["models"] == []

    def test_populated_cache_stats(self):
        fake = _make_fake_st()
        import sys
        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=fake)
        sys.modules["sentence_transformers"] = mock_module
        try:
            get_shared_sentence_transformer(model_name="test-model", device="cpu")
            stats = get_model_cache_stats()
            assert stats["size"] == 1
            assert "test-model:cpu" in stats["models"]
        finally:
            del sys.modules["sentence_transformers"]
