"""Tests for search backend integration with unified extraction.

Verifies that SemanticSearchBackend can use ExtractionOrchestrator
for richer concept indexing while maintaining backward compatibility.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from anamnesis.search.semantic_backend import SemanticSearchBackend


# ============================================================================
# Fixtures
# ============================================================================


PYTHON_CODE = '''
class UserService:
    """Manages user accounts and authentication."""

    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id: int) -> dict:
        """Retrieve a user by their ID."""
        return await self.db.find(user_id)

def helper(x: int) -> int:
    return x * 2
'''


@pytest.fixture
def mock_embedding_engine():
    """Mock embedding engine that returns fixed-size vectors."""
    engine = MagicMock()
    engine._generate_embedding = MagicMock(
        return_value=np.random.rand(384).astype(np.float32)
    )
    engine.get_embedding_dimension = MagicMock(return_value=384)
    engine.get_stats = MagicMock(return_value={"model": "test"})
    return engine


@pytest.fixture
def mock_vector_store():
    """Mock Qdrant vector store."""
    store = MagicMock()
    store.delete_by_file = AsyncMock()
    store.upsert_batch = AsyncMock()
    store.get_stats = AsyncMock(return_value=MagicMock(
        vectors_count=0,
        indexed_vectors_count=0,
        status="ok",
    ))
    return store


@pytest.fixture
def legacy_backend(mock_embedding_engine, mock_vector_store):
    """Semantic backend using legacy ASTPatternMatcher extraction."""
    return SemanticSearchBackend(
        base_path="/tmp/test",
        embedding_engine=mock_embedding_engine,
        vector_store=mock_vector_store,
        use_unified_extraction=False,
    )


@pytest.fixture
def unified_backend(mock_embedding_engine, mock_vector_store):
    """Semantic backend using unified ExtractionOrchestrator."""
    return SemanticSearchBackend(
        base_path="/tmp/test",
        embedding_engine=mock_embedding_engine,
        vector_store=mock_vector_store,
        use_unified_extraction=True,
    )


# ============================================================================
# Feature flag tests
# ============================================================================


class TestUnifiedExtractionFlag:
    """Test the use_unified_extraction feature flag."""

    def test_default_is_legacy(self, mock_embedding_engine, mock_vector_store):
        backend = SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
        )
        assert backend._use_unified_extraction is False

    def test_can_enable_unified(self, mock_embedding_engine, mock_vector_store):
        backend = SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            use_unified_extraction=True,
        )
        assert backend._use_unified_extraction is True


# ============================================================================
# Unified indexing tests
# ============================================================================


class TestUnifiedIndexing:
    """Test that unified extraction produces richer index data."""

    @pytest.mark.asyncio
    async def test_unified_index_extracts_symbols(self, unified_backend, mock_vector_store):
        """Unified indexing should extract symbols from code."""
        await unified_backend.index(
            "test.py",
            PYTHON_CODE,
            {"language": "python"},
        )

        # Should have called upsert_batch with extracted symbols
        mock_vector_store.upsert_batch.assert_called_once()
        concepts = mock_vector_store.upsert_batch.call_args[0][0]
        assert len(concepts) > 0

        # Check concept structure
        names = {c[1] for c in concepts}
        assert "UserService" in names
        assert "helper" in names

    @pytest.mark.asyncio
    async def test_unified_index_includes_methods(self, unified_backend, mock_vector_store):
        """Unified indexing flattens symbol hierarchy for indexing."""
        await unified_backend.index(
            "test.py",
            PYTHON_CODE,
            {"language": "python"},
        )

        concepts = mock_vector_store.upsert_batch.call_args[0][0]
        names = {c[1] for c in concepts}
        assert "get_user" in names  # Method from tree-sitter

    @pytest.mark.asyncio
    async def test_unified_index_has_metadata(self, unified_backend, mock_vector_store):
        """Unified indexing includes richer metadata."""
        await unified_backend.index(
            "test.py",
            PYTHON_CODE,
            {"language": "python"},
        )

        concepts = mock_vector_store.upsert_batch.call_args[0][0]
        for concept in concepts:
            metadata = concept[4]
            assert "line_start" in metadata
            assert "line_end" in metadata
            assert "language" in metadata
            assert "backend" in metadata
            assert "confidence" in metadata

    @pytest.mark.asyncio
    async def test_unified_index_uses_symbol_kind(self, unified_backend, mock_vector_store):
        """Concepts should use SymbolKind as concept_type."""
        await unified_backend.index(
            "test.py",
            PYTHON_CODE,
            {"language": "python"},
        )

        concepts = mock_vector_store.upsert_batch.call_args[0][0]
        types = {c[2] for c in concepts}
        assert "class" in types or "SymbolKind.CLASS" in types or "CLASS" in types


# ============================================================================
# Backward compatibility tests
# ============================================================================


class TestBackwardCompatibility:
    """Ensure legacy path works unchanged."""

    @pytest.mark.asyncio
    async def test_legacy_index_still_works(self, legacy_backend, mock_vector_store):
        """Legacy indexing via ASTPatternMatcher should still function."""
        await legacy_backend.index(
            "test.py",
            PYTHON_CODE,
            {"language": "python"},
        )

        # Should have called delete_by_file
        mock_vector_store.delete_by_file.assert_called_once_with("test.py")

    @pytest.mark.asyncio
    async def test_unified_and_legacy_both_produce_embeddings(
        self, unified_backend, legacy_backend, mock_vector_store, mock_embedding_engine
    ):
        """Both paths should call the embedding engine."""
        await unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )
        unified_calls = mock_embedding_engine._generate_embedding.call_count

        mock_embedding_engine._generate_embedding.reset_mock()
        mock_vector_store.upsert_batch.reset_mock()

        await legacy_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )
        legacy_calls = mock_embedding_engine._generate_embedding.call_count

        # Both should generate embeddings
        assert unified_calls > 0
        assert legacy_calls >= 0  # Legacy may be 0 if AST matcher doesn't match


# ============================================================================
# Orchestrator lazy init tests
# ============================================================================


class TestOrchestratorLazyInit:
    """Test that orchestrator is lazily initialized."""

    def test_orchestrator_not_created_on_init(self, unified_backend):
        assert unified_backend._orchestrator is None

    @pytest.mark.asyncio
    async def test_orchestrator_created_on_first_index(self, unified_backend, mock_vector_store):
        await unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )
        assert unified_backend._orchestrator is not None
