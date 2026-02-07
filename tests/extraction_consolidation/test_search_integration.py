"""Tests for search backend integration with unified extraction.

Verifies that SemanticSearchBackend can use ExtractionOrchestrator
for richer concept indexing while maintaining backward compatibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from anamnesis.interfaces.search import SearchQuery, SearchType
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
        # Note: legacy may be 0 if AST matcher doesn't match the sample code.
        # See TestSearchPipelineWithDeterministicEmbedding for a meaningful
        # unified-vs-legacy comparison using real (hash-based) embeddings.
        # Legacy call count verified in TestSearchPipelineWithDeterministicEmbedding.test_both_paths_produce_embeddings


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


# ============================================================================
# Deterministic embedding test infrastructure
# ============================================================================


def _hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based embedding for testing.

    Uses SHA-256 to produce a fixed-dimension vector from text.
    Same text always produces the same vector. NOT a semantic
    embedding — exists solely to test pipeline mechanics.

    Bytes are converted to uint8 then scaled to [0, 1] to avoid
    NaN/Inf issues from interpreting raw bytes as float32.
    """
    h = hashlib.sha256(text.encode()).digest()
    repeated = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.array([b / 255.0 for b in repeated], dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


class HashEmbeddingEngine:
    """Deterministic embedding engine using SHA-256 hashing.

    Drop-in replacement for EmbeddingEngine in tests. Produces
    deterministic, dimension-correct vectors without a real model.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def _generate_embedding(self, text: str) -> np.ndarray:
        return _hash_embedding(text, self._dim)

    def get_embedding_dimension(self) -> int:
        return self._dim

    def get_stats(self) -> dict:
        return {"model": "hash-test", "dimension": self._dim}


class InMemoryVectorStore:
    """Minimal in-memory vector store for testing.

    Brute-force cosine similarity search over a dict of stored
    vectors. Implements the same protocol as QdrantVectorStore.
    """

    def __init__(self, vector_size: int = 64):
        self._vectors: dict[str, dict] = {}
        self._vector_size = vector_size

    async def connect(self) -> None:
        pass

    async def upsert_batch(
        self,
        points: list[tuple[str, str, str, list[float], dict]],
    ) -> list[str]:
        point_ids = []
        for file_path, name, concept_type, embedding, metadata in points:
            point_id = hashlib.sha256(
                f"{file_path}:{name}:{concept_type}".encode()
            ).hexdigest()
            self._vectors[point_id] = {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "file_path": file_path,
                    "name": name,
                    "concept_type": concept_type,
                    **metadata,
                },
            }
            point_ids.append(point_id)
        return point_ids

    async def delete_by_file(self, file_path: str) -> bool:
        to_delete = [
            pid
            for pid, data in self._vectors.items()
            if data["payload"].get("file_path") == file_path
        ]
        for pid in to_delete:
            del self._vectors[pid]
        return True

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[dict] = None,
    ) -> list[dict]:
        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        results = []
        for point_id, data in self._vectors.items():
            vec = np.array(data["vector"], dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue

            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            if score < score_threshold:
                continue

            if filter_conditions:
                payload = data["payload"]
                if not all(payload.get(k) == v for k, v in filter_conditions.items()):
                    continue

            results.append({
                "id": point_id,
                "score": score,
                "payload": data["payload"],
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    async def get_stats(self):
        @dataclass
        class Stats:
            vectors_count: int = 0
            indexed_vectors_count: int = 0
            status: str = "ok"

        return Stats(
            vectors_count=len(self._vectors),
            indexed_vectors_count=len(self._vectors),
        )

    async def close(self) -> None:
        self._vectors.clear()


# ============================================================================
# Deterministic pipeline integration tests
# ============================================================================


class TestSearchPipelineWithDeterministicEmbedding:
    """Integration tests using deterministic hash-based embeddings.

    Proves the actual pipeline works: code is parsed, symbols are
    extracted, embeddings are generated, vectors are stored, and
    retrieval by similarity returns correct results. Zero mocks.
    """

    DIM = 64

    @pytest.fixture
    def hash_engine(self):
        return HashEmbeddingEngine(dim=self.DIM)

    @pytest.fixture
    def mem_store(self):
        return InMemoryVectorStore(vector_size=self.DIM)

    @pytest.fixture
    def real_unified_backend(self, hash_engine, mem_store):
        return SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=hash_engine,
            vector_store=mem_store,
            use_unified_extraction=True,
        )

    @pytest.mark.asyncio
    async def test_unified_index_stores_real_vectors(self, real_unified_backend, mem_store):
        """Unified indexing extracts symbols and stores real embedding vectors."""
        await real_unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )

        stats = await mem_store.get_stats()
        assert stats.vectors_count > 0, "No vectors stored after indexing"

        for point_data in mem_store._vectors.values():
            vec = point_data["vector"]
            assert len(vec) == self.DIM, f"Expected {self.DIM}-dim vector, got {len(vec)}"

    @pytest.mark.asyncio
    async def test_unified_index_extracts_expected_symbols(self, real_unified_backend, mem_store):
        """Symbols extracted by unified backend match known code structure."""
        await real_unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )

        stored_names = {
            data["payload"]["name"] for data in mem_store._vectors.values()
        }
        assert "UserService" in stored_names, f"UserService not found in {stored_names}"
        assert "helper" in stored_names, f"helper not found in {stored_names}"

    @pytest.mark.asyncio
    async def test_different_symbols_produce_different_embeddings(
        self, real_unified_backend, mem_store
    ):
        """Different symbols produce different embedding vectors (hash property)."""
        await real_unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )

        vectors_by_name = {}
        for data in mem_store._vectors.values():
            name = data["payload"]["name"]
            vectors_by_name[name] = np.array(data["vector"])

        assert len(vectors_by_name) >= 2, "Need at least 2 symbols for comparison"

        names = list(vectors_by_name.keys())
        v1 = vectors_by_name[names[0]]
        v2 = vectors_by_name[names[1]]
        assert not np.allclose(v1, v2), (
            f"Vectors for '{names[0]}' and '{names[1]}' should differ"
        )

    @pytest.mark.asyncio
    async def test_search_retrieves_indexed_symbols(self, real_unified_backend, mem_store):
        """After indexing, search returns results from stored vectors."""
        await real_unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )

        results = await real_unified_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))

        assert len(results) > 0, "Search returned no results after indexing"
        for r in results:
            assert r.file_path == "test.py"
            assert r.score >= 0.0
            assert r.metadata.get("name") is not None

    @pytest.mark.asyncio
    async def test_delete_by_file_removes_vectors(self, real_unified_backend, mem_store):
        """Deleting by file removes all vectors for that file."""
        await real_unified_backend.index(
            "test.py", PYTHON_CODE, {"language": "python"}
        )

        stats_before = await mem_store.get_stats()
        assert stats_before.vectors_count > 0

        await mem_store.delete_by_file("test.py")

        stats_after = await mem_store.get_stats()
        assert stats_after.vectors_count == 0, "Vectors not deleted"

    @pytest.mark.asyncio
    async def test_both_paths_produce_embeddings(self, hash_engine):
        """Both unified and legacy extraction produce real embeddings.

        Replaces the tautological 'assert legacy_calls >= 0' with a
        meaningful check that both paths actually index something.
        """
        unified_store = InMemoryVectorStore(vector_size=self.DIM)
        unified_backend = SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=hash_engine,
            vector_store=unified_store,
            use_unified_extraction=True,
        )
        await unified_backend.index("test.py", PYTHON_CODE, {"language": "python"})
        unified_count = (await unified_store.get_stats()).vectors_count

        legacy_store = InMemoryVectorStore(vector_size=self.DIM)
        legacy_backend = SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=hash_engine,
            vector_store=legacy_store,
            use_unified_extraction=False,
        )
        await legacy_backend.index("test.py", PYTHON_CODE, {"language": "python"})
        legacy_count = (await legacy_store.get_stats()).vectors_count

        assert unified_count > 0, "Unified extraction produced zero embeddings"
        assert legacy_count > 0, "Legacy extraction produced zero embeddings"
