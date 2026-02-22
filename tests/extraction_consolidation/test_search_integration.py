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

from anamnesis.interfaces.search import SearchQuery, SearchResult, SearchType
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


# ============================================================================
# Semantic search with real Qdrant (in-memory) integration tests
# ============================================================================


PYTHON_CODE_ALT = '''
class OrderProcessor:
    """Handles order creation and fulfillment."""

    def process_order(self, order_id: int) -> bool:
        """Process a pending order."""
        return True

    def cancel_order(self, order_id: int) -> None:
        """Cancel an existing order."""
        pass

def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax for a given amount."""
    return amount * rate
'''


PYTHON_CODE_V2 = '''
class RevisedService:
    """A completely different class after re-indexing."""

    def new_method(self) -> str:
        return "revised"
'''


class TestSemanticSearchWithQdrant:
    """Integration tests using real in-memory Qdrant vector store.

    These tests exercise the full pipeline end-to-end: code parsing via
    tree-sitter, hash-based embedding generation, Qdrant vector storage
    and retrieval. Zero mocks -- real Qdrant (in-memory), real tree-sitter
    parsing, real (hash-based deterministic) embeddings.
    """

    DIM = 64

    @pytest.fixture
    def hash_engine(self):
        return HashEmbeddingEngine(dim=self.DIM)

    @pytest.fixture
    async def qdrant_store(self):
        """Real in-memory Qdrant vector store."""
        from qdrant_client import QdrantClient

        from anamnesis.storage.qdrant_store import QdrantConfig, QdrantVectorStore

        config = QdrantConfig(
            path=None,
            collection_name="test_semantic_search",
            vector_size=self.DIM,
        )
        store = QdrantVectorStore(config)
        store._client = QdrantClient(location=":memory:")
        await store._ensure_collection()
        store._initialized = True
        yield store
        await store.close()

    @pytest.fixture
    def qdrant_backend(self, hash_engine, qdrant_store):
        """SemanticSearchBackend backed by real in-memory Qdrant."""
        return SemanticSearchBackend(
            base_path="/tmp/test",
            embedding_engine=hash_engine,
            vector_store=qdrant_store,
            use_unified_extraction=True,
        )

    # ------------------------------------------------------------------
    # 1. Index and search round trip
    # ------------------------------------------------------------------

    async def test_index_and_search_round_trip(self, qdrant_backend):
        """Index Python code and retrieve it by searching for a symbol name."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        results = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))

        assert len(results) > 0, "Expected at least one result after indexing"
        names_found = {r.metadata.get("name") for r in results}
        assert "UserService" in names_found, f"UserService not in results: {names_found}"

    # ------------------------------------------------------------------
    # 2. Search with language filter
    # ------------------------------------------------------------------

    async def test_search_with_matching_language_filter(self, qdrant_backend):
        """Search with language='python' returns results for Python-indexed code."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        results = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            language="python",
            limit=10,
        ))

        assert len(results) > 0, "Expected results with matching language filter"
        for r in results:
            assert r.file_path == "test.py"

    async def test_search_with_non_matching_language_filter(self, qdrant_backend):
        """Search with language='go' returns empty for Python-only indexed code."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        results = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            language="go",
            limit=10,
        ))

        assert len(results) == 0, f"Expected no results with mismatched language, got {len(results)}"

    # ------------------------------------------------------------------
    # 3. Multiple files indexed
    # ------------------------------------------------------------------

    async def test_multiple_files_indexed(self, qdrant_backend):
        """Index two files, search returns results with correct file paths."""
        await qdrant_backend.index("user.py", PYTHON_CODE, {"language": "python"})
        await qdrant_backend.index("order.py", PYTHON_CODE_ALT, {"language": "python"})

        # Search for something from the first file
        results_user = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=20,
        ))
        file_paths = {r.file_path for r in results_user}
        assert "user.py" in file_paths, f"user.py not in result file paths: {file_paths}"

        # Search for something from the second file
        results_order = await qdrant_backend.search(SearchQuery(
            query="OrderProcessor",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=20,
        ))
        file_paths_order = {r.file_path for r in results_order}
        assert "order.py" in file_paths_order, (
            f"order.py not in result file paths: {file_paths_order}"
        )

    # ------------------------------------------------------------------
    # 4. Re-index overwrites old content
    # ------------------------------------------------------------------

    async def test_reindex_overwrites_old_content(self, qdrant_backend):
        """Re-indexing a file replaces old symbols with new ones."""
        await qdrant_backend.index("svc.py", PYTHON_CODE, {"language": "python"})

        # Verify old symbol is searchable
        results_before = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))
        names_before = {r.metadata.get("name") for r in results_before}
        assert "UserService" in names_before

        # Re-index with completely different content
        await qdrant_backend.index("svc.py", PYTHON_CODE_V2, {"language": "python"})

        # Search for new symbol
        results_after = await qdrant_backend.search(SearchQuery(
            query="RevisedService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))
        names_after = {r.metadata.get("name") for r in results_after}
        assert "RevisedService" in names_after, f"New symbol not found: {names_after}"

        # Old symbol from svc.py should be gone (it was deleted before re-index)
        # Only results from svc.py should have the new symbol
        svc_results = [r for r in results_after if r.file_path == "svc.py"]
        svc_names = {r.metadata.get("name") for r in svc_results}
        assert "UserService" not in svc_names, "Old symbol should be removed after re-index"

    # ------------------------------------------------------------------
    # 5. Search result structure validation
    # ------------------------------------------------------------------

    async def test_search_result_structure(self, qdrant_backend):
        """Verify SearchResult has expected fields and search_type == SEMANTIC."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        results = await qdrant_backend.search(SearchQuery(
            query="helper",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))

        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.file_path, str)
            assert len(r.file_path) > 0
            assert isinstance(r.score, float)
            assert r.search_type == SearchType.SEMANTIC
            assert isinstance(r.metadata, dict)
            assert "name" in r.metadata
            assert "concept_type" in r.metadata
            assert "qdrant_id" in r.metadata
            assert isinstance(r.matches, list)
            assert len(r.matches) > 0

    # ------------------------------------------------------------------
    # 6. Results sorted by score descending
    # ------------------------------------------------------------------

    async def test_results_sorted_by_score_descending(self, qdrant_backend):
        """Search results should be ordered by similarity score, highest first."""
        await qdrant_backend.index("user.py", PYTHON_CODE, {"language": "python"})
        await qdrant_backend.index("order.py", PYTHON_CODE_ALT, {"language": "python"})

        results = await qdrant_backend.search(SearchQuery(
            query="class",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=20,
        ))

        assert len(results) >= 2, "Expected at least 2 results for sort order check"
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Results not sorted by score: {scores}"
            )

    # ------------------------------------------------------------------
    # 7. High similarity threshold filters results
    # ------------------------------------------------------------------

    async def test_high_threshold_filters_results(self, qdrant_backend):
        """A very high similarity threshold should return fewer or no results."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        # With threshold 0.0, we get all matches
        results_all = await qdrant_backend.search(SearchQuery(
            query="something_unrelated_to_code",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=50,
        ))

        # With threshold 0.99, hash embeddings are unlikely to reach this score
        # unless query text is identical to indexed text
        results_high = await qdrant_backend.search(SearchQuery(
            query="something_unrelated_to_code",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.99,
            limit=50,
        ))

        assert len(results_high) <= len(results_all), (
            "High threshold should return equal or fewer results"
        )

    # ------------------------------------------------------------------
    # 8. Low threshold includes all indexed symbols
    # ------------------------------------------------------------------

    async def test_low_threshold_includes_all(self, qdrant_backend, qdrant_store):
        """Threshold of 0.0 should return all indexed symbols (up to limit)."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        stats = await qdrant_store.get_stats()
        indexed_count = stats.points_count

        results = await qdrant_backend.search(SearchQuery(
            query="any query text",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=100,
        ))

        # With threshold 0.0, we should get back all indexed vectors
        # (Qdrant may still filter out exact-zero scores, but hash embeddings
        # produce non-zero cosine similarity for non-zero vectors)
        assert len(results) == indexed_count, (
            f"Expected {indexed_count} results with threshold=0.0, got {len(results)}"
        )

    # ------------------------------------------------------------------
    # 9. Delete by file then search
    # ------------------------------------------------------------------

    async def test_delete_then_search(self, qdrant_backend, qdrant_store):
        """After deleting vectors for a file, search should return no results for it."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        # Confirm vectors exist
        stats_before = await qdrant_store.get_stats()
        assert stats_before.points_count > 0, "No vectors after indexing"

        # Delete vectors for the file
        await qdrant_store.delete_by_file("test.py")

        # Search should return nothing
        results = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))

        assert len(results) == 0, f"Expected no results after deletion, got {len(results)}"

    # ------------------------------------------------------------------
    # 10. Empty index returns no results
    # ------------------------------------------------------------------

    async def test_empty_index_returns_no_results(self, qdrant_backend):
        """Searching an empty index should return an empty list, not an error."""
        results = await qdrant_backend.search(SearchQuery(
            query="anything",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))

        assert results == [], f"Expected empty list from empty index, got {results}"

    # ------------------------------------------------------------------
    # 11. Index directory with temp directory
    # ------------------------------------------------------------------

    async def test_index_directory(self, qdrant_backend, qdrant_store, tmp_path):
        """index_directory indexes all matching files in a directory."""
        # Create temp Python files
        (tmp_path / "module_a.py").write_text(PYTHON_CODE, encoding="utf-8")
        (tmp_path / "module_b.py").write_text(PYTHON_CODE_ALT, encoding="utf-8")
        # Create a non-Python file that should be skipped by default patterns
        (tmp_path / "readme.txt").write_text("just a readme", encoding="utf-8")

        # Override base_path so index_directory works with our temp dir
        qdrant_backend._base_path = tmp_path

        count = await qdrant_backend.index_directory(
            directory=str(tmp_path),
            patterns=["*.py"],
        )

        assert count == 2, f"Expected 2 files indexed, got {count}"

        stats = await qdrant_store.get_stats()
        assert stats.points_count > 0, "No vectors stored after indexing directory"

    # ------------------------------------------------------------------
    # 12. Qdrant stats reflect indexed data
    # ------------------------------------------------------------------

    async def test_qdrant_stats_after_indexing(self, qdrant_backend, qdrant_store):
        """get_stats returns accurate counts after indexing."""
        stats_empty = await qdrant_store.get_stats()
        assert stats_empty.points_count == 0

        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})

        stats_after = await qdrant_store.get_stats()
        assert stats_after.points_count > 0
        assert stats_after.collection_name == "test_semantic_search"

    # ------------------------------------------------------------------
    # 13. Deterministic IDs allow upsert deduplication
    # ------------------------------------------------------------------

    async def test_upsert_deduplication(self, qdrant_backend, qdrant_store):
        """Indexing the same file twice should not double the vector count."""
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})
        stats_first = await qdrant_store.get_stats()
        count_first = stats_first.points_count

        # Index the exact same file again (without content change)
        await qdrant_backend.index("test.py", PYTHON_CODE, {"language": "python"})
        stats_second = await qdrant_store.get_stats()
        count_second = stats_second.points_count

        assert count_second == count_first, (
            f"Expected {count_first} vectors after re-index, got {count_second} "
            "(upsert should deduplicate)"
        )

    # ------------------------------------------------------------------
    # 14. Delete only affects targeted file
    # ------------------------------------------------------------------

    async def test_delete_only_affects_targeted_file(self, qdrant_backend, qdrant_store):
        """Deleting vectors for one file should not affect another file's vectors."""
        await qdrant_backend.index("keep.py", PYTHON_CODE, {"language": "python"})
        await qdrant_backend.index("remove.py", PYTHON_CODE_ALT, {"language": "python"})

        stats_before = await qdrant_store.get_stats()
        total_before = stats_before.points_count

        await qdrant_store.delete_by_file("remove.py")

        stats_after = await qdrant_store.get_stats()
        total_after = stats_after.points_count

        assert total_after < total_before, "Delete should reduce vector count"
        assert total_after > 0, "Vectors for keep.py should still exist"

        # Confirm keep.py results are still searchable
        results = await qdrant_backend.search(SearchQuery(
            query="UserService",
            search_type=SearchType.SEMANTIC,
            similarity_threshold=0.0,
            limit=10,
        ))
        file_paths = {r.file_path for r in results}
        assert "keep.py" in file_paths, "keep.py vectors should survive deletion of remove.py"
        assert "remove.py" not in file_paths, "remove.py vectors should be gone"
