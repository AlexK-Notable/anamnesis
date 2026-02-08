"""Integration tests for QdrantVectorStore with a real in-memory Qdrant instance.

Unlike test_qdrant_store.py (which uses MagicMock), these tests exercise actual
Qdrant operations -- upsert, search, filter, delete, stats -- against an in-memory
Qdrant backend.  This catches payload/filter/threshold bugs that mocks cannot detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.storage.qdrant_store import QdrantConfig, QdrantVectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 64  # Small for speed; sufficient for cosine similarity tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vector(seed: int = 0) -> list[float]:
    """Return a normalized random vector of VECTOR_SIZE dimensions."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(VECTOR_SIZE).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _similar_vector(base: list[float], noise: float = 0.05, seed: int = 99) -> list[float]:
    """Return a vector close to *base* (high cosine similarity)."""
    rng = np.random.default_rng(seed)
    arr = np.array(base, dtype=np.float32) + noise * rng.standard_normal(VECTOR_SIZE).astype(
        np.float32
    )
    arr /= np.linalg.norm(arr)
    return arr.tolist()


def _orthogonal_vector(base: list[float], seed: int = 42) -> list[float]:
    """Return a vector roughly orthogonal to *base* (low cosine similarity)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(VECTOR_SIZE).astype(np.float32)
    base_arr = np.array(base, dtype=np.float32)
    # Gram-Schmidt: remove the component along base
    arr -= np.dot(arr, base_arr) * base_arr
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        # Extremely unlikely, but handle gracefully
        arr = rng.standard_normal(VECTOR_SIZE).astype(np.float32)
    else:
        arr /= norm
    return arr.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store():
    """QdrantVectorStore backed by a real in-memory Qdrant client."""
    from qdrant_client import QdrantClient

    config = QdrantConfig(
        path=None,
        url=None,
        collection_name="test_integration",
        vector_size=VECTOR_SIZE,
        on_disk_payload=False,
    )
    qs = QdrantVectorStore(config)
    # Inject in-memory client, then initialise collection
    qs._client = QdrantClient(location=":memory:")
    await qs._ensure_collection()
    qs._initialized = True

    yield qs

    await qs.close()


# ===========================================================================
# 1. Upsert + search round-trip
# ===========================================================================


class TestUpsertSearchRoundTrip:
    """Upsert a point, search for it, verify payload is preserved."""

    async def test_upsert_then_search_returns_payload(self, store: QdrantVectorStore):
        vec = _random_vector(seed=1)
        await store.upsert(
            file_path="src/auth.py",
            name="AuthService",
            concept_type="class",
            embedding=vec,
            metadata={"language": "python", "lines": 42},
        )

        results = await store.search(query_vector=vec, limit=5, score_threshold=0.0)

        assert len(results) >= 1
        top = results[0]
        assert top["payload"]["file_path"] == "src/auth.py"
        assert top["payload"]["name"] == "AuthService"
        assert top["payload"]["concept_type"] == "class"
        assert top["payload"]["language"] == "python"
        assert top["payload"]["lines"] == 42

    async def test_identical_vector_score_near_one(self, store: QdrantVectorStore):
        vec = _random_vector(seed=2)
        await store.upsert("a.py", "Foo", "function", vec, {"language": "python"})

        results = await store.search(query_vector=vec, limit=1, score_threshold=0.0)

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=0.01)

    async def test_dissimilar_vector_low_score(self, store: QdrantVectorStore):
        vec_a = _random_vector(seed=10)
        vec_b = _orthogonal_vector(vec_a, seed=11)

        await store.upsert("a.py", "A", "class", vec_a, {})

        results = await store.search(query_vector=vec_b, limit=1, score_threshold=None)

        # Orthogonal vectors should have score near 0
        assert len(results) == 1
        assert abs(results[0]["score"]) < 0.3

    async def test_search_orders_by_similarity(self, store: QdrantVectorStore):
        base = _random_vector(seed=20)
        close = _similar_vector(base, noise=0.05, seed=21)
        far = _orthogonal_vector(base, seed=22)

        await store.upsert("close.py", "Close", "class", close, {})
        await store.upsert("far.py", "Far", "class", far, {})

        results = await store.search(query_vector=base, limit=10, score_threshold=None)

        assert len(results) == 2
        assert results[0]["payload"]["name"] == "Close"
        assert results[1]["payload"]["name"] == "Far"
        assert results[0]["score"] > results[1]["score"]


# ===========================================================================
# 2. Batch upsert round-trip
# ===========================================================================


class TestBatchUpsertRoundTrip:
    """Batch upsert followed by individual searches."""

    async def test_batch_upsert_all_searchable(self, store: QdrantVectorStore):
        points = []
        vectors = []
        for i in range(5):
            vec = _random_vector(seed=100 + i)
            vectors.append(vec)
            points.append((f"file_{i}.py", f"Symbol_{i}", "function", vec, {"index": i}))

        ids = await store.upsert_batch(points)
        assert len(ids) == 5

        # Each vector should be the top result when searching with itself
        for i, vec in enumerate(vectors):
            results = await store.search(query_vector=vec, limit=1, score_threshold=0.5)
            assert len(results) >= 1
            assert results[0]["payload"]["name"] == f"Symbol_{i}"

    async def test_empty_batch_is_noop(self, store: QdrantVectorStore):
        ids = await store.upsert_batch([])
        assert ids == []

        stats = await store.get_stats()
        assert stats.points_count == 0

    async def test_batch_upsert_preserves_metadata(self, store: QdrantVectorStore):
        vec_a = _random_vector(seed=200)
        vec_b = _random_vector(seed=201)
        points = [
            ("a.py", "A", "class", vec_a, {"language": "python", "depth": 3}),
            ("b.rs", "B", "struct", vec_b, {"language": "rust", "depth": 1}),
        ]
        await store.upsert_batch(points)

        results = await store.search(query_vector=vec_a, limit=1, score_threshold=0.5)
        assert results[0]["payload"]["language"] == "python"
        assert results[0]["payload"]["depth"] == 3

        results = await store.search(query_vector=vec_b, limit=1, score_threshold=0.5)
        assert results[0]["payload"]["language"] == "rust"
        assert results[0]["payload"]["depth"] == 1


# ===========================================================================
# 3. Filtered search
# ===========================================================================


class TestFilteredSearch:
    """Search with filter_conditions on payload fields."""

    async def _seed_multilang(self, store: QdrantVectorStore):
        """Insert 4 points: 2 Python (class, function), 2 Rust (struct, function)."""
        vecs = {k: _random_vector(seed=300 + i) for i, k in enumerate(["pc", "pf", "rs", "rf"])}
        await store.upsert_batch([
            ("auth.py", "Auth", "class", vecs["pc"], {"language": "python"}),
            ("auth.py", "validate", "function", vecs["pf"], {"language": "python"}),
            ("lib.rs", "Config", "struct", vecs["rs"], {"language": "rust"}),
            ("lib.rs", "init", "function", vecs["rf"], {"language": "rust"}),
        ])
        return vecs

    async def test_filter_by_language(self, store: QdrantVectorStore):
        vecs = await self._seed_multilang(store)

        results = await store.search(
            query_vector=vecs["pc"],
            limit=10,
            score_threshold=None,
            filter_conditions={"language": "python"},
        )

        languages = {r["payload"]["language"] for r in results}
        assert languages == {"python"}
        assert len(results) == 2

    async def test_filter_by_concept_type(self, store: QdrantVectorStore):
        vecs = await self._seed_multilang(store)

        results = await store.search(
            query_vector=vecs["pf"],
            limit=10,
            score_threshold=None,
            filter_conditions={"concept_type": "function"},
        )

        types = {r["payload"]["concept_type"] for r in results}
        assert types == {"function"}
        assert len(results) == 2

    async def test_combined_filters(self, store: QdrantVectorStore):
        vecs = await self._seed_multilang(store)

        results = await store.search(
            query_vector=vecs["rf"],
            limit=10,
            score_threshold=None,
            filter_conditions={"language": "rust", "concept_type": "function"},
        )

        assert len(results) == 1
        assert results[0]["payload"]["name"] == "init"

    async def test_empty_filter_returns_all(self, store: QdrantVectorStore):
        vecs = await self._seed_multilang(store)

        results = await store.search(
            query_vector=vecs["pc"],
            limit=10,
            score_threshold=None,
            filter_conditions={},
        )

        assert len(results) == 4

    async def test_none_value_in_filter_ignored(self, store: QdrantVectorStore):
        vecs = await self._seed_multilang(store)

        results = await store.search(
            query_vector=vecs["pc"],
            limit=10,
            score_threshold=None,
            filter_conditions={"language": "python", "concept_type": None},
        )

        # concept_type=None should be skipped, so only language filter applies
        assert len(results) == 2
        assert all(r["payload"]["language"] == "python" for r in results)


# ===========================================================================
# 4. Delete operations
# ===========================================================================


class TestDeleteOperations:
    """delete_by_file and delete_collection against real Qdrant."""

    async def test_delete_by_file_removes_only_target(self, store: QdrantVectorStore):
        vec_a = _random_vector(seed=400)
        vec_b = _random_vector(seed=401)

        await store.upsert("remove_me.py", "A", "class", vec_a, {})
        await store.upsert("keep_me.py", "B", "class", vec_b, {})

        result = await store.delete_by_file("remove_me.py")
        assert result is True

        # Only keep_me.py should remain
        results = await store.search(query_vector=vec_b, limit=10, score_threshold=None)
        file_paths = {r["payload"]["file_path"] for r in results}
        assert "remove_me.py" not in file_paths
        assert "keep_me.py" in file_paths

    async def test_delete_by_file_removes_multiple_symbols(self, store: QdrantVectorStore):
        vec1 = _random_vector(seed=410)
        vec2 = _random_vector(seed=411)
        vec_other = _random_vector(seed=412)

        await store.upsert("target.py", "Foo", "class", vec1, {})
        await store.upsert("target.py", "bar", "function", vec2, {})
        await store.upsert("other.py", "Baz", "class", vec_other, {})

        await store.delete_by_file("target.py")

        stats = await store.get_stats()
        assert stats.points_count == 1

    async def test_delete_nonexistent_file_no_error(self, store: QdrantVectorStore):
        """Deleting vectors for a file that has no points should not raise."""
        result = await store.delete_by_file("does_not_exist.py")
        assert result is True

    async def test_delete_collection(self, store: QdrantVectorStore):
        vec = _random_vector(seed=420)
        await store.upsert("a.py", "X", "class", vec, {})

        result = await store.delete_collection()
        assert result is True

        # After deleting the collection, stats should report error (collection gone)
        stats = await store.get_stats()
        assert stats.status == "error"


# ===========================================================================
# 5. Score threshold
# ===========================================================================


class TestScoreThreshold:
    """Verify score_threshold filters results with real similarity scores."""

    async def test_high_threshold_excludes_dissimilar(self, store: QdrantVectorStore):
        base = _random_vector(seed=500)
        ortho = _orthogonal_vector(base, seed=501)
        close = _similar_vector(base, noise=0.02, seed=502)

        await store.upsert("close.py", "Close", "class", close, {})
        await store.upsert("ortho.py", "Ortho", "class", ortho, {})

        results = await store.search(query_vector=base, limit=10, score_threshold=0.8)

        # Only the close vector should pass the 0.8 threshold
        names = {r["payload"]["name"] for r in results}
        assert "Close" in names
        assert "Ortho" not in names

    async def test_no_threshold_includes_all(self, store: QdrantVectorStore):
        base = _random_vector(seed=510)
        ortho = _orthogonal_vector(base, seed=511)
        close = _similar_vector(base, noise=0.02, seed=512)

        await store.upsert("close.py", "Close", "class", close, {})
        await store.upsert("ortho.py", "Ortho", "class", ortho, {})

        results = await store.search(query_vector=base, limit=10, score_threshold=None)

        assert len(results) == 2

    async def test_threshold_one_excludes_non_identical(self, store: QdrantVectorStore):
        base = _random_vector(seed=520)
        close = _similar_vector(base, noise=0.1, seed=521)

        await store.upsert("a.py", "A", "class", base, {})
        await store.upsert("b.py", "B", "class", close, {})

        results = await store.search(query_vector=base, limit=10, score_threshold=1.0)

        # Only the identical vector should pass threshold=1.0
        assert len(results) == 1
        assert results[0]["payload"]["name"] == "A"


# ===========================================================================
# 6. Stats accuracy
# ===========================================================================


class TestStatsAccuracy:
    """get_stats() reflects actual collection state."""

    async def test_stats_empty_collection(self, store: QdrantVectorStore):
        stats = await store.get_stats()
        assert stats.points_count == 0
        assert stats.collection_name == "test_integration"

    async def test_stats_after_upserts(self, store: QdrantVectorStore):
        for i in range(3):
            vec = _random_vector(seed=600 + i)
            await store.upsert(f"f{i}.py", f"S{i}", "class", vec, {})

        stats = await store.get_stats()
        assert stats.points_count == 3

    async def test_stats_after_delete(self, store: QdrantVectorStore):
        for i in range(4):
            vec = _random_vector(seed=610 + i)
            await store.upsert(f"file{i}.py", f"Sym{i}", "function", vec, {})

        await store.delete_by_file("file0.py")

        stats = await store.get_stats()
        assert stats.points_count == 3


# ===========================================================================
# 7. Idempotent upsert
# ===========================================================================


class TestIdempotentUpsert:
    """Upserting the same (file_path, name, concept_type) twice updates in place."""

    async def test_duplicate_key_overwrites_embedding(self, store: QdrantVectorStore):
        vec_old = _random_vector(seed=700)
        vec_new = _random_vector(seed=701)

        id1 = await store.upsert("a.py", "Foo", "class", vec_old, {"version": 1})
        id2 = await store.upsert("a.py", "Foo", "class", vec_new, {"version": 2})

        # Same deterministic ID
        assert id1 == id2

        # Only one point should exist
        stats = await store.get_stats()
        assert stats.points_count == 1

    async def test_duplicate_key_updates_payload(self, store: QdrantVectorStore):
        vec = _random_vector(seed=710)

        await store.upsert("a.py", "Bar", "function", vec, {"language": "python", "v": 1})
        await store.upsert("a.py", "Bar", "function", vec, {"language": "python", "v": 2})

        results = await store.search(query_vector=vec, limit=1, score_threshold=0.0)

        assert len(results) == 1
        assert results[0]["payload"]["v"] == 2

    async def test_different_keys_create_separate_points(self, store: QdrantVectorStore):
        vec = _random_vector(seed=720)

        await store.upsert("a.py", "Foo", "class", vec, {})
        await store.upsert("a.py", "Foo", "function", vec, {})  # different concept_type
        await store.upsert("b.py", "Foo", "class", vec, {})  # different file_path

        stats = await store.get_stats()
        assert stats.points_count == 3


# ===========================================================================
# 8. Collection creation
# ===========================================================================


class TestCollectionCreation:
    """_ensure_collection is idempotent and creates proper indexes."""

    async def test_ensure_collection_idempotent(self, store: QdrantVectorStore):
        """Calling _ensure_collection() again does not raise or duplicate the collection."""
        await store._ensure_collection()
        await store._ensure_collection()

        # Still works normally
        vec = _random_vector(seed=800)
        await store.upsert("a.py", "A", "class", vec, {})

        results = await store.search(query_vector=vec, limit=1, score_threshold=0.0)
        assert len(results) == 1

    async def test_payload_indexes_enable_filtered_search(self, store: QdrantVectorStore):
        """Indexes on file_path, language, concept_type allow efficient filtered search.

        We verify indirectly by confirming filtered searches return correct results,
        which requires working payload indexes.
        """
        vec = _random_vector(seed=810)
        await store.upsert("x.py", "X", "class", vec, {"language": "go"})
        await store.upsert("y.py", "Y", "function", vec, {"language": "python"})

        # Filter by language
        results = await store.search(
            query_vector=vec, limit=10, score_threshold=0.0,
            filter_conditions={"language": "go"},
        )
        assert len(results) == 1
        assert results[0]["payload"]["name"] == "X"

        # Filter by concept_type
        results = await store.search(
            query_vector=vec, limit=10, score_threshold=None,
            filter_conditions={"concept_type": "function"},
        )
        assert len(results) == 1
        assert results[0]["payload"]["name"] == "Y"

        # Filter by file_path
        results = await store.search(
            query_vector=vec, limit=10, score_threshold=0.0,
            filter_conditions={"file_path": "x.py"},
        )
        assert len(results) == 1
        assert results[0]["payload"]["name"] == "X"
