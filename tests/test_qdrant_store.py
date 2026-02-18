"""Tests for QdrantVectorStore.

Covers:
- QdrantConfig defaults and custom configuration
- QdrantStats defaults
- Deterministic ID generation (_generate_id)
- Connection lifecycle: connect (embedded/server), close, is_connected
- Connect guard: second connect() is a no-op
- Connect error paths: ImportError, general exceptions
- Collection creation: _ensure_collection (exists / doesn't exist)
- Upsert: payload construction, not-connected guard
- Batch upsert: empty list short-circuit, payload construction, not-connected guard
- Search: result transformation, filter construction (None values excluded), not-connected guard
- Delete by file: filter construction, not-connected guard
- Delete collection: delegation, not-connected guard
- Stats: successful retrieval, disconnected state, exception fallback
- Close: state cleanup, idempotent when already closed
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from anamnesis.storage.qdrant_store import QdrantConfig, QdrantStats, QdrantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(dim: int = 384) -> list[float]:
    """Create a fake embedding vector of the given dimension."""
    return [0.1] * dim


def _make_scored_point(point_id: str, score: float, payload: dict) -> SimpleNamespace:
    """Simulate a Qdrant ScoredPoint returned from client.search()."""
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def _make_collection_info(
    name: str,
    vectors_count: int = 100,
    indexed_vectors_count: int = 90,
    points_count: int = 100,
    status_value: str = "green",
    vector_size: int = 384,
) -> SimpleNamespace:
    """Simulate the CollectionInfo object returned by client.get_collection()."""
    return SimpleNamespace(
        vectors_count=vectors_count,
        indexed_vectors_count=indexed_vectors_count,
        points_count=points_count,
        status=SimpleNamespace(value=status_value),
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=vector_size),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path) -> QdrantConfig:
    """QdrantConfig pointing to a temp directory (embedded mode)."""
    return QdrantConfig(path=str(tmp_path / "qdrant"), collection_name="test_collection")


@pytest.fixture
def mock_qdrant_client():
    """A MagicMock standing in for qdrant_client.QdrantClient."""
    client = MagicMock(name="QdrantClient")
    # get_collections returns an object with a .collections list
    client.get_collections.return_value = SimpleNamespace(collections=[])
    return client


@pytest.fixture
def store_connected(config, mock_qdrant_client) -> QdrantVectorStore:
    """A QdrantVectorStore that appears connected (client injected directly).

    This bypasses connect() so tests can exercise individual methods
    without needing to mock the entire import + init chain.
    """
    store = QdrantVectorStore(config)
    store._client = mock_qdrant_client
    store._initialized = True
    return store


# ===========================================================================
# 1. Dataclass defaults
# ===========================================================================


class TestQdrantConfig:
    """Verify QdrantConfig default and custom values."""

    def test_defaults(self):
        cfg = QdrantConfig()
        assert cfg.path == ".anamnesis/qdrant"
        assert cfg.url is None
        assert cfg.api_key is None
        assert cfg.collection_name == "code_embeddings"
        assert cfg.vector_size == 384
        assert cfg.prefer_grpc is True
        assert cfg.timeout == 30.0
        assert cfg.on_disk_payload is True

    def test_custom_values(self):
        cfg = QdrantConfig(
            path="/custom/path",
            url="http://qdrant:6333",
            api_key="secret",
            collection_name="my_coll",
            vector_size=768,
            prefer_grpc=False,
            timeout=60.0,
            on_disk_payload=False,
        )
        assert cfg.url == "http://qdrant:6333"
        assert cfg.api_key == "secret"
        assert cfg.vector_size == 768
        assert cfg.prefer_grpc is False


class TestQdrantStats:
    """Verify QdrantStats defaults."""

    def test_defaults(self):
        stats = QdrantStats()
        assert stats.vectors_count == 0
        assert stats.indexed_vectors_count == 0
        assert stats.points_count == 0
        assert stats.status == "unknown"
        assert stats.collection_name == ""


# ===========================================================================
# 2. Deterministic ID generation
# ===========================================================================


class TestGenerateId:
    """_generate_id produces deterministic, unique SHA256 hashes."""

    def test_deterministic(self, config):
        store = QdrantVectorStore(config)
        id1 = store._generate_id("src/auth.py", "AuthService", "class")
        id2 = store._generate_id("src/auth.py", "AuthService", "class")
        assert id1 == id2

    def test_matches_uuid5(self, config):
        store = QdrantVectorStore(config)
        result = store._generate_id("a.py", "Foo", "function")
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, "a.py:Foo:function"))
        assert result == expected

    def test_different_inputs_produce_different_ids(self, config):
        store = QdrantVectorStore(config)
        id_a = store._generate_id("a.py", "Foo", "class")
        id_b = store._generate_id("b.py", "Foo", "class")
        id_c = store._generate_id("a.py", "Bar", "class")
        id_d = store._generate_id("a.py", "Foo", "function")
        assert len({id_a, id_b, id_c, id_d}) == 4


# ===========================================================================
# 3. Connection lifecycle
# ===========================================================================


class TestConnectionLifecycle:
    """Tests for connect(), close(), and is_connected."""

    def test_not_connected_initially(self, config):
        store = QdrantVectorStore(config)
        assert not store.is_connected

    def test_is_connected_after_injection(self, store_connected):
        assert store_connected.is_connected

    async def test_connect_embedded_mode(self, config, tmp_path):
        """connect() in embedded mode creates a directory and sets up client."""
        mock_client_instance = MagicMock(name="client_instance")
        mock_client_instance.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=config.collection_name)]
        )
        mock_client_instance.get_collection.return_value = _make_collection_info(
            name=config.collection_name, vector_size=config.vector_size,
        )

        mock_cls_constructor = MagicMock(return_value=mock_client_instance)
        with patch.dict(
            "sys.modules",
            {"qdrant_client": MagicMock(QdrantClient=mock_cls_constructor, models=MagicMock())},
        ):
            store = QdrantVectorStore(config)
            await store.connect()

            assert store.is_connected
            mock_cls_constructor.assert_called_once_with(
                path=config.path,
                prefer_grpc=config.prefer_grpc,
            )

    async def test_connect_server_mode(self, tmp_path):
        """connect() in server mode passes url, api_key, timeout."""
        server_config = QdrantConfig(
            path=None,
            url="http://qdrant:6333",
            api_key="key123",
            collection_name="test_coll",
        )

        mock_client_instance = MagicMock(name="client_instance")
        mock_client_instance.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="test_coll")]
        )
        mock_client_instance.get_collection.return_value = _make_collection_info(
            name="test_coll", vector_size=server_config.vector_size,
        )

        mock_cls_constructor = MagicMock(return_value=mock_client_instance)
        with patch.dict(
            "sys.modules",
            {"qdrant_client": MagicMock(QdrantClient=mock_cls_constructor, models=MagicMock())},
        ):
            store = QdrantVectorStore(server_config)
            await store.connect()

            assert store.is_connected
            mock_cls_constructor.assert_called_once_with(
                url="http://qdrant:6333",
                api_key="key123",
                prefer_grpc=server_config.prefer_grpc,
                timeout=server_config.timeout,
            )

    async def test_connect_is_idempotent(self, store_connected):
        """Calling connect() when already initialized is a no-op."""
        original_client = store_connected._client
        await store_connected.connect()
        assert store_connected._client is original_client

    async def test_connect_import_error_propagates(self, config):
        """connect() raises ImportError if qdrant-client is not installed."""
        with patch.dict("sys.modules", {"qdrant_client": None}):
            store = QdrantVectorStore(config)
            with pytest.raises((ImportError, ModuleNotFoundError)):
                await store.connect()

    async def test_close_clears_state(self, store_connected, mock_qdrant_client):
        """close() calls client.close() and resets state."""
        await store_connected.close()

        assert not store_connected.is_connected
        assert store_connected._client is None
        assert store_connected._initialized is False
        mock_qdrant_client.close.assert_called_once()

    async def test_close_idempotent_when_already_closed(self, config):
        """close() on an unconnected store does not raise."""
        store = QdrantVectorStore(config)
        await store.close()  # Should not raise
        assert not store.is_connected


# ===========================================================================
# 4. _ensure_collection
# ===========================================================================


class TestEnsureCollection:
    """Collection creation logic in _ensure_collection."""

    async def test_skips_creation_when_collection_exists(self, store_connected, mock_qdrant_client):
        """When collection already exists, create_collection is NOT called."""
        mock_qdrant_client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=store_connected._config.collection_name)]
        )
        mock_qdrant_client.get_collection.return_value = _make_collection_info(
            name=store_connected._config.collection_name,
            vector_size=store_connected._config.vector_size,
        )
        # Patch the qdrant_client.models import inside _ensure_collection
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected._ensure_collection()

        mock_qdrant_client.create_collection.assert_not_called()

    async def test_creates_collection_and_indexes_when_missing(
        self, store_connected, mock_qdrant_client
    ):
        """When collection does not exist, creates it with indexes."""
        mock_qdrant_client.get_collections.return_value = SimpleNamespace(collections=[])

        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected._ensure_collection()

        mock_qdrant_client.create_collection.assert_called_once()
        # Three payload indexes: file_path, language, concept_type
        assert mock_qdrant_client.create_payload_index.call_count == 3

        actual_fields = [
            c.kwargs["field_name"]
            for c in mock_qdrant_client.create_payload_index.call_args_list
        ]
        assert set(actual_fields) == {"file_path", "language", "concept_type"}

    async def test_raises_when_client_is_none(self, config):
        """_ensure_collection raises RuntimeError if client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await store._ensure_collection()

    async def test_concurrent_ensure_collection(self, store_connected, mock_qdrant_client):
        """Concurrent _ensure_collection calls both succeed without error."""
        import asyncio

        mock_qdrant_client.get_collections.return_value = SimpleNamespace(collections=[])

        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            results = await asyncio.gather(
                store_connected._ensure_collection(),
                store_connected._ensure_collection(),
                return_exceptions=True,
            )

        # Both calls should succeed (return None) — no exceptions
        for r in results:
            assert not isinstance(r, Exception), f"_ensure_collection raised: {r}"


# ===========================================================================
# 5. Upsert
# ===========================================================================


class TestUpsert:
    """Tests for single-point upsert."""

    async def test_upsert_returns_deterministic_id(self, store_connected, mock_qdrant_client):
        """upsert() returns the SHA256-based point ID."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            point_id = await store_connected.upsert(
                file_path="src/auth.py",
                name="AuthService",
                concept_type="class",
                embedding=_fake_embedding(),
                metadata={"language": "python"},
            )

        expected_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "src/auth.py:AuthService:class"))
        assert point_id == expected_id

    async def test_upsert_constructs_correct_payload(self, store_connected, mock_qdrant_client):
        """upsert() merges file_path, name, concept_type, and metadata into payload."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.upsert(
                file_path="src/db.py",
                name="DatabasePool",
                concept_type="class",
                embedding=_fake_embedding(3),
                metadata={"language": "python", "complexity": 5},
            )

        # Inspect the PointStruct that was created
        call_args = mock_qdrant_client.upsert.call_args
        points_arg = call_args.kwargs.get("points") or call_args[1].get("points")
        assert len(points_arg) == 1

        point = points_arg[0]
        # mock_models.PointStruct was called with these kwargs
        ps_call = mock_models.PointStruct.call_args
        assert ps_call.kwargs["vector"] == _fake_embedding(3)

        payload = ps_call.kwargs["payload"]
        assert payload["file_path"] == "src/db.py"
        assert payload["name"] == "DatabasePool"
        assert payload["concept_type"] == "class"
        assert payload["language"] == "python"
        assert payload["complexity"] == 5

    async def test_upsert_raises_when_not_connected(self, config):
        """upsert() raises RuntimeError when client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert("a.py", "Foo", "class", [0.1], {})


# ===========================================================================
# 6. Batch upsert
# ===========================================================================


class TestUpsertBatch:
    """Tests for batch upsert."""

    async def test_empty_batch_returns_empty(self, store_connected):
        """upsert_batch([]) returns [] without calling client."""
        result = await store_connected.upsert_batch([])
        assert result == []
        store_connected._client.upsert.assert_not_called()

    async def test_batch_returns_correct_ids(self, store_connected, mock_qdrant_client):
        """upsert_batch returns one ID per input point."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            points = [
                ("a.py", "Foo", "class", _fake_embedding(3), {"language": "python"}),
                ("b.py", "bar", "function", _fake_embedding(3), {"language": "python"}),
            ]
            ids = await store_connected.upsert_batch(points)

        assert len(ids) == 2
        assert ids[0] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "a.py:Foo:class"))
        assert ids[1] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "b.py:bar:function"))
        # One call to client.upsert with all points batched
        mock_qdrant_client.upsert.assert_called_once()

    async def test_batch_constructs_all_point_payloads(self, store_connected, mock_qdrant_client):
        """Each point in the batch gets correct payload with merged metadata."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            points = [
                ("x.py", "A", "class", [1.0], {"k": "v1"}),
                ("y.py", "B", "function", [2.0], {"k": "v2"}),
            ]
            await store_connected.upsert_batch(points)

        # PointStruct was called twice
        assert mock_models.PointStruct.call_count == 2
        payloads = [call.kwargs["payload"] for call in mock_models.PointStruct.call_args_list]
        assert payloads[0] == {"file_path": "x.py", "name": "A", "concept_type": "class", "k": "v1"}
        assert payloads[1] == {"file_path": "y.py", "name": "B", "concept_type": "function", "k": "v2"}

    async def test_batch_raises_when_not_connected(self, config):
        """upsert_batch raises RuntimeError when client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert_batch([("a.py", "X", "class", [0.1], {})])


# ===========================================================================
# 7. Search
# ===========================================================================


class TestSearch:
    """Tests for vector search."""

    async def test_search_transforms_results(self, store_connected, mock_qdrant_client):
        """search() converts ScoredPoints into {id, score, payload} dicts."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[
                _make_scored_point("id1", 0.95, {"name": "Foo"}),
                _make_scored_point("id2", 0.80, {"name": "Bar"}),
            ]
        )

        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            results = await store_connected.search(query_vector=[0.1, 0.2, 0.3], limit=5)

        assert len(results) == 2
        assert results[0] == {"id": "id1", "score": 0.95, "payload": {"name": "Foo"}}
        assert results[1] == {"id": "id2", "score": 0.80, "payload": {"name": "Bar"}}

    async def test_search_empty_results(self, store_connected, mock_qdrant_client):
        """search() returns empty list when Qdrant returns no results."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])

        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            results = await store_connected.search(query_vector=[0.1], limit=10)

        assert results == []

    async def test_search_passes_parameters(self, store_connected, mock_qdrant_client):
        """search() forwards limit, score_threshold, and collection_name."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        mock_models = MagicMock()

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.search(
                query_vector=[0.5],
                limit=20,
                score_threshold=0.7,
            )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"
        assert call_kwargs["query"] == [0.5]
        assert call_kwargs["limit"] == 20
        assert call_kwargs["score_threshold"] == 0.7
        assert call_kwargs["query_filter"] is None

    async def test_search_builds_filter_from_conditions(self, store_connected, mock_qdrant_client):
        """search() constructs a Qdrant Filter from filter_conditions dict."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        mock_models = MagicMock()

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.search(
                query_vector=[0.5],
                filter_conditions={"language": "python", "concept_type": "class"},
            )

        # FieldCondition should have been called twice (one per filter key)
        assert mock_models.FieldCondition.call_count == 2
        # Filter should have been constructed with must conditions
        mock_models.Filter.assert_called_once()

    async def test_search_skips_none_values_in_filter(self, store_connected, mock_qdrant_client):
        """None values in filter_conditions are excluded from the Qdrant filter."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        mock_models = MagicMock()

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.search(
                query_vector=[0.5],
                filter_conditions={"language": "python", "concept_type": None},
            )

        # Only one FieldCondition (the None value was skipped)
        assert mock_models.FieldCondition.call_count == 1
        field_call = mock_models.FieldCondition.call_args
        assert field_call.kwargs["key"] == "language"

    async def test_search_no_filter_when_all_values_none(self, store_connected, mock_qdrant_client):
        """When all filter values are None, no filter is passed to Qdrant."""
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])
        mock_models = MagicMock()

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.search(
                query_vector=[0.5],
                filter_conditions={"language": None},
            )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    async def test_search_raises_when_not_connected(self, config):
        """search() raises RuntimeError when client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.search(query_vector=[0.1])


# ===========================================================================
# 8. Delete by file
# ===========================================================================


class TestDeleteByFile:
    """Tests for delete_by_file."""

    async def test_delete_by_file_returns_true(self, store_connected, mock_qdrant_client):
        """delete_by_file() returns True on success."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            result = await store_connected.delete_by_file("src/old.py")

        assert result is True
        mock_qdrant_client.delete.assert_called_once()

    async def test_delete_by_file_uses_correct_filter(self, store_connected, mock_qdrant_client):
        """delete_by_file() constructs a filter on the file_path field."""
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"qdrant_client": MagicMock(models=mock_models)}):
            await store_connected.delete_by_file("src/old.py")

        # FieldCondition was called with key="file_path" and value="src/old.py"
        mock_models.FieldCondition.assert_called_once()
        fc_kwargs = mock_models.FieldCondition.call_args.kwargs
        assert fc_kwargs["key"] == "file_path"
        mock_models.MatchValue.assert_called_once_with(value="src/old.py")

    async def test_delete_by_file_raises_when_not_connected(self, config):
        """delete_by_file() raises RuntimeError when client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.delete_by_file("anything.py")


# ===========================================================================
# 9. Delete collection
# ===========================================================================


class TestDeleteCollection:
    """Tests for delete_collection."""

    async def test_delete_collection_returns_true(self, store_connected, mock_qdrant_client):
        """delete_collection() returns True on success."""
        result = await store_connected.delete_collection()
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    async def test_delete_collection_raises_when_not_connected(self, config):
        """delete_collection() raises RuntimeError when client is None."""
        store = QdrantVectorStore(config)
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.delete_collection()


# ===========================================================================
# 10. Stats
# ===========================================================================


class TestGetStats:
    """Tests for get_stats."""

    async def test_stats_when_connected(self, store_connected, mock_qdrant_client):
        """get_stats() returns populated QdrantStats from collection info."""
        mock_qdrant_client.get_collection.return_value = _make_collection_info(
            name="test_collection",
            vectors_count=500,
            indexed_vectors_count=480,
            points_count=500,
            status_value="green",
        )

        stats = await store_connected.get_stats()

        assert isinstance(stats, QdrantStats)
        assert stats.vectors_count == 500
        assert stats.indexed_vectors_count == 480
        assert stats.points_count == 500
        assert stats.status == "green"
        assert stats.collection_name == "test_collection"

    async def test_stats_handles_none_counts(self, store_connected, mock_qdrant_client):
        """get_stats() treats None counts as 0."""
        mock_qdrant_client.get_collection.return_value = SimpleNamespace(
            vectors_count=None,
            indexed_vectors_count=None,
            points_count=None,
            status=SimpleNamespace(value="green"),
        )

        stats = await store_connected.get_stats()

        assert stats.vectors_count == 0
        assert stats.indexed_vectors_count == 0
        assert stats.points_count == 0

    async def test_stats_handles_none_status(self, store_connected, mock_qdrant_client):
        """get_stats() treats None status as 'unknown'."""
        mock_qdrant_client.get_collection.return_value = SimpleNamespace(
            vectors_count=10,
            indexed_vectors_count=10,
            points_count=10,
            status=None,
        )

        stats = await store_connected.get_stats()
        assert stats.status == "unknown"

    async def test_stats_disconnected(self, config):
        """get_stats() returns 'disconnected' status when client is None."""
        store = QdrantVectorStore(config)
        stats = await store.get_stats()

        assert stats.status == "disconnected"
        assert stats.vectors_count == 0

    async def test_stats_exception_returns_error(self, store_connected, mock_qdrant_client):
        """get_stats() returns 'error' status when get_collection raises."""
        mock_qdrant_client.get_collection.side_effect = Exception("connection lost")

        stats = await store_connected.get_stats()

        assert stats.status == "error"
        assert stats.collection_name == "test_collection"


# ===========================================================================
# 11. Default config
# ===========================================================================


class TestDefaultConfig:
    """Constructor uses default QdrantConfig when None is passed."""

    def test_default_config_when_none(self):
        store = QdrantVectorStore(None)
        assert store._config.collection_name == "code_embeddings"
        assert store._config.vector_size == 384

    def test_explicit_config_used(self):
        cfg = QdrantConfig(collection_name="custom", vector_size=768)
        store = QdrantVectorStore(cfg)
        assert store._config.collection_name == "custom"
        assert store._config.vector_size == 768


# ===========================================================================
# 12. Dimension validation
# ===========================================================================


class TestDimensionValidation:
    """Tests for _validate_collection_dimensions."""

    async def test_matching_dimensions_pass(self, store_connected, mock_qdrant_client):
        """Matching dimensions do not raise."""
        mock_qdrant_client.get_collection.return_value = _make_collection_info(
            name="test_collection", vector_size=store_connected._config.vector_size,
        )
        # Should not raise
        await store_connected._validate_collection_dimensions()

    async def test_mismatched_dimensions_raise(self, store_connected, mock_qdrant_client):
        """Mismatched dimensions raise RuntimeError."""
        mock_qdrant_client.get_collection.return_value = _make_collection_info(
            name="test_collection", vector_size=768,
        )
        with pytest.raises(RuntimeError, match="Vector dimension mismatch"):
            await store_connected._validate_collection_dimensions()

    async def test_dict_vectors_config(self, store_connected, mock_qdrant_client):
        """Handles dict-style vectors_config (named vector spaces)."""
        mock_qdrant_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"": SimpleNamespace(size=384)},
                )
            ),
        )
        # Should not raise — dimensions match
        await store_connected._validate_collection_dimensions()

    async def test_dict_vectors_config_mismatch(self, store_connected, mock_qdrant_client):
        """Dict-style vectors_config with wrong size raises."""
        mock_qdrant_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"": SimpleNamespace(size=1024)},
                )
            ),
        )
        with pytest.raises(RuntimeError, match="Vector dimension mismatch"):
            await store_connected._validate_collection_dimensions()

    async def test_unknown_vectors_format_warns(self, store_connected, mock_qdrant_client):
        """Unknown vectors_config format logs warning but does not raise."""
        mock_qdrant_client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=42),  # Unexpected type
            ),
        )
        # Should not raise — stored_dim is None, so comparison is skipped
        await store_connected._validate_collection_dimensions()

    async def test_get_collection_exception_warns(self, store_connected, mock_qdrant_client):
        """Exception during dimension check logs warning but does not raise."""
        mock_qdrant_client.get_collection.side_effect = Exception("network error")
        # Should not raise — caught by the outer except
        await store_connected._validate_collection_dimensions()
