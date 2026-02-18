"""Qdrant vector store for concurrent-safe semantic search.

This module provides a Qdrant-backed vector store that supports:
- Embedded mode (local storage, multi-process safe)
- Server mode (distributed, multi-machine)
- ACID-compliant operations for multi-agent access
- Automatic collection management and indexing
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass
class QdrantConfig:
    """Qdrant connection configuration.

    Supports two modes:
    1. Embedded (default): Local storage at `path`, multi-process safe
    2. Server: Connect to remote Qdrant instance via `url`
    """

    # Local embedded mode (default) - best for single-machine, multi-agent
    path: Optional[str] = ".anamnesis/qdrant"

    # Remote server mode (for distributed deployments)
    url: Optional[str] = None
    api_key: Optional[str] = None

    # Collection settings
    collection_name: str = "code_embeddings"
    vector_size: int = 384  # all-MiniLM-L6-v2 default

    # Concurrency settings
    prefer_grpc: bool = True  # Better for concurrent access
    timeout: float = 30.0

    # Index settings
    on_disk_payload: bool = True  # Store payloads on disk for large codebases


@dataclass
class QdrantStats:
    """Statistics about the Qdrant collection."""

    vectors_count: int = 0
    indexed_vectors_count: int = 0
    points_count: int = 0
    status: str = "unknown"
    collection_name: str = ""


class QdrantVectorStore:
    """Vector store backed by Qdrant for concurrent-safe semantic search.

    Features:
    - Embedded mode (local storage) or server mode (distributed)
    - ACID-compliant operations
    - Multi-tenant support via collection namespacing
    - Efficient batch operations
    - Automatic schema and index management

    Usage:
        config = QdrantConfig(path=".anamnesis/qdrant")
        store = QdrantVectorStore(config)
        await store.connect()

        # Index concepts
        await store.upsert(
            file_path="src/auth.py",
            name="AuthService",
            concept_type="class",
            embedding=[0.1, 0.2, ...],
            metadata={"language": "python"}
        )

        # Search
        results = await store.search(query_vector=[0.1, 0.2, ...], limit=10)
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize Qdrant vector store.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or QdrantConfig()
        self._client: Optional["QdrantClient"] = None
        self._initialized = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._initialized and self._client is not None

    async def connect(self) -> None:
        """Initialize Qdrant connection.

        Creates the client and ensures the collection exists.
        """
        if self._initialized:
            return

        try:
            from qdrant_client import QdrantClient

            if self._config.url:
                # Remote server mode
                logger.info(f"Connecting to Qdrant server: {self._config.url}")
                self._client = QdrantClient(
                    url=self._config.url,
                    api_key=self._config.api_key,
                    prefer_grpc=self._config.prefer_grpc,
                    timeout=self._config.timeout,
                )
            else:
                # Embedded mode (local storage)
                path = Path(self._config.path)
                path.mkdir(parents=True, exist_ok=True)

                logger.info(f"Initializing embedded Qdrant at: {path}")
                self._client = QdrantClient(
                    path=str(path),
                    prefer_grpc=self._config.prefer_grpc,
                )

            await self._ensure_collection()
            self._initialized = True
            logger.info(f"Qdrant connected: {self._config.collection_name}")

        except ImportError:
            logger.error("qdrant-client not installed. Run: uv add qdrant-client")
            raise

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist, validate dimensions if it does."""
        if self._client is None:
            raise RuntimeError("Client not initialized")

        from qdrant_client import models

        resp = await asyncio.to_thread(self._client.get_collections)
        collections = resp.collections
        exists = any(c.name == self._config.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating Qdrant collection: {self._config.collection_name}")

            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=self._config.collection_name,
                vectors_config=models.VectorParams(
                    size=self._config.vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True,  # Store vectors on disk for large codebases
                ),
                on_disk_payload=self._config.on_disk_payload,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                ),
            )

            # Create payload indexes for efficient filtering
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=self._config.collection_name,
                field_name="file_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=self._config.collection_name,
                field_name="language",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=self._config.collection_name,
                field_name="concept_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Created collection with indexes: {self._config.collection_name}")
        else:
            await self._validate_collection_dimensions()

    async def _validate_collection_dimensions(self) -> None:
        """Validate that existing collection dimensions match configured vector_size.

        Raises:
            RuntimeError: If dimensions mismatch, indicating the embedding model
                changed without re-indexing the vector store.
        """
        try:
            info = await asyncio.to_thread(
                self._client.get_collection, self._config.collection_name
            )
            vectors_config = info.config.params.vectors
            if hasattr(vectors_config, "size"):
                stored_dim = vectors_config.size
            elif isinstance(vectors_config, dict):
                default = vectors_config.get("", vectors_config.get(None))
                stored_dim = default.size if default else None
            else:
                stored_dim = None

            if stored_dim is not None and stored_dim != self._config.vector_size:
                raise RuntimeError(
                    f"Vector dimension mismatch: collection has {stored_dim}d vectors, "
                    f"but embedding model produces {self._config.vector_size}d vectors. "
                    f"Delete and re-index the collection to resolve."
                )
            elif stored_dim is not None:
                logger.debug("Qdrant dimension validation passed: %dd", stored_dim)
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(
                "Could not validate Qdrant collection dimensions: %s", e
            )

    def _generate_id(self, file_path: str, name: str, concept_type: str) -> str:
        """Generate deterministic UUID for deduplication.

        Uses UUID-5 (SHA-1, deterministic) so the same inputs always produce
        the same ID.  Qdrant requires valid UUID strings as point IDs.

        Args:
            file_path: Source file path.
            name: Concept name.
            concept_type: Type of concept.

        Returns:
            Deterministic UUID string.
        """
        content = f"{file_path}:{name}:{concept_type}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

    async def upsert(
        self,
        file_path: str,
        name: str,
        concept_type: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Upsert a vector with metadata.

        Args:
            file_path: Source file path.
            name: Concept name (class name, function name, etc).
            concept_type: Type of concept (class, function, etc).
            embedding: Vector embedding.
            metadata: Additional metadata.

        Returns:
            The point ID.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        from qdrant_client import models

        point_id = self._generate_id(file_path, name, concept_type)

        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self._config.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        "name": name,
                        "concept_type": concept_type,
                        **metadata,
                    },
                )
            ],
        )

        return point_id

    async def upsert_batch(
        self,
        points: list[tuple[str, str, str, list[float], dict]],
    ) -> list[str]:
        """Batch upsert for efficiency.

        Args:
            points: List of (file_path, name, concept_type, embedding, metadata).

        Returns:
            List of point IDs.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if not points:
            return []

        from qdrant_client import models

        qdrant_points = []
        point_ids = []

        for file_path, name, concept_type, embedding, metadata in points:
            point_id = self._generate_id(file_path, name, concept_type)
            point_ids.append(point_id)

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        "name": name,
                        "concept_type": concept_type,
                        **metadata,
                    },
                )
            )

        # Batch upsert - Qdrant handles batching internally for efficiency
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self._config.collection_name,
            points=qdrant_points,
        )

        logger.debug(f"Batch upserted {len(points)} vectors")
        return point_ids

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: Optional[float] = 0.5,
        filter_conditions: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score (0.0-1.0).
            filter_conditions: Optional filters like {"language": "python"}.

        Returns:
            List of {id, score, payload} dicts, sorted by similarity.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        from qdrant_client import models

        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if value is not None:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            if must_conditions:
                qdrant_filter = models.Filter(must=must_conditions)

        response = await asyncio.to_thread(
            self._client.query_points,
            collection_name=self._config.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in response.points
        ]

    async def delete_by_file(self, file_path: str) -> bool:
        """Delete all vectors for a file (for re-indexing).

        Args:
            file_path: Path of the file to delete vectors for.

        Returns:
            True if operation completed.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        from qdrant_client import models

        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )

        logger.debug(f"Deleted vectors for file: {file_path}")
        return True

    async def delete_collection(self) -> bool:
        """Delete the entire collection.

        Use with caution - this removes all indexed data.

        Returns:
            True if operation completed.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        await asyncio.to_thread(self._client.delete_collection, self._config.collection_name)
        logger.warning(f"Deleted collection: {self._config.collection_name}")
        return True

    async def get_stats(self) -> QdrantStats:
        """Get collection statistics.

        Returns:
            QdrantStats with collection information.
        """
        if self._client is None:
            return QdrantStats(status="disconnected")

        try:
            info = await asyncio.to_thread(
                self._client.get_collection, self._config.collection_name
            )
            return QdrantStats(
                vectors_count=getattr(info, "vectors_count", None) or 0,
                indexed_vectors_count=getattr(info, "indexed_vectors_count", None) or 0,
                points_count=info.points_count or 0,
                status=info.status.value if info.status else "unknown",
                collection_name=self._config.collection_name,
            )
        except Exception as e:
            logger.warning(f"Failed to get Qdrant stats: {e}")
            return QdrantStats(status="error", collection_name=self._config.collection_name)

    async def close(self) -> None:
        """Close connection to Qdrant."""
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._initialized = False
            logger.info("Qdrant connection closed")
