"""Embedding engine for semantic code search using sentence-transformers.

Provides vector embedding generation and similarity search for code concepts.
Supports offline/CPU mode with graceful degradation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from loguru import logger

from anamnesis.interfaces.engines import CodeMetadata, SemanticSearchResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding engine."""

    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: Optional[str] = None
    device: str = "cpu"  # "cpu", "cuda", "mps"
    batch_size: int = 32
    normalize_embeddings: bool = True
    max_sequence_length: int = 512


@dataclass
class IndexedConcept:
    """A concept with its embedding stored in the index."""

    id: str
    name: str
    concept_type: str
    file_path: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class EmbeddingEngine:
    """Engine for generating and searching code embeddings.

    Uses sentence-transformers for generating semantic embeddings of code
    concepts and provides similarity search functionality.

    Features:
    - Lazy model loading (only loads when first needed)
    - In-memory vector index with numpy
    - Graceful fallback to text search if embeddings unavailable
    - Batch processing for efficiency

    Usage:
        engine = EmbeddingEngine()

        # Index concepts
        engine.add_concept("auth", "class", "/src/auth.py", {"lines": 50})
        engine.add_concept("login", "function", "/src/auth.py", {"lines": 10})

        # Search
        results = engine.search("authentication logic", limit=5)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or EmbeddingConfig()
        self._model: Optional["SentenceTransformer"] = None
        self._model_loaded = False
        self._model_error: Optional[str] = None

        # In-memory index
        self._concepts: dict[str, IndexedConcept] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_to_index: dict[str, int] = {}
        self._index_dirty = False

    @property
    def config(self) -> EmbeddingConfig:
        """Get engine configuration."""
        return self._config

    @property
    def model_available(self) -> bool:
        """Check if the embedding model is loaded and available."""
        return self._model_loaded and self._model is not None

    @property
    def concept_count(self) -> int:
        """Get number of indexed concepts."""
        return len(self._concepts)

    def _load_model(self) -> bool:
        """Load the embedding model lazily.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model_loaded:
            return self._model is not None

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._config.model_name}")
            self._model = SentenceTransformer(
                self._config.model_name,
                cache_folder=self._config.cache_dir,
                device=self._config.device,
            )
            self._model.max_seq_length = self._config.max_sequence_length
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
            return True

        except ImportError as e:
            self._model_error = f"sentence-transformers not installed: {e}"
            logger.warning(self._model_error)
            self._model_loaded = True
            return False

        except Exception as e:
            self._model_error = f"Failed to load embedding model: {e}"
            logger.warning(self._model_error)
            self._model_loaded = True
            return False

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector or None if model unavailable.
        """
        if not self._load_model() or self._model is None:
            return None

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize_embeddings,
        )
        return embedding

    def _generate_embeddings_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Matrix of embeddings (N x embedding_dim) or None if unavailable.
        """
        if not self._load_model() or self._model is None:
            return None

        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize_embeddings,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def _generate_concept_id(
        self, name: str, concept_type: str, file_path: str
    ) -> str:
        """Generate unique ID for a concept.

        Args:
            name: Concept name.
            concept_type: Type of concept (class, function, etc.).
            file_path: File path where concept is defined.

        Returns:
            Unique hash-based ID.
        """
        content = f"{name}:{concept_type}:{file_path}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_index(self) -> None:
        """Rebuild the embedding matrix from indexed concepts."""
        if not self._index_dirty or not self._concepts:
            return

        concept_ids = list(self._concepts.keys())
        embeddings = []

        for cid in concept_ids:
            concept = self._concepts[cid]
            if concept.embedding is not None:
                embeddings.append(concept.embedding)

        if embeddings:
            self._embeddings_matrix = np.vstack(embeddings)
            self._id_to_index = {cid: i for i, cid in enumerate(concept_ids)}
        else:
            self._embeddings_matrix = None
            self._id_to_index = {}

        self._index_dirty = False

    def add_concept(
        self,
        name: str,
        concept_type: str,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a concept to the index.

        Args:
            name: Concept name (class name, function name, etc.).
            concept_type: Type of concept (class, function, module, etc.).
            file_path: File path where the concept is defined.
            metadata: Additional metadata about the concept.

        Returns:
            The concept ID.
        """
        concept_id = self._generate_concept_id(name, concept_type, file_path)

        # Generate embedding text combining name and type
        embed_text = f"{concept_type}: {name}"
        embedding = self._generate_embedding(embed_text)

        if embedding is None:
            # Fallback: store without embedding for text search
            embedding = np.array([])

        concept = IndexedConcept(
            id=concept_id,
            name=name,
            concept_type=concept_type,
            file_path=file_path,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._concepts[concept_id] = concept
        self._index_dirty = True

        return concept_id

    def add_concepts_batch(
        self,
        concepts: list[tuple[str, str, str, Optional[dict]]],
    ) -> list[str]:
        """Add multiple concepts efficiently.

        Args:
            concepts: List of (name, concept_type, file_path, metadata) tuples.

        Returns:
            List of concept IDs.
        """
        if not concepts:
            return []

        # Generate embedding texts
        embed_texts = [f"{c[1]}: {c[0]}" for c in concepts]

        # Generate embeddings in batch
        embeddings = self._generate_embeddings_batch(embed_texts)

        concept_ids = []
        for i, (name, concept_type, file_path, metadata) in enumerate(concepts):
            concept_id = self._generate_concept_id(name, concept_type, file_path)

            if embeddings is not None and len(embeddings) > i:
                embedding = embeddings[i]
            else:
                embedding = np.array([])

            concept = IndexedConcept(
                id=concept_id,
                name=name,
                concept_type=concept_type,
                file_path=file_path,
                embedding=embedding,
                metadata=metadata or {},
            )

            self._concepts[concept_id] = concept
            concept_ids.append(concept_id)

        self._index_dirty = True
        return concept_ids

    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept from the index.

        Args:
            concept_id: ID of concept to remove.

        Returns:
            True if removed, False if not found.
        """
        if concept_id in self._concepts:
            del self._concepts[concept_id]
            self._index_dirty = True
            return True
        return False

    def get_concept(self, concept_id: str) -> Optional[IndexedConcept]:
        """Get a concept by ID.

        Args:
            concept_id: Concept ID.

        Returns:
            IndexedConcept or None if not found.
        """
        return self._concepts.get(concept_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        concept_type: Optional[str] = None,
        file_path_filter: Optional[str] = None,
    ) -> list[SemanticSearchResult]:
        """Search for semantically similar concepts.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            concept_type: Optional filter by concept type.
            file_path_filter: Optional filter by file path prefix.

        Returns:
            List of SemanticSearchResult sorted by similarity.
        """
        if not self._concepts:
            return []

        # Filter concepts if needed
        candidates = list(self._concepts.values())
        if concept_type:
            candidates = [c for c in candidates if c.concept_type == concept_type]
        if file_path_filter:
            candidates = [
                c for c in candidates if c.file_path.startswith(file_path_filter)
            ]

        if not candidates:
            return []

        # Try vector search first
        query_embedding = self._generate_embedding(query)

        if query_embedding is not None and len(query_embedding) > 0:
            return self._vector_search(query_embedding, candidates, limit)
        else:
            # Fallback to text matching
            return self._text_search(query, candidates, limit)

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        candidates: list[IndexedConcept],
        limit: int,
    ) -> list[SemanticSearchResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector.
            candidates: List of candidate concepts.
            limit: Maximum results.

        Returns:
            Sorted search results.
        """
        # Filter to concepts with valid embeddings
        valid_candidates = [
            c for c in candidates if c.embedding is not None and len(c.embedding) > 0
        ]

        if not valid_candidates:
            return []

        # Build embedding matrix for candidates
        embeddings = np.vstack([c.embedding for c in valid_candidates])

        # Compute cosine similarities
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            concept = valid_candidates[idx]
            similarity = float(similarities[idx])

            results.append(
                SemanticSearchResult(
                    concept=concept.name,
                    similarity=similarity,
                    file_path=concept.file_path,
                )
            )

        return results

    def _text_search(
        self,
        query: str,
        candidates: list[IndexedConcept],
        limit: int,
    ) -> list[SemanticSearchResult]:
        """Fallback text-based search.

        Args:
            query: Search query.
            candidates: List of candidate concepts.
            limit: Maximum results.

        Returns:
            Sorted search results based on text matching.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for concept in candidates:
            name_lower = concept.name.lower()
            type_lower = concept.concept_type.lower()

            # Score based on various matches
            score = 0.0

            # Exact match
            if query_lower == name_lower:
                score += 1.0
            # Substring match
            elif query_lower in name_lower:
                score += 0.7
            elif name_lower in query_lower:
                score += 0.5
            # Word overlap
            else:
                name_words = set(name_lower.split("_"))
                overlap = len(query_words & name_words)
                if overlap > 0:
                    score += 0.3 * overlap

            # Type match bonus
            if type_lower in query_lower:
                score += 0.2

            if score > 0:
                scored.append((concept, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for concept, score in scored[:limit]:
            results.append(
                SemanticSearchResult(
                    concept=concept.name,
                    similarity=min(score, 1.0),  # Cap at 1.0
                    file_path=concept.file_path,
                )
            )

        return results

    def clear(self) -> None:
        """Clear all indexed concepts."""
        self._concepts.clear()
        self._embeddings_matrix = None
        self._id_to_index.clear()
        self._index_dirty = False

    def get_stats(self) -> dict:
        """Get statistics about the embedding engine.

        Returns:
            Dictionary with stats.
        """
        return {
            "model_name": self._config.model_name,
            "model_available": self.model_available,
            "model_error": self._model_error,
            "concept_count": self.concept_count,
            "device": self._config.device,
            "index_dirty": self._index_dirty,
        }
