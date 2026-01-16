"""Tests for EmbeddingEngine.

Tests the embedding engine for semantic code search.
Uses real sentence-transformers model - tests may be slow on first run.
"""

import pytest
import numpy as np

from anamnesis.intelligence.embedding_engine import (
    EmbeddingConfig,
    EmbeddingEngine,
    IndexedConcept,
)
from anamnesis.interfaces.engines import SemanticSearchResult


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert config.max_sequence_length == 512

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="paraphrase-MiniLM-L3-v2",
            device="cuda",
            batch_size=64,
        )

        assert config.model_name == "paraphrase-MiniLM-L3-v2"
        assert config.device == "cuda"
        assert config.batch_size == 64


class TestEmbeddingEngineInit:
    """Tests for EmbeddingEngine initialization."""

    def test_create_engine_default(self):
        """Create engine with default config."""
        engine = EmbeddingEngine()

        assert engine.config.model_name == "all-MiniLM-L6-v2"
        assert engine.concept_count == 0
        assert engine.model_available is False  # Not loaded yet

    def test_create_engine_custom_config(self):
        """Create engine with custom config."""
        config = EmbeddingConfig(model_name="paraphrase-MiniLM-L3-v2")
        engine = EmbeddingEngine(config)

        assert engine.config.model_name == "paraphrase-MiniLM-L3-v2"


class TestIndexedConcept:
    """Tests for IndexedConcept dataclass."""

    def test_create_indexed_concept(self):
        """Test IndexedConcept creation."""
        embedding = np.array([0.1, 0.2, 0.3])
        concept = IndexedConcept(
            id="abc123",
            name="UserService",
            concept_type="class",
            file_path="/src/services/user.py",
            embedding=embedding,
            metadata={"lines": 100},
        )

        assert concept.id == "abc123"
        assert concept.name == "UserService"
        assert concept.concept_type == "class"
        assert np.array_equal(concept.embedding, embedding)


class TestAddConcepts:
    """Tests for adding concepts to the engine."""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine for each test."""
        return EmbeddingEngine()

    def test_add_single_concept(self, engine):
        """Add a single concept."""
        concept_id = engine.add_concept(
            name="UserService",
            concept_type="class",
            file_path="/src/user.py",
            metadata={"lines": 50},
        )

        assert concept_id is not None
        assert len(concept_id) == 16  # SHA256 truncated
        assert engine.concept_count == 1

    def test_add_multiple_concepts(self, engine):
        """Add multiple concepts."""
        engine.add_concept("auth", "function", "/src/auth.py")
        engine.add_concept("login", "function", "/src/auth.py")
        engine.add_concept("User", "class", "/src/models.py")

        assert engine.concept_count == 3

    def test_add_concepts_batch(self, engine):
        """Add concepts in batch for efficiency."""
        concepts = [
            ("auth", "function", "/src/auth.py", {"lines": 20}),
            ("login", "function", "/src/auth.py", {"lines": 15}),
            ("User", "class", "/src/models.py", None),
            ("Database", "class", "/src/db.py", {"connections": 5}),
        ]

        ids = engine.add_concepts_batch(concepts)

        assert len(ids) == 4
        assert engine.concept_count == 4

    def test_concept_id_deterministic(self, engine):
        """Same concept generates same ID."""
        id1 = engine.add_concept("test", "function", "/src/test.py")
        engine.remove_concept(id1)
        id2 = engine.add_concept("test", "function", "/src/test.py")

        assert id1 == id2

    def test_different_concepts_different_ids(self, engine):
        """Different concepts get different IDs."""
        id1 = engine.add_concept("test1", "function", "/src/test.py")
        id2 = engine.add_concept("test2", "function", "/src/test.py")

        assert id1 != id2


class TestGetConcept:
    """Tests for retrieving concepts."""

    @pytest.fixture
    def engine(self):
        """Engine with some concepts."""
        engine = EmbeddingEngine()
        engine.add_concept("UserService", "class", "/src/user.py")
        engine.add_concept("AuthService", "class", "/src/auth.py")
        return engine

    def test_get_existing_concept(self, engine):
        """Get a concept that exists."""
        concept_id = engine.add_concept("NewConcept", "function", "/src/new.py")
        concept = engine.get_concept(concept_id)

        assert concept is not None
        assert concept.name == "NewConcept"
        assert concept.concept_type == "function"

    def test_get_nonexistent_concept(self, engine):
        """Get a concept that doesn't exist."""
        concept = engine.get_concept("nonexistent")

        assert concept is None


class TestRemoveConcept:
    """Tests for removing concepts."""

    @pytest.fixture
    def engine(self):
        """Engine with some concepts."""
        engine = EmbeddingEngine()
        engine.add_concept("ToRemove", "function", "/src/remove.py")
        return engine

    def test_remove_existing_concept(self, engine):
        """Remove a concept that exists."""
        concept_id = engine.add_concept("NewConcept", "function", "/src/new.py")
        initial_count = engine.concept_count

        result = engine.remove_concept(concept_id)

        assert result is True
        assert engine.concept_count == initial_count - 1
        assert engine.get_concept(concept_id) is None

    def test_remove_nonexistent_concept(self, engine):
        """Remove a concept that doesn't exist."""
        result = engine.remove_concept("nonexistent")

        assert result is False


class TestSearch:
    """Tests for semantic search functionality."""

    @pytest.fixture
    def engine(self):
        """Engine with indexed concepts for search testing."""
        engine = EmbeddingEngine()
        concepts = [
            ("UserAuthentication", "class", "/src/auth/user_auth.py", None),
            ("LoginHandler", "function", "/src/auth/login.py", None),
            ("PasswordValidator", "function", "/src/auth/password.py", None),
            ("DatabaseConnection", "class", "/src/db/connection.py", None),
            ("QueryBuilder", "class", "/src/db/query.py", None),
            ("UserProfile", "class", "/src/models/user.py", None),
            ("EmailService", "class", "/src/services/email.py", None),
        ]
        engine.add_concepts_batch(concepts)
        return engine

    def test_search_returns_results(self, engine):
        """Search returns list of SemanticSearchResult."""
        results = engine.search("authentication", limit=5)

        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SemanticSearchResult)
            assert hasattr(r, "concept")
            assert hasattr(r, "similarity")
            assert hasattr(r, "file_path")

    def test_search_respects_limit(self, engine):
        """Search respects the limit parameter."""
        results = engine.search("user", limit=2)

        assert len(results) <= 2

    def test_search_filter_by_type(self, engine):
        """Search filters by concept type."""
        results = engine.search("user", concept_type="class")

        for r in results:
            concept = engine.get_concept(
                engine._generate_concept_id(r.concept, "class", r.file_path)
            )
            if concept:
                assert concept.concept_type == "class"

    def test_search_filter_by_path(self, engine):
        """Search filters by file path prefix."""
        results = engine.search("connection", file_path_filter="/src/db/")

        for r in results:
            assert r.file_path.startswith("/src/db/")

    def test_search_empty_index(self):
        """Search on empty index returns empty list."""
        engine = EmbeddingEngine()
        results = engine.search("anything")

        assert results == []

    def test_search_similarity_sorted(self, engine):
        """Results are sorted by similarity descending."""
        results = engine.search("authentication logic", limit=10)

        if len(results) > 1:
            similarities = [r.similarity for r in results]
            assert similarities == sorted(similarities, reverse=True)


class TestClear:
    """Tests for clearing the index."""

    def test_clear_removes_all(self):
        """Clear removes all indexed concepts."""
        engine = EmbeddingEngine()
        engine.add_concept("test1", "function", "/src/test1.py")
        engine.add_concept("test2", "function", "/src/test2.py")
        assert engine.concept_count == 2

        engine.clear()

        assert engine.concept_count == 0


class TestStats:
    """Tests for engine statistics."""

    def test_get_stats(self):
        """Get engine statistics."""
        engine = EmbeddingEngine()
        engine.add_concept("test", "function", "/src/test.py")

        stats = engine.get_stats()

        assert "model_name" in stats
        assert "model_available" in stats
        assert "concept_count" in stats
        assert stats["concept_count"] == 1
        assert stats["device"] == "cpu"


class TestFallbackTextSearch:
    """Tests for text-based fallback search."""

    @pytest.fixture
    def engine_no_embeddings(self):
        """Engine configured to skip embedding model loading."""
        # Create engine but don't add concepts with embeddings
        engine = EmbeddingEngine()
        # Manually add concepts without generating embeddings
        concept = IndexedConcept(
            id="test1",
            name="UserService",
            concept_type="class",
            file_path="/src/user.py",
            embedding=np.array([]),  # Empty embedding forces text search
        )
        engine._concepts["test1"] = concept
        concept2 = IndexedConcept(
            id="test2",
            name="AuthService",
            concept_type="class",
            file_path="/src/auth.py",
            embedding=np.array([]),
        )
        engine._concepts["test2"] = concept2
        return engine

    def test_text_search_exact_match(self, engine_no_embeddings):
        """Text search finds exact name matches."""
        # Force fallback by making model unavailable
        engine = engine_no_embeddings
        engine._model_loaded = True  # Pretend model was tried
        engine._model = None  # But no model available

        results = engine.search("UserService")

        assert len(results) > 0
        assert results[0].concept == "UserService"

    def test_text_search_substring_match(self, engine_no_embeddings):
        """Text search finds substring matches."""
        engine = engine_no_embeddings
        engine._model_loaded = True
        engine._model = None

        results = engine.search("User")

        # Should find UserService
        concepts_found = [r.concept for r in results]
        assert "UserService" in concepts_found


class TestIntegration:
    """Integration tests with real model (may be slow)."""

    @pytest.fixture
    def engine_with_model(self):
        """Engine with model loaded."""
        engine = EmbeddingEngine()
        # Add concepts to trigger model loading
        engine.add_concept("Authentication", "class", "/src/auth.py")
        engine.add_concept("Login", "function", "/src/login.py")
        engine.add_concept("Database", "class", "/src/db.py")
        return engine

    @pytest.mark.slow
    def test_semantic_search_finds_related(self, engine_with_model):
        """Semantic search finds conceptually related items."""
        engine = engine_with_model

        # Search for auth-related concepts
        results = engine.search("user authentication login")

        # Should find auth-related concepts higher
        if results:
            top_concepts = [r.concept for r in results[:2]]
            assert any(
                "auth" in c.lower() or "login" in c.lower()
                for c in top_concepts
            )

    @pytest.mark.slow
    def test_model_loads_on_demand(self):
        """Model loads only when needed (lazy loading)."""
        engine = EmbeddingEngine()

        # Model not loaded yet
        assert engine._model is None
        assert engine.model_available is False

        # Adding concept triggers model load
        engine.add_concept("test", "function", "/src/test.py")

        # Model should now be loaded
        assert engine._model_loaded is True
