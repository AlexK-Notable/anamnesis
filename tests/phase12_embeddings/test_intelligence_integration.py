"""Integration tests for EmbeddingEngine with IntelligenceService."""

import pytest

from anamnesis.intelligence.embedding_engine import EmbeddingConfig, EmbeddingEngine
from anamnesis.intelligence.semantic_engine import ConceptType, SemanticConcept
from anamnesis.services.intelligence_service import IntelligenceService


class TestIntelligenceServiceEmbeddingIntegration:
    """Tests for IntelligenceService embedding integration."""

    @pytest.fixture
    def service(self):
        """Create service with embedding engine."""
        return IntelligenceService()

    def test_service_has_embedding_engine(self, service):
        """Service should have an embedding engine."""
        assert service.embedding_engine is not None
        assert isinstance(service.embedding_engine, EmbeddingEngine)

    def test_service_with_custom_embedding_config(self):
        """Service can be created with custom embedding config."""
        config = EmbeddingConfig(model_name="paraphrase-MiniLM-L3-v2")
        service = IntelligenceService(embedding_config=config)

        assert service.embedding_engine.config.model_name == "paraphrase-MiniLM-L3-v2"

    def test_service_with_provided_embedding_engine(self):
        """Service can use a provided embedding engine."""
        engine = EmbeddingEngine()
        engine.add_concept("PreExisting", "class", "/pre/existing.py")

        service = IntelligenceService(embedding_engine=engine)

        assert service.embedding_engine is engine
        assert service.embedding_engine.concept_count == 1

    def test_load_concepts_indexes_in_embedding_engine(self, service):
        """Loading concepts should index them in embedding engine."""
        concepts = [
            SemanticConcept(
                name="UserAuthentication",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/auth.py",
            ),
            SemanticConcept(
                name="login_handler",
                concept_type=ConceptType.FUNCTION,
                confidence=0.8,
                file_path="/src/handlers.py",
            ),
        ]

        service.load_concepts(concepts)

        # Concepts should be in both memory cache and embedding index
        assert len(service._concepts) == 2
        assert service.embedding_engine.concept_count == 2

    def test_search_semantically_similar_basic(self, service):
        """Basic semantic search should work."""
        concepts = [
            SemanticConcept(
                name="UserAuthentication",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/auth.py",
            ),
            SemanticConcept(
                name="DatabaseConnection",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/db.py",
            ),
        ]
        service.load_concepts(concepts)

        results = service.search_semantically_similar("authentication")

        assert len(results) > 0
        # Should find auth-related concepts
        concept_names = [r.concept for r in results]
        assert "UserAuthentication" in concept_names

    def test_search_semantically_similar_with_limit(self, service):
        """Semantic search should respect limit."""
        concepts = [
            SemanticConcept(
                name=f"Concept{i}",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path=f"/src/module{i}.py",
            )
            for i in range(10)
        ]
        service.load_concepts(concepts)

        results = service.search_semantically_similar("concept", limit=3)

        assert len(results) <= 3

    def test_search_semantically_similar_filter_by_type(self, service):
        """Semantic search can filter by concept type."""
        concepts = [
            SemanticConcept(
                name="UserClass",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/user.py",
            ),
            SemanticConcept(
                name="user_function",
                concept_type=ConceptType.FUNCTION,
                confidence=0.9,
                file_path="/src/user.py",
            ),
        ]
        service.load_concepts(concepts)

        results = service.search_semantically_similar("user", concept_type="class")

        for r in results:
            # Check via embedding engine that type is class
            concept_id = service.embedding_engine._generate_concept_id(
                r.concept, "class", r.file_path
            )
            concept = service.embedding_engine.get_concept(concept_id)
            if concept:
                assert concept.concept_type == "class"

    def test_search_semantically_similar_filter_by_path(self, service):
        """Semantic search can filter by file path."""
        concepts = [
            SemanticConcept(
                name="AuthService",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/auth/service.py",
            ),
            SemanticConcept(
                name="DbService",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/db/service.py",
            ),
        ]
        service.load_concepts(concepts)

        results = service.search_semantically_similar(
            "service", file_path_filter="/src/auth/"
        )

        for r in results:
            assert r.file_path.startswith("/src/auth/")

    def test_get_embedding_stats(self, service):
        """Can get embedding engine statistics."""
        concepts = [
            SemanticConcept(
                name="TestConcept",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/test.py",
            ),
        ]
        service.load_concepts(concepts)

        stats = service.get_embedding_stats()

        assert "model_name" in stats
        assert "concept_count" in stats
        assert stats["concept_count"] == 1

    def test_clear_clears_embedding_engine(self, service):
        """Clearing service should clear embedding engine too."""
        concepts = [
            SemanticConcept(
                name="TestConcept",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/test.py",
            ),
        ]
        service.load_concepts(concepts)
        assert service.embedding_engine.concept_count == 1

        service.clear()

        assert service.embedding_engine.concept_count == 0

    def test_concepts_with_relationships_indexed_correctly(self, service):
        """Concepts with relationships should be indexed with metadata."""
        concepts = [
            SemanticConcept(
                name="UserService",
                concept_type=ConceptType.CLASS,
                confidence=0.9,
                file_path="/src/user.py",
                relationships=["AuthService", "DatabaseService"],
            ),
        ]
        service.load_concepts(concepts)

        # Verify the concept was indexed
        assert service.embedding_engine.concept_count == 1

        # Search for it
        results = service.search_semantically_similar("user service")
        assert len(results) > 0
        assert results[0].concept == "UserService"

    def test_concepts_with_line_range_indexed_correctly(self, service):
        """Concepts with line ranges should be indexed with metadata."""
        concepts = [
            SemanticConcept(
                name="process_data",
                concept_type=ConceptType.FUNCTION,
                confidence=0.85,
                file_path="/src/processor.py",
                line_range=(10, 50),
            ),
        ]
        service.load_concepts(concepts)

        # Verify indexed
        assert service.embedding_engine.concept_count == 1

        # Check metadata was stored
        for concept in service.embedding_engine._concepts.values():
            if concept.name == "process_data":
                assert "line_range" in concept.metadata
                assert concept.metadata["line_range"] == (10, 50)
