"""Behavioral tests for IntelligenceService persistence.

Tests that prove data persists to and loads from SyncSQLiteBackend.
Uses real database, no mocks.
"""

import pytest

from anamnesis.intelligence.pattern_engine import DetectedPattern, PatternType
from anamnesis.intelligence.semantic_engine import ConceptType, SemanticConcept
from anamnesis.services.intelligence_service import IntelligenceService
from anamnesis.storage.sync_backend import SyncSQLiteBackend


@pytest.fixture
def backend():
    """Create in-memory backend for testing."""
    backend = SyncSQLiteBackend(":memory:")
    backend.connect()
    yield backend
    backend.close()


@pytest.fixture
def service_with_backend(backend):
    """Create service with backend."""
    return IntelligenceService(backend=backend)


@pytest.fixture
def sample_concepts():
    """Sample concepts for testing."""
    return [
        SemanticConcept(
            name="UserService",
            concept_type=ConceptType.CLASS,
            confidence=0.95,
            file_path="/src/services/user.py",
            description="Handles user operations",
        ),
        SemanticConcept(
            name="authenticate",
            concept_type=ConceptType.FUNCTION,
            confidence=0.9,
            file_path="/src/auth.py",
        ),
    ]


@pytest.fixture
def sample_patterns():
    """Sample patterns for testing."""
    return [
        DetectedPattern(
            pattern_type=PatternType.SINGLETON,
            description="Singleton instance pattern",
            confidence=0.85,
            file_path="/src/config.py",
            frequency=3,
        ),
        DetectedPattern(
            pattern_type=PatternType.FACTORY,
            description="Factory method pattern",
            confidence=0.88,
            file_path="/src/factory.py",
            frequency=5,
        ),
    ]


class TestPersistenceBehavior:
    """Behavioral tests for persistence."""

    def test_load_concepts_persists_to_backend(self, service_with_backend, backend, sample_concepts):
        """Loading concepts persists them to the database."""
        # When: Load concepts into service
        service_with_backend.load_concepts(sample_concepts)

        # Then: Concepts are in the backend
        stored = backend.search_concepts("", limit=100)
        assert len(stored) == 2

        # And: Data is preserved
        names = {c.name for c in stored}
        assert "UserService" in names
        assert "authenticate" in names

    def test_load_patterns_persists_to_backend(self, service_with_backend, backend, sample_patterns):
        """Loading patterns persists them to the database."""
        # When: Load patterns into service
        service_with_backend.load_patterns(sample_patterns)

        # Then: Patterns are in the backend
        stored = backend.get_all_patterns()
        assert len(stored) == 2

        # And: Data is preserved
        frequencies = {p.frequency for p in stored}
        assert 3 in frequencies
        assert 5 in frequencies

    def test_contribute_insight_persists_to_backend(self, service_with_backend, backend):
        """Contributing insights persists them to the database."""
        # When: Contribute an insight
        success, insight_id, message = service_with_backend.contribute_insight(
            insight_type="bug_pattern",
            content={"practice": "Null check missing", "reasoning": "Found 3 cases"},
            confidence=0.85,
            source_agent="test-agent",
        )

        # Then: Success
        assert success is True
        assert insight_id.startswith("insight_")

        # And: Insight is in backend
        stored = backend.get_insight(insight_id)
        assert stored is not None
        assert stored.confidence == 0.85

    def test_load_from_backend_restores_state(self, backend, sample_concepts, sample_patterns):
        """Service can restore state from backend after restart."""
        # Given: First service loads data
        service1 = IntelligenceService(backend=backend)
        service1.load_concepts(sample_concepts)
        service1.load_patterns(sample_patterns)

        # When: New service instance loads from backend
        service2 = IntelligenceService(backend=backend)
        service2.load_from_backend()

        # Then: State is restored
        assert len(service2._concepts) == 2
        assert len(service2._patterns) == 2

        # And: Names match
        concept_names = {c.name for c in service2._concepts}
        assert "UserService" in concept_names
        assert "authenticate" in concept_names

    def test_learning_status_queries_backend(self, service_with_backend, backend, sample_concepts):
        """Learning status reflects backend counts."""
        # Given: Empty service
        status = service_with_backend._get_learning_status("/project")
        assert status["concepts_stored"] == 0
        assert status["has_intelligence"] is False

        # When: Load concepts
        service_with_backend.load_concepts(sample_concepts)

        # Then: Status reflects persisted data
        status = service_with_backend._get_learning_status("/project")
        assert status["concepts_stored"] == 2
        assert status["has_intelligence"] is True
        assert status["persisted"] is True

    def test_service_without_backend_works_in_memory(self, sample_concepts):
        """Service works without backend (memory-only mode)."""
        # Given: Service without backend
        service = IntelligenceService()

        # When: Load concepts
        service.load_concepts(sample_concepts)

        # Then: Concepts are in memory
        assert len(service._concepts) == 2

        # And: Status shows not persisted
        status = service._get_learning_status("/project")
        assert status["persisted"] is False

    def test_backend_property_returns_backend(self, service_with_backend, backend):
        """Backend property returns the configured backend."""
        assert service_with_backend.backend is backend

    def test_backend_property_returns_none_without_backend(self):
        """Backend property returns None when no backend configured."""
        service = IntelligenceService()
        assert service.backend is None


class TestTypeConversion:
    """Tests for type conversion between engine and storage types."""

    def test_concept_round_trip_preserves_data(self, backend):
        """Concept data survives engine→storage→engine round trip."""
        from anamnesis.services.type_converters import (
            engine_concept_to_storage,
            storage_concept_to_engine,
        )

        # Given: Engine concept with full data
        original = SemanticConcept(
            name="TestClass",
            concept_type=ConceptType.CLASS,
            confidence=0.95,
            file_path="/test.py",
            line_range=(10, 50),
            description="A test class",
            relationships=["BaseClass", "Interface"],
        )

        # When: Convert to storage and back
        storage = engine_concept_to_storage(original)
        restored = storage_concept_to_engine(storage)

        # Then: Key data preserved
        assert restored.name == original.name
        assert restored.confidence == original.confidence
        assert restored.file_path == original.file_path
        assert restored.description == original.description

    def test_pattern_round_trip_preserves_data(self, backend):
        """Pattern data survives engine→storage→engine round trip."""
        from anamnesis.services.type_converters import (
            detected_pattern_to_storage,
            storage_pattern_to_detected,
        )

        # Given: Engine pattern with full data
        original = DetectedPattern(
            pattern_type=PatternType.FACTORY,
            description="Factory pattern",
            confidence=0.88,
            file_path="/factory.py",
            frequency=5,
            code_snippet="def create():",
        )

        # When: Convert to storage and back
        storage = detected_pattern_to_storage(original)
        restored = storage_pattern_to_detected(storage)

        # Then: Key data preserved
        assert restored.description == original.description
        assert restored.confidence == original.confidence
        assert restored.frequency == original.frequency
        assert restored.file_path == original.file_path
