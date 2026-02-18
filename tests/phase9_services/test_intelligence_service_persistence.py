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
        """Concept data survives unified→storage conversion."""
        from anamnesis.extraction.converters import unified_symbol_to_storage_concept
        from anamnesis.extraction.types import SymbolKind, UnifiedSymbol

        # Given: UnifiedSymbol with full data
        original = UnifiedSymbol(
            name="TestClass",
            kind=SymbolKind.CLASS,
            file_path="test.py",
            start_line=10,
            end_line=50,
            confidence=0.95,
            docstring="A test class",
            references=["BaseClass"],
            dependencies=["Interface"],
            backend="tree_sitter",
        )

        # When: Convert to storage
        storage = unified_symbol_to_storage_concept(original)

        # Then: Key data preserved
        assert storage.name == original.name
        assert storage.confidence == original.confidence
        assert storage.file_path == original.file_path
        assert storage.description == original.docstring
        assert storage.line_start == original.start_line
        assert storage.line_end == original.end_line
        assert len(storage.relationships) == 2  # 1 reference + 1 dependency

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

    def test_insight_round_trip_preserves_data(self, backend):
        """Insight data survives service→storage→service round trip."""
        from anamnesis.services.type_converters import (
            service_insight_to_storage,
            storage_insight_to_service,
        )

        # Given: Service insight with full data
        original_content = {
            "title": "Null check missing",
            "description": "Found unchecked null in 3 functions",
            "affected_files": ["/src/handler.py"],
            "suggested_action": "Add null guards",
        }
        storage = service_insight_to_storage(
            insight_id="insight_test123",
            insight_type="bug_pattern",
            content=original_content,
            confidence=0.85,
            source_agent="test-agent",
            impact_prediction={"severity": "medium"},
        )

        # When: Convert back to service type
        restored = storage_insight_to_service(storage)

        # Then: Key data preserved
        assert restored.insight_id == "insight_test123"
        assert restored.insight_type == "bug_pattern"
        assert restored.confidence == 0.85
        assert restored.source_agent == "test-agent"
        assert restored.impact_prediction == {"severity": "medium"}
        assert restored.content == original_content

    def test_insight_persists_and_restores_via_load_from_backend(self, backend):
        """Insights survive contribute → new service instance cycle."""
        # Given: Service contributes an insight
        service1 = IntelligenceService(backend=backend)
        success, insight_id, _ = service1.contribute_insight(
            insight_type="optimization",
            content={"title": "Cache opportunity", "description": "Hot loop detected"},
            confidence=0.92,
            source_agent="perf-agent",
        )
        assert success

        # When: New service auto-loads from backend on init
        service2 = IntelligenceService(backend=backend)

        # Then: Insight is restored automatically
        assert len(service2._insights) >= 1
        match = next((i for i in service2._insights if i.insight_id == insight_id), None)
        assert match is not None
        assert match.insight_type == "optimization"
        assert match.source_agent == "perf-agent"
        assert match.confidence == 0.92
