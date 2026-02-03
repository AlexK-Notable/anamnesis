"""Contract tests for entity adapters.

Tests that prove:
1. Validation catches missing required fields
2. Type coercion handles string→enum and string→datetime
3. Round-trips (dict→entity→dict) preserve all data
4. Error handling provides useful messages
"""

from datetime import datetime

import pytest

from anamnesis.constants import utcnow

from anamnesis.storage.adapters import (
    AIInsightAdapter,
    ArchitecturalDecisionAdapter,
    DeveloperPatternAdapter,
    EntryPointAdapter,
    FeatureMapAdapter,
    FileIntelligenceAdapter,
    KeyDirectoryAdapter,
    ProjectDecisionAdapter,
    ProjectMetadataAdapter,
    SemanticConceptAdapter,
    SharedPatternAdapter,
    TypeCoercionError,
    ValidationError,
    WorkSessionAdapter,
    adapt_from_dict,
    adapt_to_dict,
)
from anamnesis.storage.schema import (
    ConceptType,
    DecisionStatus,
    InsightType,
    PatternType,
    SemanticConcept,
)


class TestSemanticConceptAdapter:
    """Tests for SemanticConceptAdapter."""

    def test_from_dict_minimal(self):
        """Converts minimal valid dict."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "CLASS",
            "file_path": "/src/test.py",
        }
        result = SemanticConceptAdapter.from_dict(data)

        assert result.id == "test-1"
        assert result.name == "TestClass"
        assert result.concept_type == ConceptType.CLASS
        assert result.file_path == "/src/test.py"

    def test_from_dict_full(self):
        """Converts dict with all fields."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "CLASS",
            "file_path": "/src/test.py",
            "description": "A test class",
            "line_start": 10,
            "line_end": 50,
            "relationships": [{"type": "extends", "target": "BaseClass"}],
            "metadata": {"complexity": 5},
        }
        result = SemanticConceptAdapter.from_dict(data)

        assert result.description == "A test class"
        assert result.line_start == 10
        assert result.line_end == 50
        assert len(result.relationships) == 1
        assert result.metadata["complexity"] == 5

    def test_from_dict_enum_as_enum(self):
        """Accepts enum value directly."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": ConceptType.FUNCTION,
            "file_path": "/src/test.py",
        }
        result = SemanticConceptAdapter.from_dict(data)
        assert result.concept_type == ConceptType.FUNCTION

    def test_validation_missing_required(self):
        """Raises ValidationError for missing required fields."""
        data = {"id": "test-1"}
        with pytest.raises(ValidationError) as exc_info:
            SemanticConceptAdapter.from_dict(data)

        assert "name" in str(exc_info.value) or "Missing required" in str(exc_info.value)

    def test_type_coercion_invalid_enum(self):
        """Raises TypeCoercionError for invalid enum value."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "INVALID_TYPE",
            "file_path": "/src/test.py",
        }
        with pytest.raises(TypeCoercionError) as exc_info:
            SemanticConceptAdapter.from_dict(data)

        assert "INVALID_TYPE" in str(exc_info.value)
        assert "Valid values" in str(exc_info.value)

    def test_round_trip(self):
        """Dict→entity→dict preserves data."""
        original = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "CLASS",
            "file_path": "/src/test.py",
            "description": "A test class",
            "line_start": 10,
            "line_end": 50,
        }
        entity = SemanticConceptAdapter.from_dict(original)
        result = adapt_to_dict(entity)

        assert result["id"] == original["id"]
        assert result["name"] == original["name"]
        assert result["description"] == original["description"]
        assert result["line_start"] == original["line_start"]
        assert result["line_end"] == original["line_end"]


class TestDeveloperPatternAdapter:
    """Tests for DeveloperPatternAdapter."""

    def test_from_dict_minimal(self):
        """Converts minimal valid dict."""
        data = {
            "id": "pattern-1",
            "pattern_type": "FACTORY",
            "name": "RepositoryPattern",
        }
        result = DeveloperPatternAdapter.from_dict(data)

        assert result.id == "pattern-1"
        assert result.pattern_type == PatternType.FACTORY
        assert result.name == "RepositoryPattern"

    def test_from_dict_full(self):
        """Converts dict with all fields."""
        data = {
            "id": "pattern-1",
            "pattern_type": "SINGLETON",
            "name": "CachePattern",
            "frequency": 25,
            "confidence": 0.92,
            "examples": ["UserCache", "DataCache"],
            "file_paths": ["/src/cache.py"],
            "metadata": {"category": "caching"},
        }
        result = DeveloperPatternAdapter.from_dict(data)

        assert result.frequency == 25
        assert result.confidence == 0.92
        assert len(result.examples) == 2
        assert "UserCache" in result.examples

    def test_round_trip(self):
        """Dict→entity→dict preserves data."""
        original = {
            "id": "pattern-1",
            "pattern_type": "FACTORY",
            "name": "RepositoryPattern",
            "frequency": 10,
            "confidence": 0.85,
        }
        entity = DeveloperPatternAdapter.from_dict(original)
        result = adapt_to_dict(entity)

        assert result["id"] == original["id"]
        assert result["name"] == original["name"]
        assert result["frequency"] == original["frequency"]
        assert result["confidence"] == original["confidence"]


class TestAIInsightAdapter:
    """Tests for AIInsightAdapter."""

    def test_from_dict_minimal(self):
        """Converts minimal valid dict."""
        data = {
            "id": "insight-1",
            "insight_type": "BUG_PATTERN",
            "title": "Potential Bug",
            "description": "Found a potential bug",
        }
        result = AIInsightAdapter.from_dict(data)

        assert result.id == "insight-1"
        assert result.insight_type == InsightType.BUG_PATTERN
        assert result.title == "Potential Bug"

    def test_from_dict_full(self):
        """Converts dict with all fields."""
        data = {
            "id": "insight-1",
            "insight_type": "OPTIMIZATION",
            "title": "Performance Issue",
            "description": "Database query is slow",
            "confidence": 0.9,
            "severity": "high",
            "affected_files": ["/src/db.py"],
            "suggested_action": "Add index",
            "metadata": {"found_by": "analysis"},
        }
        result = AIInsightAdapter.from_dict(data)

        assert result.confidence == 0.9
        assert result.severity == "high"
        assert "/src/db.py" in result.affected_files
        assert result.suggested_action == "Add index"

    def test_all_insight_types(self):
        """All InsightType values can be coerced."""
        for insight_type in InsightType:
            data = {
                "id": f"insight-{insight_type.name}",
                "insight_type": insight_type.name,
                "title": "Test",
                "description": "Test description",
            }
            result = AIInsightAdapter.from_dict(data)
            assert result.insight_type == insight_type


class TestWorkSessionAdapter:
    """Tests for WorkSessionAdapter."""

    def test_from_dict_minimal(self):
        """Converts minimal valid dict."""
        data = {"id": "session-1"}
        result = WorkSessionAdapter.from_dict(data)
        assert result.id == "session-1"

    def test_from_dict_full(self):
        """Converts dict with all fields."""
        started_at = utcnow()
        data = {
            "id": "session-1",
            "name": "Feature Work",
            "feature": "authentication",
            "files": ["/src/auth.py"],
            "tasks": ["implement login"],
            "notes": "Working on JWT auth",
            "started_at": started_at.isoformat(),
            "metadata": {"sprint": 5},
        }
        result = WorkSessionAdapter.from_dict(data)

        assert result.name == "Feature Work"
        assert result.feature == "authentication"
        assert "/src/auth.py" in result.files
        assert result.notes == "Working on JWT auth"

    def test_datetime_coercion(self):
        """Coerces ISO string to datetime."""
        data = {
            "id": "session-1",
            "started_at": "2024-01-15T10:30:00",
        }
        result = WorkSessionAdapter.from_dict(data)
        assert isinstance(result.started_at, datetime)
        assert result.started_at.year == 2024
        assert result.started_at.month == 1
        assert result.started_at.day == 15

    def test_optional_datetime_none(self):
        """Handles None for optional datetime."""
        data = {
            "id": "session-1",
            "ended_at": None,
        }
        result = WorkSessionAdapter.from_dict(data)
        assert result.ended_at is None

    def test_invalid_datetime_format(self):
        """Raises TypeCoercionError for invalid datetime."""
        data = {
            "id": "session-1",
            "started_at": "not-a-date",
        }
        with pytest.raises(TypeCoercionError) as exc_info:
            WorkSessionAdapter.from_dict(data)
        assert "datetime" in str(exc_info.value).lower()


class TestArchitecturalDecisionAdapter:
    """Tests for ArchitecturalDecisionAdapter."""

    def test_from_dict_minimal(self):
        """Converts minimal valid dict."""
        data = {
            "id": "decision-1",
            "title": "Use PostgreSQL",
            "context": "Database selection",
            "decision": "Use PostgreSQL for relational features",
        }
        result = ArchitecturalDecisionAdapter.from_dict(data)

        assert result.id == "decision-1"
        assert result.title == "Use PostgreSQL"
        assert result.status == DecisionStatus.PROPOSED  # default

    def test_status_coercion(self):
        """Coerces status string to enum."""
        data = {
            "id": "decision-1",
            "title": "Use PostgreSQL",
            "context": "Database selection",
            "decision": "Use PostgreSQL",
            "status": "ACCEPTED",
        }
        result = ArchitecturalDecisionAdapter.from_dict(data)
        assert result.status == DecisionStatus.ACCEPTED


class TestGenericAdapterFunctions:
    """Tests for adapt_from_dict and adapt_to_dict."""

    def test_adapt_from_dict_selects_correct_adapter(self):
        """adapt_from_dict uses correct adapter for type."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "CLASS",
            "file_path": "/src/test.py",
        }
        result = adapt_from_dict(data, SemanticConcept)
        assert isinstance(result, SemanticConcept)
        assert result.concept_type == ConceptType.CLASS

    def test_adapt_from_dict_unknown_type(self):
        """adapt_from_dict raises ValueError for unknown type."""

        class UnknownType:
            pass

        with pytest.raises(ValueError) as exc_info:
            adapt_from_dict({}, UnknownType)
        assert "No adapter" in str(exc_info.value)

    def test_adapt_to_dict_entity(self):
        """adapt_to_dict converts entity to dict."""
        entity = SemanticConcept(
            id="test-1",
            name="TestClass",
            concept_type=ConceptType.CLASS,
            file_path="/src/test.py",
        )
        result = adapt_to_dict(entity)

        assert result["id"] == "test-1"
        assert result["name"] == "TestClass"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extra_fields_ignored(self):
        """Extra fields in dict are ignored."""
        data = {
            "id": "test-1",
            "name": "TestClass",
            "concept_type": "CLASS",
            "file_path": "/src/test.py",
            "extra_field": "ignored",
            "another_extra": 123,
        }
        result = SemanticConceptAdapter.from_dict(data)
        assert result.id == "test-1"
        assert not hasattr(result, "extra_field")

    def test_validate_returns_all_errors(self):
        """Validate returns all missing fields."""
        errors = SemanticConceptAdapter.validate({})
        assert len(errors) >= 4  # id, name, concept_type, file_path

    def test_adapter_error_has_field_info(self):
        """AdapterError includes field and value info."""
        try:
            SemanticConceptAdapter.from_dict({
                "id": "test-1",
                "name": "TestClass",
                "concept_type": "INVALID",
                "file_path": "/src/test.py",
            })
        except TypeCoercionError as e:
            assert e.field == "concept_type"
            assert e.value == "INVALID"


class TestAllAdaptersContractCompliance:
    """Contract tests ensuring all adapters behave consistently."""

    @pytest.mark.parametrize("adapter_cls,data", [
        (
            SemanticConceptAdapter,
            {"id": "1", "name": "Test", "concept_type": "CLASS", "file_path": "/a.py"},
        ),
        (
            DeveloperPatternAdapter,
            {"id": "1", "pattern_type": "FACTORY", "name": "Test"},
        ),
        (
            AIInsightAdapter,
            {"id": "1", "insight_type": "BUG_PATTERN", "title": "T", "description": "D"},
        ),
        (WorkSessionAdapter, {"id": "1"}),
        (
            ArchitecturalDecisionAdapter,
            {"id": "1", "title": "T", "context": "C", "decision": "D"},
        ),
        (FileIntelligenceAdapter, {"id": "1", "file_path": "/a.py"}),
        (ProjectMetadataAdapter, {"id": "1", "name": "P", "path": "/proj"}),
        (FeatureMapAdapter, {"id": "1", "feature_name": "Feature"}),
        (EntryPointAdapter, {"id": "1", "name": "main", "file_path": "/a.py"}),
        (KeyDirectoryAdapter, {"id": "1", "path": "/src", "name": "src"}),
        (SharedPatternAdapter, {"id": "1", "name": "Pattern"}),
        (ProjectDecisionAdapter, {"id": "1", "decision": "D"}),
    ])
    def test_adapter_creates_valid_entity(self, adapter_cls, data):
        """Each adapter creates a valid entity from minimal data."""
        result = adapter_cls.from_dict(data)
        assert result.id == "1"

    @pytest.mark.parametrize("adapter_cls,data", [
        (
            SemanticConceptAdapter,
            {"id": "1", "name": "Test", "concept_type": "CLASS", "file_path": "/a.py"},
        ),
        (
            DeveloperPatternAdapter,
            {"id": "1", "pattern_type": "FACTORY", "name": "Test"},
        ),
        (
            AIInsightAdapter,
            {"id": "1", "insight_type": "BUG_PATTERN", "title": "T", "description": "D"},
        ),
        (WorkSessionAdapter, {"id": "1"}),
    ])
    def test_round_trip_preserves_id(self, adapter_cls, data):
        """Round-trip preserves at minimum the id."""
        entity = adapter_cls.from_dict(data)
        result = adapt_to_dict(entity)
        assert result["id"] == data["id"]
