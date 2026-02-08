"""
Phase 3 Tests: Schema Dataclasses

Tests for the storage schema dataclasses including:
- Dataclass creation and defaults
- Serialization (to_dict)
- Deserialization (from_dict)
- Enum handling
- Datetime handling
"""

from datetime import datetime


from anamnesis.constants import utcnow

from anamnesis.storage.schema import (
    AIInsight,
    ConceptType,
    DeveloperPattern,
    EntryPoint,
    FeatureMap,
    FileIntelligence,
    InsightType,
    KeyDirectory,
    PatternType,
    ProjectDecision,
    ProjectMetadata,
    SemanticConcept,
    SharedPattern,
    WorkSession,
)


class TestConceptTypeEnum:
    """Tests for ConceptType enum."""

    def test_all_concept_types_exist(self):
        """All expected concept types exist."""
        expected = [
            "class",
            "function",
            "method",
            "interface",
            "type",
            "variable",
            "constant",
            "module",
            "package",
            "enum",
            "property",
            "decorator",
        ]
        for name in expected:
            assert ConceptType(name) is not None

    def test_concept_type_values(self):
        """Concept type values are correct strings."""
        assert ConceptType.CLASS.value == "class"
        assert ConceptType.FUNCTION.value == "function"
        assert ConceptType.METHOD.value == "method"
        assert ConceptType.INTERFACE.value == "interface"
        assert ConceptType.PACKAGE.value == "package"
        assert ConceptType.DECORATOR.value == "decorator"


class TestPatternTypeEnum:
    """Tests for PatternType enum."""

    def test_all_pattern_types_exist(self):
        """All expected pattern types exist."""
        expected = [
            "factory",
            "singleton",
            "observer",
            "strategy",
            "decorator",
            "adapter",
            "facade",
            "proxy",
            "builder",
            "dependency_injection",
            "repository",
            "service",
            "controller",
            "middleware",
            "custom",
        ]
        for name in expected:
            assert PatternType(name) is not None

    def test_pattern_type_values(self):
        """Pattern type values are correct strings."""
        assert PatternType.FACTORY.value == "factory"
        assert PatternType.SINGLETON.value == "singleton"
        assert PatternType.REPOSITORY.value == "repository"
        assert PatternType.CUSTOM.value == "custom"


class TestInsightTypeEnum:
    """Tests for InsightType enum."""

    def test_all_insight_types_exist(self):
        """All expected insight types exist."""
        expected = [
            "bug_pattern",
            "optimization",
            "refactor_suggestion",
            "best_practice",
            "security_concern",
            "performance_issue",
            "code_smell",
        ]
        for name in expected:
            assert InsightType(name) is not None

    def test_insight_type_values(self):
        """Insight type values are correct strings."""
        assert InsightType.BUG_PATTERN.value == "bug_pattern"
        assert InsightType.CODE_SMELL.value == "code_smell"


class TestSemanticConcept:
    """Tests for SemanticConcept dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        concept = SemanticConcept(
            id="test-id",
            name="TestClass",
            concept_type=ConceptType.CLASS,
            file_path="/path/to/file.py",
        )
        assert concept.id == "test-id"
        assert concept.name == "TestClass"
        assert concept.concept_type == ConceptType.CLASS
        assert concept.file_path == "/path/to/file.py"

    def test_default_values(self):
        """Default values are set correctly."""
        concept = SemanticConcept(
            id="test-id",
            name="test",
            concept_type=ConceptType.FUNCTION,
            file_path="/path.py",
        )
        assert concept.description == ""
        assert concept.signature == ""
        assert concept.relationships == []
        assert concept.metadata == {}
        assert concept.confidence == 1.0
        assert concept.line_start == 0
        assert concept.line_end == 0
        assert isinstance(concept.created_at, datetime)
        assert isinstance(concept.updated_at, datetime)

    def test_to_dict(self):
        """Serializes to dictionary correctly."""
        now = utcnow()
        concept = SemanticConcept(
            id="test-id",
            name="TestFunc",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
            description="A test function",
            signature="def test_func(x: int) -> str",
            relationships=[{"type": "calls", "target": "other_func"}],
            metadata={"complexity": 5},
            created_at=now,
            updated_at=now,
            confidence=0.95,
            line_start=10,
            line_end=20,
        )

        data = concept.to_dict()

        assert data["id"] == "test-id"
        assert data["name"] == "TestFunc"
        assert data["concept_type"] == "function"
        assert data["file_path"] == "/test.py"
        assert data["description"] == "A test function"
        assert data["signature"] == "def test_func(x: int) -> str"
        assert data["relationships"] == [{"type": "calls", "target": "other_func"}]
        assert data["metadata"] == {"complexity": 5}
        assert data["confidence"] == 0.95
        assert data["line_start"] == 10
        assert data["line_end"] == 20
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)

    def test_from_dict(self):
        """Deserializes from dictionary correctly."""
        data = {
            "id": "test-id",
            "name": "TestClass",
            "concept_type": "class",
            "file_path": "/test.py",
            "description": "A test class",
            "signature": "class TestClass",
            "relationships": [],
            "metadata": {},
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "confidence": 0.9,
            "line_start": 1,
            "line_end": 50,
        }

        concept = SemanticConcept.from_dict(data)

        assert concept.id == "test-id"
        assert concept.name == "TestClass"
        assert concept.concept_type == ConceptType.CLASS
        assert concept.file_path == "/test.py"
        assert isinstance(concept.created_at, datetime)
        assert isinstance(concept.updated_at, datetime)

    def test_roundtrip_serialization(self):
        """to_dict and from_dict are inverses."""
        original = SemanticConcept(
            id="roundtrip-test",
            name="RoundtripFunc",
            concept_type=ConceptType.METHOD,
            file_path="/roundtrip.py",
            description="Testing roundtrip",
            relationships=[{"rel": "test"}],
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = SemanticConcept.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.concept_type == original.concept_type
        assert restored.file_path == original.file_path
        assert restored.description == original.description
        assert restored.relationships == original.relationships
        assert restored.metadata == original.metadata


class TestDeveloperPattern:
    """Tests for DeveloperPattern dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        pattern = DeveloperPattern(
            id="pattern-1",
            pattern_type=PatternType.FACTORY,
            name="FactoryPattern",
        )
        assert pattern.id == "pattern-1"
        assert pattern.pattern_type == PatternType.FACTORY
        assert pattern.name == "FactoryPattern"

    def test_default_values(self):
        """Default values are set correctly."""
        pattern = DeveloperPattern(
            id="p",
            pattern_type=PatternType.SINGLETON,
            name="test",
        )
        assert pattern.examples == []
        assert pattern.frequency == 1  # Default is 1
        assert pattern.file_paths == []
        assert pattern.confidence == 1.0
        assert pattern.metadata == {}

    def test_to_dict(self):
        """Serializes to dictionary correctly."""
        pattern = DeveloperPattern(
            id="pattern-1",
            pattern_type=PatternType.REPOSITORY,
            name="repository_pattern",
            examples=["UserRepository", "OrderRepository"],
            frequency=15,
            file_paths=["/src/repos/user.py", "/src/repos/order.py"],
            confidence=0.85,
            metadata={"layer": "data"},
        )

        data = pattern.to_dict()

        assert data["id"] == "pattern-1"
        assert data["pattern_type"] == "repository"
        assert data["name"] == "repository_pattern"
        assert data["examples"] == ["UserRepository", "OrderRepository"]
        assert data["frequency"] == 15
        assert data["confidence"] == 0.85

    def test_from_dict(self):
        """Deserializes from dictionary correctly."""
        data = {
            "id": "pattern-2",
            "pattern_type": "singleton",
            "name": "singleton_pattern",
            "examples": [],
            "frequency": 100,
            "file_paths": [],
            "confidence": 0.99,
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        pattern = DeveloperPattern.from_dict(data)

        assert pattern.id == "pattern-2"
        assert pattern.pattern_type == PatternType.SINGLETON
        assert pattern.frequency == 100


class TestFileIntelligence:
    """Tests for FileIntelligence dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        intel = FileIntelligence(
            id="file-1",
            file_path="/src/main.py",
        )
        assert intel.id == "file-1"
        assert intel.file_path == "/src/main.py"

    def test_default_values(self):
        """Default values are set correctly."""
        intel = FileIntelligence(
            id="f",
            file_path="/test.py",
        )
        assert intel.language == ""
        assert intel.summary == ""
        assert intel.concepts == []
        assert intel.imports == []
        assert intel.exports == []
        assert intel.dependencies == []
        assert intel.dependents == []
        assert intel.complexity_score == 0.0
        assert intel.metrics == {}
        assert intel.metadata == {}
        assert isinstance(intel.last_analyzed, datetime)
        assert intel.content_hash == ""

    def test_to_dict(self):
        """Serializes to dictionary correctly."""
        now = utcnow()
        intel = FileIntelligence(
            id="file-2",
            file_path="/src/utils.py",
            language="python",
            summary="Utility functions",
            concepts=["helper_func", "util_class"],
            imports=["os", "sys"],
            exports=["main"],
            dependencies=["../lib"],
            dependents=["/src/main.py"],
            complexity_score=3.5,
            metrics={"lines_of_code": 150, "cyclomatic": 5},
            last_analyzed=now,
            content_hash="abc123",
        )

        data = intel.to_dict()

        assert data["file_path"] == "/src/utils.py"
        assert data["language"] == "python"
        assert data["concepts"] == ["helper_func", "util_class"]
        assert data["complexity_score"] == 3.5
        assert data["metrics"]["lines_of_code"] == 150
        assert data["last_analyzed"] == now.isoformat()

    def test_from_dict(self):
        """Deserializes from dictionary correctly."""
        data = {
            "id": "file-3",
            "file_path": "/test.ts",
            "language": "typescript",
            "summary": "",
            "concepts": [],
            "imports": ["express"],
            "exports": [],
            "dependencies": [],
            "dependents": [],
            "complexity_score": 2.0,
            "metrics": {"lines_of_code": 50},
            "metadata": {},
            "last_analyzed": "2024-01-01T00:00:00",
            "content_hash": "",
        }

        intel = FileIntelligence.from_dict(data)

        assert intel.language == "typescript"
        assert intel.imports == ["express"]
        assert isinstance(intel.last_analyzed, datetime)


class TestSharedPattern:
    """Tests for SharedPattern dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        pattern = SharedPattern(
            id="shared-1",
            name="Error Boundary",
        )
        assert pattern.id == "shared-1"
        assert pattern.name == "Error Boundary"

    def test_default_values(self):
        """Default values are set correctly."""
        pattern = SharedPattern(
            id="s",
            name="test",
        )
        assert pattern.description == ""
        assert pattern.pattern_code == ""
        assert pattern.occurrences == []
        assert pattern.frequency == 0
        assert pattern.category == ""
        assert pattern.confidence == 1.0

    def test_create_and_serialize(self):
        """Can create and serialize shared pattern."""
        pattern = SharedPattern(
            id="shared-1",
            name="Error Boundary",
            description="Catches rendering errors",
            pattern_code="try { render() } catch { fallback() }",
            frequency=25,
            confidence=0.9,
        )

        data = pattern.to_dict()
        restored = SharedPattern.from_dict(data)

        assert restored.id == "shared-1"
        assert restored.frequency == 25
        assert restored.confidence == 0.9
        assert restored.pattern_code == "try { render() } catch { fallback() }"


class TestAIInsight:
    """Tests for AIInsight dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        insight = AIInsight(
            id="insight-1",
            insight_type=InsightType.BUG_PATTERN,
            title="Potential null pointer",
            description="Variable may be null when accessed",
        )
        assert insight.id == "insight-1"
        assert insight.insight_type == InsightType.BUG_PATTERN
        assert insight.title == "Potential null pointer"

    def test_default_values(self):
        """Default values are set correctly."""
        insight = AIInsight(
            id="i",
            insight_type=InsightType.OPTIMIZATION,
            title="test",
            description="test",
        )
        assert insight.affected_files == []
        assert insight.severity == "info"
        assert insight.confidence == 1.0
        assert insight.suggested_action == ""
        assert insight.code_snippet == ""
        assert insight.metadata == {}
        assert insight.acknowledged is False
        assert insight.resolved is False

    def test_to_dict_and_from_dict(self):
        """Serializes and deserializes correctly."""
        insight = AIInsight(
            id="insight-2",
            insight_type=InsightType.OPTIMIZATION,
            title="Consider caching",
            description="Database queries could be cached",
            affected_files=["/src/api.py"],
            severity="warning",
            confidence=0.75,
            suggested_action="Add Redis caching layer",
            metadata={"priority": "high"},
        )

        data = insight.to_dict()
        restored = AIInsight.from_dict(data)

        assert restored.id == "insight-2"
        assert restored.insight_type == InsightType.OPTIMIZATION
        assert restored.confidence == 0.75
        assert restored.affected_files == ["/src/api.py"]
        assert restored.severity == "warning"


class TestProjectMetadata:
    """Tests for ProjectMetadata dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        meta = ProjectMetadata(
            id="proj-1",
            name="myproject",
            path="/home/user/myproject",
        )
        assert meta.id == "proj-1"
        assert meta.path == "/home/user/myproject"
        assert meta.name == "myproject"

    def test_default_values(self):
        """Default values are set correctly."""
        meta = ProjectMetadata(
            id="p",
            name="test",
            path="/test",
        )
        assert meta.tech_stack == []
        assert meta.build_tools == []
        assert meta.vcs_type == "git"
        assert meta.default_branch == "main"
        assert meta.file_count == 0
        assert meta.total_lines == 0
        assert meta.languages == {}
        assert meta.metadata == {}

    def test_roundtrip(self):
        """Roundtrip serialization works."""
        meta = ProjectMetadata(
            id="proj-2",
            name="api-service",
            path="/projects/api",
            tech_stack=["python", "fastapi", "sqlalchemy"],
            build_tools=["pip", "poetry"],
            vcs_type="git",
            default_branch="main",
            file_count=150,
            total_lines=10000,
            languages={"python": 8000, "sql": 2000},
        )

        data = meta.to_dict()
        restored = ProjectMetadata.from_dict(data)

        assert restored.name == "api-service"
        assert restored.tech_stack == ["python", "fastapi", "sqlalchemy"]
        assert restored.languages == {"python": 8000, "sql": 2000}


class TestFeatureMap:
    """Tests for FeatureMap dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        feature = FeatureMap(
            id="feat-1",
            feature_name="Authentication",
        )
        assert feature.id == "feat-1"
        assert feature.feature_name == "Authentication"

    def test_default_values(self):
        """Default values are set correctly."""
        feature = FeatureMap(
            id="f",
            feature_name="test",
        )
        assert feature.description == ""
        assert feature.files == []
        assert feature.entry_points == []
        assert feature.keywords == []
        assert feature.confidence == 1.0

    def test_create_and_serialize(self):
        """Can create and serialize feature map."""
        feature = FeatureMap(
            id="feat-1",
            feature_name="Authentication",
            description="User authentication feature",
            files=["/src/auth/login.py", "/src/auth/logout.py"],
            entry_points=["login", "logout"],
            keywords=["auth", "login", "password"],
        )

        data = feature.to_dict()
        restored = FeatureMap.from_dict(data)

        assert restored.id == "feat-1"
        assert restored.feature_name == "Authentication"
        assert len(restored.files) == 2


class TestEntryPoint:
    """Tests for EntryPoint dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        entry = EntryPoint(
            id="entry-1",
            name="main",
            file_path="/src/main.py",
        )
        assert entry.id == "entry-1"
        assert entry.name == "main"
        assert entry.file_path == "/src/main.py"

    def test_default_values(self):
        """Default values are set correctly."""
        entry = EntryPoint(
            id="e",
            name="test",
            file_path="/test.py",
        )
        assert entry.entry_type == "main"
        assert entry.description == ""
        assert entry.exports == []
        assert entry.metadata == {}

    def test_create_and_serialize(self):
        """Can create and serialize entry point."""
        entry = EntryPoint(
            id="entry-1",
            name="main",
            file_path="/src/main.py",
            entry_type="script",
            description="Main entry point",
            exports=["run_app"],
        )

        data = entry.to_dict()
        restored = EntryPoint.from_dict(data)

        assert restored.id == "entry-1"
        assert restored.entry_type == "script"


class TestKeyDirectory:
    """Tests for KeyDirectory dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        key_dir = KeyDirectory(
            id="dir-1",
            path="/src/components",
            name="components",
        )
        assert key_dir.id == "dir-1"
        assert key_dir.path == "/src/components"
        assert key_dir.name == "components"

    def test_default_values(self):
        """Default values are set correctly."""
        key_dir = KeyDirectory(
            id="d",
            path="/src",
            name="src",
        )
        assert key_dir.purpose == ""
        assert key_dir.file_count == 0
        assert key_dir.languages == []
        assert key_dir.patterns == []
        assert key_dir.metadata == {}

    def test_create_and_serialize(self):
        """Can create and serialize key directory."""
        key_dir = KeyDirectory(
            id="dir-1",
            path="/src/components",
            name="components",
            purpose="React components",
            file_count=50,
            languages=["typescript", "css"],
            patterns=["component", "hook"],
        )

        data = key_dir.to_dict()
        restored = KeyDirectory.from_dict(data)

        assert restored.id == "dir-1"
        assert restored.purpose == "React components"
        assert restored.file_count == 50


class TestWorkSession:
    """Tests for WorkSession dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        session = WorkSession(
            id="session-1",
        )
        assert session.id == "session-1"

    def test_default_values(self):
        """Default values are set correctly."""
        session = WorkSession(
            id="s",
        )
        assert session.name == ""
        assert session.feature == ""
        assert session.files == []
        assert session.tasks == []
        assert session.notes == ""
        assert session.metadata == {}
        assert session.ended_at is None

    def test_is_active_property(self):
        """is_active property works correctly."""
        session = WorkSession(id="s1")
        assert session.is_active is True

        session.ended_at = utcnow()
        assert session.is_active is False

    def test_roundtrip(self):
        """Roundtrip serialization works."""
        session = WorkSession(
            id="session-2",
            name="Feature implementation",
            feature="authentication",
            files=["/src/auth.py", "/tests/test_auth.py"],
            tasks=["implement login", "write tests"],
            notes="Working on OAuth2 integration",
        )

        data = session.to_dict()
        restored = WorkSession.from_dict(data)

        assert restored.feature == "authentication"
        assert restored.files == ["/src/auth.py", "/tests/test_auth.py"]
        assert restored.is_active is True


class TestProjectDecision:
    """Tests for ProjectDecision dataclass."""

    def test_create_with_required_fields(self):
        """Can create with minimal required fields."""
        decision = ProjectDecision(
            id="dec-1",
            decision="Use microservices",
        )
        assert decision.id == "dec-1"
        assert decision.decision == "Use microservices"

    def test_default_values(self):
        """Default values are set correctly."""
        decision = ProjectDecision(
            id="d",
            decision="test",
        )
        assert decision.context == ""
        assert decision.rationale == ""
        assert decision.session_id == ""
        assert decision.related_files == []
        assert decision.tags == []
        assert decision.metadata == {}

    def test_create_and_serialize(self):
        """Can create and serialize project decision."""
        decision = ProjectDecision(
            id="dec-1",
            decision="Use microservices",
            context="Need better scalability",
            rationale="Better scalability",
            session_id="session-123",
            related_files=["/src/services/"],
            tags=["architecture"],
        )

        data = decision.to_dict()
        restored = ProjectDecision.from_dict(data)

        assert restored.id == "dec-1"
        assert restored.decision == "Use microservices"
        assert restored.session_id == "session-123"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_concept_with_string_type(self):
        """Can create concept with string type instead of enum."""
        concept = SemanticConcept(
            id="test",
            name="test",
            concept_type="custom_type",  # String instead of enum
            file_path="/test.py",
        )

        data = concept.to_dict()
        assert data["concept_type"] == "custom_type"

    def test_from_dict_with_missing_optional_fields(self):
        """from_dict handles missing optional fields."""
        data = {
            "id": "min-concept",
            "name": "minimal",
            "concept_type": "function",
            "file_path": "/test.py",
        }

        concept = SemanticConcept.from_dict(data)

        assert concept.id == "min-concept"
        assert concept.description == ""
        assert concept.relationships == []

    def test_pattern_from_dict_with_unknown_type(self):
        """from_dict handles unknown pattern type gracefully."""
        data = {
            "id": "p1",
            "pattern_type": "future_type",  # Unknown type
            "name": "test",
            "examples": [],
            "file_paths": [],
            "frequency": 1,
            "confidence": 1.0,
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        # Should handle gracefully - pattern_type stays as string
        pattern = DeveloperPattern.from_dict(data)
        assert pattern is not None
        assert pattern.pattern_type == "future_type"

    def test_datetime_with_microseconds(self):
        """Handles datetime with microseconds."""
        data = {
            "id": "dt-test",
            "name": "test",
            "concept_type": "function",
            "file_path": "/test.py",
            "description": "",
            "signature": "",
            "relationships": [],
            "metadata": {},
            "created_at": "2024-01-01T12:30:45.123456",
            "updated_at": "2024-01-01T12:30:45.123456",
            "confidence": 1.0,
            "line_start": 0,
            "line_end": 0,
        }

        concept = SemanticConcept.from_dict(data)

        assert concept.created_at.microsecond == 123456

    def test_empty_relationships_and_metadata(self):
        """Handles empty relationships and metadata."""
        concept = SemanticConcept(
            id="empty-test",
            name="test",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
            relationships=[],
            metadata={},
        )

        data = concept.to_dict()
        restored = SemanticConcept.from_dict(data)

        assert restored.relationships == []
        assert restored.metadata == {}

    def test_insight_from_dict_with_unknown_type(self):
        """from_dict handles unknown insight type gracefully."""
        data = {
            "id": "i1",
            "insight_type": "new_insight_type",  # Unknown type
            "title": "test",
            "description": "test",
            "affected_files": [],
            "severity": "info",
            "confidence": 1.0,
            "suggested_action": "",
            "code_snippet": "",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
            "acknowledged": False,
            "resolved": False,
        }

        insight = AIInsight.from_dict(data)
        assert insight is not None
        assert insight.insight_type == "new_insight_type"

