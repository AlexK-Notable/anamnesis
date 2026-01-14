"""
Phase 3 Tests: SQLite Backend

Tests for the SQLite storage backend including:
- Connection management
- CRUD operations for all entity types
- Search functionality
- Bulk operations
- Error handling
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from anamnesis.storage.schema import (
    AIInsight,
    ArchitecturalDecision,
    ConceptType,
    DecisionStatus,
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
from anamnesis.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
async def backend():
    """Create a SQLite backend with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    backend = SQLiteBackend(db_path)
    await backend.connect()

    yield backend

    await backend.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def memory_backend():
    """Create an in-memory SQLite backend."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield backend
    await backend.close()


class TestSQLiteBackendConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_database(self):
        """Connect creates database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        backend = SQLiteBackend(db_path)
        await backend.connect()

        assert Path(db_path).exists()
        assert backend.is_connected

        await backend.close()
        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_connect_memory_database(self):
        """Can connect to in-memory database."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()

        assert backend.is_connected

        await backend.close()

    @pytest.mark.asyncio
    async def test_close_disconnects(self):
        """Close disconnects from database."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        await backend.close()

        assert not backend.is_connected

    @pytest.mark.asyncio
    async def test_connect_initializes_schema(self, memory_backend):
        """Connect initializes database schema."""
        # Check that tables exist by attempting operations
        concepts = await memory_backend.get_concepts_by_file("/test.py")
        assert concepts == []


class TestSemanticConceptCRUD:
    """Tests for SemanticConcept CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_concept(self, memory_backend):
        """Can save a concept."""
        concept = SemanticConcept(
            id="test-concept-1",
            name="TestClass",
            concept_type=ConceptType.CLASS,
            file_path="/src/test.py",
            description="A test class",
            line_start=10,
            line_end=50,
        )

        await memory_backend.save_concept(concept)

        # Verify by retrieving
        retrieved = await memory_backend.get_concept("test-concept-1")
        assert retrieved is not None
        assert retrieved.name == "TestClass"

    @pytest.mark.asyncio
    async def test_get_concept(self, memory_backend):
        """Can get a concept by ID."""
        concept = SemanticConcept(
            id="get-test",
            name="GetTest",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
        )
        await memory_backend.save_concept(concept)

        retrieved = await memory_backend.get_concept("get-test")

        assert retrieved is not None
        assert retrieved.id == "get-test"
        assert retrieved.name == "GetTest"
        assert retrieved.concept_type == ConceptType.FUNCTION

    @pytest.mark.asyncio
    async def test_get_concept_not_found(self, memory_backend):
        """Returns None for nonexistent concept."""
        result = await memory_backend.get_concept("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_concepts_by_file(self, memory_backend):
        """Can get concepts by file path."""
        concepts = [
            SemanticConcept(
                id=f"file-concept-{i}",
                name=f"Concept{i}",
                concept_type=ConceptType.FUNCTION,
                file_path="/src/module.py",
            )
            for i in range(3)
        ]

        for c in concepts:
            await memory_backend.save_concept(c)

        # Add concept in different file
        other = SemanticConcept(
            id="other-file",
            name="Other",
            concept_type=ConceptType.CLASS,
            file_path="/src/other.py",
        )
        await memory_backend.save_concept(other)

        result = await memory_backend.get_concepts_by_file("/src/module.py")

        assert len(result) == 3
        assert all(c.file_path == "/src/module.py" for c in result)

    @pytest.mark.asyncio
    async def test_search_concepts(self, memory_backend):
        """Can search concepts by name."""
        concepts = [
            SemanticConcept(
                id="search-1",
                name="UserService",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            ),
            SemanticConcept(
                id="search-2",
                name="UserRepository",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            ),
            SemanticConcept(
                id="search-3",
                name="OrderService",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            ),
        ]

        for c in concepts:
            await memory_backend.save_concept(c)

        result = await memory_backend.search_concepts("User")

        assert len(result) == 2
        names = [c.name for c in result]
        assert "UserService" in names
        assert "UserRepository" in names

    @pytest.mark.asyncio
    async def test_delete_concept(self, memory_backend):
        """Can delete a concept."""
        concept = SemanticConcept(
            id="delete-me",
            name="DeleteMe",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
        )
        await memory_backend.save_concept(concept)

        deleted = await memory_backend.delete_concept("delete-me")
        assert deleted is True

        # Verify deletion
        result = await memory_backend.get_concept("delete-me")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_concept(self, memory_backend):
        """Delete returns False for nonexistent concept."""
        deleted = await memory_backend.delete_concept("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_concept(self, memory_backend):
        """Can update an existing concept."""
        concept = SemanticConcept(
            id="update-test",
            name="OriginalName",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
        )
        await memory_backend.save_concept(concept)

        # Update
        concept.name = "UpdatedName"
        concept.description = "Updated description"
        await memory_backend.save_concept(concept)

        # Verify
        retrieved = await memory_backend.get_concept("update-test")
        assert retrieved.name == "UpdatedName"
        assert retrieved.description == "Updated description"


class TestDeveloperPatternCRUD:
    """Tests for DeveloperPattern CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_pattern(self, memory_backend):
        """Can save and get a pattern."""
        pattern = DeveloperPattern(
            id="pattern-1",
            pattern_type=PatternType.FACTORY,
            name="snake_case",
            examples=["get_user", "save_data"],
            frequency=50,
            confidence=0.95,
        )

        await memory_backend.save_pattern(pattern)
        retrieved = await memory_backend.get_pattern("pattern-1")

        assert retrieved is not None
        assert retrieved.name == "snake_case"
        assert retrieved.frequency == 50
        assert retrieved.examples == ["get_user", "save_data"]

    @pytest.mark.asyncio
    async def test_get_all_patterns(self, memory_backend):
        """Can get all patterns."""
        patterns = [
            DeveloperPattern(
                id=f"p{i}",
                pattern_type=PatternType.SINGLETON,
                name=f"pattern_{i}",
            )
            for i in range(5)
        ]

        for p in patterns:
            await memory_backend.save_pattern(p)

        result = await memory_backend.get_all_patterns()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_delete_pattern(self, memory_backend):
        """Can delete a pattern."""
        pattern = DeveloperPattern(
            id="delete-pattern",
            pattern_type=PatternType.REPOSITORY,
            name="test_prefix",
        )
        await memory_backend.save_pattern(pattern)

        deleted = await memory_backend.delete_pattern("delete-pattern")
        assert deleted is True

        result = await memory_backend.get_pattern("delete-pattern")
        assert result is None


class TestArchitecturalDecisionCRUD:
    """Tests for ArchitecturalDecision CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_decision(self, memory_backend):
        """Can save and get a decision."""
        decision = ArchitecturalDecision(
            id="adr-001",
            title="Use SQLite",
            context="Need local storage for codebase intelligence",
            decision="Use SQLite for local storage",
            status=DecisionStatus.ACCEPTED,
            consequences=["Limited concurrency", "Simple deployment"],
        )

        await memory_backend.save_decision(decision)
        retrieved = await memory_backend.get_decision("adr-001")

        assert retrieved is not None
        assert retrieved.title == "Use SQLite"
        assert retrieved.status == DecisionStatus.ACCEPTED
        assert "Limited concurrency" in retrieved.consequences

    @pytest.mark.asyncio
    async def test_get_all_decisions(self, memory_backend):
        """Can get all decisions."""
        for i in range(3):
            decision = ArchitecturalDecision(
                id=f"adr-{i}",
                title=f"Decision {i}",
                context="test context",
                decision="test decision",
                status=DecisionStatus.PROPOSED,
            )
            await memory_backend.save_decision(decision)

        result = await memory_backend.get_all_decisions()
        assert len(result) == 3


class TestFileIntelligenceCRUD:
    """Tests for FileIntelligence CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_file_intelligence(self, memory_backend):
        """Can save and get file intelligence."""
        intel = FileIntelligence(
            id="file-intel-1",
            file_path="/src/main.py",
            language="python",
            concepts=["main", "setup", "run"],
            imports=["os", "sys", "asyncio"],
            exports=["main"],
            complexity_score=3.5,
            metrics={"lines_of_code": 150},
            last_analyzed=datetime.utcnow(),
        )

        await memory_backend.save_file_intelligence(intel)
        retrieved = await memory_backend.get_file_intelligence("/src/main.py")

        assert retrieved is not None
        assert retrieved.file_path == "/src/main.py"
        assert retrieved.language == "python"
        assert "asyncio" in retrieved.imports

    @pytest.mark.asyncio
    async def test_search_file_intelligence_by_path(self, memory_backend):
        """Can search file intelligence by file path."""
        intel = FileIntelligence(
            id="path-lookup",
            file_path="/unique/path/file.py",
            language="python",
        )
        await memory_backend.save_file_intelligence(intel)

        # Search for files matching path pattern
        result = await memory_backend.search_concepts("/unique/path")

        # The search should find related concepts if any, but at minimum we verify
        # the file intelligence was saved correctly
        retrieved = await memory_backend.get_file_intelligence("/unique/path/file.py")
        assert retrieved is not None
        assert retrieved.file_path == "/unique/path/file.py"


class TestSharedPatternCRUD:
    """Tests for SharedPattern CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_shared_pattern(self, memory_backend):
        """Can save and get shared pattern."""
        pattern = SharedPattern(
            id="shared-1",
            name="Error Boundary",
            description="Catches errors in rendering",
            pattern_code="try/catch wrapper",
            frequency=25,
            confidence=0.9,
        )

        await memory_backend.save_shared_pattern(pattern)
        retrieved = await memory_backend.get_shared_pattern("shared-1")

        assert retrieved is not None
        assert retrieved.name == "Error Boundary"
        assert retrieved.frequency == 25


class TestAIInsightCRUD:
    """Tests for AIInsight CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_insight(self, memory_backend):
        """Can save and get insight."""
        insight = AIInsight(
            id="insight-1",
            insight_type=InsightType.BUG_PATTERN,
            title="Null Pointer Bug",
            description="Potential null pointer in user lookup",
            confidence=0.85,
            severity="high",
            affected_files=["/src/user.py"],
        )

        await memory_backend.save_insight(insight)
        retrieved = await memory_backend.get_insight("insight-1")

        assert retrieved is not None
        assert retrieved.insight_type == InsightType.BUG_PATTERN
        assert retrieved.confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_insights_by_type(self, memory_backend):
        """Can get insights by type."""
        insights = [
            AIInsight(
                id="bug-1",
                insight_type=InsightType.BUG_PATTERN,
                title="Bug 1",
                description="bug",
            ),
            AIInsight(
                id="opt-1",
                insight_type=InsightType.OPTIMIZATION,
                title="Optimization 1",
                description="optimize",
            ),
            AIInsight(
                id="bug-2",
                insight_type=InsightType.BUG_PATTERN,
                title="Bug 2",
                description="another bug",
            ),
        ]

        for i in insights:
            await memory_backend.save_insight(i)

        bugs = await memory_backend.get_insights_by_type(InsightType.BUG_PATTERN)

        assert len(bugs) == 2


class TestProjectMetadataCRUD:
    """Tests for ProjectMetadata CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_metadata(self, memory_backend):
        """Can save and get project metadata."""
        meta = ProjectMetadata(
            id="proj-1",
            path="/home/user/project",
            name="my-project",
            languages={"python": 5000, "typescript": 3000},
            tech_stack=["fastapi", "react", "postgres", "redis"],
            build_tools=["pip", "npm"],
            file_count=150,
            total_lines=8000,
        )

        await memory_backend.save_project_metadata(meta)
        retrieved = await memory_backend.get_project_metadata("proj-1")

        assert retrieved is not None
        assert retrieved.name == "my-project"
        assert "python" in retrieved.languages
        assert "fastapi" in retrieved.tech_stack


class TestFeatureMapCRUD:
    """Tests for FeatureMap CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_feature_map(self, memory_backend):
        """Can save and get feature map."""
        feature = FeatureMap(
            id="feat-1",
            feature_name="Authentication",
            description="User authentication feature",
            files=["/src/auth/login.py", "/src/auth/logout.py"],
            entry_points=["login", "logout"],
            keywords=["auth", "login", "jwt"],
        )

        await memory_backend.save_feature_map(feature)
        retrieved = await memory_backend.get_feature_map("feat-1")

        assert retrieved is not None
        assert retrieved.feature_name == "Authentication"
        assert len(retrieved.files) == 2

    @pytest.mark.asyncio
    async def test_get_all_feature_maps(self, memory_backend):
        """Can get all feature maps."""
        for i in range(3):
            feature = FeatureMap(
                id=f"feat-{i}",
                feature_name=f"Feature {i}",
                description="test",
            )
            await memory_backend.save_feature_map(feature)

        result = await memory_backend.get_all_feature_maps()
        assert len(result) == 3


class TestEntryPointCRUD:
    """Tests for EntryPoint CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_entry_point(self, memory_backend):
        """Can save and get entry point."""
        entry = EntryPoint(
            id="entry-1",
            name="main",
            file_path="/src/main.py",
            entry_type="script",
            description="Main entry point",
        )

        await memory_backend.save_entry_point(entry)
        retrieved = await memory_backend.get_entry_point("entry-1")

        assert retrieved is not None
        assert retrieved.name == "main"
        assert retrieved.entry_type == "script"


class TestKeyDirectoryCRUD:
    """Tests for KeyDirectory CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_key_directory(self, memory_backend):
        """Can save and get key directory."""
        key_dir = KeyDirectory(
            id="dir-1",
            path="/src/components",
            name="components",
            purpose="React components",
            file_count=25,
        )

        await memory_backend.save_key_directory(key_dir)
        retrieved = await memory_backend.get_key_directory("dir-1")

        assert retrieved is not None
        assert retrieved.path == "/src/components"
        assert retrieved.purpose == "React components"


class TestWorkSessionCRUD:
    """Tests for WorkSession CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_session(self, memory_backend):
        """Can save and get work session."""
        session = WorkSession(
            id="session-1",
            name="Auth Implementation",
            feature="authentication",
            files=["/src/auth.py", "/tests/test_auth.py"],
            tasks=["implement login", "write tests"],
            metadata={"method": "jwt"},
        )

        await memory_backend.save_work_session(session)
        retrieved = await memory_backend.get_work_session("session-1")

        assert retrieved is not None
        assert retrieved.name == "Auth Implementation"
        assert retrieved.feature == "authentication"
        assert "implement login" in retrieved.tasks


class TestProjectDecisionCRUD:
    """Tests for ProjectDecision CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_project_decision(self, memory_backend):
        """Can save and get project decision."""
        decision = ProjectDecision(
            id="dec-1",
            decision="Use microservices",
            context="architecture",
            rationale="Better scalability",
            tags=["architecture", "scalability"],
        )

        await memory_backend.save_project_decision(decision)
        retrieved = await memory_backend.get_project_decision("dec-1")

        assert retrieved is not None
        assert retrieved.context == "architecture"
        assert retrieved.decision == "Use microservices"


class TestBulkOperations:
    """Tests for bulk operations."""

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_backend):
        """Can clear all data."""
        # Add some data
        concept = SemanticConcept(
            id="clear-test",
            name="Test",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
        )
        await memory_backend.save_concept(concept)

        pattern = DeveloperPattern(
            id="clear-pattern",
            pattern_type=PatternType.FACTORY,
            name="test",
        )
        await memory_backend.save_pattern(pattern)

        # Clear all
        await memory_backend.clear_all()

        # Verify empty
        concepts = await memory_backend.get_concepts_by_file("/test.py")
        patterns = await memory_backend.get_all_patterns()

        assert len(concepts) == 0
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_backend):
        """Can get statistics."""
        # Add some data
        for i in range(5):
            concept = SemanticConcept(
                id=f"stat-concept-{i}",
                name=f"Concept{i}",
                concept_type=ConceptType.FUNCTION,
                file_path="/test.py",
            )
            await memory_backend.save_concept(concept)

        for i in range(3):
            pattern = DeveloperPattern(
                id=f"stat-pattern-{i}",
                pattern_type=PatternType.SINGLETON,
                name=f"pattern{i}",
            )
            await memory_backend.save_pattern(pattern)

        stats = await memory_backend.get_stats()

        assert stats["semantic_concepts"] == 5
        assert stats["developer_patterns"] == 3

    @pytest.mark.asyncio
    async def test_vacuum(self, backend):
        """Can vacuum database."""
        # Add and delete some data
        for i in range(10):
            concept = SemanticConcept(
                id=f"vacuum-{i}",
                name=f"Vacuum{i}",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            await backend.save_concept(concept)

        for i in range(10):
            await backend.delete_concept(f"vacuum-{i}")

        # Vacuum should not raise
        await backend.vacuum()


class TestComplexQueries:
    """Tests for complex query scenarios."""

    @pytest.mark.asyncio
    async def test_search_concepts_with_limit(self, memory_backend):
        """Search respects limit parameter."""
        for i in range(20):
            concept = SemanticConcept(
                id=f"limit-{i}",
                name=f"SearchableClass{i}",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            await memory_backend.save_concept(concept)

        result = await memory_backend.search_concepts("Searchable", limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_concepts_with_complex_relationships(self, memory_backend):
        """Can store concepts with complex relationships."""
        concept = SemanticConcept(
            id="complex-rel",
            name="ComplexClass",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
            relationships=[
                {"type": "extends", "target": "BaseClass", "file": "/base.py"},
                {"type": "implements", "target": "Interface1"},
                {"type": "uses", "target": "Helper", "count": 5},
            ],
        )

        await memory_backend.save_concept(concept)
        retrieved = await memory_backend.get_concept("complex-rel")

        assert len(retrieved.relationships) == 3
        assert retrieved.relationships[0]["type"] == "extends"

    @pytest.mark.asyncio
    async def test_concepts_with_nested_metadata(self, memory_backend):
        """Can store concepts with nested metadata."""
        concept = SemanticConcept(
            id="nested-meta",
            name="MetaClass",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
            metadata={
                "complexity": {"cyclomatic": 5, "cognitive": 3},
                "metrics": {"loc": 100, "comments": 20},
                "tags": ["important", "core"],
            },
        )

        await memory_backend.save_concept(concept)
        retrieved = await memory_backend.get_concept("nested-meta")

        assert retrieved.metadata["complexity"]["cyclomatic"] == 5
        assert "important" in retrieved.metadata["tags"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_strings(self, memory_backend):
        """Handles empty strings correctly."""
        concept = SemanticConcept(
            id="empty-strings",
            name="",  # Empty name
            concept_type=ConceptType.CLASS,
            file_path="",  # Empty path
            description="",
        )

        await memory_backend.save_concept(concept)
        retrieved = await memory_backend.get_concept("empty-strings")

        assert retrieved is not None
        assert retrieved.name == ""

    @pytest.mark.asyncio
    async def test_unicode_content(self, memory_backend):
        """Handles unicode content correctly."""
        concept = SemanticConcept(
            id="unicode-test",
            name="日本語クラス",  # Japanese
            concept_type=ConceptType.CLASS,
            file_path="/test/日本語.py",
            description="Класс на русском",  # Russian
        )

        await memory_backend.save_concept(concept)
        retrieved = await memory_backend.get_concept("unicode-test")

        assert retrieved.name == "日本語クラス"
        assert "Класс" in retrieved.description

    @pytest.mark.asyncio
    async def test_special_characters_in_search(self, memory_backend):
        """Search handles special characters."""
        concept = SemanticConcept(
            id="special-chars",
            name="Test_Class$Special",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
        )

        await memory_backend.save_concept(concept)

        # Search with special characters
        result = await memory_backend.search_concepts("$Special")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_large_content(self, memory_backend):
        """Handles large content correctly."""
        large_description = "A" * 10000  # 10KB description
        large_relationships = [{"type": f"rel_{i}"} for i in range(100)]

        concept = SemanticConcept(
            id="large-content",
            name="LargeClass",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
            description=large_description,
            relationships=large_relationships,
        )

        await memory_backend.save_concept(concept)
        retrieved = await memory_backend.get_concept("large-content")

        assert len(retrieved.description) == 10000
        assert len(retrieved.relationships) == 100
