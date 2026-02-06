"""Tests for SyncSQLiteBackend.

Tests the synchronous wrapper for SQLiteBackend including:
- Connection management (sync)
- CRUD operations for key entity types
- Batch context for efficient multiple operations
- Round-trip data integrity
"""

import asyncio
import tempfile
from pathlib import Path

from anamnesis.constants import utcnow

import pytest

from anamnesis.storage.schema import (
    AIInsight,
    ConceptType,
    DeveloperPattern,
    InsightType,
    PatternType,
    SemanticConcept,
    WorkSession,
)
from anamnesis.storage.sync_backend import SyncSQLiteBackend


@pytest.fixture
def backend():
    """Create a sync SQLite backend with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    backend = SyncSQLiteBackend(db_path)
    backend.connect()

    yield backend

    backend.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def memory_backend():
    """Create an in-memory sync SQLite backend."""
    backend = SyncSQLiteBackend(":memory:")
    backend.connect()
    yield backend
    backend.close()


class TestSyncBackendConnection:
    """Tests for connection management."""

    def test_connect_creates_database(self):
        """Connect creates database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        backend = SyncSQLiteBackend(db_path)
        backend.connect()

        assert Path(db_path).exists()
        assert backend.is_connected

        backend.close()
        Path(db_path).unlink(missing_ok=True)

    def test_connect_memory_database(self):
        """Can connect to in-memory database."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()

        assert backend.is_connected

        backend.close()

    def test_close_disconnects(self):
        """Close disconnects from database."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()
        backend.close()

        assert not backend.is_connected

    def test_connect_initializes_schema(self, memory_backend):
        """Connect initializes database schema."""
        # Check that tables exist by attempting operations
        concepts = memory_backend.get_concepts_by_file("/test.py")
        assert concepts == []

    def test_sync_backend_works_from_async_context(self):
        """SyncSQLiteBackend works when called from within a running async loop.

        This is the critical scenario for FastMCP: the MCP server's tool
        dispatch runs inside an async event loop, so SyncSQLiteBackend must
        not call run_until_complete() on the caller's loop. Instead it uses
        a dedicated background thread with its own event loop.
        """

        async def _use_sync_backend_from_async():
            backend = SyncSQLiteBackend(":memory:")
            backend.connect()

            concept = SemanticConcept(
                id="async-ctx-test",
                name="AsyncContextTest",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept)

            retrieved = backend.get_concept("async-ctx-test")
            assert retrieved is not None
            assert retrieved.name == "AsyncContextTest"

            backend.close()

        # asyncio.run() creates and runs an event loop on the current thread.
        # If SyncSQLiteBackend tried to use run_until_complete() on *this*
        # thread's loop, it would raise RuntimeError. The thread-based bridge
        # avoids this by executing coroutines on a separate background thread.
        asyncio.run(_use_sync_backend_from_async())

    def test_close_is_idempotent(self):
        """Calling close() multiple times does not raise."""
        backend = SyncSQLiteBackend(":memory:")
        backend.connect()
        backend.close()
        # Second close should be a no-op, not an error
        backend.close()


class TestSyncSemanticConceptCRUD:
    """Tests for SemanticConcept CRUD operations via sync wrapper."""

    def test_save_concept(self, memory_backend):
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

        memory_backend.save_concept(concept)

        # Verify by retrieving
        retrieved = memory_backend.get_concept("test-concept-1")
        assert retrieved is not None
        assert retrieved.name == "TestClass"

    def test_get_concept(self, memory_backend):
        """Can get a concept by ID."""
        concept = SemanticConcept(
            id="get-test",
            name="GetTest",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
        )
        memory_backend.save_concept(concept)

        retrieved = memory_backend.get_concept("get-test")

        assert retrieved is not None
        assert retrieved.id == "get-test"
        assert retrieved.name == "GetTest"
        assert retrieved.concept_type == ConceptType.FUNCTION

    def test_get_concept_not_found(self, memory_backend):
        """Returns None for nonexistent concept."""
        result = memory_backend.get_concept("nonexistent")
        assert result is None

    def test_search_concepts(self, memory_backend):
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
            memory_backend.save_concept(c)

        result = memory_backend.search_concepts("User")

        assert len(result) == 2
        names = [c.name for c in result]
        assert "UserService" in names
        assert "UserRepository" in names

    def test_delete_concept(self, memory_backend):
        """Can delete a concept."""
        concept = SemanticConcept(
            id="delete-me",
            name="DeleteMe",
            concept_type=ConceptType.FUNCTION,
            file_path="/test.py",
        )
        memory_backend.save_concept(concept)

        deleted = memory_backend.delete_concept("delete-me")
        assert deleted is True

        # Verify deletion
        result = memory_backend.get_concept("delete-me")
        assert result is None


class TestSyncDeveloperPatternCRUD:
    """Tests for DeveloperPattern CRUD operations via sync wrapper."""

    def test_save_and_get_pattern(self, memory_backend):
        """Can save and get a pattern."""
        pattern = DeveloperPattern(
            id="pattern-1",
            pattern_type=PatternType.FACTORY,
            name="snake_case",
            examples=["get_user", "save_data"],
            frequency=50,
            confidence=0.95,
        )

        memory_backend.save_pattern(pattern)
        retrieved = memory_backend.get_pattern("pattern-1")

        assert retrieved is not None
        assert retrieved.name == "snake_case"
        assert retrieved.frequency == 50
        assert retrieved.examples == ["get_user", "save_data"]

    def test_get_all_patterns(self, memory_backend):
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
            memory_backend.save_pattern(p)

        result = memory_backend.get_all_patterns()

        assert len(result) == 5


class TestSyncWorkSessionCRUD:
    """Tests for WorkSession CRUD operations via sync wrapper."""

    def test_save_and_get_session(self, memory_backend):
        """Can save and get work session."""
        session = WorkSession(
            id="session-1",
            name="Auth Implementation",
            feature="authentication",
            files=["/src/auth.py", "/tests/test_auth.py"],
            tasks=["implement login", "write tests"],
            metadata={"method": "jwt"},
        )

        memory_backend.save_work_session(session)
        retrieved = memory_backend.get_work_session("session-1")

        assert retrieved is not None
        assert retrieved.name == "Auth Implementation"
        assert retrieved.feature == "authentication"
        assert "implement login" in retrieved.tasks

    def test_get_active_sessions(self, memory_backend):
        """Can get active sessions."""
        active_session = WorkSession(
            id="active-1",
            name="Active Session",
        )
        ended_session = WorkSession(
            id="ended-1",
            name="Ended Session",
            ended_at=utcnow(),
        )

        memory_backend.save_work_session(active_session)
        memory_backend.save_work_session(ended_session)

        result = memory_backend.get_active_sessions()

        assert len(result) == 1
        assert result[0].id == "active-1"


class TestSyncAIInsightCRUD:
    """Tests for AIInsight CRUD operations via sync wrapper."""

    def test_save_and_get_insight(self, memory_backend):
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

        memory_backend.save_insight(insight)
        retrieved = memory_backend.get_insight("insight-1")

        assert retrieved is not None
        assert retrieved.insight_type == InsightType.BUG_PATTERN
        assert retrieved.confidence == 0.85

    def test_get_insights_by_type(self, memory_backend):
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
            memory_backend.save_insight(i)

        bugs = memory_backend.get_insights_by_type(InsightType.BUG_PATTERN)

        assert len(bugs) == 2


class TestSyncBatchContext:
    """Tests for batch context efficiency."""

    def test_batch_context_multiple_operations(self, memory_backend):
        """Batch context allows multiple operations efficiently."""
        concepts = [
            SemanticConcept(
                id=f"batch-{i}",
                name=f"BatchClass{i}",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            for i in range(10)
        ]

        # Using batch context for multiple operations
        with memory_backend.batch_context():
            for concept in concepts:
                memory_backend.save_concept(concept)

        # Verify all were saved
        for i in range(10):
            retrieved = memory_backend.get_concept(f"batch-{i}")
            assert retrieved is not None
            assert retrieved.name == f"BatchClass{i}"

    def test_batch_context_mixed_operations(self, memory_backend):
        """Batch context works with mixed operation types."""
        with memory_backend.batch_context():
            # Save a concept
            concept = SemanticConcept(
                id="mixed-1",
                name="MixedClass",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            memory_backend.save_concept(concept)

            # Save a pattern
            pattern = DeveloperPattern(
                id="mixed-pattern-1",
                pattern_type=PatternType.FACTORY,
                name="test_pattern",
            )
            memory_backend.save_pattern(pattern)

            # Save an insight
            insight = AIInsight(
                id="mixed-insight-1",
                insight_type=InsightType.REFACTOR_SUGGESTION,
                title="Refactor suggestion",
                description="test",
            )
            memory_backend.save_insight(insight)

        # Verify all types were saved
        assert memory_backend.get_concept("mixed-1") is not None
        assert memory_backend.get_pattern("mixed-pattern-1") is not None
        assert memory_backend.get_insight("mixed-insight-1") is not None


class TestSyncBulkOperations:
    """Tests for bulk operations."""

    def test_clear_all(self, memory_backend):
        """Can clear all data."""
        # Add some data
        concept = SemanticConcept(
            id="clear-test",
            name="Test",
            concept_type=ConceptType.CLASS,
            file_path="/test.py",
        )
        memory_backend.save_concept(concept)

        pattern = DeveloperPattern(
            id="clear-pattern",
            pattern_type=PatternType.FACTORY,
            name="test",
        )
        memory_backend.save_pattern(pattern)

        # Clear all
        memory_backend.clear_all()

        # Verify empty
        concepts = memory_backend.get_concepts_by_file("/test.py")
        patterns = memory_backend.get_all_patterns()

        assert len(concepts) == 0
        assert len(patterns) == 0

    def test_get_stats(self, memory_backend):
        """Can get statistics."""
        # Add some data
        for i in range(5):
            concept = SemanticConcept(
                id=f"stat-concept-{i}",
                name=f"Concept{i}",
                concept_type=ConceptType.FUNCTION,
                file_path="/test.py",
            )
            memory_backend.save_concept(concept)

        for i in range(3):
            pattern = DeveloperPattern(
                id=f"stat-pattern-{i}",
                pattern_type=PatternType.SINGLETON,
                name=f"pattern{i}",
            )
            memory_backend.save_pattern(pattern)

        stats = memory_backend.get_stats()

        assert stats["semantic_concepts"] == 5
        assert stats["developer_patterns"] == 3

    def test_vacuum(self, backend):
        """Can vacuum database."""
        # Add and delete some data
        for i in range(10):
            concept = SemanticConcept(
                id=f"vacuum-{i}",
                name=f"Vacuum{i}",
                concept_type=ConceptType.CLASS,
                file_path="/test.py",
            )
            backend.save_concept(concept)

        for i in range(10):
            backend.delete_concept(f"vacuum-{i}")

        # Vacuum should not raise
        backend.vacuum()


class TestSyncRoundTrip:
    """Tests verifying data integrity through round-trips.

    These are contract tests that prove the sync wrapper doesn't
    lose or corrupt data when bridging to the async backend.
    """

    def test_concept_round_trip_full_fields(self, memory_backend):
        """Concept with all fields survives round-trip."""
        original = SemanticConcept(
            id="round-trip-concept",
            name="CompleteClass",
            concept_type=ConceptType.CLASS,
            file_path="/src/complete.py",
            description="A complete class for testing",
            line_start=10,
            line_end=100,
            relationships=[
                {"type": "extends", "target": "BaseClass"},
                {"type": "uses", "target": "Helper"},
            ],
            metadata={
                "complexity": 5,
                "tags": ["important", "core"],
            },
        )

        memory_backend.save_concept(original)
        retrieved = memory_backend.get_concept("round-trip-concept")

        assert retrieved.id == original.id
        assert retrieved.name == original.name
        assert retrieved.concept_type == original.concept_type
        assert retrieved.file_path == original.file_path
        assert retrieved.description == original.description
        assert retrieved.line_start == original.line_start
        assert retrieved.line_end == original.line_end
        assert retrieved.relationships == original.relationships
        assert retrieved.metadata == original.metadata

    def test_pattern_round_trip_full_fields(self, memory_backend):
        """Pattern with all fields survives round-trip."""
        original = DeveloperPattern(
            id="round-trip-pattern",
            pattern_type=PatternType.FACTORY,
            name="RepositoryPattern",
            examples=["UserRepository", "ProductRepository", "OrderRepository"],
            frequency=25,
            confidence=0.92,
            file_paths=["/src/repository.py"],
            metadata={"category": "data-access"},
        )

        memory_backend.save_pattern(original)
        retrieved = memory_backend.get_pattern("round-trip-pattern")

        assert retrieved.id == original.id
        assert retrieved.pattern_type == original.pattern_type
        assert retrieved.name == original.name
        assert retrieved.examples == original.examples
        assert retrieved.frequency == original.frequency
        assert retrieved.confidence == original.confidence
        assert retrieved.file_paths == original.file_paths
        assert retrieved.metadata == original.metadata

    def test_work_session_round_trip_full_fields(self, memory_backend):
        """Work session with all fields survives round-trip."""
        started_at = utcnow()
        original = WorkSession(
            id="round-trip-session",
            name="Feature Implementation",
            feature="authentication",
            files=["/src/auth.py", "/src/login.py", "/tests/test_auth.py"],
            tasks=["implement login", "add JWT", "write tests"],
            notes="Use bcrypt for password hashing. Store sessions in redis.",
            started_at=started_at,
            metadata={
                "priority": "high",
                "sprint": 5,
            },
        )

        memory_backend.save_work_session(original)
        retrieved = memory_backend.get_work_session("round-trip-session")

        assert retrieved.id == original.id
        assert retrieved.name == original.name
        assert retrieved.feature == original.feature
        assert retrieved.files == original.files
        assert retrieved.tasks == original.tasks
        assert retrieved.notes == original.notes
        assert retrieved.metadata == original.metadata

    def test_insight_round_trip_full_fields(self, memory_backend):
        """AI insight with all fields survives round-trip."""
        original = AIInsight(
            id="round-trip-insight",
            insight_type=InsightType.BUG_PATTERN,
            title="Memory Leak in Cache",
            description="The cache implementation doesn't properly clean up expired entries",
            confidence=0.87,
            severity="high",
            affected_files=["/src/cache.py", "/src/memory.py"],
            suggested_action="Add TTL cleanup routine",
            metadata={"found_by": "static_analysis"},
        )

        memory_backend.save_insight(original)
        retrieved = memory_backend.get_insight("round-trip-insight")

        assert retrieved.id == original.id
        assert retrieved.insight_type == original.insight_type
        assert retrieved.title == original.title
        assert retrieved.description == original.description
        assert retrieved.confidence == original.confidence
        assert retrieved.severity == original.severity
        assert retrieved.affected_files == original.affected_files
        assert retrieved.suggested_action == original.suggested_action
        assert retrieved.metadata == original.metadata
