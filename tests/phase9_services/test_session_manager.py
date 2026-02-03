"""Tests for SessionManager service.

Tests session lifecycle management and decision recording.
"""

import pytest

from anamnesis.constants import utcnow

from anamnesis.services.session_manager import (
    DecisionInfo,
    SessionInfo,
    SessionManager,
)
from anamnesis.storage.sync_backend import SyncSQLiteBackend


@pytest.fixture
def backend():
    """Create in-memory backend for testing."""
    backend = SyncSQLiteBackend(":memory:")
    backend.connect()
    yield backend
    backend.close()


@pytest.fixture
def manager(backend):
    """Create SessionManager with backend."""
    return SessionManager(backend)


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_creation(self):
        """Test SessionInfo creation."""
        now = utcnow()
        info = SessionInfo(
            session_id="session_123",
            name="Test Session",
            feature="authentication",
            files=["/src/auth.py"],
            tasks=["Implement login"],
            is_active=True,
            started_at=now,
            updated_at=now,
        )
        assert info.session_id == "session_123"
        assert info.name == "Test Session"
        assert info.is_active is True

    def test_to_dict(self):
        """Test SessionInfo.to_dict()."""
        now = utcnow()
        info = SessionInfo(
            session_id="session_123",
            name="Test",
            feature="auth",
            files=[],
            tasks=[],
            is_active=True,
            started_at=now,
            updated_at=now,
        )
        d = info.to_dict()
        assert d["session_id"] == "session_123"
        assert d["is_active"] is True
        assert d["ended_at"] is None


class TestDecisionInfo:
    """Tests for DecisionInfo dataclass."""

    def test_creation(self):
        """Test DecisionInfo creation."""
        now = utcnow()
        info = DecisionInfo(
            decision_id="decision_123",
            decision="Use JWT",
            context="API design",
            rationale="Stateless is better",
            session_id="session_456",
            related_files=["/src/auth.py"],
            tags=["security", "api"],
            created_at=now,
        )
        assert info.decision_id == "decision_123"
        assert info.decision == "Use JWT"

    def test_to_dict(self):
        """Test DecisionInfo.to_dict()."""
        now = utcnow()
        info = DecisionInfo(
            decision_id="decision_123",
            decision="Use JWT",
            context="",
            rationale="",
            session_id="",
            related_files=[],
            tags=[],
            created_at=now,
        )
        d = info.to_dict()
        assert d["decision_id"] == "decision_123"
        assert "created_at" in d


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_manager(self, manager, backend):
        """Test manager creation."""
        assert manager.backend is backend
        assert manager.active_session_id is None

    def test_start_session(self, manager):
        """Test starting a new session."""
        session = manager.start_session(
            name="Test Session",
            feature="authentication",
            files=["/src/auth.py"],
            tasks=["Implement login"],
        )

        assert session.session_id.startswith("session_")
        assert session.name == "Test Session"
        assert session.feature == "authentication"
        assert session.is_active is True
        assert len(session.files) == 1
        assert manager.active_session_id == session.session_id

    def test_end_session(self, manager):
        """Test ending a session."""
        session = manager.start_session(name="Test")
        assert manager.active_session_id == session.session_id

        result = manager.end_session()
        assert result is True
        assert manager.active_session_id is None

        # Verify session is marked as ended
        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.is_active is False
        assert retrieved.ended_at is not None

    def test_end_session_by_id(self, manager):
        """Test ending a specific session by ID."""
        session1 = manager.start_session(name="Session 1")
        session2 = manager.start_session(name="Session 2")

        # End session 1 (not the active one)
        result = manager.end_session(session1.session_id)
        assert result is True

        # Active session should still be session 2
        assert manager.active_session_id == session2.session_id

        # Session 1 should be ended
        retrieved = manager.get_session(session1.session_id)
        assert retrieved.is_active is False

    def test_get_session(self, manager):
        """Test retrieving a session."""
        session = manager.start_session(name="Test", feature="auth")

        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.name == "Test"
        assert retrieved.feature == "auth"

    def test_get_session_not_found(self, manager):
        """Test retrieving non-existent session."""
        result = manager.get_session("nonexistent")
        assert result is None

    def test_get_active_sessions(self, manager):
        """Test getting active sessions."""
        session1 = manager.start_session(name="Session 1")
        session2 = manager.start_session(name="Session 2")
        manager.end_session(session1.session_id)

        active = manager.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == session2.session_id

    def test_get_recent_sessions(self, manager):
        """Test getting recent sessions."""
        session1 = manager.start_session(name="Session 1")
        session2 = manager.start_session(name="Session 2")
        session3 = manager.start_session(name="Session 3")

        recent = manager.get_recent_sessions(limit=2)
        assert len(recent) <= 2

    def test_update_session(self, manager):
        """Test updating a session."""
        session = manager.start_session(name="Test")

        updated = manager.update_session(
            files=["/src/new.py"],
            tasks=["New task"],
            notes="Updated notes",
        )

        assert updated is not None
        assert "/src/new.py" in updated.files
        assert "New task" in updated.tasks

    def test_add_file_to_session(self, manager):
        """Test adding a file to session."""
        session = manager.start_session(name="Test")
        result = manager.add_file_to_session("/src/new.py")
        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert "/src/new.py" in retrieved.files

    def test_add_task_to_session(self, manager):
        """Test adding a task to session."""
        session = manager.start_session(name="Test")
        result = manager.add_task_to_session("New task")
        assert result is True

        retrieved = manager.get_session(session.session_id)
        assert "New task" in retrieved.tasks


class TestDecisionRecording:
    """Tests for decision recording."""

    def test_record_decision(self, manager):
        """Test recording a decision."""
        session = manager.start_session(name="Test")

        decision = manager.record_decision(
            decision="Use JWT for authentication",
            context="API design discussion",
            rationale="Better for stateless APIs",
            related_files=["/src/auth.py"],
            tags=["security", "api"],
        )

        assert decision.decision_id.startswith("decision_")
        assert decision.decision == "Use JWT for authentication"
        assert decision.session_id == session.session_id

    def test_record_decision_without_session(self, manager):
        """Test recording a decision without an active session."""
        decision = manager.record_decision(
            decision="Standalone decision",
            rationale="Not linked to any session",
        )

        assert decision.decision_id.startswith("decision_")
        assert decision.session_id == ""

    def test_record_decision_with_explicit_session(self, manager):
        """Test recording a decision linked to specific session."""
        session = manager.start_session(name="Test")
        manager.end_session()

        # Record decision linked to ended session
        decision = manager.record_decision(
            decision="Retrospective decision",
            session_id=session.session_id,
        )

        assert decision.session_id == session.session_id

    def test_get_decision(self, manager):
        """Test retrieving a decision."""
        decision = manager.record_decision(decision="Test decision")

        retrieved = manager.get_decision(decision.decision_id)
        assert retrieved is not None
        assert retrieved.decision == "Test decision"

    def test_get_decision_not_found(self, manager):
        """Test retrieving non-existent decision."""
        result = manager.get_decision("nonexistent")
        assert result is None

    def test_get_decisions_by_session(self, manager):
        """Test getting decisions for a session."""
        session = manager.start_session(name="Test")

        manager.record_decision(decision="Decision 1")
        manager.record_decision(decision="Decision 2")
        manager.record_decision(decision="Decision 3")

        decisions = manager.get_decisions_by_session(session.session_id)
        assert len(decisions) == 3

    def test_get_recent_decisions(self, manager):
        """Test getting recent decisions."""
        manager.record_decision(decision="Decision 1")
        manager.record_decision(decision="Decision 2")
        manager.record_decision(decision="Decision 3")

        recent = manager.get_recent_decisions(limit=2)
        assert len(recent) <= 2

    def test_session_decision_count(self, manager):
        """Test that session includes decision count."""
        session = manager.start_session(name="Test")

        manager.record_decision(decision="Decision 1")
        manager.record_decision(decision="Decision 2")

        retrieved = manager.get_session(session.session_id)
        assert retrieved.decision_count == 2


class TestSessionManagerPersistence:
    """Integration tests for persistence."""

    def test_session_survives_manager_restart(self, backend):
        """Test sessions persist after manager restart."""
        # Create session with first manager
        manager1 = SessionManager(backend)
        session = manager1.start_session(
            name="Persistent Session",
            feature="test-feature",
        )
        session_id = session.session_id

        # Create new manager with same backend
        manager2 = SessionManager(backend)

        # Session should be retrievable
        retrieved = manager2.get_session(session_id)
        assert retrieved is not None
        assert retrieved.name == "Persistent Session"
        assert retrieved.feature == "test-feature"

    def test_decisions_survive_manager_restart(self, backend):
        """Test decisions persist after manager restart."""
        # Create decision with first manager
        manager1 = SessionManager(backend)
        decision = manager1.record_decision(
            decision="Persistent Decision",
            rationale="Test persistence",
        )
        decision_id = decision.decision_id

        # Create new manager with same backend
        manager2 = SessionManager(backend)

        # Decision should be retrievable
        retrieved = manager2.get_decision(decision_id)
        assert retrieved is not None
        assert retrieved.decision == "Persistent Decision"
        assert retrieved.rationale == "Test persistence"
