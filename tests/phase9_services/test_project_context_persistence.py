"""Tests for ProjectContext persistent backend wiring.

Verifies that ProjectContext creates a persistent SyncSQLiteBackend
and shares it across SessionManager, IntelligenceService, and
LearningService. Tests data survival across ProjectContext recreation.
"""

import pytest
from pathlib import Path

from anamnesis.services.project_registry import ProjectContext


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project directory."""
    return str(tmp_path)


class TestBackendCreation:
    """Tests for _get_backend() lazy creation."""

    def test_backend_creates_intelligence_db(self, project_dir):
        """_get_backend() creates .anamnesis/intelligence.db on first call."""
        ctx = ProjectContext(path=project_dir)
        try:
            backend = ctx._get_backend()
            assert backend is not None
            db_path = Path(project_dir) / ".anamnesis" / "intelligence.db"
            assert db_path.exists()
        finally:
            ctx.cleanup()

    def test_backend_creates_anamnesis_directory(self, project_dir):
        """.anamnesis/ directory is created if it doesn't exist."""
        ctx = ProjectContext(path=project_dir)
        try:
            ctx._get_backend()
            assert (Path(project_dir) / ".anamnesis").is_dir()
        finally:
            ctx.cleanup()

    def test_backend_is_singleton(self, project_dir):
        """Multiple _get_backend() calls return the same instance."""
        ctx = ProjectContext(path=project_dir)
        try:
            b1 = ctx._get_backend()
            b2 = ctx._get_backend()
            assert b1 is b2
        finally:
            ctx.cleanup()


class TestSharedBackend:
    """Tests that all services share the same backend."""

    def test_session_manager_uses_shared_backend(self, project_dir):
        """SessionManager uses the shared persistent backend."""
        ctx = ProjectContext(path=project_dir)
        try:
            sm = ctx.get_session_manager()
            assert sm._backend is ctx._get_backend()
        finally:
            ctx.cleanup()

    def test_intelligence_service_uses_shared_backend(self, project_dir):
        """IntelligenceService uses the shared persistent backend."""
        ctx = ProjectContext(path=project_dir)
        try:
            isvc = ctx.get_intelligence_service()
            assert isvc.backend is ctx._get_backend()
        finally:
            ctx.cleanup()

    def test_learning_service_uses_shared_backend(self, project_dir):
        """LearningService uses the shared persistent backend."""
        ctx = ProjectContext(path=project_dir)
        try:
            lsvc = ctx.get_learning_service()
            assert lsvc.backend is ctx._get_backend()
        finally:
            ctx.cleanup()

    def test_all_three_services_share_same_backend(self, project_dir):
        """All three services point to the same backend instance."""
        ctx = ProjectContext(path=project_dir)
        try:
            sm = ctx.get_session_manager()
            isvc = ctx.get_intelligence_service()
            lsvc = ctx.get_learning_service()
            assert sm._backend is isvc.backend is lsvc.backend
        finally:
            ctx.cleanup()


class TestSessionPersistence:
    """Tests that sessions survive ProjectContext recreation."""

    def test_session_survives_restart(self, project_dir):
        """A session created in one context is visible in a new context."""
        # Create session in first context
        ctx1 = ProjectContext(path=project_dir)
        try:
            sm1 = ctx1.get_session_manager()
            info = sm1.start_session(name="test-session", feature="testing")
            session_id = info.session_id
        finally:
            ctx1.cleanup()

        # Verify session exists in second context
        ctx2 = ProjectContext(path=project_dir)
        try:
            sm2 = ctx2.get_session_manager()
            restored = sm2.get_session(session_id)
            assert restored is not None
            assert restored.name == "test-session"
            assert restored.feature == "testing"
        finally:
            ctx2.cleanup()

    def test_decision_survives_restart(self, project_dir):
        """A decision recorded in one context is visible in a new context."""
        # Record decision in first context
        ctx1 = ProjectContext(path=project_dir)
        try:
            sm1 = ctx1.get_session_manager()
            sm1.start_session(name="decision-test")
            decision_info = sm1.record_decision(
                decision="Use shared backend",
                context="architecture planning",
                rationale="simpler than separate DBs",
            )
            decision_id = decision_info.decision_id
        finally:
            ctx1.cleanup()

        # Verify decision exists in second context
        ctx2 = ProjectContext(path=project_dir)
        try:
            sm2 = ctx2.get_session_manager()
            decisions = sm2.get_recent_decisions(limit=10)
            assert any(d.decision_id == decision_id for d in decisions)
            match = next(d for d in decisions if d.decision_id == decision_id)
            assert match.decision == "Use shared backend"
            assert match.rationale == "simpler than separate DBs"
        finally:
            ctx2.cleanup()


class TestInsightPersistence:
    """Tests that AI insights survive ProjectContext recreation."""

    def test_insight_survives_restart(self, project_dir):
        """An insight contributed in one context is visible after restart."""
        # Contribute insight in first context
        ctx1 = ProjectContext(path=project_dir)
        try:
            isvc1 = ctx1.get_intelligence_service()
            success, insight_id, msg = isvc1.contribute_insight(
                insight_type="bug_pattern",
                content={"title": "Null check missing", "description": "Found unchecked null"},
                confidence=0.9,
                source_agent="test-agent",
            )
            assert success
        finally:
            ctx1.cleanup()

        # Verify insight is loadable in second context
        ctx2 = ProjectContext(path=project_dir)
        try:
            isvc2 = ctx2.get_intelligence_service()
            isvc2.load_from_backend()
            insights = isvc2.get_insights()
            assert len(insights) >= 1
            match = next(
                (i for i in insights if i.insight_id == insight_id), None
            )
            assert match is not None
            assert match.insight_type == "bug_pattern"
            assert match.source_agent == "test-agent"
        finally:
            ctx2.cleanup()


class TestCleanup:
    """Tests for cleanup behavior with persistent backends."""

    def test_cleanup_closes_backend(self, project_dir):
        """cleanup() closes the shared backend."""
        ctx = ProjectContext(path=project_dir)
        backend = ctx._get_backend()
        assert not backend._closed
        ctx.cleanup()
        assert backend._closed

    def test_cleanup_is_idempotent(self, project_dir):
        """cleanup() can be called multiple times without error."""
        ctx = ProjectContext(path=project_dir)
        ctx._get_backend()
        ctx.cleanup()
        ctx.cleanup()  # Should not raise

    def test_cleanup_clears_backend_reference(self, project_dir):
        """cleanup() sets _backend to None."""
        ctx = ProjectContext(path=project_dir)
        ctx._get_backend()
        ctx.cleanup()
        assert ctx._backend is None

    def test_db_file_survives_cleanup(self, project_dir):
        """The intelligence.db file is not deleted by cleanup()."""
        ctx = ProjectContext(path=project_dir)
        ctx._get_backend()
        ctx.cleanup()
        db_path = Path(project_dir) / ".anamnesis" / "intelligence.db"
        assert db_path.exists()


class TestToDict:
    """Tests for to_dict() with backend status."""

    def test_to_dict_shows_backend_inactive(self, project_dir):
        """to_dict() shows backend as False when not yet created."""
        ctx = ProjectContext(path=project_dir)
        d = ctx.to_dict()
        assert d["services"]["backend"] is False

    def test_to_dict_shows_backend_active(self, project_dir):
        """to_dict() shows backend as True after creation."""
        ctx = ProjectContext(path=project_dir)
        try:
            ctx._get_backend()
            d = ctx.to_dict()
            assert d["services"]["backend"] is True
        finally:
            ctx.cleanup()
