"""Tests for MCP session tools.

Behavioral tests for start_session, end_session, record_decision, etc.
Tests the tool implementations directly, not via MCP protocol.
"""

import pytest

from anamnesis.mcp_server.tools.session import (
    _manage_decisions_impl,
    _manage_sessions_impl,
)


# Use the shared reset_server_state fixture from conftest.py
@pytest.fixture(autouse=True)
def _activate_project(reset_server_state):
    """Every test gets a fresh project context via the conftest fixture."""


class TestStartSession:
    """Tests for manage_sessions(action='start')."""

    def test_start_session_returns_success(self):
        """Starting a session returns success."""
        result = _manage_sessions_impl(action="start", name="Test Session")

        assert result["success"] is True
        assert result["data"]["name"] == "Test Session"
        assert result["data"]["is_active"] is True

    def test_start_session_with_feature_and_files(self):
        """Start session with feature and files."""
        result = _manage_sessions_impl(
            action="start",
            name="Auth Work",
            feature="authentication",
            files=["/src/auth.py", "/src/users.py"],
            tasks=["Implement login", "Add JWT"],
        )

        assert result["success"] is True
        session = result["data"]
        assert session["feature"] == "authentication"
        assert "/src/auth.py" in session["files"]
        assert "Implement login" in session["tasks"]

    def test_start_session_generates_unique_id(self):
        """Each session gets a unique ID."""
        result1 = _manage_sessions_impl(action="start", name="Session 1")
        result2 = _manage_sessions_impl(action="start", name="Session 2")

        assert result1["data"]["session_id"] != result2["data"]["session_id"]


class TestEndSession:
    """Tests for manage_sessions(action='end')."""

    def test_end_active_session(self):
        """End the currently active session."""
        # Start a session
        start_result = _manage_sessions_impl(action="start", name="Test")
        session_id = start_result["data"]["session_id"]

        # End it
        result = _manage_sessions_impl(action="end")

        assert result["success"] is True
        assert result["data"]["session_id"] == session_id
        assert result["data"]["is_active"] is False
        assert result["data"]["ended_at"] is not None

    def test_end_specific_session(self):
        """End a specific session by ID."""
        # Start two sessions
        result1 = _manage_sessions_impl(action="start", name="Session 1")
        result2 = _manage_sessions_impl(action="start", name="Session 2")

        # End the first one
        result = _manage_sessions_impl(action="end", session_id=result1["data"]["session_id"])

        assert result["success"] is True

        # Second session should still be active
        list_result = _manage_sessions_impl(action="list", active_only=True)
        active_sessions = list_result["data"]
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == result2["data"]["session_id"]

    def test_end_session_without_active(self):
        """End session when no active session returns error."""
        result = _manage_sessions_impl(action="end")

        assert result["success"] is False
        assert "No active session" in result["error"]


class TestRecordDecision:
    """Tests for record_decision tool."""

    def test_record_decision_with_session(self):
        """Record decision linked to active session."""
        # Start a session
        _manage_sessions_impl(action="start", name="Test Session")

        # Record a decision
        result = _manage_decisions_impl(
            action="record",
            decision="Use JWT for authentication",
            context="API design",
            rationale="Stateless is better for scaling",
            related_files=["/src/auth.py"],
            tags=["security", "api"],
        )

        assert result["success"] is True
        decision = result["data"]
        assert decision["decision"] == "Use JWT for authentication"
        assert decision["context"] == "API design"
        assert decision["rationale"] == "Stateless is better for scaling"
        assert "/src/auth.py" in decision["related_files"]
        assert "security" in decision["tags"]

    def test_record_decision_without_session(self):
        """Record decision without active session (standalone)."""
        result = _manage_decisions_impl(
            action="record",
            decision="Use PostgreSQL",
            rationale="Better for complex queries",
        )

        assert result["success"] is True
        assert result["data"]["session_id"] == ""

    def test_record_decision_generates_unique_id(self):
        """Each decision gets a unique ID."""
        result1 = _manage_decisions_impl(action="record", decision="Decision 1")
        result2 = _manage_decisions_impl(action="record", decision="Decision 2")

        assert result1["data"]["decision_id"] != result2["data"]["decision_id"]


class TestGetSession:
    """Tests for manage_sessions(action='list') (single-session lookup)."""

    def test_get_active_session(self):
        """Get the currently active session."""
        _manage_sessions_impl(action="start", name="Active Session", feature="test")

        # manage_sessions list with active_only returns active sessions
        result = _manage_sessions_impl(action="list", active_only=True)

        assert result["success"] is True
        assert result["data"][0]["name"] == "Active Session"
        assert result["data"][0]["feature"] == "test"

    def test_get_session_by_id(self):
        """Get a specific session by ID."""
        start_result = _manage_sessions_impl(action="start", name="Specific")
        session_id = start_result["data"]["session_id"]

        result = _manage_sessions_impl(action="list", session_id=session_id)

        assert result["success"] is True
        assert result["data"][0]["session_id"] == session_id

    def test_get_session_not_found(self):
        """Get non-existent session returns error."""
        result = _manage_sessions_impl(action="list", session_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_get_session_no_active(self):
        """Get sessions when none active returns empty list."""
        result = _manage_sessions_impl(action="list", active_only=True)

        assert result["success"] is True
        assert result["metadata"]["total"] == 0


class TestListSessions:
    """Tests for manage_sessions(action='list')."""

    def test_list_all_sessions(self):
        """List all recent sessions."""
        _manage_sessions_impl(action="start", name="Session 1")
        _manage_sessions_impl(action="start", name="Session 2")
        _manage_sessions_impl(action="start", name="Session 3")

        result = _manage_sessions_impl(action="list")

        assert result["success"] is True
        assert result["metadata"]["total"] == 3
        assert len(result["data"]) == 3

    def test_list_active_only(self):
        """List only active sessions."""
        result1 = _manage_sessions_impl(action="start", name="Session 1")
        _manage_sessions_impl(action="start", name="Session 2")
        _manage_sessions_impl(action="end", session_id=result1["data"]["session_id"])

        result = _manage_sessions_impl(action="list", active_only=True)

        assert result["success"] is True
        assert result["metadata"]["total"] == 1
        assert result["data"][0]["name"] == "Session 2"

    def test_list_sessions_with_limit(self):
        """List sessions respects limit."""
        for i in range(5):
            _manage_sessions_impl(action="start", name=f"Session {i}")

        result = _manage_sessions_impl(action="list", limit=3)

        assert result["success"] is True
        assert len(result["data"]) <= 3


class TestGetDecisions:
    """Tests for manage_decisions tool (list action)."""

    def test_get_recent_decisions(self):
        """Get recent decisions across all sessions."""
        _manage_decisions_impl(action="record", decision="Decision 1")
        _manage_decisions_impl(action="record", decision="Decision 2")
        _manage_decisions_impl(action="record", decision="Decision 3")

        result = _manage_decisions_impl(action="list")

        assert result["success"] is True
        assert result["metadata"]["total"] == 3

    def test_get_decisions_by_session(self):
        """Get decisions for a specific session."""
        # Session 1 with 2 decisions
        start_result = _manage_sessions_impl(action="start", name="Session 1")
        session_id = start_result["data"]["session_id"]
        _manage_decisions_impl(action="record", decision="S1 Decision 1")
        _manage_decisions_impl(action="record", decision="S1 Decision 2")

        # Session 2 with 1 decision
        _manage_sessions_impl(action="start", name="Session 2")
        _manage_decisions_impl(action="record", decision="S2 Decision 1")

        # Get only Session 1 decisions
        result = _manage_decisions_impl(action="list", session_id=session_id)

        assert result["success"] is True
        assert result["metadata"]["total"] == 2

    def test_get_decisions_with_limit(self):
        """Get decisions respects limit."""
        for i in range(5):
            _manage_decisions_impl(action="record", decision=f"Decision {i}")

        result = _manage_decisions_impl(action="list", limit=2)

        assert result["success"] is True
        assert len(result["data"]) <= 2


class TestSessionIntegration:
    """Integration tests for session workflows."""

    def test_full_session_workflow(self):
        """Test complete session workflow."""
        # Start session
        start_result = _manage_sessions_impl(
            action="start",
            name="Feature Development",
            feature="user-auth",
            files=["/src/auth.py"],
            tasks=["Implement login"],
        )
        assert start_result["success"] is True
        session_id = start_result["data"]["session_id"]

        # Record decisions
        _manage_decisions_impl(
            action="record",
            decision="Use JWT",
            rationale="Better for stateless APIs",
        )
        _manage_decisions_impl(
            action="record",
            decision="Bcrypt for passwords",
            rationale="Industry standard",
        )

        # Verify decisions linked to session
        session_result = _manage_sessions_impl(action="list", session_id=session_id)
        assert session_result["data"][0]["decision_count"] == 2

        # End session
        end_result = _manage_sessions_impl(action="end")
        assert end_result["success"] is True
        assert end_result["data"]["is_active"] is False

        # Verify decisions still accessible
        decisions_result = _manage_decisions_impl(action="list", session_id=session_id)
        assert decisions_result["metadata"]["total"] == 2


class TestManageSessions:
    """Tests for the consolidated manage_sessions tool."""

    def test_start_via_manage_sessions(self):
        """action='start' creates a session."""
        result = _manage_sessions_impl(
            action="start", name="Consolidated Test", feature="consolidation"
        )
        assert result["success"] is True
        assert result["data"]["name"] == "Consolidated Test"
        assert result["data"]["feature"] == "consolidation"
        assert result["data"]["is_active"] is True

    def test_end_via_manage_sessions(self):
        """action='end' ends the active session."""
        start = _manage_sessions_impl(action="start", name="To End")
        session_id = start["data"]["session_id"]

        result = _manage_sessions_impl(action="end")
        assert result["success"] is True
        assert result["data"]["session_id"] == session_id
        assert result["data"]["is_active"] is False

    def test_end_specific_via_manage_sessions(self):
        """action='end' with session_id ends a specific session."""
        s1 = _manage_sessions_impl(action="start", name="Session 1")
        _manage_sessions_impl(action="start", name="Session 2")

        result = _manage_sessions_impl(
            action="end", session_id=s1["data"]["session_id"]
        )
        assert result["success"] is True

    def test_list_via_manage_sessions(self):
        """action='list' returns sessions."""
        _manage_sessions_impl(action="start", name="Session A")
        _manage_sessions_impl(action="start", name="Session B")

        result = _manage_sessions_impl(action="list")
        assert result["success"] is True
        assert result["metadata"]["total"] == 2

    def test_list_active_only_via_manage_sessions(self):
        """action='list' with active_only filters correctly."""
        s1 = _manage_sessions_impl(action="start", name="Session 1")
        _manage_sessions_impl(action="start", name="Session 2")
        _manage_sessions_impl(action="end", session_id=s1["data"]["session_id"])

        result = _manage_sessions_impl(action="list", active_only=True)
        assert result["success"] is True
        assert result["metadata"]["total"] == 1
        assert result["data"][0]["name"] == "Session 2"

    def test_list_by_id_via_manage_sessions(self):
        """action='list' with session_id returns that session."""
        start = _manage_sessions_impl(action="start", name="Specific")
        sid = start["data"]["session_id"]

        result = _manage_sessions_impl(action="list", session_id=sid)
        assert result["success"] is True
        assert result["data"][0]["session_id"] == sid

    def test_unknown_action(self):
        """Unknown action returns error."""
        result = _manage_sessions_impl(action="invalid")
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_end_without_active(self):
        """action='end' with no active session returns error."""
        result = _manage_sessions_impl(action="end")
        assert result["success"] is False
        assert "No active session" in result["error"]

    def test_full_workflow_via_consolidated(self):
        """Full start → list → end → list workflow."""
        start = _manage_sessions_impl(
            action="start",
            name="Full Workflow",
            feature="testing",
            files=["/src/test.py"],
            tasks=["Write tests"],
        )
        assert start["success"] is True
        sid = start["data"]["session_id"]

        listing = _manage_sessions_impl(action="list", session_id=sid)
        assert listing["data"][0]["feature"] == "testing"

        end = _manage_sessions_impl(action="end")
        assert end["success"] is True
        assert end["data"]["is_active"] is False

        post_end = _manage_sessions_impl(action="list", session_id=sid)
        assert post_end["data"][0]["is_active"] is False
