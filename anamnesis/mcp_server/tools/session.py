"""Session tools — work session tracking and decision recording."""

from typing import Optional

from anamnesis.mcp_server._shared import (
    _failure_response,
    _get_session_manager,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("start_session")
def _start_session_impl(
    name: str = "",
    feature: str = "",
    files: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
) -> dict:
    """Implementation for start_session tool."""
    session_manager = _get_session_manager()

    session = session_manager.start_session(
        name=name,
        feature=feature,
        files=files or [],
        tasks=tasks or [],
    )

    return {
        "success": True,
        "session": session.to_dict(),
        "message": f"Session '{session.session_id}' started",
    }


@_with_error_handling("end_session")
def _end_session_impl(
    session_id: Optional[str] = None,
) -> dict:
    """Implementation for end_session tool."""
    session_manager = _get_session_manager()

    target_id = session_id or session_manager.active_session_id
    if not target_id:
        return _failure_response("No active session to end")

    success = session_manager.end_session(target_id)

    if success:
        ended_session = session_manager.get_session(target_id)
        return {
            "success": True,
            "session": ended_session.to_dict() if ended_session else None,
            "message": f"Session '{target_id}' ended",
        }
    else:
        return _failure_response(f"Session '{target_id}' not found")


@_with_error_handling("record_decision")
def _record_decision_impl(
    decision: str,
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Implementation for record_decision tool."""
    session_manager = _get_session_manager()

    decision_info = session_manager.record_decision(
        decision=decision,
        context=context,
        rationale=rationale,
        session_id=session_id,
        related_files=related_files,
        tags=tags,
    )

    return {
        "success": True,
        "decision": decision_info.to_dict(),
        "message": f"Decision '{decision_info.decision_id}' recorded",
    }


@_with_error_handling("get_session")
def _get_session_impl(
    session_id: Optional[str] = None,
) -> dict:
    """Implementation for get_session tool."""
    session_manager = _get_session_manager()

    target_id = session_id or session_manager.active_session_id
    if not target_id:
        return _failure_response("No active session", session=None)

    session = session_manager.get_session(target_id)
    if session:
        return {
            "success": True,
            "session": session.to_dict(),
        }
    else:
        return _failure_response(f"Session '{target_id}' not found", session=None)


@_with_error_handling("list_sessions")
def _list_sessions_impl(
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Implementation for list_sessions tool."""
    limit = max(1, min(limit, 100))
    session_manager = _get_session_manager()

    if active_only:
        sessions = session_manager.get_active_sessions()
    else:
        sessions = session_manager.get_recent_sessions(limit=limit)

    return {
        "success": True,
        "sessions": [s.to_dict() for s in sessions],
        "total": len(sessions),
        "active_session_id": session_manager.active_session_id,
    }


@_with_error_handling("get_decisions")
def _get_decisions_impl(
    session_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Implementation for get_decisions tool."""
    limit = max(1, min(limit, 100))
    session_manager = _get_session_manager()

    if session_id:
        decisions = session_manager.get_decisions_by_session(session_id)
    else:
        decisions = session_manager.get_recent_decisions(limit=limit)

    return {
        "success": True,
        "decisions": [d.to_dict() for d in decisions],
        "total": len(decisions),
    }


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def start_session(
    name: str = "",
    feature: str = "",
    files: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
) -> dict:
    """Start a new work session to track development context.

    Use this to begin tracking a focused piece of work. Sessions help
    organize decisions, files, and tasks related to a specific feature
    or bug fix.

    Args:
        name: Name or description of the session
        feature: Feature being worked on (e.g., "authentication", "search")
        files: Initial list of files being worked on
        tasks: Initial list of tasks to complete

    Returns:
        Session info with session_id and status
    """
    return _start_session_impl(name, feature, files, tasks)


@mcp.tool
def end_session(
    session_id: Optional[str] = None,
) -> dict:
    """End a work session.

    Marks the session as completed and records the end time.
    If no session_id is provided, ends the currently active session.

    Args:
        session_id: Session ID to end (optional, defaults to active session)

    Returns:
        Result with ended session info
    """
    return _end_session_impl(session_id)


@mcp.tool
def record_decision(
    decision: str,
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Record a project decision for future reference.

    Use this to capture important decisions made during development,
    including the reasoning and context. Decisions can be linked to
    sessions or recorded independently.

    Args:
        decision: The decision made (e.g., "Use JWT for authentication")
        context: Context for the decision (e.g., "API design discussion")
        rationale: Why this decision was made
        session_id: Session to link to (optional, defaults to active session)
        related_files: Files related to the decision
        tags: Tags for categorization (e.g., ["security", "api"])

    Returns:
        Decision info with decision_id
    """
    return _record_decision_impl(decision, context, rationale, session_id, related_files, tags)


@mcp.tool
def get_session(
    session_id: Optional[str] = None,
) -> dict:
    """Get information about a work session.

    Retrieves session details including files, tasks, and decision count.
    If no session_id is provided, returns the currently active session.

    Args:
        session_id: Session ID to retrieve (optional, defaults to active session)

    Returns:
        Session info or error if not found
    """
    return _get_session_impl(session_id)


@mcp.tool
def list_sessions(
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """List work sessions.

    Get a list of recent sessions or only active (non-ended) sessions.

    Args:
        active_only: Only return active sessions (default False)
        limit: Maximum number of sessions to return (default 10)

    Returns:
        List of sessions with count and active session ID
    """
    return _list_sessions_impl(active_only, limit)


@mcp.tool
def get_decisions(
    session_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Get project decisions.

    Retrieve decisions for a specific session or recent decisions across
    all sessions.

    Args:
        session_id: Filter by session ID (optional)
        limit: Maximum number of decisions to return (default 10)

    Returns:
        List of decisions with count
    """
    return _get_decisions_impl(session_id, limit)
