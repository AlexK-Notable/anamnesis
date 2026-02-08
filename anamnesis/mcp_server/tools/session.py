"""Session tools — work session tracking and decision recording."""

from typing import Literal, Optional

from anamnesis.mcp_server._shared import (
    _failure_response,
    _get_session_manager,
    _success_response,
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

    return _success_response(
        session.to_dict(),
        message=f"Session '{session.session_id}' started",
    )


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
        return _success_response(
            ended_session.to_dict() if ended_session else None,
            message=f"Session '{target_id}' ended",
        )
    else:
        return _failure_response(f"Session '{target_id}' not found")


@_with_error_handling("get_sessions")
def _get_sessions_impl(
    session_id: Optional[str] = None,
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Implementation for get_sessions tool.

    If session_id is provided, return that single session.
    If active_only, return active sessions.
    Otherwise, return recent sessions up to limit.
    """
    session_manager = _get_session_manager()

    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            return _success_response(
                [session.to_dict()],
                total=1,
                active_session_id=session_manager.active_session_id,
            )
        else:
            return _failure_response(f"Session '{session_id}' not found", session=None)
    elif active_only:
        sessions = session_manager.get_active_sessions()
    else:
        # If no session_id, check for active session first (backward compat for get_session())
        limit = max(1, min(limit, 100))
        sessions = session_manager.get_recent_sessions(limit=limit)

    return _success_response(
        [s.to_dict() for s in sessions],
        total=len(sessions),
        active_session_id=session_manager.active_session_id,
    )


@_with_error_handling("manage_decisions")
def _manage_decisions_impl(
    action: str = "list",
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    limit: int = 10,
) -> dict:
    """Implementation for manage_decisions tool.

    action="record": record a new decision (requires decision text).
    action="list": list decisions for a session or recent decisions.
    """
    session_manager = _get_session_manager()

    if action == "record":
        if not decision:
            return _failure_response("'decision' text is required when action='record'")
        decision_info = session_manager.record_decision(
            decision=decision,
            context=context,
            rationale=rationale,
            session_id=session_id,
            related_files=related_files,
            tags=tags,
        )
        return _success_response(
            decision_info.to_dict(),
            message=f"Decision '{decision_info.decision_id}' recorded",
        )
    elif action == "list":
        limit = max(1, min(limit, 100))
        if session_id:
            decisions = session_manager.get_decisions_by_session(session_id)
        else:
            decisions = session_manager.get_recent_decisions(limit=limit)
        return _success_response(
            [d.to_dict() for d in decisions],
            total=len(decisions),
        )
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: record, list"
        )


# Backward-compat aliases for tests that import old names
def _get_session_impl(session_id: Optional[str] = None) -> dict:
    """Backward-compat wrapper: delegates to _get_sessions_impl.

    The old API returned "session" (singular dict), the new API returns
    "sessions" (list). This wrapper adds the singular key for compat.
    """
    if session_id:
        result = _get_sessions_impl(session_id=session_id)
    else:
        # Replicate old behavior: get active session or return error
        session_manager = _get_session_manager()
        target_id = session_manager.active_session_id
        if not target_id:
            return _failure_response("No active session", session=None)
        result = _get_sessions_impl(session_id=target_id)
    # Old API returned "session" (singular), not "data" (list)
    if result.get("success") and isinstance(result.get("data"), list) and result["data"]:
        result["session"] = result["data"][0]
    return result


def _list_sessions_impl(active_only: bool = False, limit: int = 10) -> dict:
    """Backward-compat wrapper: delegates to _get_sessions_impl."""
    return _get_sessions_impl(active_only=active_only, limit=limit)


def _record_decision_impl(
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Backward-compat wrapper: delegates to _manage_decisions_impl."""
    return _manage_decisions_impl(
        action="record",
        decision=decision,
        context=context,
        rationale=rationale,
        session_id=session_id,
        related_files=related_files,
        tags=tags,
    )


def _get_decisions_impl(
    session_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Backward-compat wrapper: delegates to _manage_decisions_impl."""
    return _manage_decisions_impl(
        action="list",
        session_id=session_id,
        limit=limit,
    )


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
def get_sessions(
    session_id: Optional[str] = None,
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Get work sessions by ID, active status, or recent history.

    Retrieves session details including files, tasks, and decision count.
    If session_id is provided, returns that single session. If active_only
    is True, returns only active sessions. Otherwise returns recent sessions.

    Args:
        session_id: Get a specific session by ID (optional)
        active_only: Only return active sessions (default False)
        limit: Maximum number of sessions to return (default 10)

    Returns:
        List of sessions with count and active session ID
    """
    return _get_sessions_impl(session_id, active_only, limit)


@mcp.tool
def manage_decisions(
    action: Literal["record", "list"] = "list",
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    limit: int = 10,
) -> dict:
    """Record or list project decisions.

    Use action="record" to capture important decisions made during
    development, including the reasoning and context. Decisions can be
    linked to sessions or recorded independently.

    Use action="list" to retrieve decisions for a specific session or
    recent decisions across all sessions.

    Args:
        action: "record" to save a new decision, "list" to retrieve decisions
        decision: The decision made (required for action="record")
        context: Context for the decision (e.g., "API design discussion")
        rationale: Why this decision was made
        session_id: Session to link to (record) or filter by (list)
        related_files: Files related to the decision (for record)
        tags: Tags for categorization (e.g., ["security", "api"])
        limit: Maximum decisions to return when listing (default 10)

    Returns:
        Decision info (record) or list of decisions (list)
    """
    return _manage_decisions_impl(
        action, decision, context, rationale, session_id, related_files, tags, limit
    )
