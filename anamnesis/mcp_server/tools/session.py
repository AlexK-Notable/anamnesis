"""Session tools — work session tracking and decision recording."""

from typing import Literal

from anamnesis.utils.security import (
    MAX_NAME_LENGTH,
    MAX_RATIONALE_LENGTH,
    clamp_integer,
    validate_string_length,
)

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


@_with_error_handling("manage_sessions")
def _manage_sessions_impl(
    action: str = "list",
    name: str = "",
    feature: str = "",
    files: list[str] | None = None,
    tasks: list[str] | None = None,
    session_id: str | None = None,
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Implementation for manage_sessions tool.

    action="start": start a new session (uses name, feature, files, tasks).
    action="end": end a session (uses session_id, defaults to active).
    action="list": list sessions (uses session_id, active_only, limit).
    """
    session_manager = _get_session_manager()

    if action == "start":
        if name:
            validate_string_length(name, "name", max_length=MAX_NAME_LENGTH)
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
    elif action == "end":
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
    elif action == "list":
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                return _success_response(
                    [session.to_dict()],
                    total=1,
                    active_session_id=session_manager.active_session_id,
                )
            else:
                return _failure_response(
                    f"Session '{session_id}' not found", session=None
                )
        elif active_only:
            sessions = session_manager.get_active_sessions()
        else:
            limit = clamp_integer(limit, "limit", 1, 100)
            sessions = session_manager.get_recent_sessions(limit=limit)
        return _success_response(
            [s.to_dict() for s in sessions],
            total=len(sessions),
            active_session_id=session_manager.active_session_id,
        )
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: start, end, list"
        )


@_with_error_handling("manage_decisions")
def _manage_decisions_impl(
    action: str = "list",
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: str | None = None,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
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
        validate_string_length(decision, "decision", min_length=1, max_length=MAX_RATIONALE_LENGTH)
        if rationale:
            validate_string_length(rationale, "rationale", max_length=MAX_RATIONALE_LENGTH)
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
        limit = clamp_integer(limit, "limit", 1, 100)
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



# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def manage_sessions(
    action: Literal["start", "end", "list"] = "list",
    name: str = "",
    feature: str = "",
    files: list[str] | None = None,
    tasks: list[str] | None = None,
    session_id: str | None = None,
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Manage work sessions — start, end, or list sessions.

    Use action="start" to begin tracking a focused piece of work. Sessions
    help organize decisions, files, and tasks related to a specific feature.

    Use action="end" to mark a session as completed. Defaults to ending
    the currently active session if no session_id is provided.

    Use action="list" to retrieve sessions by ID, active status, or
    recent history.

    Args:
        action: "start" to begin, "end" to finish, "list" to retrieve sessions
        name: Session name (for action="start")
        feature: Feature being worked on (for action="start")
        files: Initial list of files (for action="start")
        tasks: Initial list of tasks (for action="start")
        session_id: Session ID to end or get (for action="end" or "list")
        active_only: Only return active sessions (for action="list")
        limit: Maximum sessions to return (for action="list", default 10)

    Returns:
        Session info (start/end) or list of sessions (list)
    """
    return _manage_sessions_impl(
        action, name, feature, files, tasks, session_id, active_only, limit
    )


@mcp.tool
def manage_decisions(
    action: Literal["record", "list"] = "list",
    decision: str = "",
    context: str = "",
    rationale: str = "",
    session_id: str | None = None,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
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
