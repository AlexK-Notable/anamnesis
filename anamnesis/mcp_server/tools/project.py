"""Project management tools — config, activation, and project listing."""

from typing import Literal

from anamnesis.mcp_server._shared import (
    _failure_response,
    _registry,
    _success_response,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementation
# =============================================================================


@_with_error_handling("manage_project")
def _manage_project_impl(action: str = "status", path: str = "") -> dict:
    """Implementation for manage_project tool — dispatches by action."""
    if action == "status":
        projects = _registry.list_projects()
        return _success_response(
            {
                "registry": _registry.to_dict(),
                "projects": [p.to_dict() for p in projects],
            },
            total=len(projects),
            active_path=_registry.active_path,
        )
    elif action == "activate":
        if not path:
            return _failure_response(
                "path is required when action='activate'"
            )
        ctx = _registry.activate(path)
        return _success_response(
            {"activated": ctx.to_dict(), "registry": _registry.to_dict()},
        )
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: status, activate"
        )


# =============================================================================
# MCP Tool Registration
# =============================================================================


@mcp.tool
def manage_project(
    action: Literal["status", "activate"] = "status",
    path: str = "",
) -> dict:
    """Manage project context — view status or switch active project.

    Each project gets isolated services (database, intelligence, sessions),
    preventing cross-project data contamination.

    Args:
        action: What to do:
            - "status": Get current config, active project, and all known projects (default)
            - "activate": Switch to a different project directory
        path: Project directory path (required when action="activate")

    Returns:
        When action="status":
            registry, projects list, total count, active_path
        When action="activate":
            activated project details and updated registry
    """
    return _manage_project_impl(action, path)
