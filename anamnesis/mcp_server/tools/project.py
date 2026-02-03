"""Project management tools — config, activation, and project listing."""

from typing import Optional

from anamnesis.mcp_server._shared import (
    _registry,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("get_project_config")
def _get_project_config_impl() -> dict:
    """Implementation for get_project_config tool (read-only)."""
    return {
        "success": True,
        "registry": _registry.to_dict(),
    }


@_with_error_handling("activate_project")
def _activate_project_impl(path: str) -> dict:
    """Implementation for activate_project tool."""
    ctx = _registry.activate(path)
    return {
        "success": True,
        "activated": ctx.to_dict(),
        "registry": _registry.to_dict(),
    }


@_with_error_handling("list_projects")
def _list_projects_impl() -> dict:
    """Implementation for list_projects tool."""
    projects = _registry.list_projects()
    return {
        "success": True,
        "projects": [p.to_dict() for p in projects],
        "total": len(projects),
        "active_path": _registry.active_path,
    }


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def get_project_config(
    activate: Optional[str] = None,
) -> dict:
    """Get the current project configuration and registry state.

    Returns the active project context and all known projects.

    For multi-project workflows, use `activate_project(path)` to switch
    between projects. Each project gets isolated services, preventing
    cross-project data contamination.

    Args:
        activate: Deprecated — use activate_project() instead.
            If provided, activates the project for backward compatibility.

    Returns:
        Registry state with project details and active project info
    """
    if activate:
        return _activate_project_impl(activate)
    return _get_project_config_impl()


@mcp.tool
def activate_project(path: str) -> dict:
    """Switch the active project context to a different directory.

    Each project gets isolated services (database, intelligence, sessions),
    preventing cross-project data contamination. After activation, all
    subsequent tool calls operate on the newly activated project.

    Args:
        path: Absolute path to the project directory to activate.

    Returns:
        Activated project details and updated registry state
    """
    return _activate_project_impl(path)


@mcp.tool
def list_projects() -> dict:
    """List all known projects in the registry.

    Shows projects that have been activated during this server session,
    sorted by most recently activated first.

    Returns:
        List of projects with their service status
    """
    return _list_projects_impl()
