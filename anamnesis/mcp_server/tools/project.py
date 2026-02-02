"""Project management tools — config and project listing."""

from typing import Optional

from anamnesis.mcp_server._shared import (
    _registry,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("activate_project")
def _get_project_config_impl(activate: Optional[str] = None) -> dict:
    """Implementation for get_project_config tool."""
    if activate:
        ctx = _registry.activate(activate)
        return {
            "success": True,
            "activated": ctx.to_dict(),
            "registry": _registry.to_dict(),
        }
    return {
        "success": True,
        "registry": _registry.to_dict(),
    }


@_with_error_handling("list_projects")
def _list_projects_impl() -> dict:
    """Implementation for list_projects tool."""
    projects = _registry.list_projects()
    return {
        "success": True,
        "projects": [p.to_dict() for p in projects],
        "count": len(projects),
        "active_path": _registry.active_path,
    }


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def get_project_config(
    activate: Optional[str] = None,
) -> dict:
    """Get or change the active project configuration.

    Without `activate`, returns the current project configuration and
    registry state. With `activate`, switches the active project context
    first, then returns the updated configuration.

    Args:
        activate: Optional path to activate as the current project.
            If provided, switches context before returning config.

    Returns:
        Registry state with project details and active project info
    """
    return _get_project_config_impl(activate)


@mcp.tool
def list_projects() -> dict:
    """List all known projects in the registry.

    Shows projects that have been activated during this server session,
    sorted by most recently activated first.

    Returns:
        List of projects with their service status
    """
    return _list_projects_impl()
