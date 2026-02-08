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
# Raw helpers (undecorated — called by the merged dispatch tool)
# =============================================================================


def _get_project_config_helper() -> dict:
    """Return current project config (no error wrapping)."""
    return _success_response({"registry": _registry.to_dict()})


def _activate_project_helper(path: str) -> dict:
    """Activate a project by path (no error wrapping)."""
    ctx = _registry.activate(path)
    return _success_response(
        {"activated": ctx.to_dict(), "registry": _registry.to_dict()},
    )


def _list_projects_helper() -> dict:
    """List all known projects (no error wrapping)."""
    projects = _registry.list_projects()
    return _success_response(
        [p.to_dict() for p in projects],
        total=len(projects),
        active_path=_registry.active_path,
    )


# =============================================================================
# Decorated _impl functions (kept for backward test compatibility)
# =============================================================================


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("get_project_config")
def _get_project_config_impl() -> dict:
    """Implementation for get_project_config tool (read-only)."""
    return _get_project_config_helper()


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("activate_project")
def _activate_project_impl(path: str) -> dict:
    """Implementation for activate_project tool."""
    return _activate_project_helper(path)


# TODO(cleanup): Remove this backward-compat wrapper — tests migrated to canonical function
@_with_error_handling("list_projects")
def _list_projects_impl() -> dict:
    """Implementation for list_projects tool."""
    return _list_projects_helper()


# =============================================================================
# Merged Implementation
# =============================================================================


@_with_error_handling("manage_project")
def _manage_project_impl(action: str = "status", path: str = "") -> dict:
    """Implementation for manage_project tool — dispatches by action."""
    if action == "status":
        config = _get_project_config_helper()
        projects = _list_projects_helper()
        config_data = config.get("data", {})
        projects_data = projects.get("data", [])
        projects_meta = projects.get("metadata", {})
        return _success_response(
            {
                "registry": config_data.get("registry"),
                "projects": projects_data,
            },
            total=projects_meta.get("total", 0),
            active_path=projects_meta.get("active_path"),
        )
    elif action == "activate":
        if not path:
            return _failure_response(
                "path is required when action='activate'"
            )
        return _activate_project_helper(path)
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
