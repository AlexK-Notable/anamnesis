"""MCP Server implementation for Anamnesis - Codebase Intelligence.

This module is the coordinator for the MCP server. It imports all tool
modules (triggering @mcp.tool registration) and re-exports _impl names
for backward test compatibility.

Architecture:
    _shared.py      — FastMCP instance, service accessors, error handling, helpers
    tools/          — _impl functions + @mcp.tool registrations per domain
    server.py       — this file: coordinator + create_server()
    __init__.py     — public API: create_server, mcp
"""

from fastmcp import FastMCP

# Import shared infrastructure (re-exported for tests that import from server)
from anamnesis.mcp_server._shared import (  # noqa: F401
    _categorize_references,
    _check_names_against_convention,
    _detect_naming_style,
    _format_blueprint_as_memory,
    _get_active_context,
    _get_codebase_service,
    _get_current_path,
    _get_intelligence_service,
    _get_learning_service,
    _get_memory_service,
    _get_search_service,
    _get_session_manager,
    _REFLECT_PROMPTS,
    _registry,
    _server_start_time,
    _set_current_path,
    _get_symbol_service,
    _toon_encoder,
    _with_error_handling,
    _ensure_semantic_search,
    mcp,
)

# Import all tool modules — this triggers @mcp.tool registration on the
# shared `mcp` instance.  The _impl re-exports keep existing tests working
# (they do `from anamnesis.mcp_server.server import _foo_impl`).
from anamnesis.mcp_server.tools import (  # noqa: F401
    # intelligence
    _get_semantic_insights_impl,
    _get_pattern_recommendations_impl,
    _predict_coding_approach_impl,
    _get_developer_profile_impl,
    _contribute_insights_impl,
    _get_project_blueprint_impl,
    # learning
    _auto_learn_if_needed_impl,
    # monitoring
    _get_system_status_impl,
    # search
    _search_codebase_impl,
    _analyze_codebase_impl,
    # session
    _start_session_impl,
    _end_session_impl,
    _record_decision_impl,
    _get_session_impl,
    _list_sessions_impl,
    _get_decisions_impl,
    # project
    _activate_project_impl,
    _get_project_config_impl,
    _list_projects_impl,
    # memory + metacognition
    _write_memory_impl,
    _read_memory_impl,
    _list_memories_impl,
    _delete_memory_impl,
    _edit_memory_impl,
    _search_memories_impl,
    _reflect_impl,
    # lsp
    _get_lsp_manager,
    _get_symbol_retriever,
    _get_code_editor,
    _find_symbol_impl,
    _get_symbols_overview_impl,
    _find_referencing_symbols_impl,
    _replace_symbol_body_impl,
    _insert_after_symbol_impl,
    _insert_before_symbol_impl,
    _rename_symbol_impl,
    _enable_lsp_impl,
    _get_lsp_status_impl,
    _check_conventions_impl,
)


# =============================================================================
# Server Factory
# =============================================================================


def create_server() -> FastMCP:
    """Create and return the configured MCP server instance.

    Returns:
        Configured FastMCP server with all tools registered
    """
    return mcp


# Allow running directly
if __name__ == "__main__":
    mcp.run()
