"""MCP Server implementation for Anamnesis - Codebase Intelligence.

This module is the coordinator for the MCP server. It imports all tool
modules (triggering @mcp.tool registration) and re-exports canonical
_impl names so tests can import from either tools/<module>.py or server.py.

Architecture:
    _shared.py      -- FastMCP instance, service accessors, error handling, helpers
    tools/          -- _impl functions + @mcp.tool registrations per domain
    server.py       -- this file: coordinator + create_server()
    __init__.py     -- public API: create_server, mcp
"""

from fastmcp import FastMCP

# Import shared infrastructure (re-exported for tests that import from server)
from anamnesis.mcp_server._shared import (  # noqa: F401
    _categorize_references,
    _check_names_against_convention,
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
    _success_response,
    _toon_encoder,
    _with_error_handling,
    _ensure_semantic_search,
    mcp,
)

# Import all tool modules -- this triggers @mcp.tool registration on the
# shared `mcp` instance.  Re-exports below are canonical names only.
from anamnesis.mcp_server.tools import (  # noqa: F401
    # intelligence
    _manage_concepts_impl,
    _get_coding_guidance_impl,
    _get_developer_profile_impl,
    _analyze_project_impl,
    # learning
    _auto_learn_if_needed_impl,
    # monitoring
    _get_system_status_impl,
    # search
    _search_codebase_impl,
    # session
    _start_session_impl,
    _end_session_impl,
    _get_sessions_impl,
    _manage_decisions_impl,
    # project
    _manage_project_impl,
    # memory + metacognition
    _write_memory_impl,
    _read_memory_impl,
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
    _insert_near_symbol_impl,
    _manage_lsp_impl,
    _rename_symbol_impl,
    _analyze_code_quality_impl,
    _investigate_symbol_impl,
    _match_sibling_style_impl,
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
