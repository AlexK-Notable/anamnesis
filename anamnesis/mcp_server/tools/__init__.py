"""MCP tool modules for Anamnesis.

Each module defines _impl functions (testable logic) and @mcp.tool
registrations (MCP-facing wrappers). Importing a module triggers tool
registration on the shared `mcp` FastMCP instance from _shared.py.
"""

from anamnesis.mcp_server.tools.intelligence import (  # noqa: F401
    _contribute_insights_impl,
    _get_developer_profile_impl,
    _get_pattern_recommendations_impl,
    _get_project_blueprint_impl,
    _get_semantic_insights_impl,
    _predict_coding_approach_impl,
)
from anamnesis.mcp_server.tools.learning import (  # noqa: F401
    _auto_learn_if_needed_impl,
)
from anamnesis.mcp_server.tools.lsp import (  # noqa: F401
    _analyze_code_quality_impl,
    _check_conventions_impl,
    _enable_lsp_impl,
    _find_referencing_symbols_impl,
    _find_symbol_impl,
    _get_code_editor,
    _get_lsp_manager,
    _get_lsp_status_impl,
    _get_symbol_retriever,
    _get_symbols_overview_impl,
    _insert_after_symbol_impl,
    _insert_before_symbol_impl,
    _rename_symbol_impl,
    _replace_symbol_body_impl,
)
from anamnesis.mcp_server.tools.memory import (  # noqa: F401
    _delete_memory_impl,
    _edit_memory_impl,
    _list_memories_impl,
    _read_memory_impl,
    _reflect_impl,
    _search_memories_impl,
    _write_memory_impl,
)
from anamnesis.mcp_server.tools.monitoring import (  # noqa: F401
    _get_system_status_impl,
)
from anamnesis.mcp_server.tools.project import (  # noqa: F401
    _activate_project_impl,
    _get_project_config_impl,
    _list_projects_impl,
    _manage_project_impl,
)
from anamnesis.mcp_server.tools.search import (  # noqa: F401
    _analyze_codebase_impl,
    _search_codebase_impl,
)
from anamnesis.mcp_server.tools.session import (  # noqa: F401
    _end_session_impl,
    _get_decisions_impl,
    _get_session_impl,
    _list_sessions_impl,
    _record_decision_impl,
    _start_session_impl,
)
