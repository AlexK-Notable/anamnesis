"""MCP tool modules for Anamnesis.

Each module defines _impl functions (testable logic) and @mcp.tool
registrations (MCP-facing wrappers). Importing a module triggers tool
registration on the shared `mcp` FastMCP instance from _shared.py.
"""

from anamnesis.mcp_server.tools.intelligence import (  # noqa: F401
    _analyze_project_impl,
    _get_coding_guidance_impl,
    _get_developer_profile_impl,
    _manage_concepts_impl,
)
from anamnesis.mcp_server.tools.learning import (  # noqa: F401
    _auto_learn_if_needed_impl,
)
from anamnesis.mcp_server.tools.lsp import (  # noqa: F401
    _analyze_code_quality_impl,
    _find_referencing_symbols_impl,
    _find_symbol_impl,
    _get_code_editor,
    _get_lsp_manager,
    _get_symbol_retriever,
    _get_symbols_overview_impl,
    _go_to_definition_impl,
    _insert_near_symbol_impl,
    _investigate_symbol_impl,
    _manage_lsp_impl,
    _match_sibling_style_impl,
    _rename_symbol_impl,
    _replace_symbol_body_impl,
)
from anamnesis.mcp_server.tools.memory import (  # noqa: F401
    _delete_memory_impl,
    _edit_memory_impl,
    _read_memory_impl,
    _reflect_impl,
    _search_memories_impl,
    _write_memory_impl,
)
from anamnesis.mcp_server.tools.monitoring import (  # noqa: F401
    _get_system_status_impl,
)
from anamnesis.mcp_server.tools.project import (  # noqa: F401
    _manage_project_impl,
)
from anamnesis.mcp_server.tools.search import (  # noqa: F401
    _search_codebase_impl,
)
from anamnesis.mcp_server.tools.session import (  # noqa: F401
    _end_session_impl,
    _get_sessions_impl,
    _manage_decisions_impl,
    _start_session_impl,
)
