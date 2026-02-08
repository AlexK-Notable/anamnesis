"""Search tools — codebase search."""

from typing import Literal, Optional

from anamnesis.interfaces.search import SearchQuery, SearchType
from anamnesis.utils.logger import logger

from anamnesis.mcp_server._shared import (
    _ensure_semantic_search,
    _failure_response,
    _get_current_path,
    _get_search_service,
    _success_response,
    _with_error_handling,
    mcp,
)

# Re-export _analyze_codebase_impl from intelligence (where it was merged)
# for backward compatibility with tests that import from search.py
from anamnesis.mcp_server.tools.intelligence import _analyze_codebase_impl  # noqa: F401


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("search_codebase")
async def _search_codebase_impl(
    query: str,
    search_type: str = "text",
    limit: int = 50,
    language: Optional[str] = None,
) -> dict:
    """Implementation for search_codebase tool.

    Routes to appropriate search backend based on search_type:
    - text: Simple substring matching (fast, always available)
    - pattern: Regex and AST structural patterns
    - semantic: Vector similarity search (requires indexing)
    """
    limit = max(1, min(limit, 500))
    current_path = _get_current_path()

    # Map string to SearchType enum
    type_map = {
        "text": SearchType.TEXT,
        "pattern": SearchType.PATTERN,
        "semantic": SearchType.SEMANTIC,
    }
    search_type_enum = type_map.get(search_type.lower())
    if search_type_enum is None:
        return _failure_response(
            f"Unknown search_type '{search_type}'. Choose from: {', '.join(type_map)}"
        )

    # Initialize semantic search lazily if requested
    if search_type_enum == SearchType.SEMANTIC:
        semantic_available = await _ensure_semantic_search()
        if not semantic_available:
            logger.warning("Semantic search not available, falling back to text search")

    # Get search service
    search_service = _get_search_service()

    # Build search query
    search_query = SearchQuery(
        query=query,
        search_type=search_type_enum,
        limit=limit,
        language=language,
    )

    # Execute search (errors propagate to _with_error_handling decorator)
    results = await search_service.search(search_query)

    return _success_response(
        [
            {
                "file": r.file_path,
                "matches": r.matches,
                "score": r.score,
            }
            for r in results
        ],
        query=query,
        search_type=search_type,
        total=len(results),
        path=current_path,
        available_types=[t.value for t in search_service.get_available_backends()],
    )


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
async def search_codebase(
    query: str,
    search_type: Literal["text", "pattern", "semantic"] = "text",
    limit: int = 50,
    language: Optional[str] = None,
) -> dict:
    """Search for code by text matching or patterns.

    Use "text" type for finding specific strings/keywords in code.
    Use "pattern" type for regex/AST patterns.

    Args:
        query: Search query - literal text string or regex pattern
        search_type: Type of search ("text", "pattern", "semantic")
        limit: Maximum number of results (default 50)
        language: Filter results by programming language

    Returns:
        Search results with file paths and matched content
    """
    return await _search_codebase_impl(query, search_type, limit, language)
