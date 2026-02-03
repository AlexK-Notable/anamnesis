"""Search tools — codebase search and analysis."""

from pathlib import Path
from typing import Optional

from anamnesis.interfaces.search import SearchQuery, SearchType
from anamnesis.utils.logger import logger

from anamnesis.mcp_server._shared import (
    _ensure_semantic_search,
    _failure_response,
    _get_codebase_service,
    _get_current_path,
    _get_search_service,
    _with_error_handling,
    mcp,
)


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
    current_path = _get_current_path()

    # Map string to SearchType enum
    type_map = {
        "text": SearchType.TEXT,
        "pattern": SearchType.PATTERN,
        "semantic": SearchType.SEMANTIC,
    }
    search_type_enum = type_map.get(search_type.lower(), SearchType.TEXT)

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

    # Execute search
    try:
        results = await search_service.search(search_query)

        return {
            "success": True,
            "results": [
                {
                    "file": r.file_path,
                    "matches": r.matches,
                    "score": r.score,
                }
                for r in results
            ],
            "query": query,
            "search_type": search_type,
            "total": len(results),
            "path": current_path,
            "available_types": [t.value for t in search_service.get_available_backends()],
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return _failure_response(
            str(e),
            results=[],
            query=query,
            search_type=search_type,
            total=0,
            path=current_path,
        )


@_with_error_handling("analyze_codebase")
def _analyze_codebase_impl(
    path: Optional[str] = None,
    include_file_content: bool = False,
) -> dict:
    """Implementation for analyze_codebase tool."""
    codebase_service = _get_codebase_service()
    path = path or _get_current_path()

    analysis_result = codebase_service.analyze_codebase(
        path=path,
        include_complexity=True,
        include_dependencies=True,
    )

    result = {
        "success": True,
        "path": path,
        "analysis": analysis_result.to_dict() if hasattr(analysis_result, "to_dict") else analysis_result,
    }

    if include_file_content and hasattr(analysis_result, "file_contents"):
        result["file_contents"] = analysis_result.file_contents
    elif include_file_content:
        # Read file content directly if the analysis doesn't provide it
        target = Path(path)
        if target.is_file():
            try:
                result["file_contents"] = {str(target): target.read_text(encoding="utf-8", errors="replace")[:50000]}
            except OSError:
                result["file_contents"] = {}

    return result


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
async def search_codebase(
    query: str,
    search_type: str = "text",
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


@mcp.tool
def analyze_codebase(
    path: Optional[str] = None,
    include_file_content: bool = False,
) -> dict:
    """One-time analysis of a specific file or directory.

    Returns AST structure, complexity metrics, and detected patterns.
    For project-wide understanding, use get_project_blueprint instead.

    Args:
        path: Path to file or directory to analyze
        include_file_content: Include full file content in response

    Returns:
        Analysis results with structure and metrics
    """
    return _analyze_codebase_impl(path, include_file_content)
