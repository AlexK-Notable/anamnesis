"""Memory and metacognition tools — CRUD for project memories + reflect."""

from typing import Literal, Optional

from anamnesis.mcp_server._shared import (
    _failure_response,
    _get_memory_service,
    _REFLECT_PROMPTS,
    _success_response,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Memory Implementations
# =============================================================================


@_with_error_handling("write_memory")
def _write_memory_impl(
    name: str,
    content: str,
) -> dict:
    """Implementation for write_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.write_memory(name, content)
    return _success_response(result.to_dict())


@_with_error_handling("read_memory")
def _read_memory_impl(
    name: str,
) -> dict:
    """Implementation for read_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.read_memory(name)
    if result is None:
        return _failure_response(f"Memory '{name}' not found")
    return _success_response(result.to_dict())


@_with_error_handling("delete_memory")
def _delete_memory_impl(
    name: str,
) -> dict:
    """Implementation for delete_memory tool."""
    memory_service = _get_memory_service()
    deleted = memory_service.delete_memory(name)
    if not deleted:
        return _failure_response(f"Memory '{name}' not found")
    return _success_response({"deleted": name})


@_with_error_handling("edit_memory")
def _edit_memory_impl(
    name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Implementation for edit_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.edit_memory(name, old_text, new_text)
    if result is None:
        return _failure_response(f"Memory '{name}' not found")
    return _success_response(result.to_dict())


@_with_error_handling("search_memories")
def _search_memories_impl(
    query: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """Implementation for search_memories tool.

    When query is None or empty, lists all memories instead of searching.
    """
    memory_service = _get_memory_service()
    if not query:
        # List all memories (replaces former list_memories tool)
        memories = memory_service.list_memories()
        return _success_response(
            [m.to_dict() for m in memories],
            total=len(memories),
        )
    results = memory_service.search_memories(query, limit=limit)
    return _success_response(
        results,
        query=query,
        total=len(results),
    )


# Backward-compat alias for tests that import _list_memories_impl
def _list_memories_impl() -> dict:
    """Backward-compat wrapper: delegates to _search_memories_impl(query=None)."""
    return _search_memories_impl()


# =============================================================================
# Metacognition Implementation
# =============================================================================


@_with_error_handling("reflect")
def _reflect_impl(focus: str = "collected_information") -> dict:
    """Implementation for reflect tool."""
    prompt = _REFLECT_PROMPTS.get(focus)
    if prompt is None:
        return _failure_response(f"Unknown focus '{focus}'. Choose from: {', '.join(_REFLECT_PROMPTS.keys())}")
    return _success_response(
        {"focus": focus, "prompt": prompt},
    )


# =============================================================================
# Memory MCP Tool Registrations
# =============================================================================


@mcp.tool
def write_memory(
    name: str,
    content: str,
) -> dict:
    """Write information about this project that can be useful for future tasks.

    Stores a markdown file in `.anamnesis/memories/` within the project root.
    The memory name should be meaningful and descriptive.

    Args:
        name: Name for the memory (e.g., "architecture-decisions",
            "api-patterns"). Letters, numbers, hyphens, underscores, dots only.
        content: The content to write (markdown format recommended)

    Returns:
        Result with the written memory details
    """
    return _write_memory_impl(name, content)


@mcp.tool
def read_memory(
    name: str,
) -> dict:
    """Read the content of a memory file.

    Use this to retrieve previously stored project knowledge. Only read
    memories that are relevant to the current task.

    Args:
        name: Name of the memory to read

    Returns:
        Memory content and metadata, or error if not found
    """
    return _read_memory_impl(name)


@mcp.tool
def delete_memory(
    name: str,
) -> dict:
    """Delete a memory file.

    Remove a memory that is no longer relevant or correct.

    Args:
        name: Name of the memory to delete

    Returns:
        Success status
    """
    return _delete_memory_impl(name)


@mcp.tool
def edit_memory(
    name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Edit an existing memory by replacing text.

    Use this to update specific parts of a memory without rewriting
    the entire content.

    Args:
        name: Name of the memory to edit
        old_text: The exact text to find and replace
        new_text: The replacement text

    Returns:
        Updated memory content and metadata
    """
    return _edit_memory_impl(name, old_text, new_text)


@mcp.tool
def search_memories(
    query: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """Search project memories or list all memories.

    When a query is provided, finds memories relevant to the natural language
    query using embedding-based search (falls back to substring matching).
    When query is omitted, lists all available memories with metadata.

    Args:
        query: What you're looking for (e.g., "authentication decisions").
            Omit to list all memories.
        limit: Maximum results to return (default 5, applies to search only)

    Returns:
        Matching memories ranked by relevance, or all memories if no query
    """
    return _search_memories_impl(query, limit)


# =============================================================================
# Metacognition MCP Tool Registration
# =============================================================================


@mcp.tool
def reflect(
    focus: Literal["collected_information", "task_adherence", "whether_done"] = "collected_information",
) -> dict:
    """Reflect on your current work with metacognitive prompts.

    Provides structured reflection prompts to help maintain quality and
    focus during complex tasks. Call at natural checkpoints.

    Args:
        focus: What to reflect on:
            - "collected_information": After search/exploration — is the info sufficient?
            - "task_adherence": Before code changes — still on track with the goal?
            - "whether_done": Before declaring done — truly complete and communicated?

    Returns:
        A reflective prompt to guide your thinking
    """
    return _reflect_impl(focus)
