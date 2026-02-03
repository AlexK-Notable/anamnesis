"""Memory and metacognition tools — CRUD for project memories + reflect."""


from anamnesis.mcp_server._shared import (
    _get_memory_service,
    _REFLECT_PROMPTS,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Memory Implementations
# =============================================================================


@_with_error_handling("write_memory")
def _write_memory_impl(
    memory_file_name: str,
    content: str,
) -> dict:
    """Implementation for write_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.write_memory(memory_file_name, content)
    return {
        "success": True,
        "memory": result.to_dict(),
    }


@_with_error_handling("read_memory")
def _read_memory_impl(
    memory_file_name: str,
) -> dict:
    """Implementation for read_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.read_memory(memory_file_name)
    if result is None:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "memory": result.to_dict(),
    }


@_with_error_handling("list_memories")
def _list_memories_impl() -> dict:
    """Implementation for list_memories tool."""
    memory_service = _get_memory_service()
    memories = memory_service.list_memories()
    return {
        "success": True,
        "memories": [m.to_dict() for m in memories],
        "count": len(memories),
    }


@_with_error_handling("delete_memory")
def _delete_memory_impl(
    memory_file_name: str,
) -> dict:
    """Implementation for delete_memory tool."""
    memory_service = _get_memory_service()
    deleted = memory_service.delete_memory(memory_file_name)
    if not deleted:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "deleted": memory_file_name,
    }


@_with_error_handling("edit_memory")
def _edit_memory_impl(
    memory_file_name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Implementation for edit_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.edit_memory(memory_file_name, old_text, new_text)
    if result is None:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "memory": result.to_dict(),
    }


@_with_error_handling("search_memories")
def _search_memories_impl(
    query: str,
    limit: int = 5,
) -> dict:
    """Implementation for search_memories tool."""
    memory_service = _get_memory_service()
    results = memory_service.search_memories(query, limit=limit)
    return {
        "success": True,
        "results": results,
        "query": query,
        "total": len(results),
    }


# =============================================================================
# Metacognition Implementation
# =============================================================================


def _reflect_impl(focus: str = "collected_information") -> dict:
    """Implementation for reflect tool."""
    prompt = _REFLECT_PROMPTS.get(focus)
    if prompt is None:
        return {
            "success": False,
            "error": f"Unknown focus '{focus}'. Choose from: {', '.join(_REFLECT_PROMPTS.keys())}",
        }
    return {
        "success": True,
        "focus": focus,
        "prompt": prompt,
    }


# =============================================================================
# Memory MCP Tool Registrations
# =============================================================================


@mcp.tool
def write_memory(
    memory_file_name: str,
    content: str,
) -> dict:
    """Write information about this project that can be useful for future tasks.

    Stores a markdown file in `.anamnesis/memories/` within the project root.
    The memory name should be meaningful and descriptive.

    Args:
        memory_file_name: Name for the memory (e.g., "architecture-decisions",
            "api-patterns"). Letters, numbers, hyphens, underscores, dots only.
        content: The content to write (markdown format recommended)

    Returns:
        Result with the written memory details
    """
    return _write_memory_impl(memory_file_name, content)


@mcp.tool
def read_memory(
    memory_file_name: str,
) -> dict:
    """Read the content of a memory file.

    Use this to retrieve previously stored project knowledge. Only read
    memories that are relevant to the current task.

    Args:
        memory_file_name: Name of the memory to read

    Returns:
        Memory content and metadata, or error if not found
    """
    return _read_memory_impl(memory_file_name)


@mcp.tool
def list_memories() -> dict:
    """List available memories for this project.

    Returns names and metadata for all stored memories. Use this to
    discover what project knowledge is available before reading specific
    memories.

    Returns:
        List of memory entries with names, sizes, and timestamps
    """
    return _list_memories_impl()


@mcp.tool
def delete_memory(
    memory_file_name: str,
) -> dict:
    """Delete a memory file.

    Remove a memory that is no longer relevant or correct.

    Args:
        memory_file_name: Name of the memory to delete

    Returns:
        Success status
    """
    return _delete_memory_impl(memory_file_name)


@mcp.tool
def edit_memory(
    memory_file_name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Edit an existing memory by replacing text.

    Use this to update specific parts of a memory without rewriting
    the entire content.

    Args:
        memory_file_name: Name of the memory to edit
        old_text: The exact text to find and replace
        new_text: The replacement text

    Returns:
        Updated memory content and metadata
    """
    return _edit_memory_impl(memory_file_name, old_text, new_text)


@mcp.tool
def search_memories(
    query: str,
    limit: int = 5,
) -> dict:
    """Search project memories by semantic similarity.

    Finds memories relevant to a natural language query. Uses embedding-based
    search when available, falls back to substring matching.

    Args:
        query: What you're looking for (e.g., "authentication decisions")
        limit: Maximum results to return (default 5)

    Returns:
        Matching memories ranked by relevance with snippets
    """
    return _search_memories_impl(query, limit)


# =============================================================================
# Metacognition MCP Tool Registration
# =============================================================================


@mcp.tool
def reflect(
    focus: str = "collected_information",
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
