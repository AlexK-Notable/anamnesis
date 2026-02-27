"""Memory and metacognition tools — CRUD for project memories + reflect."""

import secrets
from datetime import UTC, datetime
from typing import Literal

from anamnesis.mcp_server._shared import (
    _REFLECT_PROMPTS,
    _failure_response,
    _get_memory_service,
    _success_response,
    _thought_chains,
    _with_error_handling,
    mcp,
)
from anamnesis.utils.security import (
    MAX_CONTENT_LENGTH,
    MAX_NAME_LENGTH,
    MAX_QUERY_LENGTH,
    clamp_integer,
    validate_string_length,
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
    validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
    validate_string_length(content, "content", min_length=1, max_length=MAX_CONTENT_LENGTH)
    memory_service = _get_memory_service()
    result = memory_service.write_memory(name, content)
    return _success_response(result.to_dict())


@_with_error_handling("read_memory")
def _read_memory_impl(
    name: str,
) -> dict:
    """Implementation for read_memory tool."""
    validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
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
    validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
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
    validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
    validate_string_length(old_text, "old_text", min_length=1, max_length=MAX_CONTENT_LENGTH)
    memory_service = _get_memory_service()
    result = memory_service.edit_memory(name, old_text, new_text)
    if result is None:
        return _failure_response(f"Memory '{name}' not found")
    return _success_response(result.to_dict())


@_with_error_handling("search_memories")
def _search_memories_impl(
    query: str | None = None,
    limit: int = 5,
) -> dict:
    """Implementation for search_memories tool.

    When query is None or empty, lists all memories instead of searching.
    """
    if query is not None and query:
        validate_string_length(query, "query", min_length=1, max_length=MAX_QUERY_LENGTH)
    limit = clamp_integer(limit, "limit", 1, 100)
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


# =============================================================================
# Metacognition Implementation
# =============================================================================


@_with_error_handling("reflect")
def _reflect_impl(
    thought: str = "",
    thought_number: int = 1,
    total_thoughts: int = 1,
    next_thought_needed: bool = False,
    focus: str = "collected_information",
    is_revision: bool = False,
    revises_thought: int | None = None,
    branch_id: str | None = None,
    branch_from_thought: int | None = None,
    chain_id: str | None = None,
) -> dict:
    """Implementation for reflect tool — legacy and sequential thinking modes."""
    # Validate focus
    prompt = _REFLECT_PROMPTS.get(focus)
    if prompt is None:
        return _failure_response(
            f"Unknown focus '{focus}'. Choose from: {', '.join(_REFLECT_PROMPTS)}"
        )

    # Legacy mode — no thought provided, return just the prompt
    if not thought:
        return _success_response({"focus": focus, "prompt": prompt})

    # Sequential thinking mode — store thought in chain
    if chain_id is None:
        chain_id = f"chain_{secrets.token_hex(6)}"

    # Auto-adjust total if thought_number exceeds it
    if thought_number > total_thoughts:
        total_thoughts = thought_number

    # Build thought record
    record: dict = {
        "thought": thought,
        "thought_number": thought_number,
        "total_thoughts": total_thoughts,
        "focus": focus,
        "is_revision": is_revision,
        "revises_thought": revises_thought,
        "branch_id": branch_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Store in chain
    if chain_id not in _thought_chains:
        _thought_chains[chain_id] = []
    _thought_chains[chain_id].append(record)

    chain = _thought_chains[chain_id]

    # Collect metadata about chain shape
    branches = sorted({t["branch_id"] for t in chain if t["branch_id"]})
    revisions = sorted({t["revises_thought"] for t in chain if t["revises_thought"]})

    return _success_response(
        {
            "chain_id": chain_id,
            "thought_number": thought_number,
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "focus": focus,
            "focus_prompt": prompt,
            "chain_length": len(chain),
            "branches": branches,
            "revisions": revisions,
        },
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
    query: str | None = None,
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
    thought: str = "",
    thought_number: int = 1,
    total_thoughts: int = 1,
    next_thought_needed: bool = False,
    focus: Literal[
        "collected_information", "task_adherence", "whether_done", "approach_selection"
    ] = "collected_information",
    is_revision: bool = False,
    revises_thought: int | None = None,
    branch_id: str | None = None,
    branch_from_thought: int | None = None,
    chain_id: str | None = None,
) -> dict:
    """A sequential thinking tool for structured reflection during complex tasks.

    Use this to break down reasoning into steps, building a chain of thoughts.
    Each call stores your thought and returns metadata to guide the next step.

    **When to use:** At natural checkpoints — after exploration, before
    implementation, when comparing approaches, before claiming done.

    **How to think:** Provide your reasoning as `thought`. Set
    `next_thought_needed=True` to continue the chain, `False` when you've
    reached a conclusion. Adjust `total_thoughts` as your understanding evolves.

    **Branching:** Use `branch_id` and `branch_from_thought` to explore
    alternative approaches side by side. Good with focus="approach_selection".

    **Revising:** Set `is_revision=True` and `revises_thought=N` to correct
    an earlier thought when new information contradicts it.

    **Legacy mode:** Omit `thought` to get just the focus prompt (backward
    compatible with older callers).

    Args:
        thought: Your reasoning for this step. Omit for legacy prompt-only mode.
        thought_number: Current step number (1-indexed).
        total_thoughts: Estimated total steps. Auto-adjusts upward if exceeded.
        next_thought_needed: True to continue reasoning, False when done.
        focus: Reflection lens — each returns a guiding prompt alongside your thought:
            - "collected_information": Is the gathered info sufficient and relevant?
            - "task_adherence": Still on track with the original goal?
            - "whether_done": Truly complete and communicated?
            - "approach_selection": Weighing alternatives — which approach fits best?
        is_revision: True if this thought revises an earlier one.
        revises_thought: Which thought number is being revised.
        branch_id: Name for an alternative exploration branch.
        branch_from_thought: Which thought number the branch diverges from.
        chain_id: Links thoughts across calls. Auto-generated on first call.

    Returns:
        Chain metadata: chain_id, thought_number, total_thoughts,
        next_thought_needed, focus_prompt, chain_length, branches, revisions
    """
    return _reflect_impl(
        thought=thought,
        thought_number=thought_number,
        total_thoughts=total_thoughts,
        next_thought_needed=next_thought_needed,
        focus=focus,
        is_revision=is_revision,
        revises_thought=revises_thought,
        branch_id=branch_id,
        branch_from_thought=branch_from_thought,
        chain_id=chain_id,
    )
