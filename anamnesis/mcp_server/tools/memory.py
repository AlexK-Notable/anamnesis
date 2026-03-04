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


@_with_error_handling("manage_memories")
def _manage_memories_impl(
    action: str = "search",
    name: str = "",
    content: str = "",
    old_text: str = "",
    new_text: str = "",
    query: str | None = None,
    limit: int = 5,
) -> dict:
    """Implementation for manage_memories tool.

    action="write": write a new memory (requires name, content).
    action="read": read a memory by name (requires name).
    action="edit": edit a memory by replacing text (requires name, old_text, new_text).
    action="delete": delete a memory (requires name).
    action="search": search memories by query, or list all if query is omitted.
    """
    memory_service = _get_memory_service()

    if action == "write":
        if not name:
            return _failure_response("'name' is required when action='write'")
        validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
        if not content:
            return _failure_response("'content' is required when action='write'")
        validate_string_length(content, "content", min_length=1, max_length=MAX_CONTENT_LENGTH)
        result = memory_service.write_memory(name, content)
        return _success_response(result.to_dict())
    elif action == "read":
        if not name:
            return _failure_response("'name' is required when action='read'")
        validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
        result = memory_service.read_memory(name)
        if result is None:
            return _failure_response(f"Memory '{name}' not found")
        return _success_response(result.to_dict())
    elif action == "edit":
        if not name:
            return _failure_response("'name' is required when action='edit'")
        validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
        if not old_text:
            return _failure_response("'old_text' is required when action='edit'")
        validate_string_length(old_text, "old_text", min_length=1, max_length=MAX_CONTENT_LENGTH)
        result = memory_service.edit_memory(name, old_text, new_text)
        if result is None:
            return _failure_response(f"Memory '{name}' not found")
        return _success_response(result.to_dict())
    elif action == "delete":
        if not name:
            return _failure_response("'name' is required when action='delete'")
        validate_string_length(name, "name", min_length=1, max_length=MAX_NAME_LENGTH)
        deleted = memory_service.delete_memory(name)
        if not deleted:
            return _failure_response(f"Memory '{name}' not found")
        return _success_response({"deleted": name})
    elif action == "search":
        if query is not None and query:
            validate_string_length(query, "query", min_length=1, max_length=MAX_QUERY_LENGTH)
        limit = clamp_integer(limit, "limit", 1, 100)
        if not query:
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
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: write, read, edit, delete, search"
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
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def manage_memories(
    action: Literal["write", "read", "edit", "delete", "search"] = "search",
    name: str = "",
    content: str = "",
    old_text: str = "",
    new_text: str = "",
    query: str | None = None,
    limit: int = 5,
) -> dict:
    """Manage persistent project memories — write, read, edit, delete, or search.

    Use action="write" to store project knowledge as a markdown file in
    `.anamnesis/memories/`. The memory name should be meaningful.

    Use action="read" to retrieve a specific memory by name.

    Use action="edit" to update part of a memory by replacing text.

    Use action="delete" to remove a memory that is no longer relevant.

    Use action="search" to find memories by query (embedding-based with
    substring fallback), or list all memories when query is omitted.

    Args:
        action: "write", "read", "edit", "delete", or "search"
        name: Memory name (required for write/read/edit/delete)
        content: Content to write (required for action="write")
        old_text: Text to find and replace (required for action="edit")
        new_text: Replacement text (for action="edit")
        query: Search query (for action="search"; omit to list all)
        limit: Maximum results (for action="search", default 5)

    Returns:
        Memory data (write/read/edit), deletion status, or search results
    """
    return _manage_memories_impl(action, name, content, old_text, new_text, query, limit)



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
