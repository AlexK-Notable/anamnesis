"""Tests for memory service and metacognition tools.

Tests cover:
- MemoryService CRUD operations
- Name validation and path traversal prevention
- Edge cases (empty, special chars, large content)
- Metacognition tool prompt output
- MCP tool _impl functions
"""

import pytest

from anamnesis.services.memory_service import (
    MemoryInfo,
    MemoryListEntry,
    MemoryService,
)


# =============================================================================
# MemoryService Unit Tests
# =============================================================================


class TestMemoryServiceWrite:
    """Tests for MemoryService.write_memory."""

    def test_write_creates_file(self, tmp_path):
        """Writing a memory creates a markdown file."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("test-note", "# Hello\nSome content.")

        assert isinstance(result, MemoryInfo)
        assert result.name == "test-note"
        assert result.content == "# Hello\nSome content."
        assert result.size_bytes > 0

        # Verify file exists on disk
        path = tmp_path / ".anamnesis" / "memories" / "test-note.md"
        assert path.exists()
        assert path.read_text() == "# Hello\nSome content."

    def test_write_creates_directories(self, tmp_path):
        """Writing creates the .anamnesis/memories directory."""
        service = MemoryService(str(tmp_path))
        service.write_memory("first", "content")

        assert (tmp_path / ".anamnesis" / "memories").is_dir()

    def test_write_overwrites_existing(self, tmp_path):
        """Writing to existing memory overwrites content."""
        service = MemoryService(str(tmp_path))
        service.write_memory("note", "version 1")
        result = service.write_memory("note", "version 2")

        assert result.content == "version 2"
        path = tmp_path / ".anamnesis" / "memories" / "note.md"
        assert path.read_text() == "version 2"

    def test_write_strips_md_extension(self, tmp_path):
        """Writing with .md extension strips it."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("note.md", "content")

        assert result.name == "note"
        path = tmp_path / ".anamnesis" / "memories" / "note.md"
        assert path.exists()

    def test_write_unicode_content(self, tmp_path):
        """Writing unicode content works."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("unicode", "日本語テスト 🎉")

        assert result.content == "日本語テスト 🎉"

    def test_write_empty_content(self, tmp_path):
        """Writing empty content is allowed."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("empty", "")

        assert result.content == ""

    def test_write_returns_timestamps(self, tmp_path):
        """Written memory has timestamps."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("timed", "content")

        assert result.created_at is not None
        assert result.updated_at is not None


class TestMemoryServiceRead:
    """Tests for MemoryService.read_memory."""

    def test_read_existing_memory(self, tmp_path):
        """Reading an existing memory returns content."""
        service = MemoryService(str(tmp_path))
        service.write_memory("readable", "the content")

        result = service.read_memory("readable")
        assert result is not None
        assert result.name == "readable"
        assert result.content == "the content"

    def test_read_nonexistent_returns_none(self, tmp_path):
        """Reading a nonexistent memory returns None."""
        service = MemoryService(str(tmp_path))
        result = service.read_memory("does-not-exist")

        assert result is None

    def test_read_with_md_extension(self, tmp_path):
        """Reading with .md extension works."""
        service = MemoryService(str(tmp_path))
        service.write_memory("note", "content")

        result = service.read_memory("note.md")
        assert result is not None
        assert result.content == "content"


class TestMemoryServiceList:
    """Tests for MemoryService.list_memories."""

    def test_list_empty(self, tmp_path):
        """Listing with no memories returns empty list."""
        service = MemoryService(str(tmp_path))
        result = service.list_memories()

        assert result == []

    def test_list_returns_entries(self, tmp_path):
        """Listing returns all memory entries."""
        service = MemoryService(str(tmp_path))
        service.write_memory("alpha", "first")
        service.write_memory("beta", "second")

        result = service.list_memories()
        assert len(result) == 2
        names = {e.name for e in result}
        assert names == {"alpha", "beta"}

    def test_list_entries_are_summaries(self, tmp_path):
        """List entries are MemoryListEntry without full content."""
        service = MemoryService(str(tmp_path))
        service.write_memory("note", "some content here")

        result = service.list_memories()
        assert len(result) == 1
        assert isinstance(result[0], MemoryListEntry)
        assert result[0].name == "note"
        assert result[0].size_bytes > 0

    def test_list_no_memories_dir(self, tmp_path):
        """Listing when memories dir doesn't exist returns empty."""
        service = MemoryService(str(tmp_path))
        # Don't write anything — dir doesn't exist
        result = service.list_memories()
        assert result == []


class TestMemoryServiceDelete:
    """Tests for MemoryService.delete_memory."""

    def test_delete_existing(self, tmp_path):
        """Deleting an existing memory returns True."""
        service = MemoryService(str(tmp_path))
        service.write_memory("deleteme", "bye")

        assert service.delete_memory("deleteme") is True
        # Verify gone from disk
        path = tmp_path / ".anamnesis" / "memories" / "deleteme.md"
        assert not path.exists()

    def test_delete_nonexistent(self, tmp_path):
        """Deleting a nonexistent memory returns False."""
        service = MemoryService(str(tmp_path))
        assert service.delete_memory("nope") is False

    def test_delete_then_read(self, tmp_path):
        """Deleted memory is no longer readable."""
        service = MemoryService(str(tmp_path))
        service.write_memory("temp", "temporary")
        service.delete_memory("temp")

        assert service.read_memory("temp") is None


class TestMemoryServiceEdit:
    """Tests for MemoryService.edit_memory."""

    def test_edit_replaces_text(self, tmp_path):
        """Editing replaces the specified text."""
        service = MemoryService(str(tmp_path))
        service.write_memory("editable", "Hello World!")

        result = service.edit_memory("editable", "World", "Universe")
        assert result is not None
        assert result.content == "Hello Universe!"

    def test_edit_nonexistent_returns_none(self, tmp_path):
        """Editing a nonexistent memory returns None."""
        service = MemoryService(str(tmp_path))
        result = service.edit_memory("nope", "old", "new")

        assert result is None

    def test_edit_text_not_found_raises(self, tmp_path):
        """Editing with text not in content raises ValueError."""
        service = MemoryService(str(tmp_path))
        service.write_memory("note", "Hello World")

        with pytest.raises(ValueError, match="not found"):
            service.edit_memory("note", "Goodbye", "Hi")

    def test_edit_replaces_first_occurrence(self, tmp_path):
        """Edit replaces only the first occurrence."""
        service = MemoryService(str(tmp_path))
        service.write_memory("repeated", "foo bar foo baz")

        result = service.edit_memory("repeated", "foo", "qux")
        assert result is not None
        assert result.content == "qux bar foo baz"

    def test_edit_persists_to_disk(self, tmp_path):
        """Edited content is persisted to disk."""
        service = MemoryService(str(tmp_path))
        service.write_memory("persist", "original content")
        service.edit_memory("persist", "original", "modified")

        path = tmp_path / ".anamnesis" / "memories" / "persist.md"
        assert path.read_text() == "modified content"


class TestMemoryServiceNameValidation:
    """Tests for name validation and path traversal prevention."""

    def test_empty_name_raises(self, tmp_path):
        """Empty name raises ValueError."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            service.write_memory("", "content")

    def test_whitespace_name_raises(self, tmp_path):
        """Whitespace-only name raises ValueError."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            service.write_memory("   ", "content")

    def test_path_separator_rejected(self, tmp_path):
        """Names with path separators are rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid"):
            service.write_memory("../etc/passwd", "evil")

    def test_slash_rejected(self, tmp_path):
        """Names with slashes are rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid"):
            service.write_memory("sub/dir", "content")

    def test_valid_characters_accepted(self, tmp_path):
        """Valid characters (alphanumeric, hyphens, underscores, dots) work."""
        service = MemoryService(str(tmp_path))

        for name in ["simple", "with-hyphens", "with_underscores", "v1.2.3", "CamelCase"]:
            result = service.write_memory(name, "content")
            assert result.name == name

    def test_starts_with_dot_rejected(self, tmp_path):
        """Names starting with a dot are rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid"):
            service.write_memory(".hidden", "content")

    def test_starts_with_hyphen_rejected(self, tmp_path):
        """Names starting with a hyphen are rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid"):
            service.write_memory("-flag", "content")

    def test_very_long_name_rejected(self, tmp_path):
        """Names over 200 characters are rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="too long"):
            service.write_memory("a" * 201, "content")

    def test_max_length_name_accepted(self, tmp_path):
        """Name at exactly 200 characters is accepted."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("a" * 200, "content")
        assert len(result.name) == 200


class TestMemoryServiceSerialization:
    """Tests for to_dict serialization."""

    def test_memory_info_to_dict(self, tmp_path):
        """MemoryInfo.to_dict returns expected structure."""
        service = MemoryService(str(tmp_path))
        result = service.write_memory("test", "content")

        d = result.to_dict()
        assert d["name"] == "test"
        assert d["content"] == "content"
        assert d["size_bytes"] > 0
        assert "created_at" in d
        assert "updated_at" in d

    def test_memory_list_entry_to_dict(self, tmp_path):
        """MemoryListEntry.to_dict returns expected structure."""
        service = MemoryService(str(tmp_path))
        service.write_memory("test", "content")

        entries = service.list_memories()
        d = entries[0].to_dict()
        assert d["name"] == "test"
        assert "content" not in d
        assert d["size_bytes"] > 0


class TestMemoryServiceContentLimits:
    """Tests for content size limits."""

    def test_large_content_rejected(self, tmp_path):
        """Content over 1MB is rejected."""
        service = MemoryService(str(tmp_path))
        with pytest.raises(ValueError, match="too large"):
            service.write_memory("big", "x" * 1_000_001)

    def test_max_content_accepted(self, tmp_path):
        """Content at exactly 1MB is accepted."""
        service = MemoryService(str(tmp_path))
        # 1M characters in ASCII = 1M bytes
        result = service.write_memory("max", "x" * 1_000_000)
        assert result.size_bytes == 1_000_000


# =============================================================================
# Metacognition Tool Tests
# =============================================================================


class TestMetacognitionTools:
    """Tests for reflect tool — legacy prompt mode and sequential thinking."""

    # --- Legacy mode (backward compatibility) ---

    def test_reflect_collected_information(self):
        """Legacy: reflect(focus='collected_information') returns a prompt."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(focus="collected_information")
        assert result["success"] is True
        assert "prompt" in result["data"]
        assert "Completeness" in result["data"]["prompt"]
        assert "Relevance" in result["data"]["prompt"]
        assert "Confidence" in result["data"]["prompt"]

    def test_reflect_task_adherence(self):
        """Legacy: reflect(focus='task_adherence') returns a prompt."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(focus="task_adherence")
        assert result["success"] is True
        assert "prompt" in result["data"]
        assert "Original goal" in result["data"]["prompt"]
        assert "Scope" in result["data"]["prompt"]
        assert "Progress" in result["data"]["prompt"]

    def test_reflect_whether_done(self):
        """Legacy: reflect(focus='whether_done') returns a prompt."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(focus="whether_done")
        assert result["success"] is True
        assert "prompt" in result["data"]
        assert "Completeness" in result["data"]["prompt"]
        assert "Quality" in result["data"]["prompt"]
        assert "Communication" in result["data"]["prompt"]

    def test_empty_thought_legacy_mode(self):
        """Legacy: thought='' triggers legacy path (no chain storage)."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(thought="", focus="task_adherence")
        assert result["success"] is True
        # Legacy response has 'prompt', not 'chain_id'
        assert "prompt" in result["data"]
        assert "chain_id" not in result["data"]

    # --- Sequential thinking mode ---

    def test_sequential_chain_creation(self):
        """First thought creates chain and returns chain_id."""
        from anamnesis.mcp_server._shared import _thought_chains
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(
            thought="Exploring the auth module structure.",
            thought_number=1,
            total_thoughts=3,
            next_thought_needed=True,
        )
        assert result["success"] is True
        data = result["data"]
        assert data["chain_id"].startswith("chain_")
        assert data["thought_number"] == 1
        assert data["total_thoughts"] == 3
        assert data["next_thought_needed"] is True
        assert data["chain_length"] == 1
        # Chain stored in module state
        assert data["chain_id"] in _thought_chains

    def test_sequential_chain_accumulation(self):
        """Multiple thoughts accumulate in the same chain."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Step 1: identify entry points.",
            thought_number=1, total_thoughts=3, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]

        r2 = _reflect_impl(
            thought="Step 2: trace call paths.",
            thought_number=2, total_thoughts=3, next_thought_needed=True,
            chain_id=chain_id,
        )
        assert r2["data"]["chain_length"] == 2

        r3 = _reflect_impl(
            thought="Step 3: conclusion reached.",
            thought_number=3, total_thoughts=3, next_thought_needed=False,
            chain_id=chain_id,
        )
        assert r3["data"]["chain_length"] == 3
        assert r3["data"]["next_thought_needed"] is False

    def test_chain_id_persistence(self):
        """Same chain_id links thoughts across calls."""
        from anamnesis.mcp_server._shared import _thought_chains
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="First thought.",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]

        r2 = _reflect_impl(
            thought="Second thought.",
            thought_number=2, total_thoughts=2, next_thought_needed=False,
            chain_id=chain_id,
        )
        assert r2["data"]["chain_id"] == chain_id
        assert len(_thought_chains[chain_id]) == 2

    def test_auto_chain_id_generation(self):
        """Omitting chain_id generates one automatically."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(
            thought="A standalone thought.",
            thought_number=1, total_thoughts=1, next_thought_needed=False,
        )
        chain_id = result["data"]["chain_id"]
        assert chain_id.startswith("chain_")
        assert len(chain_id) == len("chain_") + 12  # token_hex(6) = 12 chars

    def test_total_thoughts_adjustment(self):
        """Exceeding total_thoughts auto-adjusts upward."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(
            thought="Thought beyond original estimate.",
            thought_number=5, total_thoughts=3, next_thought_needed=True,
        )
        assert result["data"]["total_thoughts"] == 5

    def test_next_thought_needed_loop(self):
        """next_thought_needed is returned correctly for True and False."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r_true = _reflect_impl(
            thought="Continuing...",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
        )
        assert r_true["data"]["next_thought_needed"] is True

        r_false = _reflect_impl(
            thought="Done.",
            thought_number=2, total_thoughts=2, next_thought_needed=False,
            chain_id=r_true["data"]["chain_id"],
        )
        assert r_false["data"]["next_thought_needed"] is False

    def test_revision_tracking(self):
        """Revision is linked to original thought in response."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Initial analysis.",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]

        r2 = _reflect_impl(
            thought="Corrected analysis — earlier conclusion was wrong.",
            thought_number=2, total_thoughts=2, next_thought_needed=False,
            is_revision=True, revises_thought=1,
            chain_id=chain_id,
        )
        assert r2["data"]["revisions"] == [1]

    def test_branch_creation(self):
        """branch_id creates a branch tracked in response."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Main line of reasoning.",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]
        assert r1["data"]["branches"] == []

        r2 = _reflect_impl(
            thought="Alternative: what if we use strategy B?",
            thought_number=2, total_thoughts=3, next_thought_needed=True,
            branch_id="strategy-b", branch_from_thought=1,
            chain_id=chain_id,
        )
        assert "strategy-b" in r2["data"]["branches"]

    def test_branch_from_thought(self):
        """Branching from a specific thought records branch_id on the record."""
        from anamnesis.mcp_server._shared import _thought_chains
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Base thought.",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]

        _reflect_impl(
            thought="Branched exploration.",
            thought_number=2, total_thoughts=3, next_thought_needed=True,
            branch_id="alt-a", branch_from_thought=1,
            chain_id=chain_id,
        )
        # Verify the stored record has branch_id
        records = _thought_chains[chain_id]
        assert records[1]["branch_id"] == "alt-a"

    def test_multiple_branches(self):
        """Multiple branches are all tracked in the response."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Start.",
            thought_number=1, total_thoughts=4, next_thought_needed=True,
        )
        chain_id = r1["data"]["chain_id"]

        _reflect_impl(
            thought="Branch A.",
            thought_number=2, total_thoughts=4, next_thought_needed=True,
            branch_id="branch-a", chain_id=chain_id,
        )
        r3 = _reflect_impl(
            thought="Branch B.",
            thought_number=3, total_thoughts=4, next_thought_needed=True,
            branch_id="branch-b", chain_id=chain_id,
        )
        assert sorted(r3["data"]["branches"]) == ["branch-a", "branch-b"]

    def test_focus_prompt_included(self):
        """Sequential mode includes focus_prompt alongside chain metadata."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(
            thought="Checking if I have enough info.",
            thought_number=1, total_thoughts=1, next_thought_needed=False,
            focus="collected_information",
        )
        assert "focus_prompt" in result["data"]
        assert "Completeness" in result["data"]["focus_prompt"]

    def test_approach_selection_focus(self):
        """New approach_selection focus type works in both modes."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        # Legacy mode
        legacy = _reflect_impl(focus="approach_selection")
        assert legacy["success"] is True
        assert "Alternatives" in legacy["data"]["prompt"]

        # Sequential mode
        seq = _reflect_impl(
            thought="Weighing Redis vs in-memory cache.",
            thought_number=1, total_thoughts=2, next_thought_needed=True,
            focus="approach_selection",
        )
        assert seq["success"] is True
        assert "Alternatives" in seq["data"]["focus_prompt"]

    def test_invalid_focus_with_thought(self):
        """Invalid focus returns error even with a thought provided."""
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        result = _reflect_impl(
            thought="Some reasoning.",
            thought_number=1, total_thoughts=1, next_thought_needed=False,
            focus="nonexistent_focus",
        )
        assert result["success"] is False
        assert "Unknown focus" in result["error"]

    def test_chain_isolation(self):
        """Different chain_ids don't interfere with each other."""
        from anamnesis.mcp_server._shared import _thought_chains
        from anamnesis.mcp_server.tools.memory import _reflect_impl

        r1 = _reflect_impl(
            thought="Chain A thought.",
            thought_number=1, total_thoughts=1, next_thought_needed=False,
        )
        r2 = _reflect_impl(
            thought="Chain B thought.",
            thought_number=1, total_thoughts=1, next_thought_needed=False,
        )
        chain_a = r1["data"]["chain_id"]
        chain_b = r2["data"]["chain_id"]
        assert chain_a != chain_b
        assert len(_thought_chains[chain_a]) == 1
        assert len(_thought_chains[chain_b]) == 1


# =============================================================================
# Memory MCP Tool _impl Tests
# =============================================================================


@pytest.fixture(autouse=True)
def reset_server_state(tmp_path):
    """Reset server global state and use tmp_path for memory storage."""
    import anamnesis.mcp_server._shared as shared_module

    # Reset registry and activate tmp_path as the project
    shared_module._registry.reset()
    # Create the tmp_path as a valid project directory
    shared_module._registry.activate(str(tmp_path))

    yield

    # Clean up
    shared_module._registry.reset()
    shared_module._thought_chains.clear()


class TestWriteMemoryTool:
    """Tests for manage_memories(action='write') implementation."""

    def test_write_returns_success(self):
        """manage_memories write returns success with memory details."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="write", name="test-note", content="# Test\nContent here.")
        assert result["success"] is True
        assert result["data"]["name"] == "test-note"
        assert result["data"]["content"] == "# Test\nContent here."

    def test_write_invalid_name_returns_error(self):
        """manage_memories write with invalid name returns error."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="write", name="../evil", content="content")
        assert result["success"] is False


class TestReadMemoryTool:
    """Tests for manage_memories(action='read') implementation."""

    def test_read_existing(self):
        """manage_memories read returns content for existing memory."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        _manage_memories_impl(action="write", name="readable", content="the content")
        result = _manage_memories_impl(action="read", name="readable")
        assert result["success"] is True
        assert result["data"]["content"] == "the content"

    def test_read_nonexistent(self):
        """manage_memories read returns error for nonexistent memory."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="read", name="nope")
        assert result["success"] is False
        assert "not found" in result["error"]


class TestListMemoriesTool:
    """Tests for manage_memories(action='search') list-all mode."""

    def test_list_empty(self):
        """manage_memories search with no query returns empty when no memories exist."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="search")
        assert result["success"] is True
        assert result["metadata"]["total"] == 0
        assert result["data"] == []

    def test_list_after_writes(self):
        """manage_memories search with no query returns entries after writing."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        _manage_memories_impl(action="write", name="alpha", content="first")
        _manage_memories_impl(action="write", name="beta", content="second")

        result = _manage_memories_impl(action="search")
        assert result["success"] is True
        assert result["metadata"]["total"] == 2


class TestDeleteMemoryTool:
    """Tests for manage_memories(action='delete') implementation."""

    def test_delete_existing(self):
        """manage_memories delete returns success for existing memory."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        _manage_memories_impl(action="write", name="temp", content="temporary")
        result = _manage_memories_impl(action="delete", name="temp")
        assert result["success"] is True
        assert result["data"]["deleted"] == "temp"

    def test_delete_nonexistent(self):
        """manage_memories delete returns error for nonexistent memory."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="delete", name="nope")
        assert result["success"] is False


class TestEditMemoryTool:
    """Tests for manage_memories(action='edit') implementation."""

    def test_edit_returns_updated(self):
        """manage_memories edit returns updated content."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        _manage_memories_impl(action="write", name="editable", content="Hello World!")
        result = _manage_memories_impl(action="edit", name="editable", old_text="World", new_text="Universe")
        assert result["success"] is True
        assert result["data"]["content"] == "Hello Universe!"

    def test_edit_nonexistent(self):
        """manage_memories edit returns error for nonexistent memory."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        result = _manage_memories_impl(action="edit", name="nope", old_text="old", new_text="new")
        assert result["success"] is False

    def test_edit_text_not_found(self):
        """manage_memories edit returns error when text not found."""
        from anamnesis.mcp_server.tools.memory import _manage_memories_impl

        _manage_memories_impl(action="write", name="note", content="Hello World")
        result = _manage_memories_impl(action="edit", name="note", old_text="Goodbye", new_text="Hi")
        assert result["success"] is False


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests that all new tools are registered with the MCP server."""

    def test_metacognition_tools_registered(self):
        """Consolidated reflect tool is registered."""
        from anamnesis.mcp_server._shared import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        assert "reflect" in tool_names

    def test_memory_tools_registered(self):
        """Consolidated manage_memories tool is registered."""
        from anamnesis.mcp_server._shared import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        assert "manage_memories" in tool_names

    def test_total_tool_count(self):
        """Server has expected total tool count after legacy alias removal."""
        from anamnesis.mcp_server._shared import mcp

        tool_count = len(mcp._tool_manager._tools)
        # 29 original - 5 legacy memory - 3 legacy session + manage_memories + manage_sessions = 23
        assert tool_count == 23
