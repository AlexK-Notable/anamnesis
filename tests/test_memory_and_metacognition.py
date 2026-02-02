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
    """Tests for metacognition prompt-returning tools."""

    def test_think_collected_information(self):
        """think_about_collected_information returns a prompt."""
        from anamnesis.mcp_server.server import _think_about_collected_information_impl

        result = _think_about_collected_information_impl()
        assert result["success"] is True
        assert "prompt" in result
        assert "Completeness" in result["prompt"]
        assert "Relevance" in result["prompt"]
        assert "Confidence" in result["prompt"]

    def test_think_task_adherence(self):
        """think_about_task_adherence returns a prompt."""
        from anamnesis.mcp_server.server import _think_about_task_adherence_impl

        result = _think_about_task_adherence_impl()
        assert result["success"] is True
        assert "prompt" in result
        assert "Original goal" in result["prompt"]
        assert "Scope" in result["prompt"]
        assert "Progress" in result["prompt"]

    def test_think_whether_done(self):
        """think_about_whether_you_are_done returns a prompt."""
        from anamnesis.mcp_server.server import _think_about_whether_you_are_done_impl

        result = _think_about_whether_you_are_done_impl()
        assert result["success"] is True
        assert "prompt" in result
        assert "Completeness" in result["prompt"]
        assert "Quality" in result["prompt"]
        assert "Communication" in result["prompt"]


# =============================================================================
# Memory MCP Tool _impl Tests
# =============================================================================


@pytest.fixture(autouse=True)
def reset_server_state(tmp_path):
    """Reset server global state and use tmp_path for memory storage."""
    import anamnesis.mcp_server.server as server_module

    # Reset registry and activate tmp_path as the project
    server_module._registry.reset()
    # Create the tmp_path as a valid project directory
    server_module._registry.activate(str(tmp_path))

    yield

    # Clean up
    server_module._registry.reset()


class TestWriteMemoryTool:
    """Tests for write_memory MCP tool implementation."""

    def test_write_returns_success(self):
        """write_memory returns success with memory details."""
        from anamnesis.mcp_server.server import _write_memory_impl

        result = _write_memory_impl("test-note", "# Test\nContent here.")
        assert result["success"] is True
        assert result["memory"]["name"] == "test-note"
        assert result["memory"]["content"] == "# Test\nContent here."

    def test_write_invalid_name_returns_error(self):
        """write_memory with invalid name returns error."""
        from anamnesis.mcp_server.server import _write_memory_impl

        result = _write_memory_impl("../evil", "content")
        assert result["success"] is False


class TestReadMemoryTool:
    """Tests for read_memory MCP tool implementation."""

    def test_read_existing(self):
        """read_memory returns content for existing memory."""
        from anamnesis.mcp_server.server import (
            _read_memory_impl,
            _write_memory_impl,
        )

        _write_memory_impl("readable", "the content")
        result = _read_memory_impl("readable")
        assert result["success"] is True
        assert result["memory"]["content"] == "the content"

    def test_read_nonexistent(self):
        """read_memory returns error for nonexistent memory."""
        from anamnesis.mcp_server.server import _read_memory_impl

        result = _read_memory_impl("nope")
        assert result["success"] is False
        assert "not found" in result["error"]


class TestListMemoriesTool:
    """Tests for list_memories MCP tool implementation."""

    def test_list_empty(self):
        """list_memories returns empty when no memories exist."""
        from anamnesis.mcp_server.server import _list_memories_impl

        result = _list_memories_impl()
        assert result["success"] is True
        assert result["count"] == 0
        assert result["memories"] == []

    def test_list_after_writes(self):
        """list_memories returns entries after writing."""
        from anamnesis.mcp_server.server import (
            _list_memories_impl,
            _write_memory_impl,
        )

        _write_memory_impl("alpha", "first")
        _write_memory_impl("beta", "second")

        result = _list_memories_impl()
        assert result["success"] is True
        assert result["count"] == 2


class TestDeleteMemoryTool:
    """Tests for delete_memory MCP tool implementation."""

    def test_delete_existing(self):
        """delete_memory returns success for existing memory."""
        from anamnesis.mcp_server.server import (
            _delete_memory_impl,
            _write_memory_impl,
        )

        _write_memory_impl("temp", "temporary")
        result = _delete_memory_impl("temp")
        assert result["success"] is True
        assert result["deleted"] == "temp"

    def test_delete_nonexistent(self):
        """delete_memory returns error for nonexistent memory."""
        from anamnesis.mcp_server.server import _delete_memory_impl

        result = _delete_memory_impl("nope")
        assert result["success"] is False


class TestEditMemoryTool:
    """Tests for edit_memory MCP tool implementation."""

    def test_edit_returns_updated(self):
        """edit_memory returns updated content."""
        from anamnesis.mcp_server.server import (
            _edit_memory_impl,
            _write_memory_impl,
        )

        _write_memory_impl("editable", "Hello World!")
        result = _edit_memory_impl("editable", "World", "Universe")
        assert result["success"] is True
        assert result["memory"]["content"] == "Hello Universe!"

    def test_edit_nonexistent(self):
        """edit_memory returns error for nonexistent memory."""
        from anamnesis.mcp_server.server import _edit_memory_impl

        result = _edit_memory_impl("nope", "old", "new")
        assert result["success"] is False

    def test_edit_text_not_found(self):
        """edit_memory returns error when text not found."""
        from anamnesis.mcp_server.server import (
            _edit_memory_impl,
            _write_memory_impl,
        )

        _write_memory_impl("note", "Hello World")
        result = _edit_memory_impl("note", "Goodbye", "Hi")
        assert result["success"] is False


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests that all new tools are registered with the MCP server."""

    def test_metacognition_tools_registered(self):
        """All 3 metacognition tools are registered."""
        from anamnesis.mcp_server.server import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        assert "think_about_collected_information" in tool_names
        assert "think_about_task_adherence" in tool_names
        assert "think_about_whether_you_are_done" in tool_names

    def test_memory_tools_registered(self):
        """All 5 memory tools are registered."""
        from anamnesis.mcp_server.server import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        assert "write_memory" in tool_names
        assert "read_memory" in tool_names
        assert "list_memories" in tool_names
        assert "delete_memory" in tool_names
        assert "edit_memory" in tool_names

    def test_total_tool_count(self):
        """Server has expected total tool count (20 original + 12 new + 9 LSP = 41)."""
        from anamnesis.mcp_server.server import mcp

        tool_count = len(mcp._tool_manager._tools)
        # 20 original + 3 metacognition + 6 memory + 3 project mgmt + 9 LSP = 41
        assert tool_count == 41
