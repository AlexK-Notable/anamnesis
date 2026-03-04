"""Tests for memory _impl functions — full CRUD + search + reflect.

These test the actual implementation layer, not the MCP tool wrappers.
Uses reset_server_state to provide a clean project context with active
MemoryService.
"""

import pytest

from anamnesis.mcp_server.tools.memory import (
    _manage_memories_impl,
    _reflect_impl,
)

from .conftest import _as_dict


@pytest.fixture(autouse=True)
def _activate_project(reset_server_state):
    """Every test gets a fresh project context."""


# =============================================================================
# Write
# =============================================================================


class TestWriteMemory:
    """Tests for manage_memories(action='write')."""

    def test_write_returns_success(self):
        result = _as_dict(_manage_memories_impl(action="write", name="test-mem", content="hello world"))
        assert result["success"] is True
        assert result["data"]["name"] == "test-mem"

    def test_write_stores_content(self):
        _manage_memories_impl(action="write", name="arch-notes", content="# Architecture\nService layer pattern")
        result = _as_dict(_manage_memories_impl(action="read", name="arch-notes"))
        assert result["success"] is True
        assert "Service layer pattern" in result["data"]["content"]


# =============================================================================
# Read
# =============================================================================


class TestReadMemory:
    """Tests for manage_memories(action='read')."""

    def test_read_existing(self):
        _manage_memories_impl(action="write", name="read-test", content="content here")
        result = _as_dict(_manage_memories_impl(action="read", name="read-test"))
        assert result["success"] is True
        assert result["data"]["content"] == "content here"

    def test_read_nonexistent_returns_failure(self):
        result = _as_dict(_manage_memories_impl(action="read", name="does-not-exist"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# List
# =============================================================================


class TestListMemories:
    """Tests for manage_memories(action='search', query=None) (list-all mode)."""

    def test_empty_project_returns_empty_list(self):
        result = _as_dict(_manage_memories_impl(action="search"))
        assert result["success"] is True
        assert result["metadata"]["total"] == 0
        assert result["data"] == []

    def test_lists_written_memories(self):
        _manage_memories_impl(action="write", name="mem-a", content="content a")
        _manage_memories_impl(action="write", name="mem-b", content="content b")
        result = _as_dict(_manage_memories_impl(action="search"))
        assert result["success"] is True
        assert result["metadata"]["total"] == 2
        names = {m["name"] for m in result["data"]}
        assert names == {"mem-a", "mem-b"}


# =============================================================================
# Delete
# =============================================================================


class TestDeleteMemory:
    """Tests for manage_memories(action='delete')."""

    def test_delete_existing(self):
        _manage_memories_impl(action="write", name="to-delete", content="ephemeral")
        result = _as_dict(_manage_memories_impl(action="delete", name="to-delete"))
        assert result["success"] is True
        assert result["data"]["deleted"] == "to-delete"

        # Verify gone
        read_result = _as_dict(_manage_memories_impl(action="read", name="to-delete"))
        assert read_result["success"] is False

    def test_delete_nonexistent_returns_failure(self):
        result = _as_dict(_manage_memories_impl(action="delete", name="ghost"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Edit
# =============================================================================


class TestEditMemory:
    """Tests for manage_memories(action='edit')."""

    def test_edit_replaces_text(self):
        _manage_memories_impl(action="write", name="editable", content="hello world")
        result = _as_dict(_manage_memories_impl(action="edit", name="editable", old_text="world", new_text="universe"))
        assert result["success"] is True
        assert "universe" in result["data"]["content"]
        assert "world" not in result["data"]["content"]

    def test_edit_nonexistent_returns_failure(self):
        result = _as_dict(_manage_memories_impl(action="edit", name="nope", old_text="a", new_text="b"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Search
# =============================================================================


class TestSearchMemories:
    """Tests for manage_memories(action='search')."""

    def test_search_returns_structure(self):
        _manage_memories_impl(action="write", name="auth-patterns", content="JWT token authentication flow")
        result = _as_dict(_manage_memories_impl(action="search", query="authentication"))
        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert result["metadata"]["query"] == "authentication"
        assert isinstance(result["metadata"]["total"], int)

    def test_search_finds_relevant_memory(self):
        _manage_memories_impl(action="write", name="db-notes", content="PostgreSQL connection pooling")
        _manage_memories_impl(action="write", name="api-notes", content="REST endpoint design patterns")
        result = _as_dict(_manage_memories_impl(action="search", query="database"))
        assert result["success"] is True
        # Should find at least the db-notes memory
        assert result["metadata"]["total"] >= 1

    def test_search_empty_project(self):
        result = _as_dict(_manage_memories_impl(action="search", query="anything"))
        assert result["success"] is True
        assert result["metadata"]["total"] == 0

    def test_search_respects_limit(self):
        for i in range(5):
            _manage_memories_impl(action="write", name=f"note-{i}", content=f"content about topic {i}")
        result = _as_dict(_manage_memories_impl(action="search", query="topic", limit=2))
        assert result["success"] is True
        assert result["metadata"]["total"] <= 2


# =============================================================================
# Reflect (metacognition)
# =============================================================================


class TestReflect:
    """Tests for _reflect_impl."""

    def test_collected_information(self):
        result = _reflect_impl(focus="collected_information")
        assert result["success"] is True
        assert result["data"]["focus"] == "collected_information"
        assert len(result["data"]["prompt"]) > 0

    def test_task_adherence(self):
        result = _reflect_impl(focus="task_adherence")
        assert result["success"] is True
        assert result["data"]["focus"] == "task_adherence"

    def test_whether_done(self):
        result = _reflect_impl(focus="whether_done")
        assert result["success"] is True
        assert result["data"]["focus"] == "whether_done"

    def test_unknown_focus_returns_failure(self):
        result = _reflect_impl(focus="invalid_focus")
        assert result["success"] is False
        assert "unknown" in result["error"].lower()


# =============================================================================
# Full CRUD Lifecycle
# =============================================================================


class TestMemoryCRUDLifecycle:
    """End-to-end test exercising the complete memory lifecycle.

    Verifies: write -> read -> edit -> read(updated) -> search -> delete -> read(gone).
    """

    def test_memory_crud_lifecycle(self):
        """Full create-read-update-search-delete lifecycle for a memory."""
        # Step 1: Write a new memory
        write_result = _as_dict(_manage_memories_impl(action="write", name="lifecycle-test", content="initial content about databases"))
        assert write_result["success"] is True
        assert write_result["data"]["name"] == "lifecycle-test"

        # Step 2: Read it back and verify content matches
        read_result = _as_dict(_manage_memories_impl(action="read", name="lifecycle-test"))
        assert read_result["success"] is True
        assert read_result["data"]["content"] == "initial content about databases"

        # Step 3: Edit the memory (replace a substring)
        edit_result = _as_dict(_manage_memories_impl(action="edit", name="lifecycle-test", old_text="databases", new_text="microservices"))
        assert edit_result["success"] is True
        assert "microservices" in edit_result["data"]["content"]
        assert "databases" not in edit_result["data"]["content"]

        # Step 4: Read again to confirm the edit persisted
        read_updated = _as_dict(_manage_memories_impl(action="read", name="lifecycle-test"))
        assert read_updated["success"] is True
        assert "microservices" in read_updated["data"]["content"]

        # Step 5: Search for the memory by its content
        search_result = _as_dict(_manage_memories_impl(action="search", query="microservices"))
        assert search_result["success"] is True
        assert search_result["metadata"]["total"] >= 1
        found_names = [m["name"] for m in search_result["data"]]
        assert "lifecycle-test" in found_names

        # Step 6: Delete the memory
        delete_result = _as_dict(_manage_memories_impl(action="delete", name="lifecycle-test"))
        assert delete_result["success"] is True
        assert delete_result["data"]["deleted"] == "lifecycle-test"

        # Step 7: Read again to confirm it is gone
        read_gone = _as_dict(_manage_memories_impl(action="read", name="lifecycle-test"))
        assert read_gone["success"] is False
        assert "not found" in read_gone["error"].lower()


# =============================================================================
# Consolidated manage_memories
# =============================================================================


class TestManageMemories:
    """Tests for the consolidated manage_memories tool."""

    def test_write_via_manage(self):
        result = _as_dict(_manage_memories_impl(action="write", name="mm-test", content="hello"))
        assert result["success"] is True
        assert result["data"]["name"] == "mm-test"

    def test_read_via_manage(self):
        _manage_memories_impl(action="write", name="mm-read", content="read me")
        result = _as_dict(_manage_memories_impl(action="read", name="mm-read"))
        assert result["success"] is True
        assert result["data"]["content"] == "read me"

    def test_read_nonexistent_via_manage(self):
        result = _as_dict(_manage_memories_impl(action="read", name="nope"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_edit_via_manage(self):
        _manage_memories_impl(action="write", name="mm-edit", content="old text")
        result = _as_dict(
            _manage_memories_impl(
                action="edit", name="mm-edit", old_text="old", new_text="new"
            )
        )
        assert result["success"] is True
        assert "new text" in result["data"]["content"]

    def test_delete_via_manage(self):
        _manage_memories_impl(action="write", name="mm-del", content="temp")
        result = _as_dict(_manage_memories_impl(action="delete", name="mm-del"))
        assert result["success"] is True
        assert result["data"]["deleted"] == "mm-del"

    def test_search_via_manage(self):
        _manage_memories_impl(action="write", name="mm-search", content="database patterns")
        result = _as_dict(_manage_memories_impl(action="search", query="database"))
        assert result["success"] is True
        assert result["metadata"]["total"] >= 1

    def test_list_all_via_manage(self):
        _manage_memories_impl(action="write", name="mm-a", content="a")
        _manage_memories_impl(action="write", name="mm-b", content="b")
        result = _as_dict(_manage_memories_impl(action="search"))
        assert result["success"] is True
        assert result["metadata"]["total"] == 2

    def test_unknown_action(self):
        result = _as_dict(_manage_memories_impl(action="invalid"))
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_write_missing_name(self):
        result = _as_dict(_manage_memories_impl(action="write", content="no name"))
        assert result["success"] is False
        assert "name" in result["error"].lower()

    def test_write_missing_content(self):
        result = _as_dict(_manage_memories_impl(action="write", name="no-content"))
        assert result["success"] is False
        assert "content" in result["error"].lower()

    def test_full_crud_via_manage(self):
        """Full lifecycle through the consolidated tool."""
        # Write
        w = _as_dict(_manage_memories_impl(action="write", name="crud", content="initial"))
        assert w["success"] is True

        # Read
        r = _as_dict(_manage_memories_impl(action="read", name="crud"))
        assert r["data"]["content"] == "initial"

        # Edit
        e = _as_dict(
            _manage_memories_impl(
                action="edit", name="crud", old_text="initial", new_text="updated"
            )
        )
        assert "updated" in e["data"]["content"]

        # Search
        s = _as_dict(_manage_memories_impl(action="search", query="updated"))
        assert s["metadata"]["total"] >= 1

        # Delete
        d = _as_dict(_manage_memories_impl(action="delete", name="crud"))
        assert d["success"] is True

        # Verify gone
        gone = _as_dict(_manage_memories_impl(action="read", name="crud"))
        assert gone["success"] is False
