"""Tests for memory _impl functions — full CRUD + search + reflect.

These test the actual implementation layer, not the MCP tool wrappers.
Uses reset_server_state to provide a clean project context with active
MemoryService.
"""

import pytest

from anamnesis.mcp_server.tools.memory import (
    _delete_memory_impl,
    _edit_memory_impl,
    _list_memories_impl,
    _read_memory_impl,
    _reflect_impl,
    _search_memories_impl,
    _write_memory_impl,
)

from .conftest import _as_dict


@pytest.fixture(autouse=True)
def _activate_project(reset_server_state):
    """Every test gets a fresh project context."""


# =============================================================================
# Write
# =============================================================================


class TestWriteMemory:
    """Tests for _write_memory_impl."""

    def test_write_returns_success(self):
        result = _as_dict(_write_memory_impl("test-mem", "hello world"))
        assert result["success"] is True
        assert "memory" in result
        assert result["memory"]["name"] == "test-mem"

    def test_write_stores_content(self):
        _write_memory_impl("arch-notes", "# Architecture\nService layer pattern")
        result = _as_dict(_read_memory_impl("arch-notes"))
        assert result["success"] is True
        assert "Service layer pattern" in result["memory"]["content"]


# =============================================================================
# Read
# =============================================================================


class TestReadMemory:
    """Tests for _read_memory_impl."""

    def test_read_existing(self):
        _write_memory_impl("read-test", "content here")
        result = _as_dict(_read_memory_impl("read-test"))
        assert result["success"] is True
        assert result["memory"]["content"] == "content here"

    def test_read_nonexistent_returns_failure(self):
        result = _as_dict(_read_memory_impl("does-not-exist"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# List
# =============================================================================


class TestListMemories:
    """Tests for _list_memories_impl."""

    def test_empty_project_returns_empty_list(self):
        result = _as_dict(_list_memories_impl())
        assert result["success"] is True
        assert result["total"] == 0
        assert result["memories"] == []

    def test_lists_written_memories(self):
        _write_memory_impl("mem-a", "content a")
        _write_memory_impl("mem-b", "content b")
        result = _as_dict(_list_memories_impl())
        assert result["success"] is True
        assert result["total"] == 2
        names = {m["name"] for m in result["memories"]}
        assert names == {"mem-a", "mem-b"}


# =============================================================================
# Delete
# =============================================================================


class TestDeleteMemory:
    """Tests for _delete_memory_impl."""

    def test_delete_existing(self):
        _write_memory_impl("to-delete", "ephemeral")
        result = _as_dict(_delete_memory_impl("to-delete"))
        assert result["success"] is True
        assert result["deleted"] == "to-delete"

        # Verify gone
        read_result = _as_dict(_read_memory_impl("to-delete"))
        assert read_result["success"] is False

    def test_delete_nonexistent_returns_failure(self):
        result = _as_dict(_delete_memory_impl("ghost"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Edit
# =============================================================================


class TestEditMemory:
    """Tests for _edit_memory_impl."""

    def test_edit_replaces_text(self):
        _write_memory_impl("editable", "hello world")
        result = _as_dict(_edit_memory_impl("editable", "world", "universe"))
        assert result["success"] is True
        assert "universe" in result["memory"]["content"]
        assert "world" not in result["memory"]["content"]

    def test_edit_nonexistent_returns_failure(self):
        result = _as_dict(_edit_memory_impl("nope", "a", "b"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Search
# =============================================================================


class TestSearchMemories:
    """Tests for _search_memories_impl."""

    def test_search_returns_structure(self):
        _write_memory_impl("auth-patterns", "JWT token authentication flow")
        result = _as_dict(_search_memories_impl("authentication"))
        assert result["success"] is True
        assert "results" in result
        assert result["query"] == "authentication"
        assert isinstance(result["total"], int)

    def test_search_finds_relevant_memory(self):
        _write_memory_impl("db-notes", "PostgreSQL connection pooling")
        _write_memory_impl("api-notes", "REST endpoint design patterns")
        result = _as_dict(_search_memories_impl("database"))
        assert result["success"] is True
        # Should find at least the db-notes memory
        assert result["total"] >= 1

    def test_search_empty_project(self):
        result = _as_dict(_search_memories_impl("anything"))
        assert result["success"] is True
        assert result["total"] == 0

    def test_search_respects_limit(self):
        for i in range(5):
            _write_memory_impl(f"note-{i}", f"content about topic {i}")
        result = _as_dict(_search_memories_impl("topic", limit=2))
        assert result["success"] is True
        assert result["total"] <= 2


# =============================================================================
# Reflect (metacognition)
# =============================================================================


class TestReflect:
    """Tests for _reflect_impl."""

    def test_collected_information(self):
        result = _reflect_impl("collected_information")
        assert result["success"] is True
        assert result["focus"] == "collected_information"
        assert len(result["prompt"]) > 0

    def test_task_adherence(self):
        result = _reflect_impl("task_adherence")
        assert result["success"] is True
        assert result["focus"] == "task_adherence"

    def test_whether_done(self):
        result = _reflect_impl("whether_done")
        assert result["success"] is True
        assert result["focus"] == "whether_done"

    def test_unknown_focus_returns_failure(self):
        result = _reflect_impl("invalid_focus")
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
        write_result = _as_dict(_write_memory_impl("lifecycle-test", "initial content about databases"))
        assert write_result["success"] is True
        assert write_result["memory"]["name"] == "lifecycle-test"

        # Step 2: Read it back and verify content matches
        read_result = _as_dict(_read_memory_impl("lifecycle-test"))
        assert read_result["success"] is True
        assert read_result["memory"]["content"] == "initial content about databases"

        # Step 3: Edit the memory (replace a substring)
        edit_result = _as_dict(_edit_memory_impl("lifecycle-test", "databases", "microservices"))
        assert edit_result["success"] is True
        assert "microservices" in edit_result["memory"]["content"]
        assert "databases" not in edit_result["memory"]["content"]

        # Step 4: Read again to confirm the edit persisted
        read_updated = _as_dict(_read_memory_impl("lifecycle-test"))
        assert read_updated["success"] is True
        assert "microservices" in read_updated["memory"]["content"]

        # Step 5: Search for the memory by its content
        search_result = _as_dict(_search_memories_impl("microservices"))
        assert search_result["success"] is True
        assert search_result["total"] >= 1
        found_names = [m["name"] for m in search_result["results"]]
        assert "lifecycle-test" in found_names

        # Step 6: Delete the memory
        delete_result = _as_dict(_delete_memory_impl("lifecycle-test"))
        assert delete_result["success"] is True
        assert delete_result["deleted"] == "lifecycle-test"

        # Step 7: Read again to confirm it is gone
        read_gone = _as_dict(_read_memory_impl("lifecycle-test"))
        assert read_gone["success"] is False
        assert "not found" in read_gone["error"].lower()
