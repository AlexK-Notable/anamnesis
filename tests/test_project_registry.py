"""Tests for multi-project support via ProjectRegistry.

Covers:
- ProjectContext: lazy service init, isolation, serialization
- ProjectRegistry: activate, switch, deactivate, auto-activate cwd
- MCP tool integration: activate_project, get_current_config, list_projects
- Cross-project isolation: services don't leak between projects
- Backward compatibility: single-project usage auto-activates cwd
"""

import json
import os
from pathlib import Path

import pytest

from anamnesis.services.project_registry import ProjectContext, ProjectRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Fresh ProjectRegistry for each test (no disk I/O)."""
    return ProjectRegistry(persist_path=None)


@pytest.fixture
def project_a(tmp_path):
    """First project directory."""
    d = tmp_path / "project_a"
    d.mkdir()
    (d / "main.py").write_text('print("a")\n')
    return str(d)


@pytest.fixture
def project_b(tmp_path):
    """Second project directory."""
    d = tmp_path / "project_b"
    d.mkdir()
    (d / "main.py").write_text('print("b")\n')
    return str(d)


@pytest.fixture
def project_c(tmp_path):
    """Third project directory."""
    d = tmp_path / "project_c"
    d.mkdir()
    (d / "app.py").write_text('print("c")\n')
    return str(d)


# ---------------------------------------------------------------------------
# ProjectContext tests
# ---------------------------------------------------------------------------


class TestProjectContext:
    """Tests for ProjectContext dataclass."""

    def test_name_from_path(self, project_a):
        ctx = ProjectContext(path=project_a)
        assert ctx.name == "project_a"

    def test_services_initially_none(self, project_a):
        ctx = ProjectContext(path=project_a)
        assert ctx._learning_service is None
        assert ctx._intelligence_service is None
        assert ctx._codebase_service is None
        assert ctx._session_manager is None
        assert ctx._memory_service is None
        assert ctx._search_service is None

    def test_lazy_learning_service(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_learning_service()
        assert svc is not None
        # Second call returns same instance
        assert ctx.get_learning_service() is svc

    def test_lazy_intelligence_service(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_intelligence_service()
        assert svc is not None
        assert ctx.get_intelligence_service() is svc

    def test_lazy_codebase_service(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_codebase_service()
        assert svc is not None
        assert ctx.get_codebase_service() is svc

    def test_lazy_session_manager(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_session_manager()
        assert svc is not None
        assert ctx.get_session_manager() is svc

    def test_lazy_memory_service(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_memory_service()
        assert svc is not None
        assert ctx.get_memory_service() is svc

    def test_lazy_search_service(self, project_a):
        ctx = ProjectContext(path=project_a)
        svc = ctx.get_search_service()
        assert svc is not None
        assert ctx.get_search_service() is svc

    def test_to_dict_shows_service_status(self, project_a):
        ctx = ProjectContext(path=project_a)
        d = ctx.to_dict()

        # Before any access, all services are False
        assert all(v is False for v in d["services"].values())

        # After accessing one service, only that one is True
        ctx.get_learning_service()
        d = ctx.to_dict()
        assert d["services"]["learning"] is True
        assert d["services"]["intelligence"] is False

    def test_to_dict_includes_metadata(self, project_a):
        ctx = ProjectContext(path=project_a)
        d = ctx.to_dict()
        assert d["path"] == project_a
        assert d["name"] == "project_a"
        assert d["activated_at"] is None

    def test_reset_search_clears_state(self, project_a):
        ctx = ProjectContext(path=project_a)
        ctx.get_search_service()
        assert ctx._search_service is not None

        ctx.reset_search()
        assert ctx._search_service is None
        assert ctx._semantic_initialized is False


# ---------------------------------------------------------------------------
# ProjectRegistry tests
# ---------------------------------------------------------------------------


class TestProjectRegistryActivation:
    """Tests for project activation."""

    def test_activate_creates_context(self, registry, project_a):
        ctx = registry.activate(project_a)
        assert ctx.path == str(Path(project_a).resolve())
        assert ctx.activated_at is not None

    def test_activate_sets_active(self, registry, project_a):
        registry.activate(project_a)
        assert registry.active_path == str(Path(project_a).resolve())
        assert registry.active_project is not None

    def test_activate_same_path_returns_same_context(self, registry, project_a):
        ctx1 = registry.activate(project_a)
        ctx2 = registry.activate(project_a)
        assert ctx1 is ctx2

    def test_activate_preserves_services_on_reactivation(
        self, registry, project_a, project_b
    ):
        """Switching projects and back preserves cached services."""
        ctx_a = registry.activate(project_a)
        svc_a = ctx_a.get_learning_service()

        # Switch to B
        registry.activate(project_b)

        # Switch back to A
        ctx_a2 = registry.activate(project_a)
        assert ctx_a2.get_learning_service() is svc_a

    def test_activate_nonexistent_path_raises(self, registry):
        with pytest.raises(ValueError, match="does not exist"):
            registry.activate("/definitely/not/a/real/path")

    def test_activate_file_not_dir_raises(self, registry, project_a):
        file_path = os.path.join(project_a, "main.py")
        with pytest.raises(ValueError, match="not a directory"):
            registry.activate(file_path)

    def test_activate_resolves_symlinks(self, registry, project_a, tmp_path):
        """Symlinks resolve to the same project."""
        link = tmp_path / "link_to_a"
        link.symlink_to(project_a)

        ctx1 = registry.activate(project_a)
        ctx2 = registry.activate(str(link))
        assert ctx1 is ctx2


class TestProjectRegistrySwitching:
    """Tests for switching between projects."""

    def test_switch_changes_active(self, registry, project_a, project_b):
        registry.activate(project_a)
        assert registry.active_project.name == "project_a"

        registry.activate(project_b)
        assert registry.active_project.name == "project_b"

    def test_get_active_returns_current(self, registry, project_a):
        registry.activate(project_a)
        ctx = registry.get_active()
        assert ctx.path == str(Path(project_a).resolve())

    def test_get_active_auto_activates_cwd(self, registry):
        """When no project is active, auto-activates cwd."""
        ctx = registry.get_active()
        assert ctx.path == str(Path(os.getcwd()).resolve())

    def test_get_project_by_path(self, registry, project_a, project_b):
        registry.activate(project_a)
        registry.activate(project_b)

        ctx = registry.get_project(project_a)
        assert ctx is not None
        assert ctx.name == "project_a"

    def test_get_project_unknown_returns_none(self, registry):
        assert registry.get_project("/unknown/path") is None


class TestProjectRegistryListing:
    """Tests for listing projects."""

    def test_list_empty(self, registry):
        assert registry.list_projects() == []

    def test_list_returns_all(self, registry, project_a, project_b, project_c):
        registry.activate(project_a)
        registry.activate(project_b)
        registry.activate(project_c)

        projects = registry.list_projects()
        assert len(projects) == 3

    def test_list_sorted_by_activation_newest_first(
        self, registry, project_a, project_b
    ):
        registry.activate(project_a)
        registry.activate(project_b)

        projects = registry.list_projects()
        # project_b was activated last, so it should be first
        assert projects[0].name == "project_b"
        assert projects[1].name == "project_a"


class TestProjectRegistryDeactivation:
    """Tests for deactivating projects."""

    def test_deactivate_removes_project(self, registry, project_a):
        registry.activate(project_a)
        result = registry.deactivate(project_a)

        assert result is True
        assert registry.get_project(project_a) is None

    def test_deactivate_clears_active_if_current(self, registry, project_a):
        registry.activate(project_a)
        registry.deactivate(project_a)
        assert registry.active_path is None

    def test_deactivate_unknown_returns_false(self, registry):
        assert registry.deactivate("/unknown") is False

    def test_reset_clears_everything(self, registry, project_a, project_b):
        registry.activate(project_a)
        registry.activate(project_b)
        registry.reset()

        assert registry.active_path is None
        assert registry.list_projects() == []


class TestProjectRegistrySerialization:
    """Tests for registry serialization."""

    def test_to_dict_empty(self, registry):
        d = registry.to_dict()
        assert d["active_path"] is None
        assert d["project_count"] == 0
        assert d["projects"] == []

    def test_to_dict_with_projects(self, registry, project_a, project_b):
        registry.activate(project_a)
        registry.activate(project_b)

        d = registry.to_dict()
        assert d["project_count"] == 2
        assert d["active_project"] == "project_b"
        assert len(d["projects"]) == 2


# ---------------------------------------------------------------------------
# Service isolation tests
# ---------------------------------------------------------------------------


class TestServiceIsolation:
    """Tests that services are isolated between projects."""

    def test_learning_services_are_different(self, registry, project_a, project_b):
        ctx_a = registry.activate(project_a)
        svc_a = ctx_a.get_learning_service()

        ctx_b = registry.activate(project_b)
        svc_b = ctx_b.get_learning_service()

        assert svc_a is not svc_b

    def test_intelligence_services_are_different(
        self, registry, project_a, project_b
    ):
        ctx_a = registry.activate(project_a)
        svc_a = ctx_a.get_intelligence_service()

        ctx_b = registry.activate(project_b)
        svc_b = ctx_b.get_intelligence_service()

        assert svc_a is not svc_b

    def test_session_managers_are_different(self, registry, project_a, project_b):
        ctx_a = registry.activate(project_a)
        sm_a = ctx_a.get_session_manager()

        ctx_b = registry.activate(project_b)
        sm_b = ctx_b.get_session_manager()

        assert sm_a is not sm_b

    def test_memory_services_are_different(self, registry, project_a, project_b):
        ctx_a = registry.activate(project_a)
        ms_a = ctx_a.get_memory_service()

        ctx_b = registry.activate(project_b)
        ms_b = ctx_b.get_memory_service()

        assert ms_a is not ms_b

    def test_memory_data_isolated_between_projects(
        self, registry, project_a, project_b
    ):
        """Writing memory in project A doesn't affect project B."""
        ctx_a = registry.activate(project_a)
        ms_a = ctx_a.get_memory_service()
        ms_a.write_memory("test-note", "content for A")

        ctx_b = registry.activate(project_b)
        ms_b = ctx_b.get_memory_service()
        result = ms_b.read_memory("test-note")
        assert result is None  # B should not see A's memory

    def test_session_data_isolated_between_projects(
        self, registry, project_a, project_b
    ):
        """Sessions in project A don't appear in project B."""
        ctx_a = registry.activate(project_a)
        sm_a = ctx_a.get_session_manager()
        sm_a.start_session(name="Session in A")

        ctx_b = registry.activate(project_b)
        sm_b = ctx_b.get_session_manager()
        sessions = sm_b.get_recent_sessions(limit=10)
        assert len(sessions) == 0


# ---------------------------------------------------------------------------
# MCP tool integration tests
# ---------------------------------------------------------------------------


class TestMCPProjectTools:
    """Tests for MCP project management tools via _impl functions."""

    @pytest.fixture(autouse=True)
    def reset_server_state(self):
        """Reset server global state between tests."""
        import anamnesis.mcp_server.server as server_module

        # Disable disk persistence for the global registry during tests
        orig_persist = server_module._registry._persist_path
        server_module._registry._persist_path = None
        server_module._registry.reset()
        yield
        server_module._registry.reset()
        server_module._registry._persist_path = orig_persist

    def test_activate_project_via_dedicated_tool(self, project_a):
        from anamnesis.mcp_server.server import _activate_project_impl

        result = _activate_project_impl(path=project_a)
        assert result["success"] is True
        assert result["activated"]["name"] == "project_a"

    def test_activate_project_invalid_path(self):
        from anamnesis.mcp_server.server import _activate_project_impl

        result = _activate_project_impl(path="/nonexistent/path")
        assert result["success"] is False
        assert "error" in result

    def test_get_current_config_empty(self):
        from anamnesis.mcp_server.server import _get_project_config_impl

        result = _get_project_config_impl()
        assert result["success"] is True
        assert result["registry"]["project_count"] == 0

    def test_get_current_config_with_projects(self, project_a, project_b):
        from anamnesis.mcp_server.server import (
            _activate_project_impl,
            _get_project_config_impl,
        )

        _activate_project_impl(path=project_a)
        _activate_project_impl(path=project_b)

        result = _get_project_config_impl()
        assert result["registry"]["project_count"] == 2
        assert result["registry"]["active_project"] == "project_b"

    def test_list_projects_tool(self, project_a, project_b):
        from anamnesis.mcp_server.server import (
            _activate_project_impl,
            _list_projects_impl,
        )

        _activate_project_impl(path=project_a)
        _activate_project_impl(path=project_b)

        result = _list_projects_impl()
        assert result["success"] is True
        assert result["total"] == 2

    def test_project_switch_affects_services(self, project_a, project_b):
        """Switching active project changes which services are returned."""
        from anamnesis.mcp_server.server import (
            _activate_project_impl,
            _get_learning_service,
        )

        _activate_project_impl(path=project_a)
        svc_a = _get_learning_service()

        _activate_project_impl(path=project_b)
        svc_b = _get_learning_service()

        assert svc_a is not svc_b

    def test_set_current_path_backward_compat(self, project_a):
        """_set_current_path still works as before."""
        from anamnesis.mcp_server.server import (
            _get_current_path,
            _set_current_path,
        )

        _set_current_path(project_a)
        assert _get_current_path() == str(Path(project_a).resolve())


class TestMCPToolRegistration:
    """Tests that project tools are registered."""

    def test_project_tools_registered(self):
        from anamnesis.mcp_server.server import mcp

        tools = mcp._tool_manager._tools
        assert "manage_project" in tools


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestProjectPersistence:
    """Tests for project registry persistence to disk."""

    @pytest.fixture
    def persist_file(self, tmp_path):
        """Return a path for the persist file (does not exist yet)."""
        return tmp_path / "anamnesis" / "projects.json"

    def test_save_creates_file_on_activate(self, persist_file, project_a):
        reg = ProjectRegistry(persist_path=persist_file)
        assert not persist_file.exists()

        reg.activate(project_a)
        assert persist_file.exists()

        data = json.loads(persist_file.read_text())
        resolved_a = str(Path(project_a).resolve())
        assert resolved_a in data["projects"]

    def test_load_restores_known_projects(self, persist_file, project_a, project_b):
        # Activate two projects to populate the file
        reg1 = ProjectRegistry(persist_path=persist_file)
        reg1.activate(project_a)
        reg1.activate(project_b)

        # Create a fresh registry that loads from the same file
        reg2 = ProjectRegistry(persist_path=persist_file)
        names = {p.name for p in reg2.list_projects()}
        assert "project_a" in names
        assert "project_b" in names

    def test_loaded_projects_are_not_active(self, persist_file, project_a):
        reg1 = ProjectRegistry(persist_path=persist_file)
        reg1.activate(project_a)

        reg2 = ProjectRegistry(persist_path=persist_file)
        assert reg2.active_path is None
        # The project exists but has no activated_at
        resolved = str(Path(project_a).resolve())
        ctx = reg2.get_project(resolved)
        assert ctx is not None
        assert ctx.activated_at is None

    def test_deactivate_persists_removal(self, persist_file, project_a, project_b):
        reg = ProjectRegistry(persist_path=persist_file)
        reg.activate(project_a)
        reg.activate(project_b)
        reg.deactivate(project_a)

        data = json.loads(persist_file.read_text())
        resolved_a = str(Path(project_a).resolve())
        resolved_b = str(Path(project_b).resolve())
        assert resolved_a not in data["projects"]
        assert resolved_b in data["projects"]

    def test_missing_file_is_fine(self, persist_file):
        # persist_file doesn't exist — should not raise
        reg = ProjectRegistry(persist_path=persist_file)
        assert reg.list_projects() == []

    def test_corrupted_file_logs_warning_and_continues(self, persist_file):
        persist_file.parent.mkdir(parents=True, exist_ok=True)
        persist_file.write_text("NOT VALID JSON {{{", encoding="utf-8")

        reg = ProjectRegistry(persist_path=persist_file)
        assert reg.list_projects() == []

    def test_missing_directory_skipped(self, persist_file, tmp_path):
        # Write a file referencing a directory that doesn't exist
        persist_file.parent.mkdir(parents=True, exist_ok=True)
        gone = str(tmp_path / "gone_project")
        persist_file.write_text(
            json.dumps({
                "version": 1,
                "projects": {gone: {"last_activated_at": None}},
            }),
            encoding="utf-8",
        )

        reg = ProjectRegistry(persist_path=persist_file)
        assert reg.list_projects() == []

    def test_persist_path_none_disables_persistence(self, project_a):
        reg = ProjectRegistry(persist_path=None)
        reg.activate(project_a)
        # No file anywhere — nothing to assert on disk, just no crash
        assert reg.list_projects() != []

    def test_reset_does_not_touch_disk(self, persist_file, project_a):
        reg = ProjectRegistry(persist_path=persist_file)
        reg.activate(project_a)
        assert persist_file.exists()

        # Snapshot file content before reset
        before = persist_file.read_text()
        reg.reset()
        after = persist_file.read_text()
        assert before == after  # File unchanged

    def test_version_field_present(self, persist_file, project_a):
        reg = ProjectRegistry(persist_path=persist_file)
        reg.activate(project_a)

        data = json.loads(persist_file.read_text())
        assert data["version"] == 1

    def test_atomic_write_no_partial_file(self, persist_file, project_a):
        reg = ProjectRegistry(persist_path=persist_file)
        reg.activate(project_a)

        # The file should be valid JSON (no partial writes)
        data = json.loads(persist_file.read_text())
        assert "projects" in data
        # No .tmp files left behind
        tmp_files = list(persist_file.parent.glob("*.tmp"))
        assert tmp_files == []

    def test_activate_restores_then_adds(self, persist_file, project_a, project_b):
        # First session: activate project_a
        reg1 = ProjectRegistry(persist_path=persist_file)
        reg1.activate(project_a)

        # Second session: loads project_a, then activates project_b
        reg2 = ProjectRegistry(persist_path=persist_file)
        reg2.activate(project_b)

        data = json.loads(persist_file.read_text())
        resolved_a = str(Path(project_a).resolve())
        resolved_b = str(Path(project_b).resolve())
        assert resolved_a in data["projects"]
        assert resolved_b in data["projects"]


# ---------------------------------------------------------------------------
# Boundary enforcement tests
# ---------------------------------------------------------------------------


class TestProjectRegistryBoundary:
    """Tests for allowed_roots boundary enforcement."""

    def test_activate_within_allowed_root_succeeds(self, tmp_path):
        root = tmp_path / "allowed"
        root.mkdir()
        project = root / "myproject"
        project.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[str(root)],
        )
        ctx = reg.activate(str(project))
        assert ctx.path == str(project.resolve())

    def test_activate_outside_allowed_root_raises(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[str(allowed)],
        )
        with pytest.raises(ValueError, match="outside the allowed project roots"):
            reg.activate(str(outside))

    def test_activate_exact_root_succeeds(self, tmp_path):
        root = tmp_path / "root_project"
        root.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[str(root)],
        )
        ctx = reg.activate(str(root))
        assert ctx.path == str(root.resolve())

    def test_activate_prefix_confusion_blocked(self, tmp_path):
        """``/tmp/project`` should NOT allow ``/tmp/project-evil``."""
        legit = tmp_path / "project"
        legit.mkdir()
        evil = tmp_path / "project-evil"
        evil.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[str(legit)],
        )
        # The exact root and its children are fine
        ctx = reg.activate(str(legit))
        assert ctx is not None

        # But the prefix-confusable sibling is blocked
        with pytest.raises(ValueError, match="outside the allowed project roots"):
            reg.activate(str(evil))

    def test_activate_symlink_escape_blocked(self, tmp_path):
        """A symlink inside an allowed root that points outside must be blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        # Create a symlink inside allowed/ that points to outside/
        escape_link = allowed / "escape"
        escape_link.symlink_to(outside)

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[str(allowed)],
        )
        with pytest.raises(ValueError, match="outside the allowed project roots"):
            reg.activate(str(escape_link))

    def test_allowed_roots_none_is_unrestricted(self, tmp_path):
        project = tmp_path / "anywhere"
        project.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=None,
        )
        ctx = reg.activate(str(project))
        assert ctx.path == str(project.resolve())

    def test_allowed_roots_empty_blocks_all(self, tmp_path):
        project = tmp_path / "blocked"
        project.mkdir()

        reg = ProjectRegistry(
            persist_path=None,
            allowed_roots=[],
        )
        with pytest.raises(ValueError, match="outside the allowed project roots"):
            reg.activate(str(project))
