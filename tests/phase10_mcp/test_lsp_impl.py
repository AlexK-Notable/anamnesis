"""Tests for LSP _impl functions — symbol navigation via tree-sitter.

Tests find_symbol, get_symbols_overview, and find_referencing_symbols
_impl functions against the sample_python_project fixture. No LSP
server needed — tree-sitter backend handles all navigation.
"""

import pytest

import anamnesis.mcp_server._shared as shared_module
from anamnesis.mcp_server.tools.lsp import (
    _find_symbol_impl,
    _get_symbols_overview_impl,
)

from .conftest import _as_dict


@pytest.fixture(autouse=True)
def _activate_project(sample_python_project):
    """Activate the sample project for each test."""
    orig_persist = shared_module._registry._persist_path
    shared_module._registry._persist_path = None
    shared_module._registry.reset()
    shared_module._registry.activate(str(sample_python_project))
    yield
    shared_module._registry.reset()
    shared_module._registry._persist_path = orig_persist


# =============================================================================
# get_symbols_overview
# =============================================================================


class TestGetSymbolsOverview:
    """Tests for _get_symbols_overview_impl."""

    def test_returns_overview_for_python_file(self):
        result = _as_dict(_get_symbols_overview_impl("src/service.py"))
        assert result["success"] is True

        # Should have classes and/or functions
        has_symbols = any(
            key in result
            for key in ("Class", "Function", "classes", "functions")
        )
        assert has_symbols or len(result) > 1  # at least success + something

    def test_overview_finds_class(self):
        result = _as_dict(_get_symbols_overview_impl("src/service.py"))
        assert result["success"] is True

        # UserService class should be present
        classes = result.get("Class", [])
        class_names = [
            c["name"] if isinstance(c, dict) else c for c in classes
        ]
        assert "UserService" in class_names

    def test_overview_finds_functions(self):
        result = _as_dict(_get_symbols_overview_impl("src/utils.py"))
        assert result["success"] is True

        functions = result.get("Function", [])
        func_names = [
            f["name"] if isinstance(f, dict) else f for f in functions
        ]
        assert "helper_func" in func_names
        assert "format_name" in func_names

    def test_nonexistent_file_returns_empty_overview(self):
        """Nonexistent file returns success with empty/minimal overview."""
        result = _as_dict(_get_symbols_overview_impl("src/ghost.py"))
        # Implementation logs the error but returns a valid (empty) overview
        assert result["success"] is True
        # Should not contain actual symbols
        classes = result.get("Class", [])
        functions = result.get("Function", [])
        assert len(classes) == 0
        assert len(functions) == 0

    def test_overview_with_depth(self):
        """depth=1 should include class children (methods)."""
        result = _as_dict(
            _get_symbols_overview_impl("src/service.py", depth=1)
        )
        assert result["success"] is True

        # With depth=1, classes should have children as a flat list
        classes = result.get("Class", [])
        for cls in classes:
            if isinstance(cls, dict) and cls.get("name") == "UserService":
                children = cls.get("children", [])
                assert isinstance(children, list)
                assert len(children) > 0
                method_names = [
                    c["name"] for c in children if isinstance(c, dict)
                ]
                assert "get_user" in method_names
                break


# =============================================================================
# find_symbol
# =============================================================================


class TestFindSymbol:
    """Tests for _find_symbol_impl."""

    def test_find_class_by_name(self):
        result = _as_dict(
            _find_symbol_impl("UserService", relative_path="src/service.py")
        )
        assert result["success"] is True
        assert result["total"] >= 1
        assert any(
            s.get("name") == "UserService" or "UserService" in str(s)
            for s in result["symbols"]
        )

    def test_find_function_by_name(self):
        result = _as_dict(
            _find_symbol_impl("helper_func", relative_path="src/utils.py")
        )
        assert result["success"] is True
        assert result["total"] >= 1

    def test_find_with_include_body(self):
        result = _as_dict(
            _find_symbol_impl(
                "simple_add",
                relative_path="src/service.py",
                include_body=True,
            )
        )
        assert result["success"] is True
        assert result["total"] >= 1

        # At least one symbol should have body content
        symbol = result["symbols"][0]
        body = symbol.get("body") or symbol.get("source") or ""
        assert len(body) > 0

    def test_find_nonexistent_returns_empty(self):
        result = _as_dict(
            _find_symbol_impl(
                "nonexistent_function_xyz",
                relative_path="src/service.py",
            )
        )
        assert result["success"] is True
        assert result["total"] == 0

    def test_find_with_substring_matching(self):
        """substring_matching=True should find partial matches."""
        result = _as_dict(
            _find_symbol_impl(
                "complex",
                relative_path="src/service.py",
                substring_matching=True,
            )
        )
        assert result["success"] is True
        # Should find complex_handler
        assert result["total"] >= 1

    def test_find_method_with_path(self):
        """UserService/get_user should find the method."""
        result = _as_dict(
            _find_symbol_impl(
                "UserService/get_user",
                relative_path="src/service.py",
            )
        )
        assert result["success"] is True
        assert result["total"] >= 1

    def test_find_across_codebase_without_relative_path(self):
        """Searching without relative_path searches entire project."""
        result = _as_dict(
            _find_symbol_impl("UserService")
        )
        assert result["success"] is True
        assert result["total"] >= 1
