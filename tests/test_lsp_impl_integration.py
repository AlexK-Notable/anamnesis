"""Integration tests for LSP _impl functions in anamnesis/mcp_server/tools/lsp.py.

Tests the full call chain: _impl function -> SymbolService -> SymbolRetriever/CodeEditor
using real project directories, real tree-sitter parsing, and the real project registry.
No mock objects are used anywhere — all calls go through real services.

Covers:
- _replace_symbol_body_impl
- _insert_near_symbol_impl
- _rename_symbol_impl
- _go_to_definition_impl
- _manage_lsp_impl
- _find_referencing_symbols_impl
- _analyze_code_quality_impl (diagnostics detail_level)
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_with_python(tmp_path):
    """Create a minimal Python project and activate it in the global registry.

    Sets up the project in the module-level ``_registry`` used by all
    ``_impl`` functions, then restores the previous active project on teardown.
    """
    from anamnesis.mcp_server._shared import _registry

    src = tmp_path / "src"
    src.mkdir()

    (src / "__init__.py").write_text("")

    (src / "calculator.py").write_text(
        '''\
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b


def greet(name: str) -> str:
    return f"Hello, {name}!"


PI = 3.14159
'''
    )

    (src / "helper.py").write_text(
        '''\
def utility_func():
    pass


class HelperClass:
    def method_one(self):
        pass

    def method_two(self):
        pass
'''
    )

    # Remember previous state
    previous_path = _registry.active_path

    # Activate the test project
    _registry.activate(str(tmp_path))

    yield str(tmp_path)

    # Restore: deactivate the test project and re-activate the old one
    _registry.deactivate(str(tmp_path))
    if previous_path and os.path.isdir(previous_path):
        try:
            _registry.activate(previous_path)
        except Exception:
            pass


# ===========================================================================
# _replace_symbol_body_impl tests
# ===========================================================================


class TestReplaceSymbolBodyImpl:
    """Integration tests for _replace_symbol_body_impl."""

    def test_replace_symbol_body_impl_envelope(self, project_with_python):
        """_replace_symbol_body_impl returns a well-formed response envelope.

        Without a running LSP server, the editor raises RuntimeError which
        the _with_error_handling decorator catches and converts to a failure
        response. This verifies the full integration path from _impl through
        project registry, symbol service, and error handling.
        """
        from anamnesis.mcp_server.tools.lsp import _replace_symbol_body_impl

        result = _replace_symbol_body_impl(
            name_path="greet",
            relative_path="src/calculator.py",
            body='def greet(name: str) -> str:\n    return f"Hi, {name}!"\n',
        )

        assert isinstance(result, dict)
        # Without LSP, this returns a failure envelope
        assert "success" in result
        if not result["success"]:
            assert "error" in result
            assert "error_code" in result
            assert "is_retryable" in result

    def test_replace_symbol_body_impl_nonexistent_file(self, project_with_python):
        """_replace_symbol_body_impl on a nonexistent file returns error envelope."""
        from anamnesis.mcp_server.tools.lsp import _replace_symbol_body_impl

        result = _replace_symbol_body_impl(
            name_path="greet",
            relative_path="nonexistent.py",
            body="def greet(): pass\n",
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result


# ===========================================================================
# _insert_near_symbol_impl tests
# ===========================================================================


class TestInsertNearSymbolImpl:
    """Integration tests for _insert_near_symbol_impl."""

    def test_insert_near_symbol_impl_after(self, project_with_python):
        """_insert_near_symbol_impl with position='after' returns envelope.

        Exercises the full path: project activation -> symbol service ->
        code editor -> _require_lsp check -> error handling.
        """
        from anamnesis.mcp_server.tools.lsp import _insert_near_symbol_impl

        result = _insert_near_symbol_impl(
            name_path="greet",
            relative_path="src/calculator.py",
            body="\ndef farewell(name: str) -> str:\n    return f\"Bye, {name}!\"\n",
            position="after",
        )

        assert isinstance(result, dict)
        assert "success" in result
        if not result["success"]:
            assert "error" in result
            assert "error_code" in result

    def test_insert_near_symbol_impl_before(self, project_with_python):
        """_insert_near_symbol_impl with position='before' returns envelope."""
        from anamnesis.mcp_server.tools.lsp import _insert_near_symbol_impl

        result = _insert_near_symbol_impl(
            name_path="Calculator",
            relative_path="src/calculator.py",
            body="import math\n",
            position="before",
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_insert_near_symbol_impl_invalid_position(self, project_with_python):
        """_insert_near_symbol_impl with invalid position returns failure."""
        from anamnesis.mcp_server.tools.lsp import _insert_near_symbol_impl

        result = _insert_near_symbol_impl(
            name_path="greet",
            relative_path="src/calculator.py",
            body="# comment\n",
            position="middle",
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "middle" in result.get("error", "")


# ===========================================================================
# _rename_symbol_impl tests
# ===========================================================================


class TestRenameSymbolImpl:
    """Integration tests for _rename_symbol_impl."""

    def test_rename_symbol_impl_envelope(self, project_with_python):
        """_rename_symbol_impl returns a well-formed response envelope.

        Without LSP, the rename path hits _require_lsp and the error
        handling decorator produces a failure envelope.
        """
        from anamnesis.mcp_server.tools.lsp import _rename_symbol_impl

        result = _rename_symbol_impl(
            name_path="greet",
            relative_path="src/calculator.py",
            new_name="welcome",
        )

        assert isinstance(result, dict)
        assert "success" in result
        if not result["success"]:
            assert "error" in result
            assert "is_retryable" in result

    def test_rename_symbol_impl_nonexistent_file(self, project_with_python):
        """_rename_symbol_impl on a nonexistent file returns error envelope."""
        from anamnesis.mcp_server.tools.lsp import _rename_symbol_impl

        result = _rename_symbol_impl(
            name_path="foo",
            relative_path="does_not_exist.py",
            new_name="bar",
        )

        assert isinstance(result, dict)
        assert result["success"] is False


# ===========================================================================
# _go_to_definition_impl tests
# ===========================================================================


class TestGoToDefinitionImpl:
    """Integration tests for _go_to_definition_impl."""

    def test_go_to_definition_impl_with_name_path(self, project_with_python):
        """_go_to_definition_impl with name_path returns response envelope.

        The function requires LSP for actual definition lookup. Without LSP,
        the SymbolRetriever.go_to_definition returns an error dict, which
        the _impl wraps in a success envelope (containing the error info
        from the retriever layer).
        """
        from anamnesis.mcp_server.tools.lsp import _go_to_definition_impl

        result = _go_to_definition_impl(
            relative_path="src/calculator.py",
            name_path="Calculator",
        )

        assert isinstance(result, dict)
        assert "success" in result
        # The result is a success envelope containing the data from
        # SymbolRetriever (which may indicate LSP unavailability inside data)
        if result["success"]:
            assert "data" in result

    def test_go_to_definition_impl_no_args(self, project_with_python):
        """_go_to_definition_impl with neither name_path nor line returns failure."""
        from anamnesis.mcp_server.tools.lsp import _go_to_definition_impl

        result = _go_to_definition_impl(
            relative_path="src/calculator.py",
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "name_path" in result.get("error", "") or "line" in result.get("error", "")

    def test_go_to_definition_impl_with_position(self, project_with_python):
        """_go_to_definition_impl with line+column exercises position path."""
        from anamnesis.mcp_server.tools.lsp import _go_to_definition_impl

        result = _go_to_definition_impl(
            relative_path="src/calculator.py",
            line=0,
            column=6,
        )

        assert isinstance(result, dict)
        assert "success" in result


# ===========================================================================
# _manage_lsp_impl tests
# ===========================================================================


class TestManageLspImpl:
    """Integration tests for _manage_lsp_impl."""

    def test_manage_lsp_impl_status(self, project_with_python):
        """_manage_lsp_impl action='status' returns language server info."""
        from anamnesis.mcp_server.tools.lsp import _manage_lsp_impl

        result = _manage_lsp_impl(action="status")

        assert isinstance(result, dict)
        assert result["success"] is True
        data = result["data"]
        assert "languages" in data
        # All 4 supported languages should be present
        for lang in ("python", "go", "rust", "typescript"):
            assert lang in data["languages"]
            entry = data["languages"][lang]
            assert "binary" in entry
            assert "installed" in entry
            assert isinstance(entry["installed"], bool)
            assert "running" in entry

    def test_manage_lsp_impl_status_project_root(self, project_with_python):
        """_manage_lsp_impl status includes the project root path."""
        from anamnesis.mcp_server.tools.lsp import _manage_lsp_impl

        result = _manage_lsp_impl(action="status")

        assert result["success"] is True
        assert "project_root" in result["data"]

    def test_manage_lsp_impl_invalid_action(self, project_with_python):
        """_manage_lsp_impl with unknown action returns failure envelope."""
        from anamnesis.mcp_server.tools.lsp import _manage_lsp_impl

        result = _manage_lsp_impl(action="restart")

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "restart" in result.get("error", "")

    def test_manage_lsp_impl_enable_unsupported(self, project_with_python):
        """_manage_lsp_impl enabling unsupported language returns failure."""
        from anamnesis.mcp_server.tools.lsp import _manage_lsp_impl

        result = _manage_lsp_impl(action="enable", language="cobol")

        assert isinstance(result, dict)
        assert result["success"] is False


# ===========================================================================
# _find_referencing_symbols_impl tests
# ===========================================================================


class TestFindReferencingSymbolsImpl:
    """Integration tests for _find_referencing_symbols_impl."""

    def test_find_referencing_symbols_impl_envelope(self, project_with_python):
        """_find_referencing_symbols_impl returns a well-formed envelope.

        Without LSP, the retriever returns an error indicator in the results,
        which the _impl wraps in the standard success envelope with
        categorized references.
        """
        from anamnesis.mcp_server.tools.lsp import _find_referencing_symbols_impl

        result = _find_referencing_symbols_impl(
            name_path="Calculator",
            relative_path="src/calculator.py",
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result["success"]:
            assert "data" in result
            data = result["data"]
            assert "references" in data
            assert "categories" in data

    def test_find_referencing_symbols_impl_missing_symbol(self, project_with_python):
        """_find_referencing_symbols_impl for nonexistent symbol returns envelope."""
        from anamnesis.mcp_server.tools.lsp import _find_referencing_symbols_impl

        result = _find_referencing_symbols_impl(
            name_path="NonexistentSymbol",
            relative_path="src/calculator.py",
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_find_referencing_symbols_impl_missing_file(self, project_with_python):
        """_find_referencing_symbols_impl on nonexistent file returns envelope."""
        from anamnesis.mcp_server.tools.lsp import _find_referencing_symbols_impl

        result = _find_referencing_symbols_impl(
            name_path="Calculator",
            relative_path="nonexistent.py",
        )

        assert isinstance(result, dict)
        assert "success" in result


# ===========================================================================
# _analyze_code_quality_impl (diagnostics) tests
# ===========================================================================


class TestAnalyzeCodeQualityDiagnostics:
    """Integration tests for _analyze_code_quality_impl with diagnostics detail_level."""

    def test_diagnostics_happy_path(self, project_with_python):
        """diagnostics detail_level for a known file returns well-formed envelope.

        Without a running LSP server, the diagnostics path returns an
        error from the retriever layer (LSP not available), which the
        _impl converts to a failure response.
        """
        from anamnesis.mcp_server.tools.lsp import _analyze_code_quality_impl

        result = _analyze_code_quality_impl(
            relative_path="src/calculator.py",
            detail_level="diagnostics",
        )

        assert isinstance(result, dict)
        assert "success" in result
        # The response is either a success with diagnostics data or a
        # failure because LSP is unavailable
        if result["success"]:
            data = result["data"]
            assert "diagnostics" in data
            assert "relative_path" in data
            assert data["relative_path"] == "src/calculator.py"
        else:
            assert "error" in result
            assert "error_code" in result

    def test_diagnostics_nonexistent_file(self, project_with_python):
        """diagnostics detail_level for a nonexistent file returns error envelope."""
        from anamnesis.mcp_server.tools.lsp import _analyze_code_quality_impl

        result = _analyze_code_quality_impl(
            relative_path="nonexistent.py",
            detail_level="diagnostics",
        )

        assert isinstance(result, dict)
        assert "success" in result
        # Should either fail or return empty diagnostics
        if result["success"]:
            data = result["data"]
            assert "diagnostics" in data
        else:
            assert "error" in result
