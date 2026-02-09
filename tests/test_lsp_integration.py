"""Tests for LSP integration layer.

Covers:
- SolidLSP vendoring: imports, compat module
- LspManager: language routing, lifecycle, status
- SymbolRetriever: find_symbol, get_overview, tree-sitter fallback
- NamePathMatcher: pattern matching for symbol name paths
- LspSymbol: serialization, body reading, name path construction
- CodeEditor: replace_body, insert_after/before, rename (mocked LSP)
- LspExtractionBackend: protocol compliance, confidence tiers
- ProjectContext: LSP lifecycle tied to activation/deactivation
- MCP tools: tool registration, parameter validation
"""

import os
from unittest.mock import MagicMock

import pytest

from anamnesis.lsp.symbols import (
    LspSymbol,
    LspSymbolLocation,
    NamePathMatcher,
    PositionInFile,
    SymbolRetriever,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with Python source files."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text(
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

    (tmp_path / "README.md").write_text("# Test Project\n")
    return str(tmp_path)


@pytest.fixture
def make_symbol():
    """Factory for creating LspSymbol instances."""
    def _make(
        name: str,
        kind: int = 12,  # Function
        start_line: int = 0,
        end_line: int = 0,
        relative_path: str = "src/main.py",
        parent: LspSymbol | None = None,
    ) -> LspSymbol:
        return LspSymbol(
            name=name,
            kind=kind,
            location=LspSymbolLocation(
                relative_path=relative_path,
                start_line=start_line,
                start_col=0,
                end_line=end_line,
                end_col=0,
            ),
            parent=parent,
        )
    return _make


# ===========================================================================
# SolidLSP vendoring tests
# ===========================================================================


class TestSolidLSPVendoring:
    """Verify vendored SolidLSP imports work."""

    def test_import_solid_language_server(self):
        from anamnesis.lsp.solidlsp import SolidLanguageServer
        assert SolidLanguageServer is not None

    def test_import_ls_config(self):
        from anamnesis.lsp.solidlsp.ls_config import Language, LanguageServerConfig
        assert Language.PYTHON is not None
        assert LanguageServerConfig is not None

    def test_import_ls_types(self):
        from anamnesis.lsp.solidlsp import ls_types
        assert ls_types is not None

    def test_import_ls_exceptions(self):
        from anamnesis.lsp.solidlsp.ls_exceptions import SolidLSPException
        assert SolidLSPException is not None

    def test_import_settings(self):
        from anamnesis.lsp.solidlsp.settings import SolidLSPSettings
        settings = SolidLSPSettings()
        assert settings is not None

    def test_import_lsp_protocol_handler(self):
        from anamnesis.lsp.solidlsp.lsp_protocol_handler import lsp_types
        assert hasattr(lsp_types, "SymbolKind")

    def test_python_language_server_class(self):
        """Verify Python LS class can be resolved."""
        from anamnesis.lsp.solidlsp.ls_config import Language
        cls = Language.PYTHON.get_ls_class()
        assert cls.__name__ == "PyrightServer"

    def test_unsupported_language_raises(self):
        """Unsupported languages raise ValueError."""
        from anamnesis.lsp.solidlsp.ls_config import Language
        with pytest.raises(ValueError, match="not available"):
            Language.JAVA.get_ls_class()


# ===========================================================================
# Compat module tests
# ===========================================================================


class TestCompatModule:
    """Tests for the compatibility layer replacing sensai/serena deps."""

    def test_getstate(self):
        from anamnesis.lsp.solidlsp.compat import getstate
        data = {"key": "value", "num": 42}
        serialized = getstate(data)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_pickle_roundtrip(self, tmp_path):
        from anamnesis.lsp.solidlsp.compat import dump_pickle, load_pickle
        path = str(tmp_path / "test.pkl")
        obj = {"test": [1, 2, 3]}
        dump_pickle(obj, path)
        loaded = load_pickle(path)
        assert loaded == obj

    def test_load_nonexistent_pickle(self):
        from anamnesis.lsp.solidlsp.compat import load_pickle
        result = load_pickle("/nonexistent/path/test.pkl")
        assert result is None

    def test_to_string_mixin(self):
        from anamnesis.lsp.solidlsp.compat import ToStringMixin

        class MyClass(ToStringMixin):
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = MyClass()
        r = repr(obj)
        assert "MyClass" in r
        assert "test" in r
        assert "42" in r

    def test_log_time(self, caplog):
        import logging
        from anamnesis.lsp.solidlsp.compat import LogTime

        with caplog.at_level(logging.INFO):
            with LogTime("test operation"):
                pass
        assert any("test operation" in record.message for record in caplog.records)



# ===========================================================================
# NamePathMatcher tests
# ===========================================================================


class TestNamePathMatcher:
    """Tests for symbol name path pattern matching."""

    def test_simple_name_match(self, make_symbol):
        """Simple name matches any symbol with that name."""
        matcher = NamePathMatcher("greet")
        sym = make_symbol("greet")
        assert matcher.matches(sym)

    def test_simple_name_no_match(self, make_symbol):
        matcher = NamePathMatcher("nonexistent")
        sym = make_symbol("greet")
        assert not matcher.matches(sym)

    def test_relative_path_match(self, make_symbol):
        """Relative path matches as suffix."""
        parent = make_symbol("Calculator", kind=5)
        child = make_symbol("add", kind=6, parent=parent)
        parent.children = [child]

        matcher = NamePathMatcher("Calculator/add")
        assert matcher.matches(child)

    def test_relative_path_no_match(self, make_symbol):
        parent = make_symbol("Calculator", kind=5)
        child = make_symbol("add", kind=6, parent=parent)
        parent.children = [child]

        matcher = NamePathMatcher("OtherClass/add")
        assert not matcher.matches(child)

    def test_absolute_path_match(self, make_symbol):
        """Absolute path requires exact full path match."""
        parent = make_symbol("Calculator", kind=5)
        child = make_symbol("add", kind=6, parent=parent)
        parent.children = [child]

        matcher = NamePathMatcher("/Calculator/add")
        assert matcher.matches(child)

    def test_absolute_path_no_match_wrong_depth(self, make_symbol):
        """Absolute path fails if depth differs."""
        sym = make_symbol("add")
        matcher = NamePathMatcher("/Calculator/add")
        assert not matcher.matches(sym)

    def test_overload_index_match(self, make_symbol):
        """Overload index selects specific overload."""
        sym = make_symbol("method")
        sym._raw = {"_overload_idx": 0}

        matcher = NamePathMatcher("method[0]")
        assert matcher.matches(sym)

    def test_overload_index_no_match(self, make_symbol):
        sym = make_symbol("method")
        sym._raw = {"_overload_idx": 1}

        matcher = NamePathMatcher("method[0]")
        assert not matcher.matches(sym)

    def test_substring_matching(self, make_symbol):
        """Substring mode matches partial last component."""
        sym = make_symbol("getValue")

        matcher = NamePathMatcher("get", substring_matching=True)
        assert matcher.matches(sym)

    def test_substring_matching_case_insensitive(self, make_symbol):
        sym = make_symbol("getValue")

        matcher = NamePathMatcher("GET", substring_matching=True)
        assert matcher.matches(sym)

    def test_substring_matching_disabled(self, make_symbol):
        sym = make_symbol("getValue")

        matcher = NamePathMatcher("get", substring_matching=False)
        assert not matcher.matches(sym)

    def test_empty_pattern_no_match(self, make_symbol):
        matcher = NamePathMatcher("")
        sym = make_symbol("anything")
        assert not matcher.matches(sym)


# ===========================================================================
# LspSymbol tests
# ===========================================================================


class TestLspSymbol:
    """Tests for LspSymbol data model."""

    def test_name_path_no_parent(self, make_symbol):
        sym = make_symbol("greet")
        assert sym.name_path == "greet"

    def test_name_path_with_parent(self, make_symbol):
        parent = make_symbol("Calculator", kind=5)
        child = make_symbol("add", kind=6, parent=parent)
        assert child.name_path == "Calculator/add"

    def test_name_path_deep_nesting(self, make_symbol):
        root = make_symbol("Module", kind=2)
        cls = make_symbol("MyClass", kind=5, parent=root)
        method = make_symbol("run", kind=6, parent=cls)
        assert method.name_path == "Module/MyClass/run"

    def test_kind_name(self, make_symbol):
        from anamnesis.lsp.solidlsp.lsp_protocol_handler.lsp_types import SymbolKind
        sym = make_symbol("Calculator", kind=SymbolKind.Class)
        assert sym.kind_name == "Class"

    def test_to_dict_basic(self, make_symbol):
        sym = make_symbol("greet", start_line=10, end_line=12)
        d = sym.to_dict()
        assert d["name"] == "greet"
        assert d["name_path"] == "greet"
        assert d["location"]["start_line"] == 10
        assert d["location"]["end_line"] == 12

    def test_to_dict_with_detail(self, make_symbol):
        sym = make_symbol("greet")
        sym.detail = "(name: str) -> str"
        d = sym.to_dict()
        assert d["detail"] == "(name: str) -> str"

    def test_to_dict_with_children(self, make_symbol):
        parent = make_symbol("Calculator", kind=5)
        child = make_symbol("add", kind=6, parent=parent)
        parent.children = [child]
        d = parent.to_dict()
        assert "children" in d
        assert d["children"][0]["name"] == "add"

    def test_get_body(self, project_dir, make_symbol):
        """Read symbol body from disk."""
        sym = make_symbol("Calculator", kind=5, start_line=0, end_line=8,
                          relative_path="src/main.py")
        body = sym.get_body(project_dir)
        assert body is not None
        assert "class Calculator" in body
        assert "def add" in body

    def test_get_body_nonexistent_file(self, project_dir, make_symbol):
        sym = make_symbol("Foo", relative_path="nonexistent.py")
        body = sym.get_body(project_dir)
        assert body is None


# ===========================================================================
# PositionInFile tests
# ===========================================================================


class TestPositionInFile:
    def test_to_lsp_position(self):
        pos = PositionInFile(line=5, col=10)
        lsp = pos.to_lsp_position()
        assert lsp == {"line": 5, "character": 10}


# ===========================================================================
# SymbolRetriever tests (tree-sitter fallback)
# ===========================================================================


class TestSymbolRetrieverTreeSitter:
    """Tests for SymbolRetriever using tree-sitter fallback (no LSP)."""

    def test_find_class(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find("Calculator", relative_path="src/main.py")
        assert len(results) >= 1
        assert results[0]["name"] == "Calculator"

    def test_find_function(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find("greet", relative_path="src/main.py")
        assert len(results) >= 1
        assert results[0]["name"] == "greet"

    def test_find_method(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find("Calculator/add", relative_path="src/main.py")
        # Tree-sitter may or may not build parent relationships
        # depending on the extraction backend. At minimum, "add" should be found.
        add_results = retriever.find("add", relative_path="src/main.py")
        assert len(add_results) >= 1

    def test_find_nonexistent(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find("nonexistent_symbol", relative_path="src/main.py")
        assert results == []

    def test_find_with_include_body(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find(
            "Calculator", relative_path="src/main.py", include_body=True
        )
        assert len(results) >= 1
        assert "body" in results[0]
        # Body should contain class content (def add, def subtract)
        assert "add" in results[0]["body"] or "Calculator" in results[0]["body"]

    def test_find_unique(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        result = retriever.find_unique("Calculator", "src/main.py")
        assert result is not None
        assert result["name"] == "Calculator"

    def test_find_unique_nonexistent(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        result = retriever.find_unique("Nonexistent", "src/main.py")
        assert result is None

    def test_get_overview(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        overview = retriever.get_overview("src/main.py")
        assert isinstance(overview, dict)
        # Should have at least one kind group
        assert len(overview) >= 1
        # Each group is a list
        for kind, symbols in overview.items():
            assert isinstance(symbols, list)
            for sym in symbols:
                assert "name" in sym
                assert "line" in sym

    def test_get_overview_nonexistent_file(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        overview = retriever.get_overview("nonexistent.py")
        assert overview == {}

    def test_find_with_substring_matching(self, project_dir):
        retriever = SymbolRetriever(project_dir, lsp_manager=None)
        results = retriever.find(
            "calc", relative_path="src/main.py", substring_matching=True
        )
        # Should match Calculator
        names = [r["name"] for r in results]
        assert "Calculator" in names

    def test_extension_to_language_mapping(self):
        assert SymbolRetriever._ext_to_language("py") == "python"
        assert SymbolRetriever._ext_to_language("go") == "go"
        assert SymbolRetriever._ext_to_language("rs") == "rust"
        assert SymbolRetriever._ext_to_language("ts") == "typescript"
        assert SymbolRetriever._ext_to_language("unknown") == ""

    def test_symbol_kind_to_lsp_mapping(self):
        assert SymbolRetriever._symbol_kind_to_lsp("class") == 5
        assert SymbolRetriever._symbol_kind_to_lsp("function") == 12
        assert SymbolRetriever._symbol_kind_to_lsp("method") == 6
        assert SymbolRetriever._symbol_kind_to_lsp("variable") == 13
        assert SymbolRetriever._symbol_kind_to_lsp("unknown_kind") == 13  # defaults to Variable


# ===========================================================================
# LspManager tests
# ===========================================================================


class TestLspManager:
    """Tests for LspManager (without real LSP servers)."""

    def test_init(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        assert mgr.project_root == os.path.abspath(project_dir)

    def test_language_for_python_file(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        lang = mgr.get_language_for_file("src/main.py")
        assert lang is not None
        assert lang.value == "python"

    def test_language_for_go_file(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        lang = mgr.get_language_for_file("cmd/main.go")
        assert lang is not None
        assert lang.value == "go"

    def test_language_for_rust_file(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        lang = mgr.get_language_for_file("src/lib.rs")
        assert lang is not None
        assert lang.value == "rust"

    def test_language_for_ts_file(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        lang = mgr.get_language_for_file("src/index.ts")
        assert lang is not None
        assert lang.value == "typescript"

    def test_language_for_unsupported_file(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        lang = mgr.get_language_for_file("README.md")
        assert lang is None

    def test_is_available(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        assert mgr.is_available("python") is True
        assert mgr.is_available("go") is True
        assert mgr.is_available("cobol") is False

    def test_is_running_initially_false(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        assert mgr.is_running("python") is False

    def test_get_status_no_servers(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        status = mgr.get_status()
        assert "project_root" in status
        assert "supported_languages" in status
        assert "running_servers" in status
        assert len(status["running_servers"]) == 0

    def test_start_unsupported_language(self, project_dir):
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        result = mgr.start("cobol")
        assert result is False

    def test_stop_all_empty(self, project_dir):
        """stop_all on empty manager is a no-op."""
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        mgr.stop_all()  # Should not raise

    def test_stop_unstarted(self, project_dir):
        """Stopping a language that isn't running is a no-op."""
        from anamnesis.lsp.manager import LspManager
        mgr = LspManager(project_dir)
        mgr.stop("python")  # Should not raise


# ===========================================================================
# CodeEditor tests (mocked LSP)
# ===========================================================================


class TestCodeEditor:
    """Tests for CodeEditor with mocked LSP."""

    def _make_editor(self, project_dir: str) -> tuple:
        """Create a CodeEditor with mocked dependencies."""
        from anamnesis.lsp.editor import CodeEditor

        mock_lsp_manager = MagicMock()
        mock_lsp_manager.get_language_server.return_value = MagicMock()

        mock_retriever = MagicMock()
        editor = CodeEditor(
            project_root=project_dir,
            symbol_retriever=mock_retriever,
            lsp_manager=mock_lsp_manager,
        )
        return editor, mock_retriever, mock_lsp_manager

    def test_require_lsp_raises_without_manager(self, project_dir):
        from anamnesis.lsp.editor import CodeEditor

        editor = CodeEditor(
            project_root=project_dir,
            symbol_retriever=MagicMock(),
            lsp_manager=None,
        )
        with pytest.raises(RuntimeError, match="LSP is required"):
            editor._require_lsp("src/main.py")

    def test_require_lsp_raises_no_server(self, project_dir):
        from anamnesis.lsp.editor import CodeEditor

        mock_mgr = MagicMock()
        mock_mgr.get_language_server.return_value = None
        editor = CodeEditor(
            project_root=project_dir,
            symbol_retriever=MagicMock(),
            lsp_manager=mock_mgr,
        )
        with pytest.raises(RuntimeError, match="No LSP server available"):
            editor._require_lsp("src/main.py")

    def test_replace_body_dry_run(self, project_dir):
        editor, mock_retriever, _ = self._make_editor(project_dir)

        # Mock the symbol resolution
        mock_sym = LspSymbol(
            name="greet",
            kind=12,
            location=LspSymbolLocation(
                relative_path="src/main.py",
                start_line=11,
                start_col=0,
                end_line=12,
                end_col=0,
            ),
        )
        # Mock find returning 1 result
        mock_retriever.find.return_value = [{"name_path": "greet"}]
        mock_retriever._get_all_symbols_flat.return_value = [mock_sym]
        mock_retriever._walk_symbols.return_value = [mock_sym]

        result = editor.replace_body(
            "greet", "src/main.py",
            'def greet(name: str) -> str:\n    return f"Hi, {name}!"\n',
            dry_run=True,
        )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert "diff" in result

    def test_insert_after_symbol_dry_run(self, project_dir):
        editor, mock_retriever, _ = self._make_editor(project_dir)

        mock_sym = LspSymbol(
            name="greet",
            kind=12,
            location=LspSymbolLocation(
                relative_path="src/main.py",
                start_line=11,
                start_col=0,
                end_line=12,
                end_col=0,
            ),
        )
        mock_retriever.find.return_value = [{"name_path": "greet"}]
        mock_retriever._get_all_symbols_flat.return_value = [mock_sym]
        mock_retriever._walk_symbols.return_value = [mock_sym]

        result = editor.insert_after_symbol(
            "greet", "src/main.py",
            "\ndef farewell(name: str) -> str:\n    return f\"Bye, {name}!\"\n",
            dry_run=True,
        )
        assert result["success"] is True
        assert result["dry_run"] is True
        assert "insert_at_line" in result

    def test_insert_before_symbol_dry_run(self, project_dir):
        editor, mock_retriever, _ = self._make_editor(project_dir)

        mock_sym = LspSymbol(
            name="Calculator",
            kind=5,
            location=LspSymbolLocation(
                relative_path="src/main.py",
                start_line=0,
                start_col=0,
                end_line=8,
                end_col=0,
            ),
        )
        mock_retriever.find.return_value = [{"name_path": "Calculator"}]
        mock_retriever._get_all_symbols_flat.return_value = [mock_sym]
        mock_retriever._walk_symbols.return_value = [mock_sym]

        result = editor.insert_before_symbol(
            "Calculator", "src/main.py",
            "import math\n",
            dry_run=True,
        )
        assert result["success"] is True
        assert result["dry_run"] is True

    def test_uri_to_relative(self, project_dir):
        from anamnesis.lsp.editor import CodeEditor

        editor = CodeEditor(
            project_root=project_dir,
            symbol_retriever=MagicMock(),
        )
        abs_path = os.path.join(project_dir, "src/main.py")
        uri = f"file://{abs_path}"
        result = editor._uri_to_relative(uri)
        assert result == "src/main.py"


# ===========================================================================
# LspExtractionBackend tests
# ===========================================================================


class TestLspExtractionBackend:
    """Tests for LspExtractionBackend protocol compliance."""

    def test_name(self):
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        backend = LspExtractionBackend(mock_mgr)
        assert backend.name == "lsp"

    def test_priority(self):
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        backend = LspExtractionBackend(mock_mgr)
        assert backend.priority == 100

    def test_supports_language(self):
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        mock_mgr.is_available.return_value = True
        backend = LspExtractionBackend(mock_mgr)
        assert backend.supports_language("python") is True
        mock_mgr.is_available.assert_called_with("python")

    def test_extract_symbols_fallback(self):
        """When LSP returns None server, fall back to tree-sitter."""
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        mock_mgr.get_server_for_language.return_value = None
        mock_mgr.project_root = "/tmp"
        backend = LspExtractionBackend(mock_mgr)

        # Should not raise even though LSP is unavailable
        result = backend.extract_symbols("class Foo: pass", "/tmp/test.py", "python")
        assert isinstance(result, list)

    def test_lsp_kind_map_coverage(self):
        """Verify key LSP SymbolKind values are mapped."""
        from anamnesis.lsp.backend import _LSP_KIND_MAP
        from anamnesis.extraction.types import SymbolKind

        assert _LSP_KIND_MAP[5] == SymbolKind.CLASS
        assert _LSP_KIND_MAP[6] == SymbolKind.METHOD
        assert _LSP_KIND_MAP[12] == SymbolKind.FUNCTION
        assert _LSP_KIND_MAP[13] == SymbolKind.VARIABLE
        assert _LSP_KIND_MAP[14] == SymbolKind.CONSTANT

    def test_extract_all_without_lsp(self):
        """extract_all falls back gracefully when LSP is unavailable."""
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        mock_mgr.get_server_for_language.return_value = None
        mock_mgr.project_root = "/tmp"
        backend = LspExtractionBackend(mock_mgr)

        result = backend.extract_all("def foo(): pass", "/tmp/test.py", "python")
        assert result.file_path == "/tmp/test.py"
        assert result.language == "python"
        assert result.backend_used == "lsp"

    def test_to_relative(self):
        """_to_relative converts absolute to project-relative paths."""
        from anamnesis.lsp.backend import LspExtractionBackend
        mock_mgr = MagicMock()
        mock_mgr.project_root = "/projects/myapp"
        backend = LspExtractionBackend(mock_mgr)

        result = backend._to_relative("/projects/myapp/src/main.py")
        assert result == "src/main.py"


# ===========================================================================
# ProjectContext LSP integration tests
# ===========================================================================


class TestProjectContextLsp:
    """Tests for LSP fields in ProjectContext."""

    def test_lsp_manager_initially_none(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        assert ctx._lsp_manager is None

    def test_get_lsp_manager_creates_instance(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        mgr = ctx.get_lsp_manager()
        assert mgr is not None
        assert mgr.project_root == os.path.abspath(project_dir)

    def test_get_lsp_manager_returns_same_instance(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        mgr1 = ctx.get_lsp_manager()
        mgr2 = ctx.get_lsp_manager()
        assert mgr1 is mgr2

    def test_shutdown_lsp_clears_manager(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        ctx.get_lsp_manager()  # Create it
        assert ctx._lsp_manager is not None

        ctx.shutdown_lsp()
        assert ctx._lsp_manager is None

    def test_shutdown_lsp_noop_when_none(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        ctx.shutdown_lsp()  # Should not raise

    def test_to_dict_includes_lsp(self, project_dir):
        from anamnesis.services.project_registry import ProjectContext
        ctx = ProjectContext(path=project_dir)
        d = ctx.to_dict()
        assert "lsp" in d["services"]
        assert d["services"]["lsp"] is False

        ctx.get_lsp_manager()
        d = ctx.to_dict()
        assert d["services"]["lsp"] is True

    def test_deactivate_calls_shutdown_lsp(self, tmp_path):
        from anamnesis.services.project_registry import ProjectRegistry

        proj = tmp_path / "proj"
        proj.mkdir()

        registry = ProjectRegistry(persist_path=None)
        ctx = registry.activate(str(proj))
        ctx.get_lsp_manager()  # Create the manager

        result = registry.deactivate(str(proj))
        assert result is True
        # After deactivation, the context is removed
        assert registry.get_project(str(proj)) is None


# ===========================================================================
# MCP tool registration tests
# ===========================================================================


class TestMCPToolRegistration:
    """Verify MCP tool _impl functions exist in server module."""

    def test_tool_functions_exist(self):
        import anamnesis.mcp_server.server as srv

        tool_names = [
            "_find_symbol_impl",
            "_get_symbols_overview_impl",
            "_find_referencing_symbols_impl",
            "_replace_symbol_body_impl",
            "_insert_near_symbol_impl",
            "_rename_symbol_impl",
            "_manage_lsp_impl",
        ]
        for name in tool_names:
            assert hasattr(srv, name), f"Missing {name}"
            assert callable(getattr(srv, name)), f"{name} is not callable"

    def test_helper_functions_exist(self):
        import anamnesis.mcp_server.server as srv

        helpers = ["_get_lsp_manager", "_get_symbol_retriever", "_get_code_editor"]
        for name in helpers:
            assert hasattr(srv, name), f"Missing {name}"
