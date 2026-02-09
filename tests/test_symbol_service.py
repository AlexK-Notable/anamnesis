"""Tests for SymbolService facade over LSP navigation and editing.

Covers:
- Lazy initialization: SymbolRetriever and CodeEditor not created in __init__
- Lazy init triggered on first access (.retriever, .editor)
- Singleton behavior: same instance returned on subsequent access
- Editor creation triggers retriever dependency (editor needs retriever)
- Navigation delegation: find, get_overview, find_referencing_symbols
- Editing delegation: replace_body, insert_after, insert_before, rename
"""

from unittest.mock import MagicMock

import pytest

from anamnesis.services.symbol_service import SymbolService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_lsp_manager():
    """A mock LspManager passed to SymbolService."""
    return MagicMock(name="lsp_manager")


@pytest.fixture
def svc(mock_lsp_manager):
    """SymbolService wired with a mock LspManager."""
    return SymbolService(project_root="/tmp/project", lsp_manager=mock_lsp_manager)


# ===========================================================================
# Lazy initialization tests
# ===========================================================================


class TestLazyInitDeferred:
    """Verify that SymbolRetriever and CodeEditor are NOT created in __init__."""

    def test_retriever_not_created_on_init(self, svc):
        """SymbolRetriever is None immediately after construction."""
        assert svc._retriever is None

    def test_editor_not_created_on_init(self, svc):
        """CodeEditor is None immediately after construction."""
        assert svc._editor is None


class TestLazyInitTriggered:
    """Verify that first access to .retriever / .editor creates the instances."""

    def test_retriever_created_on_first_access(self, svc, monkeypatch):
        """Accessing .retriever creates a SymbolRetriever."""
        mock_cls = MagicMock(name="SymbolRetriever")
        mock_instance = MagicMock(name="retriever_instance")
        mock_cls.return_value = mock_instance

        monkeypatch.setattr(
            "anamnesis.lsp.symbols.SymbolRetriever",
            mock_cls,
            raising=False,
        )
        # Force re-import path by clearing cached instance
        import anamnesis.lsp.symbols as symbols_mod
        monkeypatch.setattr(symbols_mod, "SymbolRetriever", mock_cls)

        result = svc.retriever

        mock_cls.assert_called_once_with(
            "/tmp/project", lsp_manager=svc._lsp_manager,
        )
        assert result is mock_instance

    def test_editor_created_on_first_access(self, svc, monkeypatch):
        """Accessing .editor creates a CodeEditor (and a retriever)."""
        mock_retriever_cls = MagicMock(name="SymbolRetriever")
        mock_retriever = MagicMock(name="retriever_instance")
        mock_retriever_cls.return_value = mock_retriever

        mock_editor_cls = MagicMock(name="CodeEditor")
        mock_editor = MagicMock(name="editor_instance")
        mock_editor_cls.return_value = mock_editor

        import anamnesis.lsp.symbols as symbols_mod
        import anamnesis.lsp.editor as editor_mod
        monkeypatch.setattr(symbols_mod, "SymbolRetriever", mock_retriever_cls)
        monkeypatch.setattr(editor_mod, "CodeEditor", mock_editor_cls)

        result = svc.editor

        mock_editor_cls.assert_called_once_with(
            "/tmp/project", mock_retriever,
            lsp_manager=svc._lsp_manager,
        )
        assert result is mock_editor


class TestSingletonBehavior:
    """Same instance returned on subsequent access."""

    def test_retriever_same_instance_on_second_access(self, svc, monkeypatch):
        """Second access to .retriever returns the cached instance."""
        mock_cls = MagicMock(name="SymbolRetriever")
        mock_instance = MagicMock(name="retriever_instance")
        mock_cls.return_value = mock_instance

        import anamnesis.lsp.symbols as symbols_mod
        monkeypatch.setattr(symbols_mod, "SymbolRetriever", mock_cls)

        first = svc.retriever
        second = svc.retriever

        assert first is second
        mock_cls.assert_called_once()

    def test_editor_same_instance_on_second_access(self, svc, monkeypatch):
        """Second access to .editor returns the cached instance."""
        import anamnesis.lsp.symbols as symbols_mod
        import anamnesis.lsp.editor as editor_mod

        monkeypatch.setattr(symbols_mod, "SymbolRetriever", MagicMock(return_value=MagicMock()))
        mock_editor_cls = MagicMock(name="CodeEditor")
        mock_editor = MagicMock(name="editor_instance")
        mock_editor_cls.return_value = mock_editor
        monkeypatch.setattr(editor_mod, "CodeEditor", mock_editor_cls)

        first = svc.editor
        second = svc.editor

        assert first is second
        mock_editor_cls.assert_called_once()


class TestEditorRetrieverDependency:
    """Accessing .editor also creates the retriever (editor needs it)."""

    def test_editor_creates_retriever_first(self, svc, monkeypatch):
        """CodeEditor receives the SymbolRetriever as its second argument."""
        mock_retriever_cls = MagicMock(name="SymbolRetriever")
        mock_retriever = MagicMock(name="retriever_instance")
        mock_retriever_cls.return_value = mock_retriever

        mock_editor_cls = MagicMock(name="CodeEditor")
        mock_editor_cls.return_value = MagicMock(name="editor_instance")

        import anamnesis.lsp.symbols as symbols_mod
        import anamnesis.lsp.editor as editor_mod
        monkeypatch.setattr(symbols_mod, "SymbolRetriever", mock_retriever_cls)
        monkeypatch.setattr(editor_mod, "CodeEditor", mock_editor_cls)

        # Access editor (which should trigger retriever creation)
        _ = svc.editor

        # Retriever was created
        mock_retriever_cls.assert_called_once()
        # Editor received the retriever as second positional arg
        call_args = mock_editor_cls.call_args
        assert call_args[0][1] is mock_retriever


# ===========================================================================
# Navigation delegation tests
# ===========================================================================


class TestNavigationDelegation:
    """Verify find, get_overview, and find_referencing_symbols delegate correctly."""

    @pytest.fixture(autouse=True)
    def _inject_mock_retriever(self, svc, monkeypatch):
        """Inject a mock retriever directly to avoid import patching in every test."""
        self.mock_retriever = MagicMock(name="retriever")
        svc._retriever = self.mock_retriever

    def test_find_delegates_to_retriever(self, svc):
        """svc.find(...) forwards all arguments to retriever.find(...)."""
        self.mock_retriever.find.return_value = [{"name": "Foo"}]

        result = svc.find(
            "Foo",
            relative_path="src/main.py",
            depth=1,
            include_body=True,
            include_info=True,
            substring_matching=True,
        )

        self.mock_retriever.find.assert_called_once_with(
            "Foo",
            relative_path="src/main.py",
            depth=1,
            include_body=True,
            include_info=True,
            substring_matching=True,
        )
        assert result == [{"name": "Foo"}]

    def test_get_overview_delegates_to_retriever(self, svc):
        """svc.get_overview(...) forwards to retriever.get_overview(...)."""
        self.mock_retriever.get_overview.return_value = {"Class": [{"name": "Foo"}]}

        result = svc.get_overview("src/main.py", depth=2)

        self.mock_retriever.get_overview.assert_called_once_with(
            "src/main.py", depth=2,
        )
        assert result == {"Class": [{"name": "Foo"}]}

    def test_find_referencing_symbols_delegates_to_retriever(self, svc):
        """svc.find_referencing_symbols(...) forwards to retriever."""
        self.mock_retriever.find_referencing_symbols.return_value = [
            {"relative_path": "src/other.py", "line": 10}
        ]

        result = svc.find_referencing_symbols("MyClass/run", "src/main.py")

        self.mock_retriever.find_referencing_symbols.assert_called_once_with(
            "MyClass/run", "src/main.py",
            include_imports=True,
            include_self=False,
        )
        assert len(result) == 1
        assert result[0]["relative_path"] == "src/other.py"

    def test_get_diagnostics_delegates_to_retriever(self, svc):
        """svc.get_diagnostics(...) forwards to retriever.get_diagnostics(...)."""
        self.mock_retriever.get_diagnostics.return_value = [
            {"severity": "error", "message": "syntax error", "line": 5}
        ]

        result = svc.get_diagnostics("src/main.py")

        self.mock_retriever.get_diagnostics.assert_called_once_with("src/main.py")
        assert len(result) == 1
        assert result[0]["severity"] == "error"


# ===========================================================================
# Editing delegation tests
# ===========================================================================


class TestEditingDelegation:
    """Verify replace_body, insert_after, insert_before, rename delegate correctly."""

    @pytest.fixture(autouse=True)
    def _inject_mock_editor(self, svc, monkeypatch):
        """Inject a mock editor (and retriever) directly."""
        self.mock_retriever = MagicMock(name="retriever")
        self.mock_editor = MagicMock(name="editor")
        svc._retriever = self.mock_retriever
        svc._editor = self.mock_editor

    def test_replace_body_delegates_to_editor(self, svc):
        """svc.replace_body(...) forwards to editor.replace_body(...)."""
        self.mock_editor.replace_body.return_value = {"success": True}

        result = svc.replace_body("MyClass/run", "src/main.py", "def run(): pass")

        self.mock_editor.replace_body.assert_called_once_with(
            "MyClass/run", "src/main.py", "def run(): pass",
        )
        assert result == {"success": True}

    def test_insert_after_delegates_to_editor(self, svc):
        """svc.insert_after(...) forwards to editor.insert_after_symbol(...)."""
        self.mock_editor.insert_after_symbol.return_value = {
            "success": True, "inserted_at_line": 15,
        }

        result = svc.insert_after("greet", "src/main.py", "def farewell(): pass")

        self.mock_editor.insert_after_symbol.assert_called_once_with(
            "greet", "src/main.py", "def farewell(): pass",
        )
        assert result["inserted_at_line"] == 15

    def test_insert_before_delegates_to_editor(self, svc):
        """svc.insert_before(...) forwards to editor.insert_before_symbol(...)."""
        self.mock_editor.insert_before_symbol.return_value = {
            "success": True, "inserted_at_line": 1,
        }

        result = svc.insert_before("Calculator", "src/main.py", "import math\n")

        self.mock_editor.insert_before_symbol.assert_called_once_with(
            "Calculator", "src/main.py", "import math\n",
        )
        assert result["inserted_at_line"] == 1

    def test_rename_delegates_to_editor(self, svc):
        """svc.rename(...) forwards to editor.rename_symbol(...)."""
        self.mock_editor.rename_symbol.return_value = {
            "success": True,
            "old_name": "run",
            "new_name": "execute",
            "files_changed": ["src/main.py"],
            "total_edits": 3,
        }

        result = svc.rename("MyClass/run", "src/main.py", "execute")

        self.mock_editor.rename_symbol.assert_called_once_with(
            "MyClass/run", "src/main.py", "execute",
        )
        assert result["old_name"] == "run"
        assert result["new_name"] == "execute"
        assert result["total_edits"] == 3
