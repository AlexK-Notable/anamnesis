"""Tests for SymbolService facade over LSP navigation and editing.

Covers:
- Lazy initialization: SymbolRetriever and CodeEditor not created in __init__
- Lazy init triggered on first access (.retriever, .editor)
- Singleton behavior: same instance returned on subsequent access
- Editor creation triggers retriever dependency (editor needs retriever)
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
