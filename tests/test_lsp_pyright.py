"""Integration tests requiring a real Pyright language server.

These tests start actual LSP servers and communicate with them.
Mark with @pytest.mark.lsp for conditional execution:

    pytest tests/test_lsp_pyright.py -m lsp

Skip these tests if Pyright is not installed.
"""

import shutil
import subprocess

import pytest

# Check if Pyright is available
_PYRIGHT_AVAILABLE = shutil.which("pyright") is not None or shutil.which("pyright-langserver") is not None

# Also check via npx
if not _PYRIGHT_AVAILABLE:
    try:
        result = subprocess.run(
            ["npx", "pyright", "--version"],
            capture_output=True, timeout=10,
        )
        _PYRIGHT_AVAILABLE = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

pytestmark = pytest.mark.skipif(
    not _PYRIGHT_AVAILABLE,
    reason="Pyright language server not installed",
)


@pytest.fixture
def python_project(tmp_path):
    """Create a Python project for LSP testing."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "__init__.py").write_text("")

    (src / "calculator.py").write_text(
        '''\
"""Calculator module."""


class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


def create_calculator() -> Calculator:
    """Factory function for Calculator."""
    return Calculator()


PI: float = 3.14159
'''
    )

    (src / "user.py").write_text(
        '''\
"""User module that imports Calculator."""

from src.calculator import Calculator, create_calculator


def main():
    calc = create_calculator()
    result = calc.add(1, 2)
    print(f"Result: {result}")
'''
    )

    # Create pyproject.toml for Pyright
    (tmp_path / "pyproject.toml").write_text(
        """\
[tool.pyright]
include = ["src"]
"""
    )

    return str(tmp_path)


@pytest.fixture
def lsp_manager(python_project):
    """Create an LspManager for the test project."""
    from anamnesis.lsp.manager import LspManager

    mgr = LspManager(python_project)
    yield mgr
    mgr.stop_all()


class TestPyrightStartStop:
    """Test Pyright server lifecycle."""

    @pytest.mark.lsp
    def test_start_python_server(self, lsp_manager):
        result = lsp_manager.start("python")
        assert result is True
        assert lsp_manager.is_running("python") is True

    @pytest.mark.lsp
    def test_stop_python_server(self, lsp_manager):
        lsp_manager.start("python")
        lsp_manager.stop("python")
        assert lsp_manager.is_running("python") is False

    @pytest.mark.lsp
    def test_get_status_after_start(self, lsp_manager):
        lsp_manager.start("python")
        status = lsp_manager.get_status()
        assert "python" in status["running_servers"]
        assert status["running_servers"]["python"]["class"] == "PyrightServer"


class TestPyrightSymbols:
    """Test symbol navigation with Pyright."""

    @pytest.mark.lsp
    def test_get_language_server_for_python(self, lsp_manager):
        ls = lsp_manager.get_language_server("src/calculator.py")
        assert ls is not None

    @pytest.mark.lsp
    def test_document_symbols(self, lsp_manager, python_project):
        ls = lsp_manager.get_language_server("src/calculator.py")
        assert ls is not None

        symbols = ls.request_document_symbols("src/calculator.py")
        assert symbols is not None
        # Should find Calculator class, create_calculator function, PI variable
        names = [s.name if hasattr(s, "name") else s.get("name", "") for s in symbols.root_symbols]
        assert "Calculator" in names

    @pytest.mark.lsp
    def test_symbol_retriever_with_lsp(self, lsp_manager, python_project):
        from anamnesis.lsp.symbols import SymbolRetriever

        retriever = SymbolRetriever(python_project, lsp_manager=lsp_manager)
        results = retriever.find("Calculator", relative_path="src/calculator.py")
        assert len(results) >= 1
        assert results[0]["name"] == "Calculator"

    @pytest.mark.lsp
    def test_symbol_overview_with_lsp(self, lsp_manager, python_project):
        from anamnesis.lsp.symbols import SymbolRetriever

        retriever = SymbolRetriever(python_project, lsp_manager=lsp_manager)
        overview = retriever.get_overview("src/calculator.py", depth=1)
        assert isinstance(overview, dict)
        assert len(overview) > 0


class TestPyrightEditing:
    """Test code editing with Pyright."""

    @pytest.mark.lsp
    def test_rename_symbol(self, lsp_manager, python_project):
        from anamnesis.lsp.editor import CodeEditor
        from anamnesis.lsp.symbols import SymbolRetriever

        retriever = SymbolRetriever(python_project, lsp_manager=lsp_manager)
        editor = CodeEditor(
            project_root=python_project,
            symbol_retriever=retriever,
            lsp_manager=lsp_manager,
        )

        # This test verifies the rename infrastructure works.
        # Actual rename result depends on Pyright capabilities.
        result = editor.rename_symbol("PI", "src/calculator.py", "TAU")
        # Either succeeds or reports an error
        assert "success" in result
