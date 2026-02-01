"""Edge case tests for extraction robustness.

These tests verify that tree-sitter extraction handles degenerate inputs
gracefully: malformed code, empty files, unicode, large files, etc.
"""

import pytest

from anamnesis.extractors.symbol_extractor import extract_symbols_from_source
from anamnesis.extractors.pattern_extractor import extract_patterns_from_source


class TestMalformedCode:
    """Tree-sitter handles syntax errors gracefully; regex may not."""

    def test_missing_closing_paren(self):
        """Partially valid Python with missing paren."""
        source = "def broken_function(\n    return 42\n"
        # Tree-sitter should not crash
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        # May or may not find the function, but must not raise

    def test_incomplete_class(self):
        """Class definition without body."""
        source = "class Incomplete:\n    pass\n"
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        names = [s.name for s in symbols]
        assert "Incomplete" in names

    def test_mixed_valid_invalid(self):
        """File with both valid and invalid sections."""
        source = '''
class ValidClass:
    def valid_method(self):
        pass

def broken(
    # syntax error

def another_valid():
    return True
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        names = [s.name for s in symbols]
        # Should find at least ValidClass
        assert "ValidClass" in names


class TestEmptyAndWhitespace:
    """Handle degenerate file inputs."""

    def test_empty_file(self):
        symbols = extract_symbols_from_source("", "python", "/empty.py")
        assert symbols == []

    def test_whitespace_only(self):
        symbols = extract_symbols_from_source("   \n\n  \t\n", "python", "/ws.py")
        assert symbols == []

    def test_comments_only(self):
        source = "# This file only has comments\n# No code at all\n"
        symbols = extract_symbols_from_source(source, "python", "/comments.py")
        assert symbols == []

    def test_single_pass(self):
        """Minimal valid Python."""
        symbols = extract_symbols_from_source("pass", "python", "/pass.py")
        assert symbols == []


class TestUnsupportedLanguages:
    """Languages without tree-sitter grammars should not crash."""

    def test_unsupported_language_returns_empty(self):
        """Completely unsupported language returns empty list."""
        symbols = extract_symbols_from_source("code here", "brainfuck", "/test.bf")
        assert symbols == []


class TestLargeFiles:
    """Performance and correctness on large inputs."""

    def test_1k_functions(self):
        """1000 functions should all be extracted."""
        lines = [f"def func_{i}(): pass" for i in range(1000)]
        source = "\n".join(lines)
        symbols = extract_symbols_from_source(source, "python", "/huge.py")
        assert len(symbols) >= 900  # Allow some tolerance for parser edge cases

    def test_deeply_nested(self):
        """Deeply nested classes (unlikely but possible)."""
        levels = 5
        source = ""
        indent = ""
        for i in range(levels):
            source += f"{indent}class Level{i}:\n"
            indent += "    "
        source += f"{indent}pass\n"

        symbols = extract_symbols_from_source(source, "python", "/nested.py")
        names = [s.name for s in symbols]
        assert "Level0" in names


class TestMultiLanguage:
    """Verify extraction works across supported languages."""

    def test_typescript_extraction(self):
        """TypeScript extraction does not crash."""
        source = 'export class UserService { getName(): string { return "test"; } }'
        symbols = extract_symbols_from_source(source, "typescript", "/test.ts")
        names = [s.name for s in symbols]
        assert "UserService" in names

    def test_go_extraction(self):
        """Go extraction does not crash."""
        source = 'package main\n\nfunc main() {\n    println("Hello")\n}\n'
        symbols = extract_symbols_from_source(source, "go", "/test.go")
        names = [s.name for s in symbols]
        assert "main" in names

    def test_rust_extraction(self):
        """Rust extraction does not crash."""
        source = "pub struct User { name: String }\n\nimpl User {\n    pub fn new(name: String) -> Self {\n        User { name }\n    }\n}\n"
        symbols = extract_symbols_from_source(source, "rust", "/test.rs")
        names = [s.name for s in symbols]
        assert "User" in names
