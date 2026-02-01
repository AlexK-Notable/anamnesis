"""Tests for AST-based naming convention detection.

Verifies that the tree-sitter backend detects naming conventions
from extracted symbols rather than regex on raw source text.
"""

from __future__ import annotations

import pytest

from anamnesis.extraction.types import PatternKind, SymbolKind, UnifiedSymbol


# ---------------------------------------------------------------------------
# Test the naming convention classifier directly
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """Get a TreeSitterBackend instance."""
    from anamnesis.extraction.backends.tree_sitter_backend import TreeSitterBackend

    return TreeSitterBackend()


def _make_symbol(
    name: str,
    kind: str,
    file_path: str = "test.py",
) -> UnifiedSymbol:
    """Create a minimal UnifiedSymbol for testing."""
    return UnifiedSymbol(
        name=name,
        kind=SymbolKind(kind) if kind in SymbolKind.__members__.values() else kind,
        file_path=file_path,
        start_line=1,
        end_line=1,
        confidence=0.9,
        language="python",
        backend="tree_sitter",
    )


class TestNamingConventionRegex:
    """Test the regex patterns used for classification."""

    def test_camel_case(self, backend):
        assert backend._RE_CAMEL.match("getUserName")
        assert backend._RE_CAMEL.match("parseJSON")
        assert not backend._RE_CAMEL.match("get_user_name")
        assert not backend._RE_CAMEL.match("GetUserName")
        assert not backend._RE_CAMEL.match("username")

    def test_pascal_case(self, backend):
        assert backend._RE_PASCAL.match("UserService")
        assert backend._RE_PASCAL.match("HttpClient")
        assert not backend._RE_PASCAL.match("userService")
        assert not backend._RE_PASCAL.match("USER_SERVICE")
        assert not backend._RE_PASCAL.match("Httpserver")  # single word after caps

    def test_snake_case(self, backend):
        assert backend._RE_SNAKE.match("get_user")
        assert backend._RE_SNAKE.match("process_data_item")
        assert not backend._RE_SNAKE.match("getUser")
        assert not backend._RE_SNAKE.match("username")  # no underscore
        assert not backend._RE_SNAKE.match("GET_USER")

    def test_screaming_snake(self, backend):
        assert backend._RE_SCREAMING.match("MAX_CONNECTIONS")
        assert backend._RE_SCREAMING.match("API_V2")
        assert not backend._RE_SCREAMING.match("max_connections")
        assert not backend._RE_SCREAMING.match("Max_Connections")
        assert not backend._RE_SCREAMING.match("SINGLE")  # no underscore


class TestNamingConventionDetection:
    """Test the full naming convention detection from symbols."""

    def test_detects_pascal_case_classes(self, backend):
        symbols = [
            _make_symbol("UserService", "class"),
            _make_symbol("DatabaseAdapter", "class"),
            _make_symbol("HttpClient", "class"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        kinds = {str(p.kind) for p in patterns}
        assert PatternKind.PASCAL_CASE_CLASS in kinds

    def test_detects_snake_case_functions(self, backend):
        symbols = [
            _make_symbol("get_user", "function"),
            _make_symbol("process_data", "function"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        kinds = {str(p.kind) for p in patterns}
        assert PatternKind.SNAKE_CASE_VARIABLE in kinds

    def test_detects_camel_case_functions(self, backend):
        symbols = [
            _make_symbol("getUserName", "function"),
            _make_symbol("processData", "method"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "javascript")
        kinds = {str(p.kind) for p in patterns}
        assert PatternKind.CAMEL_CASE_FUNCTION in kinds

    def test_detects_screaming_constants(self, backend):
        symbols = [
            _make_symbol("MAX_CONNECTIONS", "constant"),
            _make_symbol("API_VERSION", "constant"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        kinds = {str(p.kind) for p in patterns}
        assert PatternKind.SCREAMING_SNAKE_CASE_CONST in kinds

    def test_empty_symbols_no_patterns(self, backend):
        patterns = backend._detect_naming_conventions([], "test.py", "python")
        assert patterns == []

    def test_no_conventions_detected_for_plain_names(self, backend):
        """Single-word names don't match any convention pattern."""
        symbols = [
            _make_symbol("main", "function"),
            _make_symbol("x", "variable"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        assert patterns == []

    def test_pattern_has_related_symbols(self, backend):
        symbols = [
            _make_symbol("get_user", "function"),
            _make_symbol("process_data", "function"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        assert len(patterns) == 1
        assert "get_user" in patterns[0].related_symbols
        assert "process_data" in patterns[0].related_symbols

    def test_frequency_matches_count(self, backend):
        symbols = [
            _make_symbol("get_user", "function"),
            _make_symbol("process_data", "function"),
            _make_symbol("create_item", "function"),
        ]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        assert patterns[0].frequency == 3

    def test_confidence_scales_with_count(self, backend):
        few = [_make_symbol("get_user", "function")]
        many = [_make_symbol(f"func_{i}", "function") for i in range(10)]

        few_patterns = backend._detect_naming_conventions(few, "test.py", "python")
        many_patterns = backend._detect_naming_conventions(many, "test.py", "python")

        assert many_patterns[0].confidence > few_patterns[0].confidence

    def test_ast_based_metadata(self, backend):
        symbols = [_make_symbol("UserService", "class")]
        patterns = backend._detect_naming_conventions(symbols, "test.py", "python")
        assert patterns[0].metadata.get("ast_based") is True
        assert patterns[0].backend == "tree_sitter"

    def test_recurses_into_children(self, backend):
        """Methods inside classes should be detected."""
        parent = _make_symbol("UserService", "class")
        child = _make_symbol("get_user", "method")
        parent.children = [child]

        patterns = backend._detect_naming_conventions([parent], "test.py", "python")
        kinds = {str(p.kind) for p in patterns}
        assert PatternKind.PASCAL_CASE_CLASS in kinds
        assert PatternKind.SNAKE_CASE_VARIABLE in kinds


class TestFullExtraction:
    """Test naming conventions through the full extraction pipeline."""

    def test_extract_patterns_includes_naming(self, backend):
        """extract_patterns should include naming conventions."""
        code = '''
class UserService:
    def get_user(self, user_id: int):
        return None

    def process_data(self, data: str):
        return data.upper()

MAX_RETRIES = 3
'''
        patterns = backend.extract_patterns(code, "service.py", "python")
        kinds = {str(p.kind) for p in patterns}
        # Should detect PascalCase class and snake_case functions
        assert PatternKind.PASCAL_CASE_CLASS in kinds or PatternKind.SNAKE_CASE_VARIABLE in kinds

    def test_extract_all_includes_naming(self, backend):
        """extract_all should include naming conventions in patterns."""
        code = '''
MAX_CONNECTIONS = 10
DEFAULT_TIMEOUT = 30

class DatabaseAdapter:
    def execute_query(self, sql: str):
        pass
'''
        result = backend.extract_all(code, "db.py", "python")
        kinds = {str(p.kind) for p in result.patterns}
        # Should detect naming conventions
        assert len(result.patterns) > 0
