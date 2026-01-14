"""
Phase 2 Tests: Parser Manager

These tests verify that Python ParserManager matches Rust behavior:
- All 11 languages can be parsed
- AST node types match Rust output
- Error handling matches Rust behavior

Reference: rust-core/src/parsing/manager.rs
"""

import pytest

# Placeholder imports - uncomment when parsing is implemented
# from anamnesis.parsing import ParserManager
# from anamnesis.types import ParseError


class TestParserManagerCreation:
    """Tests for ParserManager initialization."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_creation(self):
        """ParserManager can be created."""
        manager = ParserManager()
        assert manager is not None

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_available_languages(self):
        """ParserManager reports correct available languages."""
        manager = ParserManager()
        languages = manager.available_languages()

        expected = [
            "typescript", "javascript", "python", "rust",
            "go", "java", "c", "cpp", "csharp", "sql", "svelte"
        ]

        for lang in expected:
            assert lang in languages, f"Missing language: {lang}"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_supports_language(self):
        """ParserManager.supports_language works correctly."""
        manager = ParserManager()

        assert manager.supports_language("typescript")
        assert manager.supports_language("javascript")
        assert manager.supports_language("python")
        assert manager.supports_language("rust")
        assert manager.supports_language("go")

        assert not manager.supports_language("unknown")
        assert not manager.supports_language("")


class TestParserManagerParsing:
    """Tests for ParserManager.parse functionality."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_typescript(self, typescript_sample):
        """Parse TypeScript code."""
        manager = ParserManager()
        filename, code = typescript_sample

        tree = manager.parse(code, "typescript")

        assert tree is not None
        assert tree.root_node.type == "program"
        assert tree.root_node.child_count > 0

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_javascript(self, javascript_sample):
        """Parse JavaScript code."""
        manager = ParserManager()
        filename, code = javascript_sample

        tree = manager.parse(code, "javascript")

        assert tree is not None
        assert tree.root_node.type == "program"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_python(self, python_sample):
        """Parse Python code."""
        manager = ParserManager()
        filename, code = python_sample

        tree = manager.parse(code, "python")

        assert tree is not None
        assert tree.root_node.type == "module"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_rust(self, rust_sample):
        """Parse Rust code."""
        manager = ParserManager()
        filename, code = rust_sample

        tree = manager.parse(code, "rust")

        assert tree is not None
        assert tree.root_node.type == "source_file"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_go(self, go_sample):
        """Parse Go code."""
        manager = ParserManager()
        filename, code = go_sample

        tree = manager.parse(code, "go")

        assert tree is not None
        assert tree.root_node.type == "source_file"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_java(self, java_sample):
        """Parse Java code."""
        manager = ParserManager()
        filename, code = java_sample

        tree = manager.parse(code, "java")

        assert tree is not None
        assert tree.root_node.type == "program"

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_unsupported_language(self):
        """Parsing unsupported language raises ParseError."""
        manager = ParserManager()

        with pytest.raises(ParseError) as exc_info:
            manager.parse("some code", "unknown")

        assert "Unsupported language" in str(exc_info.value)

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_parse_invalid_code(self):
        """Parsing syntactically invalid code still produces a tree."""
        manager = ParserManager()

        # Tree-sitter is error-tolerant
        tree = manager.parse("function {{{ invalid syntax", "javascript")

        assert tree is not None
        assert tree.root_node.type == "program"
        # Tree should contain error nodes


class TestParserManagerAllLanguages:
    """Test all 11 languages parse correctly."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_all_languages(self, all_language_samples):
        """All supported languages can parse their sample code."""
        manager = ParserManager()

        for lang, (filename, code) in all_language_samples.items():
            try:
                tree = manager.parse(code, lang)
                assert tree is not None, f"Failed to parse {lang}"
                assert tree.root_node is not None, f"No root node for {lang}"
            except Exception as e:
                pytest.fail(f"Failed to parse {lang}: {e}")


class TestParserManagerComplexity:
    """Tests for complexity analysis via ParserManager."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_analyze_complexity(self):
        """analyze_complexity returns expected metrics."""
        manager = ParserManager()

        code = """
        function test() {
            if (x > 0) {
                while (y < 10) {
                    for (let i = 0; i < 5; i++) {
                        console.log(i);
                    }
                }
            }
        }
        """

        complexity = manager.analyze_complexity(code, "javascript")

        assert "cyclomatic" in complexity
        assert "cognitive" in complexity
        assert "nesting_depth" in complexity
        assert "function_count" in complexity

        # Should have at least 1 function
        assert complexity["function_count"] >= 1

        # Should detect nesting
        assert complexity["nesting_depth"] >= 3


class TestParserManagerSymbols:
    """Tests for symbol extraction via ParserManager."""

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_symbols_typescript(self):
        """get_symbols extracts TypeScript symbols."""
        manager = ParserManager()

        code = """
        class UserService {
            private name: string;

            constructor() {}

            getName(): string {
                return this.name;
            }
        }

        function helper() {}
        """

        symbols = manager.get_symbols(code, "typescript")

        symbol_names = [s.name for s in symbols]

        assert "UserService" in symbol_names
        assert "getName" in symbol_names
        assert "helper" in symbol_names

    @pytest.mark.skip(reason="Phase 2 not implemented yet")
    def test_get_symbols_python(self):
        """get_symbols extracts Python symbols."""
        manager = ParserManager()

        code = """
class User:
    def __init__(self):
        self.name = "test"

    def get_name(self):
        return self.name

def helper():
    pass
        """

        symbols = manager.get_symbols(code, "python")

        symbol_names = [s.name for s in symbols]

        assert "User" in symbol_names
        assert "__init__" in symbol_names
        assert "get_name" in symbol_names
        assert "helper" in symbol_names
