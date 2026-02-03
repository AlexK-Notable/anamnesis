"""Tests for tree-sitter wrapper module."""

import pytest

from anamnesis.parsing.ast_types import NodeType
from anamnesis.parsing.tree_sitter_wrapper import (
    LANGUAGE_MAP,
    TreeSitterParser,
    TreeSitterQuery,
    get_supported_languages,
    is_language_supported,
)


class TestLanguageSupport:
    """Tests for language support utilities."""

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert "python" in languages
        assert "typescript" in languages
        assert "javascript" in languages
        assert "rust" in languages
        assert "go" in languages
        assert "java" in languages

    def test_is_language_supported(self):
        """Test checking language support."""
        assert is_language_supported("python") is True
        assert is_language_supported("Python") is True  # Case insensitive
        assert is_language_supported("PYTHON") is True
        assert is_language_supported("typescript") is True
        assert is_language_supported("nonexistent") is False

    def test_language_map_completeness(self):
        """Test that language map has all expected languages."""
        expected = [
            "python", "typescript", "javascript", "tsx", "jsx",
            "rust", "go", "java", "c", "cpp", "csharp",
            "ruby", "php", "swift", "kotlin", "scala",
            "html", "css", "json", "yaml", "toml",
            "markdown", "bash", "shell", "sql", "lua",
        ]
        for lang in expected:
            assert lang in LANGUAGE_MAP, f"Missing language: {lang}"


class TestTreeSitterParser:
    """Tests for TreeSitterParser class."""

    def test_init_python(self):
        """Test initializing Python parser."""
        parser = TreeSitterParser("python")
        assert parser.language_name == "python"
        assert parser.language is not None

    def test_init_typescript(self):
        """Test initializing TypeScript parser."""
        parser = TreeSitterParser("typescript")
        assert parser.language_name == "typescript"

    def test_init_case_insensitive(self):
        """Test parser initialization is case insensitive."""
        parser = TreeSitterParser("Python")
        assert parser.language_name == "python"

    def test_init_unsupported_language(self):
        """Test initialization with unsupported language."""
        with pytest.raises(ValueError) as exc_info:
            TreeSitterParser("nonexistent")
        assert "Unsupported language" in str(exc_info.value)

    def test_parse_simple_python(self):
        """Test parsing simple Python code."""
        parser = TreeSitterParser("python")
        tree = parser.parse("x = 1")
        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "module"

    def test_parse_python_function(self):
        """Test parsing Python function."""
        parser = TreeSitterParser("python")
        source = "def hello():\n    return 'world'"
        tree = parser.parse(source)
        root = tree.root_node
        
        # Find function definition
        func_node = None
        for child in root.children:
            if child.type == "function_definition":
                func_node = child
                break
        
        assert func_node is not None

    def test_parse_bytes(self):
        """Test parsing bytes input."""
        parser = TreeSitterParser("python")
        tree = parser.parse(b"x = 1")
        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_to_context(self):
        """Test parsing to ASTContext."""
        parser = TreeSitterParser("python")
        source = "def foo(): pass"
        context = parser.parse_to_context(source, "/test.py")
        
        assert context.file_path == "/test.py"
        assert context.language == "python"
        assert context.source_code == source
        assert context.is_valid
        assert context.root is not None
        assert context.root.node_type == NodeType.MODULE

    def test_parse_to_context_with_error(self):
        """Test parsing code with syntax errors."""
        parser = TreeSitterParser("python")
        source = "def broken("  # Missing closing paren
        context = parser.parse_to_context(source)
        
        assert context.is_valid  # Still parses (partial)
        assert context.has_errors  # But has errors

    def test_node_to_parsed_basic(self):
        """Test converting tree-sitter node to ParsedNode."""
        parser = TreeSitterParser("python")
        source = "x = 1"
        tree = parser.parse(source)
        
        parsed = parser._node_to_parsed(tree.root_node, source)
        assert parsed.node_type == NodeType.MODULE
        assert parsed.text == source
        assert parsed.start_line == 1
        assert parsed.end_line == 1

    def test_node_metadata(self):
        """Test that node metadata is captured."""
        parser = TreeSitterParser("python")
        source = "x = 1"
        context = parser.parse_to_context(source)
        
        root = context.root
        assert "tree_sitter_type" in root.metadata
        assert "start_byte" in root.metadata
        assert "end_byte" in root.metadata
        assert "is_named" in root.metadata

    def test_traverse(self):
        """Test AST traversal."""
        parser = TreeSitterParser("python")
        source = "x = 1\ny = 2"
        tree = parser.parse(source)

        nodes = []
        def collect(node, src):
            nodes.append(node.type)
            return True  # Continue traversal

        parser.traverse(tree, source, collect)
        assert "module" in nodes
        assert "assignment" in nodes  # Tree-sitter uses 'assignment' for Python

    def test_traverse_stop_early(self):
        """Test stopping traversal early."""
        parser = TreeSitterParser("python")
        source = "x = 1\ny = 2"
        tree = parser.parse(source)
        
        visited = []
        def stop_at_first(node, src):
            visited.append(node.type)
            if node.type == "expression_statement":
                return False  # Stop traversing children
            return True
        
        parser.traverse(tree, source, stop_at_first)
        # Should have visited fewer nodes by stopping

    def test_walk(self):
        """Test walking all nodes."""
        parser = TreeSitterParser("python")
        source = "x = 1"
        tree = parser.parse(source)
        
        nodes = list(parser.walk(tree))
        assert len(nodes) > 0
        assert nodes[0].type == "module"

    def test_find_nodes_by_type(self):
        """Test finding nodes by type."""
        parser = TreeSitterParser("python")
        source = "def foo(): pass\ndef bar(): pass"
        tree = parser.parse(source)
        
        funcs = parser.find_nodes(tree, node_types=["function_definition"])
        assert len(funcs) == 2

    def test_find_nodes_with_predicate(self):
        """Test finding nodes with custom predicate."""
        parser = TreeSitterParser("python")
        source = "x = 1\ny = 2\nz = 3"
        tree = parser.parse(source)
        
        # Find only identifiers that are 'y'
        nodes = parser.find_nodes(
            tree,
            node_types=["identifier"],
            predicate=lambda n: source[n.start_byte:n.end_byte] == "y"
        )
        assert len(nodes) == 1


class TestTreeSitterQuery:
    """Tests for TreeSitterQuery class."""

    def test_create_query(self):
        """Test creating a query."""
        parser = TreeSitterParser("python")
        query = TreeSitterQuery(parser, "(function_definition) @func")
        assert query.pattern_count >= 1
        assert "func" in query.capture_names

    def test_invalid_query(self):
        """Test creating an invalid query."""
        parser = TreeSitterParser("python")
        with pytest.raises(ValueError) as exc_info:
            TreeSitterQuery(parser, "invalid query syntax [[[")
        assert "Invalid query" in str(exc_info.value)

    def test_execute_query(self):
        """Test executing a query."""
        parser = TreeSitterParser("python")
        query = TreeSitterQuery(parser, "(function_definition) @func")
        
        source = "def foo(): pass\ndef bar(): pass"
        tree = parser.parse(source)
        
        matches = query.execute(tree, source)
        assert len(matches) >= 1

    def test_find_all(self):
        """Test finding all matching nodes."""
        parser = TreeSitterParser("python")
        query = TreeSitterQuery(
            parser,
            "(function_definition name: (identifier) @name)"
        )
        
        source = "def foo(): pass\ndef bar(): pass"
        tree = parser.parse(source)
        
        nodes = query.find_all(tree, source, capture_name="name")
        assert len(nodes) == 2
        names = [n.text for n in nodes]
        assert "foo" in names
        assert "bar" in names

    def test_find_all_classes(self):
        """Test finding class definitions."""
        parser = TreeSitterParser("python")
        query = TreeSitterQuery(
            parser,
            "(class_definition name: (identifier) @name)"
        )
        
        source = "class Foo:\n    pass\n\nclass Bar:\n    pass"
        tree = parser.parse(source)
        
        nodes = query.find_all(tree, source, capture_name="name")
        assert len(nodes) == 2


class TestNodeTypeMapping:
    """Tests for node type mapping."""

    def test_python_function_mapping(self):
        """Test Python function maps to FUNCTION."""
        parser = TreeSitterParser("python")
        result = parser._map_node_type("function_definition")
        assert result == NodeType.FUNCTION

    def test_python_class_mapping(self):
        """Test Python class maps to CLASS."""
        parser = TreeSitterParser("python")
        result = parser._map_node_type("class_definition")
        assert result == NodeType.CLASS

    def test_python_import_mapping(self):
        """Test Python imports map to IMPORT."""
        parser = TreeSitterParser("python")
        assert parser._map_node_type("import_statement") == NodeType.IMPORT
        assert parser._map_node_type("import_from_statement") == NodeType.IMPORT

    def test_unknown_type_passthrough(self):
        """Test unknown types pass through as strings."""
        parser = TreeSitterParser("python")
        result = parser._map_node_type("some_unknown_type")
        assert result == "some_unknown_type"


class TestMultiLanguage:
    """Tests for multiple language support."""

    def test_parse_typescript(self):
        """Test parsing TypeScript code."""
        parser = TreeSitterParser("typescript")
        source = "function hello(): string { return 'world'; }"
        tree = parser.parse(source)
        assert tree.root_node.type == "program"

    def test_parse_javascript(self):
        """Test parsing JavaScript code."""
        parser = TreeSitterParser("javascript")
        source = "const x = () => 42;"
        tree = parser.parse(source)
        assert tree.root_node.type == "program"

    def test_parse_rust(self):
        """Test parsing Rust code."""
        parser = TreeSitterParser("rust")
        source = "fn main() { println!(\"Hello\"); }"
        tree = parser.parse(source)
        assert tree.root_node.type == "source_file"

    def test_parse_go(self):
        """Test parsing Go code."""
        parser = TreeSitterParser("go")
        source = "package main\n\nfunc main() {}"
        tree = parser.parse(source)
        assert tree.root_node.type == "source_file"

    def test_parse_java(self):
        """Test parsing Java code."""
        parser = TreeSitterParser("java")
        source = "public class Main { public static void main(String[] args) {} }"
        tree = parser.parse(source)
        assert tree.root_node.type == "program"

    @pytest.mark.parametrize("language", [
        "python", "typescript", "javascript", "rust", "go", "java",
        "c", "cpp", "ruby", "php", "kotlin", "scala",
    ])
    def test_parser_initialization(self, language):
        """Test that all major languages can be initialized."""
        try:
            parser = TreeSitterParser(language)
            assert parser.language_name == language
        except ValueError as e:
            # Some languages might not be available in the pack
            pytest.skip(f"Language {language} not available: {e}")
