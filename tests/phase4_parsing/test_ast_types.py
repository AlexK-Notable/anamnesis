"""Tests for AST types module."""


from anamnesis.parsing.ast_types import (
    ASTContext,
    NodeType,
    ParsedNode,
    QueryCapture,
    QueryMatch,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_declaration_types(self):
        """Test declaration node types exist."""
        assert NodeType.MODULE == "module"
        assert NodeType.CLASS == "class"
        assert NodeType.FUNCTION == "function"
        assert NodeType.METHOD == "method"
        assert NodeType.VARIABLE == "variable"
        assert NodeType.CONSTANT == "constant"
        assert NodeType.INTERFACE == "interface"
        assert NodeType.TYPE_ALIAS == "type_alias"
        assert NodeType.ENUM == "enum"

    def test_statement_types(self):
        """Test statement node types exist."""
        assert NodeType.IMPORT == "import"
        assert NodeType.EXPORT == "export"
        assert NodeType.RETURN == "return"
        assert NodeType.IF == "if"
        assert NodeType.FOR == "for"
        assert NodeType.WHILE == "while"
        assert NodeType.TRY == "try"
        assert NodeType.CATCH == "catch"
        assert NodeType.THROW == "throw"

    def test_expression_types(self):
        """Test expression node types exist."""
        assert NodeType.CALL == "call"
        assert NodeType.ASSIGNMENT == "assignment"
        assert NodeType.LAMBDA == "lambda"
        assert NodeType.AWAIT == "await"
        assert NodeType.IDENTIFIER == "identifier"

    def test_string_conversion(self):
        """Test NodeType is a string enum."""
        # StrEnum value is the string, use .value to get it
        assert NodeType.FUNCTION.value == "function"
        assert NodeType.CLASS.value == "class"
        # NodeType == string comparison works due to StrEnum
        assert NodeType.FUNCTION == "function"


class TestParsedNode:
    """Tests for ParsedNode dataclass."""

    def test_basic_creation(self):
        """Test creating a basic parsed node."""
        node = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="def foo(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=15,
        )
        assert node.node_type == NodeType.FUNCTION
        assert node.text == "def foo(): pass"
        assert node.start_line == 1
        assert node.end_line == 1
        assert node.line_count == 1

    def test_node_with_name(self):
        """Test node with name attribute."""
        node = ParsedNode(
            node_type=NodeType.CLASS,
            text="class MyClass:\n    pass",
            start_line=1,
            end_line=2,
            start_col=0,
            end_col=8,
            name="MyClass",
        )
        assert node.name == "MyClass"
        assert node.line_count == 2

    def test_node_with_children(self):
        """Test node with child nodes."""
        child = ParsedNode(
            node_type=NodeType.METHOD,
            text="def method(self): pass",
            start_line=2,
            end_line=2,
            start_col=4,
            end_col=26,
            name="method",
        )
        parent = ParsedNode(
            node_type=NodeType.CLASS,
            text="class Foo:\n    def method(self): pass",
            start_line=1,
            end_line=2,
            start_col=0,
            end_col=26,
            name="Foo",
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0].name == "method"

    def test_node_with_metadata(self):
        """Test node with metadata."""
        node = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="async def fetch(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=23,
            is_async=True,
            decorators=["@cached"],
            metadata={"start_byte": 0, "end_byte": 23},
        )
        assert node.is_async is True
        assert node.decorators == ["@cached"]
        assert node.byte_range == (0, 23)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        node = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="def foo(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=15,
            name="foo",
            is_async=False,
        )
        data = node.to_dict()
        assert data["node_type"] == "function"
        assert data["name"] == "foo"
        assert data["start_line"] == 1
        assert "is_async" not in data  # False values not included

    def test_to_dict_with_async(self):
        """Test serialization includes True booleans."""
        node = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="async def foo(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=21,
            name="foo",
            is_async=True,
        )
        data = node.to_dict()
        assert data["is_async"] is True

    def test_to_dict_with_children(self):
        """Test serialization with children."""
        child = ParsedNode(
            node_type=NodeType.IDENTIFIER,
            text="x",
            start_line=1,
            end_line=1,
            start_col=4,
            end_col=5,
        )
        parent = ParsedNode(
            node_type=NodeType.ASSIGNMENT,
            text="x = 1",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=5,
            children=[child],
        )
        data = parent.to_dict()
        assert len(data["children"]) == 1
        assert data["children"][0]["node_type"] == "identifier"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "node_type": "function",
            "text": "def bar(): pass",
            "start_line": 5,
            "end_line": 5,
            "start_col": 0,
            "end_col": 15,
            "name": "bar",
            "is_async": True,
            "decorators": ["@staticmethod"],
        }
        node = ParsedNode.from_dict(data)
        assert node.node_type == NodeType.FUNCTION
        assert node.name == "bar"
        assert node.is_async is True
        assert node.decorators == ["@staticmethod"]

    def test_from_dict_unknown_type(self):
        """Test deserialization with unknown node type."""
        data = {
            "node_type": "custom_type",
            "text": "custom",
            "start_line": 1,
            "end_line": 1,
            "start_col": 0,
            "end_col": 6,
        }
        node = ParsedNode.from_dict(data)
        assert node.node_type == "custom_type"  # Kept as string

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        original = ParsedNode(
            node_type=NodeType.CLASS,
            text="class Test:\n    pass",
            start_line=1,
            end_line=2,
            start_col=0,
            end_col=8,
            name="Test",
            docstring="A test class.",
            is_private=True,
            visibility="private",
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = ParsedNode.from_dict(data)
        assert restored.node_type == original.node_type
        assert restored.name == original.name
        assert restored.docstring == original.docstring
        assert restored.visibility == original.visibility


class TestQueryCapture:
    """Tests for QueryCapture dataclass."""

    def test_basic_capture(self):
        """Test creating a query capture."""
        node = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="def test(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=16,
        )
        capture = QueryCapture(name="function.def", node=node, pattern_index=0)
        assert capture.name == "function.def"
        assert capture.node.node_type == NodeType.FUNCTION
        assert capture.pattern_index == 0


class TestQueryMatch:
    """Tests for QueryMatch dataclass."""

    def test_empty_match(self):
        """Test creating an empty query match."""
        match = QueryMatch(pattern_index=0)
        assert match.pattern_index == 0
        assert match.captures == []

    def test_match_with_captures(self):
        """Test match with multiple captures."""
        node1 = ParsedNode(
            node_type=NodeType.FUNCTION,
            text="def foo(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=15,
        )
        node2 = ParsedNode(
            node_type=NodeType.IDENTIFIER,
            text="foo",
            start_line=1,
            end_line=1,
            start_col=4,
            end_col=7,
        )
        capture1 = QueryCapture(name="function", node=node1)
        capture2 = QueryCapture(name="name", node=node2)
        match = QueryMatch(pattern_index=0, captures=[capture1, capture2])
        assert len(match.captures) == 2

    def test_get_capture(self):
        """Test getting a capture by name."""
        node = ParsedNode(
            node_type=NodeType.IDENTIFIER,
            text="test",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=4,
        )
        capture = QueryCapture(name="id", node=node)
        match = QueryMatch(pattern_index=0, captures=[capture])
        
        found = match.get_capture("id")
        assert found is not None
        assert found.node.text == "test"
        
        not_found = match.get_capture("nonexistent")
        assert not_found is None

    def test_get_captures(self):
        """Test getting multiple captures with same name."""
        node1 = ParsedNode(
            node_type=NodeType.IDENTIFIER,
            text="a",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=1,
        )
        node2 = ParsedNode(
            node_type=NodeType.IDENTIFIER,
            text="b",
            start_line=1,
            end_line=1,
            start_col=3,
            end_col=4,
        )
        captures = [
            QueryCapture(name="var", node=node1),
            QueryCapture(name="var", node=node2),
        ]
        match = QueryMatch(pattern_index=0, captures=captures)
        
        vars = match.get_captures("var")
        assert len(vars) == 2
        assert vars[0].node.text == "a"
        assert vars[1].node.text == "b"


class TestASTContext:
    """Tests for ASTContext dataclass."""

    def test_basic_context(self):
        """Test creating a basic AST context."""
        ctx = ASTContext(
            file_path="/test/file.py",
            language="python",
            source_code="def test(): pass\n",
        )
        assert ctx.file_path == "/test/file.py"
        assert ctx.language == "python"
        assert ctx.root is None
        assert not ctx.is_valid
        assert not ctx.has_errors

    def test_context_with_root(self):
        """Test context with a root node."""
        root = ParsedNode(
            node_type=NodeType.MODULE,
            text="def test(): pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=16,
        )
        ctx = ASTContext(
            file_path="/test/file.py",
            language="python",
            source_code="def test(): pass",
            root=root,
        )
        assert ctx.is_valid
        assert ctx.root.node_type == NodeType.MODULE

    def test_context_with_errors(self):
        """Test context with parsing errors."""
        ctx = ASTContext(
            file_path="/test/file.py",
            language="python",
            source_code="def broken(",
            errors=["Syntax error at line 1"],
        )
        assert ctx.has_errors
        assert len(ctx.errors) == 1

    def test_get_line(self):
        """Test getting a specific line."""
        source = "line 1\nline 2\nline 3"
        ctx = ASTContext(
            file_path="/test.py",
            language="python",
            source_code=source,
        )
        assert ctx.get_line(1) == "line 1"
        assert ctx.get_line(2) == "line 2"
        assert ctx.get_line(3) == "line 3"
        assert ctx.get_line(0) == ""  # Out of range
        assert ctx.get_line(100) == ""  # Out of range

    def test_get_lines(self):
        """Test getting a range of lines."""
        source = "a\nb\nc\nd\ne"
        ctx = ASTContext(
            file_path="/test.py",
            language="python",
            source_code=source,
        )
        assert ctx.get_lines(2, 4) == "b\nc\nd"
        assert ctx.get_lines(1, 2) == "a\nb"
        assert ctx.get_lines(0, 2) == "a\nb"  # Clamps to 1
        assert ctx.get_lines(4, 100) == "d\ne"  # Clamps to end

    def test_add_symbol(self):
        """Test adding symbols to context."""
        ctx = ASTContext(
            file_path="/test.py",
            language="python",
            source_code="class Foo: pass",
        )
        node = ParsedNode(
            node_type=NodeType.CLASS,
            text="class Foo: pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=15,
            name="Foo",
        )
        ctx.add_symbol(node)
        classes = ctx.get_symbols_by_type(NodeType.CLASS)
        assert len(classes) == 1
        assert classes[0].name == "Foo"

    def test_get_symbols_by_type_string(self):
        """Test getting symbols by string type."""
        ctx = ASTContext(
            file_path="/test.py",
            language="python",
            source_code="",
        )
        node = ParsedNode(
            node_type="custom_type",
            text="custom",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=6,
        )
        ctx.add_symbol(node)
        customs = ctx.get_symbols_by_type("custom_type")
        assert len(customs) == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        root = ParsedNode(
            node_type=NodeType.MODULE,
            text="pass",
            start_line=1,
            end_line=1,
            start_col=0,
            end_col=4,
        )
        ctx = ASTContext(
            file_path="/test.py",
            language="python",
            source_code="pass",
            root=root,
            errors=["warning"],
        )
        data = ctx.to_dict()
        assert data["file_path"] == "/test.py"
        assert data["language"] == "python"
        assert data["root"] is not None
        assert data["errors"] == ["warning"]
