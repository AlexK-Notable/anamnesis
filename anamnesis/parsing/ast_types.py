"""
AST Types for the parsing infrastructure.

Provides dataclasses for representing parsed AST nodes and query results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from anamnesis.utils.helpers import enum_value


class NodeType(str, Enum):
    """Common AST node types across languages."""

    # Declarations
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    PROPERTY = "property"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    ENUM_MEMBER = "enum_member"

    # Statements
    IMPORT = "import"
    EXPORT = "export"
    RETURN = "return"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    TRY = "try"
    CATCH = "catch"
    THROW = "throw"

    # Expressions
    CALL = "call"
    ASSIGNMENT = "assignment"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    MEMBER_ACCESS = "member_access"
    INDEX_ACCESS = "index_access"
    LAMBDA = "lambda"
    AWAIT = "await"

    # Types
    TYPE_ANNOTATION = "type_annotation"
    GENERIC = "generic"

    # Other
    COMMENT = "comment"
    DECORATOR = "decorator"
    BLOCK = "block"
    UNKNOWN = "unknown"


@dataclass
class ParsedNode:
    """A parsed AST node with metadata."""

    node_type: NodeType | str
    text: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    children: list["ParsedNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields for specific node types
    name: str | None = None
    parent_name: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    is_async: bool = False
    is_static: bool = False
    is_private: bool = False
    visibility: str = "public"

    @property
    def line_count(self) -> int:
        """Number of lines this node spans."""
        return self.end_line - self.start_line + 1

    @property
    def byte_range(self) -> tuple[int, int]:
        """Start and end byte positions if available."""
        return (
            self.metadata.get("start_byte", 0),
            self.metadata.get("end_byte", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "node_type": enum_value(self.node_type),
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_col": self.start_col,
            "end_col": self.end_col,
            "children": [child.to_dict() for child in self.children],
        }

        # Add optional fields if present
        if self.name:
            result["name"] = self.name
        if self.parent_name:
            result["parent_name"] = self.parent_name
        if self.docstring:
            result["docstring"] = self.docstring
        if self.decorators:
            result["decorators"] = self.decorators
        if self.parameters:
            result["parameters"] = self.parameters
        if self.return_type:
            result["return_type"] = self.return_type
        if self.is_async:
            result["is_async"] = True
        if self.is_static:
            result["is_static"] = True
        if self.is_private:
            result["is_private"] = True
        if self.visibility != "public":
            result["visibility"] = self.visibility
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedNode":
        """Create from dictionary."""
        node_type = data["node_type"]
        try:
            node_type = NodeType(node_type)
        except ValueError:
            pass  # Keep as string if not a known type

        return cls(
            node_type=node_type,
            text=data["text"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_col=data["start_col"],
            end_col=data["end_col"],
            children=[cls.from_dict(c) for c in data.get("children", [])],
            metadata=data.get("metadata", {}),
            name=data.get("name"),
            parent_name=data.get("parent_name"),
            docstring=data.get("docstring"),
            decorators=data.get("decorators", []),
            parameters=data.get("parameters", []),
            return_type=data.get("return_type"),
            is_async=data.get("is_async", False),
            is_static=data.get("is_static", False),
            is_private=data.get("is_private", False),
            visibility=data.get("visibility", "public"),
        )


@dataclass
class QueryCapture:
    """A single capture from a tree-sitter query."""

    name: str
    node: ParsedNode
    pattern_index: int = 0


@dataclass
class QueryMatch:
    """A match result from a tree-sitter query."""

    pattern_index: int
    captures: list[QueryCapture] = field(default_factory=list)

    def get_capture(self, name: str) -> QueryCapture | None:
        """Get a capture by name."""
        for capture in self.captures:
            if capture.name == name:
                return capture
        return None

    def get_captures(self, name: str) -> list[QueryCapture]:
        """Get all captures with a given name."""
        return [c for c in self.captures if c.name == name]


@dataclass
class ASTContext:
    """Context for AST parsing and traversal."""

    file_path: str
    language: str
    source_code: str
    root: ParsedNode | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Cached data for quick lookups
    _symbols: dict[str, list[ParsedNode]] = field(default_factory=dict)
    _imports: list[ParsedNode] = field(default_factory=list)
    _exports: list[ParsedNode] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if parsing produced errors."""
        return len(self.errors) > 0

    @property
    def is_valid(self) -> bool:
        """Check if the AST is valid (parsed without critical errors)."""
        return self.root is not None

    def get_symbols_by_type(self, node_type: NodeType | str) -> list[ParsedNode]:
        """Get all symbols of a specific type."""
        type_key = enum_value(node_type)
        return self._symbols.get(type_key, [])

    def add_symbol(self, node: ParsedNode) -> None:
        """Register a symbol in the context."""
        type_key = enum_value(node.node_type)
        if type_key not in self._symbols:
            self._symbols[type_key] = []
        self._symbols[type_key].append(node)

    def get_line(self, line_num: int) -> str:
        """Get a specific line from the source code (1-indexed)."""
        lines = self.source_code.splitlines()
        if 1 <= line_num <= len(lines):
            return lines[line_num - 1]
        return ""

    def get_lines(self, start: int, end: int) -> str:
        """Get a range of lines from the source code (1-indexed, inclusive)."""
        lines = self.source_code.splitlines()
        if start < 1:
            start = 1
        if end > len(lines):
            end = len(lines)
        return "\n".join(lines[start - 1 : end])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "root": self.root.to_dict() if self.root else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }
