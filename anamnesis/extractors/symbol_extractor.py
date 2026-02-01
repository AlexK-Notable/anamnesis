"""
Symbol Extractor - Extract symbols (functions, classes, methods) from parsed code.

This module provides comprehensive symbol extraction that goes beyond basic
parsing to include:
- Full symbol metadata (visibility, decorators, generics)
- Symbol relationships (parent/child, overrides)
- Documentation extraction
- Type information where available
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from anamnesis.parsing.ast_types import ASTContext, NodeType, ParsedNode


class SymbolKind(StrEnum):
    """Kind of extracted symbol.

    .. deprecated::
        Use ``anamnesis.extraction.types.SymbolKind`` instead.
        This enum is retained for backward compatibility with existing code
        that uses the extractors directly. New code should use the unified
        extraction types from ``anamnesis.extraction``.
    """

    # Declarations
    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    TRAIT = "trait"
    STRUCT = "struct"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    FIELD = "field"
    VARIABLE = "variable"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"

    # Special
    NAMESPACE = "namespace"
    PACKAGE = "package"
    DECORATOR = "decorator"
    LAMBDA = "lambda"


@dataclass
class ExtractedSymbol:
    """A symbol extracted from source code with full metadata."""

    # Identity
    name: str
    kind: SymbolKind | str
    file_path: str

    # Location
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    # Hierarchy
    parent_name: str | None = None
    qualified_name: str | None = None  # Full path like "module.Class.method"

    # Code
    signature: str | None = None  # Function/method signature
    body_hash: str | None = None  # Hash of body for change detection
    source_text: str | None = None  # Original source (optional)

    # Documentation
    docstring: str | None = None
    comments: list[str] = field(default_factory=list)

    # Attributes
    visibility: str = "public"  # public, private, protected, internal
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    is_exported: bool = False
    decorators: list[str] = field(default_factory=list)

    # Type info
    return_type: str | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)
    type_parameters: list[str] = field(default_factory=list)  # Generics

    # Relationships
    children: list["ExtractedSymbol"] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # Symbols this references
    dependencies: list[str] = field(default_factory=list)  # Imports used

    # Metadata
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        """Number of lines the symbol spans."""
        return self.end_line - self.start_line + 1

    @property
    def is_private(self) -> bool:
        """Check if symbol is private."""
        return self.visibility == "private" or self.name.startswith("_")

    def get_full_path(self) -> str:
        """Get the fully qualified path of this symbol."""
        if self.qualified_name:
            return self.qualified_name
        if self.parent_name:
            return f"{self.parent_name}.{self.name}"
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

        # Add optional fields if set
        if self.parent_name:
            result["parent_name"] = self.parent_name
        if self.qualified_name:
            result["qualified_name"] = self.qualified_name
        if self.signature:
            result["signature"] = self.signature
        if self.docstring:
            result["docstring"] = self.docstring
        if self.visibility != "public":
            result["visibility"] = self.visibility
        if self.is_async:
            result["is_async"] = True
        if self.is_static:
            result["is_static"] = True
        if self.is_abstract:
            result["is_abstract"] = True
        if self.is_exported:
            result["is_exported"] = True
        if self.decorators:
            result["decorators"] = self.decorators
        if self.return_type:
            result["return_type"] = self.return_type
        if self.parameters:
            result["parameters"] = self.parameters
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.language:
            result["language"] = self.language
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedSymbol":
        """Create from dictionary."""
        # Handle kind conversion
        kind_val = data.get("kind", "function")
        try:
            kind = SymbolKind(kind_val)
        except ValueError:
            kind = kind_val  # Keep as string if not a valid SymbolKind

        # Build children recursively
        children = [cls.from_dict(c) for c in data.get("children", [])]

        return cls(
            name=data["name"],
            kind=kind,
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_col=data.get("start_col", 0),
            end_col=data.get("end_col", 0),
            parent_name=data.get("parent_name"),
            qualified_name=data.get("qualified_name"),
            signature=data.get("signature"),
            body_hash=data.get("body_hash"),
            source_text=data.get("source_text"),
            docstring=data.get("docstring"),
            comments=data.get("comments", []),
            visibility=data.get("visibility", "public"),
            is_async=data.get("is_async", False),
            is_static=data.get("is_static", False),
            is_abstract=data.get("is_abstract", False),
            is_exported=data.get("is_exported", False),
            decorators=data.get("decorators", []),
            return_type=data.get("return_type"),
            parameters=data.get("parameters", []),
            type_parameters=data.get("type_parameters", []),
            children=children,
            references=data.get("references", []),
            dependencies=data.get("dependencies", []),
            language=data.get("language", ""),
            metadata=data.get("metadata", {}),
        )


class SymbolExtractor:
    """Extracts symbols from parsed AST contexts."""

    # Mapping from NodeType to SymbolKind
    NODE_TO_SYMBOL_KIND: dict[str, SymbolKind] = {
        "module": SymbolKind.MODULE,
        "class": SymbolKind.CLASS,
        "interface": SymbolKind.INTERFACE,
        "function": SymbolKind.FUNCTION,
        "method": SymbolKind.METHOD,
        "variable": SymbolKind.VARIABLE,
        "constant": SymbolKind.CONSTANT,
        "type_alias": SymbolKind.TYPE_ALIAS,
        "enum": SymbolKind.ENUM,
        "lambda": SymbolKind.LAMBDA,
        "decorator": SymbolKind.DECORATOR,
    }

    def __init__(self, include_private: bool = True, include_body: bool = False):
        """Initialize symbol extractor.

        Args:
            include_private: Whether to include private symbols
            include_body: Whether to include source text in extracted symbols
        """
        self.include_private = include_private
        self.include_body = include_body

    def extract(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract all symbols from an AST context.

        Args:
            context: Parsed AST context

        Returns:
            List of extracted symbols
        """
        if not context.is_valid or context.root is None:
            return []

        symbols: list[ExtractedSymbol] = []
        self._extract_from_node(
            context.root,
            context.file_path,
            context.language,
            context.source_code,
            symbols,
            parent_name=None,
        )
        return symbols

    def extract_from_file(
        self, file_path: str, source: str, language: str
    ) -> list[ExtractedSymbol]:
        """Extract symbols from source code.

        Args:
            file_path: Path to the file
            source: Source code
            language: Programming language

        Returns:
            List of extracted symbols
        """
        from anamnesis.parsing.tree_sitter_wrapper import TreeSitterParser

        try:
            parser = TreeSitterParser(language)
            context = parser.parse_to_context(source, file_path)
            return self.extract(context)
        except ValueError:
            # Unsupported language
            return []

    def _extract_from_node(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        symbols: list[ExtractedSymbol],
        parent_name: str | None,
        depth: int = 0,
        decorators: list[str] | None = None,
    ) -> None:
        """Recursively extract symbols from a parsed node.

        Args:
            node: Parsed node to extract from
            file_path: Path to the source file
            language: Programming language
            source: Original source code
            symbols: List to append symbols to
            parent_name: Name of parent symbol (for qualified names)
            depth: Current recursion depth
            decorators: Decorators to attach to the next symbol (from parent decorated_definition)
        """
        # Skip module nodes but process their children
        if node.node_type == NodeType.MODULE or node.node_type == "module":
            for child in node.children:
                self._extract_from_node(
                    child, file_path, language, source, symbols, parent_name, depth
                )
            return

        ts_type = node.metadata.get("tree_sitter_type", "")

        # Handle decorated definitions specially
        if ts_type == "decorated_definition" or node.node_type == NodeType.DECORATOR:
            # Collect decorators from children
            collected_decorators: list[str] = []
            inner_node = None

            for child in node.children:
                child_ts_type = child.metadata.get("tree_sitter_type", "")
                if child_ts_type == "decorator":
                    # Get the decorator text (e.g., "@staticmethod")
                    collected_decorators.append(child.text.strip())
                elif child_ts_type in ("function_definition", "async_function_definition",
                                       "class_definition"):
                    inner_node = child
                elif child.node_type in (NodeType.FUNCTION, NodeType.CLASS, NodeType.METHOD):
                    inner_node = child

            if inner_node:
                # Process the inner function/class with the collected decorators
                self._extract_from_node(
                    inner_node, file_path, language, source, symbols,
                    parent_name, depth, decorators=collected_decorators
                )
            else:
                # No inner node found, process children normally
                for child in node.children:
                    self._extract_from_node(
                        child, file_path, language, source, symbols, parent_name, depth
                    )
            return

        # Check if this node represents a symbol
        symbol = self._node_to_symbol(node, file_path, language, source, parent_name)

        if symbol:
            # Attach decorators from parent decorated_definition
            if decorators:
                symbol.decorators = decorators.copy()
                # Update is_static based on decorators
                if any(d in ("@staticmethod", "@classmethod") for d in decorators):
                    symbol.is_static = True
                if any(d in ("@abstractmethod", "@abc.abstractmethod") for d in decorators):
                    symbol.is_abstract = True

            # Skip private if not including them
            if not self.include_private and symbol.is_private:
                return

            # Extract child symbols
            child_symbols: list[ExtractedSymbol] = []
            for child in node.children:
                self._extract_from_node(
                    child,
                    file_path,
                    language,
                    source,
                    child_symbols,
                    symbol.name,
                    depth + 1,
                )

            symbol.children = child_symbols
            symbols.append(symbol)
        else:
            # Not a symbol node, but check children
            for child in node.children:
                self._extract_from_node(
                    child, file_path, language, source, symbols, parent_name, depth
                )

    def _node_to_symbol(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        parent_name: str | None,
    ) -> ExtractedSymbol | None:
        """Convert a parsed node to an extracted symbol.

        Args:
            node: Parsed node
            file_path: Path to the source file
            language: Programming language
            source: Original source code
            parent_name: Name of parent symbol

        Returns:
            ExtractedSymbol if node is a symbol, None otherwise
        """
        # Check if this is a symbol type
        node_type_str = (
            node.node_type.value
            if isinstance(node.node_type, NodeType)
            else node.node_type
        )

        # Only extract named declarations
        if not node.name:
            return None

        # Map node type to symbol kind
        kind = self.NODE_TO_SYMBOL_KIND.get(node_type_str)
        if kind is None:
            return None

        # Build qualified name
        qualified_name = f"{parent_name}.{node.name}" if parent_name else node.name

        # Determine visibility
        visibility = self._determine_visibility(node, language)

        # Extract signature for functions/methods
        signature = self._extract_signature(node, source) if kind in (
            SymbolKind.FUNCTION,
            SymbolKind.METHOD,
            SymbolKind.CONSTRUCTOR,
        ) else None

        # Get is_async from metadata
        is_async = node.metadata.get("is_async", False) or node.is_async

        symbol = ExtractedSymbol(
            name=node.name,
            kind=kind,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            start_col=node.start_col,
            end_col=node.end_col,
            parent_name=parent_name,
            qualified_name=qualified_name,
            signature=signature,
            docstring=node.docstring,
            visibility=visibility,
            is_async=is_async,
            is_static=self._is_static(node),
            is_abstract=self._is_abstract(node),
            is_exported=self._is_exported(node),
            decorators=node.decorators.copy() if node.decorators else [],
            language=language,
        )

        # Include source text if requested
        if self.include_body:
            symbol.source_text = node.text

        # Extract parameters for functions/methods
        if kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CONSTRUCTOR):
            symbol.parameters = self._extract_parameters(node, language)
            symbol.return_type = self._extract_return_type(node, language)

        return symbol

    def _determine_visibility(self, node: ParsedNode, language: str) -> str:
        """Determine the visibility of a symbol.

        Args:
            node: Parsed node
            language: Programming language

        Returns:
            Visibility string: public, private, protected, or internal
        """
        # Use explicit visibility if set
        if node.visibility:
            return node.visibility

        # Language-specific conventions
        if language == "python":
            if node.name.startswith("__") and not node.name.endswith("__"):
                return "private"
            elif node.name.startswith("_"):
                return "protected"
            return "public"

        elif language in ("javascript", "typescript"):
            # Check for # prefix (private fields)
            if node.name.startswith("#"):
                return "private"
            return "public"

        elif language in ("java", "kotlin", "csharp"):
            # These use explicit modifiers - default to package/internal
            return node.visibility or "internal"

        elif language == "go":
            # Go uses capitalization
            if node.name and node.name[0].isupper():
                return "public"
            return "private"

        elif language == "rust":
            # Rust uses pub keyword - check decorators/metadata
            if "pub" in node.decorators:
                return "public"
            return "private"

        return "public"

    def _is_static(self, node: ParsedNode) -> bool:
        """Check if a symbol is static."""
        if "@staticmethod" in node.decorators:
            return True
        if "@classmethod" in node.decorators:
            return True
        if "static" in node.metadata.get("modifiers", []):
            return True
        return False

    def _is_abstract(self, node: ParsedNode) -> bool:
        """Check if a symbol is abstract."""
        if "@abstractmethod" in node.decorators:
            return True
        if "@abc.abstractmethod" in node.decorators:
            return True
        if "abstract" in node.metadata.get("modifiers", []):
            return True
        return False

    def _is_exported(self, node: ParsedNode) -> bool:
        """Check if a symbol is exported."""
        return node.metadata.get("is_exported", False)

    def _extract_signature(self, node: ParsedNode, source: str) -> str | None:
        """Extract the signature of a function/method.

        Args:
            node: Parsed node
            source: Original source code

        Returns:
            Signature string or None
        """
        # Get the first line of the function/method
        lines = node.text.split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # For Python, return up to the colon
        if ":" in first_line:
            return first_line.rstrip(":")

        # For other languages, return the first line
        return first_line

    def _extract_parameters(
        self, node: ParsedNode, language: str
    ) -> list[dict[str, Any]]:
        """Extract parameters from a function/method node.

        Args:
            node: Parsed node
            language: Programming language

        Returns:
            List of parameter dictionaries
        """
        params: list[dict[str, Any]] = []

        # Look for parameters child node
        for child in node.children:
            ts_type = child.metadata.get("tree_sitter_type", "")
            if ts_type in ("parameters", "formal_parameters", "parameter_list"):
                # Extract individual parameters
                for param_child in child.children:
                    param_type = param_child.metadata.get("tree_sitter_type", "")
                    if param_type in (
                        "identifier",
                        "typed_parameter",
                        "default_parameter",
                        "typed_default_parameter",
                        "parameter",
                    ):
                        param_info = self._parse_parameter(param_child, language)
                        if param_info:
                            params.append(param_info)

        return params

    def _parse_parameter(
        self, node: ParsedNode, language: str
    ) -> dict[str, Any] | None:
        """Parse a single parameter node.

        Args:
            node: Parameter node
            language: Programming language

        Returns:
            Parameter dictionary or None
        """
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "identifier":
            return {"name": node.text, "type": None, "default": None}

        # For typed parameters, extract name and type
        name = None
        param_type = None
        default = None

        for child in node.children:
            child_type = child.metadata.get("tree_sitter_type", "")
            if child_type == "identifier":
                if name is None:
                    name = child.text
            elif child_type == "type":
                param_type = child.text
            elif child_type in ("default_value", "assignment"):
                default = child.text

        if name:
            return {"name": name, "type": param_type, "default": default}

        # Fallback: use entire text
        if node.name:
            return {"name": node.name, "type": None, "default": None}

        return None

    def _extract_return_type(self, node: ParsedNode, language: str) -> str | None:
        """Extract return type from a function/method node.

        Args:
            node: Parsed node
            language: Programming language

        Returns:
            Return type string or None
        """
        for child in node.children:
            ts_type = child.metadata.get("tree_sitter_type", "")
            if ts_type in ("type", "return_type", "type_annotation"):
                return child.text

        return None


def extract_symbols_from_source(
    source: str,
    language: str,
    file_path: str = "<string>",
    include_private: bool = True,
) -> list[ExtractedSymbol]:
    """Convenience function to extract symbols from source code.

    Args:
        source: Source code
        language: Programming language
        file_path: Path to the file (for context)
        include_private: Whether to include private symbols

    Returns:
        List of extracted symbols
    """
    extractor = SymbolExtractor(include_private=include_private)
    return extractor.extract_from_file(file_path, source, language)
