"""
Language-specific parsers built on top of tree-sitter.

Each parser provides specialized extraction for:
- Functions/methods
- Classes/types
- Imports/exports
- Symbols
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from anamnesis.parsing.ast_types import ASTContext, NodeType, ParsedNode
from anamnesis.parsing.tree_sitter_wrapper import (
    TreeSitterParser,
    TreeSitterQuery,
    is_language_supported,
)
from anamnesis.utils.helpers import enum_value


@dataclass
class ExtractedSymbol:
    """A symbol extracted from source code."""

    name: str
    kind: NodeType | str
    file_path: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    signature: str = ""
    docstring: str | None = None
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_type: str | None = None
    is_async: bool = False
    is_exported: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "kind": enum_value(self.kind),
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.signature:
            result["signature"] = self.signature
        if self.docstring:
            result["docstring"] = self.docstring
        if self.parent:
            result["parent"] = self.parent
        if self.children:
            result["children"] = self.children
        if self.decorators:
            result["decorators"] = self.decorators
        if self.modifiers:
            result["modifiers"] = self.modifiers
        if self.parameters:
            result["parameters"] = self.parameters
        if self.return_type:
            result["return_type"] = self.return_type
        if self.is_async:
            result["is_async"] = True
        if self.is_exported:
            result["is_exported"] = True
        return result


@dataclass
class ExtractedImport:
    """An import extracted from source code."""

    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_default: bool = False
    is_namespace: bool = False
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"module": self.module}
        if self.names:
            result["names"] = self.names
        if self.alias:
            result["alias"] = self.alias
        if self.is_default:
            result["is_default"] = True
        if self.is_namespace:
            result["is_namespace"] = True
        return result


@dataclass
class ParseResult:
    """Result of parsing a source file."""

    file_path: str
    language: str
    symbols: list[ExtractedSymbol] = field(default_factory=list)
    imports: list[ExtractedImport] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    context: ASTContext | None = None

    @property
    def has_errors(self) -> bool:
        """Check if parsing had errors."""
        return len(self.errors) > 0

    def get_classes(self) -> list[ExtractedSymbol]:
        """Get all class symbols."""
        return [s for s in self.symbols if s.kind == NodeType.CLASS]

    def get_functions(self) -> list[ExtractedSymbol]:
        """Get all function/method symbols."""
        return [
            s for s in self.symbols if s.kind in (NodeType.FUNCTION, NodeType.METHOD)
        ]

    def get_top_level_symbols(self) -> list[ExtractedSymbol]:
        """Get symbols without a parent."""
        return [s for s in self.symbols if s.parent is None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [i.to_dict() for i in self.imports],
            "exports": self.exports,
            "errors": self.errors,
        }


class LanguageParser(ABC):
    """Base class for language-specific parsers."""

    def __init__(self, language: str):
        """Initialize parser for a language.

        Args:
            language: Language name

        Raises:
            ValueError: If language is not supported
        """
        self.language = language.lower()
        self._ts_parser = TreeSitterParser(language)

    @abstractmethod
    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract symbols from parsed AST."""
        pass

    @abstractmethod
    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract imports from parsed AST."""
        pass

    def parse(self, source: str, file_path: str = "<string>") -> ParseResult:
        """Parse source code and extract symbols.

        Args:
            source: Source code to parse
            file_path: Path to the source file

        Returns:
            ParseResult with extracted information
        """
        context = self._ts_parser.parse_to_context(source, file_path)

        result = ParseResult(
            file_path=file_path,
            language=self.language,
            context=context,
            errors=context.errors.copy(),
        )

        if context.root:
            result.symbols = self.extract_symbols(context)
            result.imports = self.extract_imports(context)
            result.exports = self._extract_exports(context)

        return result

    def _extract_exports(self, context: ASTContext) -> list[str]:
        """Extract exported names (default implementation)."""
        # Override in language-specific parsers
        return []

    def _find_nodes(
        self, node: ParsedNode, node_types: list[NodeType | str]
    ) -> list[ParsedNode]:
        """Find all nodes of given types in the AST."""
        results = []
        types_set = {
            enum_value(t) for t in node_types
        }

        def _search(n: ParsedNode) -> None:
            node_type = enum_value(n.node_type)
            if node_type in types_set:
                results.append(n)
            for child in n.children:
                _search(child)

        _search(node)
        return results


class PythonParser(LanguageParser):
    """Python-specific parser."""

    def __init__(self):
        super().__init__("python")

        # Python-specific queries
        self._class_query = TreeSitterQuery(
            self._ts_parser,
            """
            (class_definition
                name: (identifier) @class.name
                body: (block) @class.body) @class.def
            """,
        )

        self._function_query = TreeSitterQuery(
            self._ts_parser,
            """
            (function_definition
                name: (identifier) @function.name
                parameters: (parameters) @function.params
                body: (block) @function.body) @function.def
            """,
        )

        self._import_query = TreeSitterQuery(
            self._ts_parser,
            """
            [
                (import_statement) @import
                (import_from_statement) @import_from
            ]
            """,
        )

    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract Python symbols."""
        symbols: list[ExtractedSymbol] = []
        if not context.root:
            return symbols

        self._extract_from_node(
            context.root, context.source_code, context.file_path, symbols, None
        )
        return symbols

    def _extract_from_node(
        self,
        node: ParsedNode,
        source: str,
        file_path: str,
        symbols: list[ExtractedSymbol],
        parent: str | None,
    ) -> None:
        """Recursively extract symbols from a node."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "class_definition":
            symbol = self._extract_class(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
                # Extract methods
                for child in node.children:
                    self._extract_from_node(
                        child, source, file_path, symbols, symbol.name
                    )
            return

        if ts_type in ("function_definition", "async_function_definition"):
            symbol = self._extract_function(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
            return

        # Continue recursion for other nodes
        for child in node.children:
            self._extract_from_node(child, source, file_path, symbols, parent)

    def _extract_class(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a class definition."""
        name = node.name
        if not name:
            # Try to find name in children
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "identifier":
                    name = child.text
                    break

        if not name:
            return None

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Extract decorators
        decorators = self._extract_decorators(node, source)

        return ExtractedSymbol(
            name=name,
            kind=NodeType.CLASS,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            start_col=node.start_col,
            end_col=node.end_col,
            signature=f"class {name}",
            docstring=docstring,
            parent=parent,
            decorators=decorators,
        )

    def _extract_function(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a function/method definition."""
        name = node.name
        if not name:
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "identifier":
                    name = child.text
                    break

        if not name:
            return None

        # Determine if async from metadata (set during tree-sitter parsing)
        is_async = node.metadata.get("is_async", False)

        # Determine kind based on parent
        kind = NodeType.METHOD if parent else NodeType.FUNCTION

        # Extract parameters
        params = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Extract decorators
        decorators = self._extract_decorators(node, source)

        # Build signature
        param_str = ", ".join(p.get("name", "") for p in params)
        async_prefix = "async " if is_async else ""
        ret_suffix = f" -> {return_type}" if return_type else ""
        signature = f"{async_prefix}def {name}({param_str}){ret_suffix}"

        return ExtractedSymbol(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            start_col=node.start_col,
            end_col=node.end_col,
            signature=signature,
            docstring=docstring,
            parent=parent,
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            is_async=is_async,
        )

    def _extract_parameters(
        self, node: ParsedNode, source: str
    ) -> list[dict[str, Any]]:
        """Extract function parameters."""
        params: list[dict[str, Any]] = []

        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "parameters":
                for param_child in child.children:
                    param_type = param_child.metadata.get("tree_sitter_type", "")
                    if param_type in ("identifier", "typed_parameter", "default_parameter"):
                        param_name = param_child.name or param_child.text.split(":")[0].split("=")[0].strip()
                        params.append({"name": param_name})

        return params

    def _extract_return_type(self, node: ParsedNode, source: str) -> str | None:
        """Extract return type annotation."""
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type":
                return child.text
        return None

    def _extract_docstring(self, node: ParsedNode, source: str) -> str | None:
        """Extract docstring from a definition."""
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "block":
                # In tree-sitter Python, docstrings are direct 'string' children of block
                for block_child in child.children:
                    ts_type = block_child.metadata.get("tree_sitter_type", "")
                    if ts_type == "string":
                        # Look for string_content child
                        for string_child in block_child.children:
                            if string_child.metadata.get("tree_sitter_type") == "string_content":
                                return string_child.text.strip()
                        # Fallback: extract from text
                        text = block_child.text
                        if text.startswith('"""') or text.startswith("'''"):
                            return text[3:-3].strip()
                        elif text.startswith('"') or text.startswith("'"):
                            return text[1:-1].strip()
                    # Also check expression_statement for backwards compatibility
                    elif ts_type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.metadata.get("tree_sitter_type") == "string":
                                text = expr_child.text
                                if text.startswith('"""') or text.startswith("'''"):
                                    return text[3:-3].strip()
                                elif text.startswith('"') or text.startswith("'"):
                                    return text[1:-1].strip()
                    else:
                        # Stop at first non-docstring statement
                        break
        return None

    def _extract_decorators(self, node: ParsedNode, source: str) -> list[str]:
        """Extract decorators from a definition."""
        decorators = []
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "decorator":
                # Remove @ prefix
                dec_text = child.text
                if dec_text.startswith("@"):
                    dec_text = dec_text[1:]
                decorators.append(dec_text)
        return decorators

    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract Python imports."""
        imports: list[ExtractedImport] = []
        if not context.root:
            return imports

        def _find_imports(node: ParsedNode) -> None:
            ts_type = node.metadata.get("tree_sitter_type", "")

            if ts_type == "import_statement":
                # import x, y, z
                names = []
                for child in node.children:
                    if child.metadata.get("tree_sitter_type") in ("dotted_name", "aliased_import"):
                        names.append(child.text)
                for name in names:
                    imports.append(
                        ExtractedImport(
                            module=name,
                            start_line=node.start_line,
                            end_line=node.end_line,
                        )
                    )

            elif ts_type == "import_from_statement":
                # from x import y, z
                # First dotted_name/relative_import is the module, rest are names
                module = ""
                names = []
                first_dotted = True
                for child in node.children:
                    child_type = child.metadata.get("tree_sitter_type", "")
                    if child_type in ("dotted_name", "relative_import"):
                        if first_dotted:
                            # First is the module name
                            module = child.text
                            first_dotted = False
                        else:
                            # Subsequent are imported names
                            names.append(child.text)
                    elif child_type in ("aliased_import", "identifier"):
                        # These are also imported names
                        names.append(child.text)

                imports.append(
                    ExtractedImport(
                        module=module,
                        names=names,
                        start_line=node.start_line,
                        end_line=node.end_line,
                    )
                )

            for child in node.children:
                _find_imports(child)

        _find_imports(context.root)
        return imports


class TypeScriptParser(LanguageParser):
    """TypeScript/JavaScript parser."""

    def __init__(self, language: str = "typescript"):
        super().__init__(language)

    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract TypeScript/JavaScript symbols."""
        symbols: list[ExtractedSymbol] = []
        if not context.root:
            return symbols

        self._extract_from_node(
            context.root, context.source_code, context.file_path, symbols, None
        )
        return symbols

    def _extract_from_node(
        self,
        node: ParsedNode,
        source: str,
        file_path: str,
        symbols: list[ExtractedSymbol],
        parent: str | None,
    ) -> None:
        """Recursively extract symbols from a node."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "class_declaration":
            symbol = self._extract_class(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
                for child in node.children:
                    self._extract_from_node(
                        child, source, file_path, symbols, symbol.name
                    )
            return

        if ts_type in ("function_declaration", "arrow_function", "method_definition"):
            symbol = self._extract_function(node, source, file_path, parent, ts_type)
            if symbol:
                symbols.append(symbol)
            return

        if ts_type == "interface_declaration":
            symbol = self._extract_interface(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
            return

        if ts_type == "type_alias_declaration":
            symbol = self._extract_type_alias(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
            return

        if ts_type == "enum_declaration":
            symbol = self._extract_enum(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)
            return

        # Continue recursion
        for child in node.children:
            self._extract_from_node(child, source, file_path, symbols, parent)

    def _extract_class(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a class declaration."""
        name = node.name
        if not name:
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "type_identifier":
                    name = child.text
                    break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.CLASS,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"class {name}",
            parent=parent,
        )

    def _extract_function(
        self,
        node: ParsedNode,
        source: str,
        file_path: str,
        parent: str | None,
        ts_type: str,
    ) -> ExtractedSymbol | None:
        """Extract a function/method."""
        name = node.name
        if not name:
            for child in node.children:
                child_type = child.metadata.get("tree_sitter_type", "")
                if child_type in ("identifier", "property_identifier"):
                    name = child.text
                    break

        if not name:
            return None

        kind = NodeType.METHOD if parent or ts_type == "method_definition" else NodeType.FUNCTION

        return ExtractedSymbol(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"function {name}()",
            parent=parent,
        )

    def _extract_interface(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract an interface declaration."""
        name = node.name
        if not name:
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "type_identifier":
                    name = child.text
                    break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.INTERFACE,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"interface {name}",
            parent=parent,
        )

    def _extract_type_alias(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a type alias declaration."""
        name = node.name
        if not name:
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "type_identifier":
                    name = child.text
                    break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.TYPE_ALIAS,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"type {name}",
            parent=parent,
        )

    def _extract_enum(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract an enum declaration."""
        name = node.name
        if not name:
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "identifier":
                    name = child.text
                    break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.ENUM,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"enum {name}",
            parent=parent,
        )

    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract TypeScript/JavaScript imports."""
        imports: list[ExtractedImport] = []
        if not context.root:
            return imports

        def _find_imports(node: ParsedNode) -> None:
            ts_type = node.metadata.get("tree_sitter_type", "")

            if ts_type == "import_statement":
                module = ""
                names = []
                is_namespace = False

                for child in node.children:
                    child_type = child.metadata.get("tree_sitter_type", "")
                    if child_type == "string":
                        # Remove quotes
                        module = child.text.strip("'\"")
                    elif child_type == "import_clause":
                        for imp_child in child.children:
                            imp_type = imp_child.metadata.get("tree_sitter_type", "")
                            if imp_type == "identifier":
                                names.append(imp_child.text)
                            elif imp_type == "named_imports":
                                for named in imp_child.children:
                                    if named.metadata.get("tree_sitter_type") in (
                                        "import_specifier",
                                        "identifier",
                                    ):
                                        names.append(named.text)
                            elif imp_type == "namespace_import":
                                is_namespace = True
                                for ns_child in imp_child.children:
                                    if ns_child.metadata.get("tree_sitter_type") == "identifier":
                                        names.append(ns_child.text)

                if module:
                    imports.append(
                        ExtractedImport(
                            module=module,
                            names=names,
                            is_namespace=is_namespace,
                            start_line=node.start_line,
                            end_line=node.end_line,
                        )
                    )

            for child in node.children:
                _find_imports(child)

        _find_imports(context.root)
        return imports


class GoParser(LanguageParser):
    """Go language parser."""

    def __init__(self):
        super().__init__("go")

    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract Go symbols."""
        symbols: list[ExtractedSymbol] = []
        if not context.root:
            return symbols

        self._extract_from_node(
            context.root, context.source_code, context.file_path, symbols, None
        )
        return symbols

    def _extract_from_node(
        self,
        node: ParsedNode,
        source: str,
        file_path: str,
        symbols: list[ExtractedSymbol],
        parent: str | None,
    ) -> None:
        """Recursively extract symbols from a node."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "type_declaration":
            for child in node.children:
                child_type = child.metadata.get("tree_sitter_type", "")
                if child_type == "type_spec":
                    symbol = self._extract_type(child, source, file_path)
                    if symbol:
                        symbols.append(symbol)

        elif ts_type == "function_declaration":
            symbol = self._extract_function(node, source, file_path, None)
            if symbol:
                symbols.append(symbol)

        elif ts_type == "method_declaration":
            symbol = self._extract_method(node, source, file_path)
            if symbol:
                symbols.append(symbol)

        # Continue recursion
        for child in node.children:
            self._extract_from_node(child, source, file_path, symbols, parent)

    def _extract_type(
        self, node: ParsedNode, source: str, file_path: str
    ) -> ExtractedSymbol | None:
        """Extract a type declaration."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type_identifier":
                name = child.text
                break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.CLASS,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"type {name}",
            is_exported=name[0].isupper(),
        )

    def _extract_function(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a function declaration."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "identifier":
                name = child.text
                break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.FUNCTION,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"func {name}()",
            parent=parent,
            is_exported=name[0].isupper(),
        )

    def _extract_method(
        self, node: ParsedNode, source: str, file_path: str
    ) -> ExtractedSymbol | None:
        """Extract a method declaration."""
        name = None
        receiver = None

        for child in node.children:
            child_type = child.metadata.get("tree_sitter_type", "")
            # Method name is in field_identifier (not identifier like functions)
            if child_type == "field_identifier":
                name = child.text
            elif child_type == "identifier":
                # Fallback for older tree-sitter versions
                if not name:
                    name = child.text
            elif child_type == "parameter_list":
                # First parameter list is receiver
                if receiver is None:
                    for param in child.children:
                        if param.metadata.get("tree_sitter_type") == "parameter_declaration":
                            for pc in param.children:
                                if pc.metadata.get("tree_sitter_type") in ("type_identifier", "pointer_type"):
                                    receiver = pc.text
                                    break

        if not name:
            return None

        return ExtractedSymbol(
            name=name,
            kind=NodeType.METHOD,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"func ({receiver}) {name}()" if receiver else f"func {name}()",
            parent=receiver,
            is_exported=name[0].isupper(),
        )

    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract Go imports."""
        imports: list[ExtractedImport] = []
        if not context.root:
            return imports

        def _find_imports(node: ParsedNode) -> None:
            ts_type = node.metadata.get("tree_sitter_type", "")

            if ts_type == "import_declaration":
                for child in node.children:
                    child_type = child.metadata.get("tree_sitter_type", "")
                    if child_type == "import_spec":
                        path = None
                        alias = None
                        for spec_child in child.children:
                            spec_type = spec_child.metadata.get("tree_sitter_type", "")
                            if spec_type == "interpreted_string_literal":
                                path = spec_child.text.strip('"')
                            elif spec_type == "package_identifier":
                                alias = spec_child.text

                        if path:
                            imports.append(
                                ExtractedImport(
                                    module=path,
                                    alias=alias,
                                    start_line=child.start_line,
                                    end_line=child.end_line,
                                )
                            )

                    elif child_type == "import_spec_list":
                        for spec in child.children:
                            if spec.metadata.get("tree_sitter_type") == "import_spec":
                                path = None
                                alias = None
                                for spec_child in spec.children:
                                    spec_type = spec_child.metadata.get("tree_sitter_type", "")
                                    if spec_type == "interpreted_string_literal":
                                        path = spec_child.text.strip('"')
                                    elif spec_type == "package_identifier":
                                        alias = spec_child.text

                                if path:
                                    imports.append(
                                        ExtractedImport(
                                            module=path,
                                            alias=alias,
                                            start_line=spec.start_line,
                                            end_line=spec.end_line,
                                        )
                                    )

            for child in node.children:
                _find_imports(child)

        _find_imports(context.root)
        return imports


class RustParser(LanguageParser):
    """Rust language parser."""

    def __init__(self):
        super().__init__("rust")

    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract Rust symbols."""
        symbols: list[ExtractedSymbol] = []
        if not context.root:
            return symbols

        self._extract_from_node(
            context.root, context.source_code, context.file_path, symbols, None
        )
        return symbols

    def _extract_from_node(
        self,
        node: ParsedNode,
        source: str,
        file_path: str,
        symbols: list[ExtractedSymbol],
        parent: str | None,
    ) -> None:
        """Recursively extract symbols from a node."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "struct_item":
            symbol = self._extract_struct(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)

        elif ts_type == "impl_item":
            # Extract impl target and methods
            impl_target = self._get_impl_target(node)
            for child in node.children:
                if child.metadata.get("tree_sitter_type") == "declaration_list":
                    for decl in child.children:
                        if decl.metadata.get("tree_sitter_type") == "function_item":
                            sym = self._extract_function(decl, source, file_path, impl_target)
                            if sym:
                                symbols.append(sym)

        elif ts_type == "function_item":
            symbol = self._extract_function(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)

        elif ts_type == "mod_item":
            symbol = self._extract_mod(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)

        elif ts_type == "enum_item":
            symbol = self._extract_enum(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)

        elif ts_type == "trait_item":
            symbol = self._extract_trait(node, source, file_path, parent)
            if symbol:
                symbols.append(symbol)

        # Continue recursion
        for child in node.children:
            self._extract_from_node(child, source, file_path, symbols, parent)

    def _get_impl_target(self, node: ParsedNode) -> str | None:
        """Get the type that an impl block targets."""
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type_identifier":
                return child.text
        return None

    def _extract_struct(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a struct definition."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type_identifier":
                name = child.text
                break

        if not name:
            return None

        is_pub = any(
            c.metadata.get("tree_sitter_type") == "visibility_modifier"
            for c in node.children
        )

        return ExtractedSymbol(
            name=name,
            kind=NodeType.CLASS,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"struct {name}",
            parent=parent,
            is_exported=is_pub,
        )

    def _extract_function(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a function definition."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "identifier":
                name = child.text
                break

        if not name:
            return None

        is_pub = any(
            c.metadata.get("tree_sitter_type") == "visibility_modifier"
            for c in node.children
        )
        is_async = any(
            c.text == "async" for c in node.children
        )

        kind = NodeType.METHOD if parent else NodeType.FUNCTION

        return ExtractedSymbol(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"fn {name}()",
            parent=parent,
            is_exported=is_pub,
            is_async=is_async,
        )

    def _extract_mod(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a module definition."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "identifier":
                name = child.text
                break

        if not name:
            return None

        is_pub = any(
            c.metadata.get("tree_sitter_type") == "visibility_modifier"
            for c in node.children
        )

        return ExtractedSymbol(
            name=name,
            kind=NodeType.MODULE,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"mod {name}",
            parent=parent,
            is_exported=is_pub,
        )

    def _extract_enum(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract an enum definition."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type_identifier":
                name = child.text
                break

        if not name:
            return None

        is_pub = any(
            c.metadata.get("tree_sitter_type") == "visibility_modifier"
            for c in node.children
        )

        return ExtractedSymbol(
            name=name,
            kind=NodeType.ENUM,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"enum {name}",
            parent=parent,
            is_exported=is_pub,
        )

    def _extract_trait(
        self, node: ParsedNode, source: str, file_path: str, parent: str | None
    ) -> ExtractedSymbol | None:
        """Extract a trait definition."""
        name = None
        for child in node.children:
            if child.metadata.get("tree_sitter_type") == "type_identifier":
                name = child.text
                break

        if not name:
            return None

        is_pub = any(
            c.metadata.get("tree_sitter_type") == "visibility_modifier"
            for c in node.children
        )

        return ExtractedSymbol(
            name=name,
            kind=NodeType.INTERFACE,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            signature=f"trait {name}",
            parent=parent,
            is_exported=is_pub,
        )

    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract Rust use declarations."""
        imports: list[ExtractedImport] = []
        if not context.root:
            return imports

        def _find_imports(node: ParsedNode) -> None:
            ts_type = node.metadata.get("tree_sitter_type", "")

            if ts_type == "use_declaration":
                # Simple extraction - get the full path
                path = node.text
                if path.startswith("use "):
                    path = path[4:]
                if path.endswith(";"):
                    path = path[:-1]

                imports.append(
                    ExtractedImport(
                        module=path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                    )
                )

            for child in node.children:
                _find_imports(child)

        _find_imports(context.root)
        return imports


# Parser registry
_PARSERS: dict[str, type[LanguageParser]] = {
    "python": PythonParser,
    "typescript": TypeScriptParser,
    "javascript": TypeScriptParser,
    "tsx": TypeScriptParser,
    "jsx": TypeScriptParser,
    "go": GoParser,
    "rust": RustParser,
}


def get_parser_for_language(language: str) -> LanguageParser:
    """Get a parser for the specified language.

    Args:
        language: Language name

    Returns:
        LanguageParser instance

    Raises:
        ValueError: If language is not supported
    """
    language = language.lower()

    parser_class = _PARSERS.get(language)
    if parser_class:
        if language in ("typescript", "javascript", "tsx", "jsx"):
            return parser_class(language)
        return parser_class()

    # Fall back to basic tree-sitter if language is supported but no specialized parser
    if is_language_supported(language):
        # Return a generic parser
        return _GenericParser(language)

    raise ValueError(
        f"Unsupported language: {language}. "
        f"Specialized parsers: {list(_PARSERS.keys())}. "
        f"Basic support: {get_supported_languages()}"
    )


def get_supported_languages() -> list[str]:
    """Get list of all supported languages."""
    from anamnesis.parsing.tree_sitter_wrapper import get_supported_languages as ts_languages

    return ts_languages()


class _GenericParser(LanguageParser):
    """Generic parser for languages without specialized support."""

    def extract_symbols(self, context: ASTContext) -> list[ExtractedSymbol]:
        """Extract basic symbols using tree-sitter types."""
        symbols: list[ExtractedSymbol] = []
        if not context.root:
            return symbols

        # Just return top-level named nodes
        for child in context.root.children:
            if child.name:
                symbols.append(
                    ExtractedSymbol(
                        name=child.name,
                        kind=child.node_type,
                        file_path=context.file_path,
                        start_line=child.start_line,
                        end_line=child.end_line,
                    )
                )

        return symbols

    def extract_imports(self, context: ASTContext) -> list[ExtractedImport]:
        """No specialized import extraction for generic parser."""
        return []
