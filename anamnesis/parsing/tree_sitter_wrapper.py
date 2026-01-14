"""
Tree-sitter wrapper for parsing source code.

Provides a clean interface to tree-sitter parsing with:
- Language detection and parser initialization
- AST traversal utilities
- Query execution support
"""

from typing import Any, Callable, Iterator

import tree_sitter_language_pack as ts_pack
from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

from anamnesis.parsing.ast_types import (
    ASTContext,
    NodeType,
    ParsedNode,
    QueryCapture,
    QueryMatch,
)


# Mapping from our language names to tree-sitter-language-pack names
LANGUAGE_MAP: dict[str, str] = {
    "python": "python",
    "typescript": "typescript",
    "javascript": "javascript",
    "tsx": "tsx",
    "jsx": "javascript",  # JSX uses JavaScript grammar
    "rust": "rust",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "csharp": "c_sharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "html": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml",
    "toml": "toml",
    "markdown": "markdown",
    "bash": "bash",
    "shell": "bash",
    "sql": "sql",
    "lua": "lua",
    "r": "r",
    "julia": "julia",
    "elixir": "elixir",
    "erlang": "erlang",
    "haskell": "haskell",
    "ocaml": "ocaml",
    "zig": "zig",
    "nim": "nim",
}


class TreeSitterParser:
    """Wrapper around tree-sitter for parsing source code."""

    def __init__(self, language: str):
        """Initialize parser for a specific language.

        Args:
            language: Language name (e.g., 'python', 'typescript')

        Raises:
            ValueError: If language is not supported
        """
        self.language_name = language.lower()
        ts_language = LANGUAGE_MAP.get(self.language_name)

        if ts_language is None:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(LANGUAGE_MAP.keys())}"
            )

        try:
            self._language: Language = ts_pack.get_language(ts_language)
            self._parser = Parser(self._language)
        except Exception as e:
            raise ValueError(f"Failed to load parser for {language}: {e}") from e

    @property
    def language(self) -> Language:
        """Get the tree-sitter Language object."""
        return self._language

    def parse(self, source: str | bytes, encoding: str = "utf-8") -> Tree:
        """Parse source code and return the tree-sitter Tree.

        Args:
            source: Source code to parse
            encoding: Character encoding (default: utf-8)

        Returns:
            tree-sitter Tree object
        """
        if isinstance(source, str):
            source = source.encode(encoding)
        return self._parser.parse(source)

    def parse_to_context(
        self, source: str, file_path: str = "<string>"
    ) -> ASTContext:
        """Parse source code and return an ASTContext.

        Args:
            source: Source code to parse
            file_path: Path to the source file (for context)

        Returns:
            ASTContext with parsed AST
        """
        context = ASTContext(
            file_path=file_path,
            language=self.language_name,
            source_code=source,
        )

        try:
            tree = self.parse(source)
            context.root = self._node_to_parsed(tree.root_node, source)

            # Check for syntax errors
            if tree.root_node.has_error:
                context.errors.append("Syntax errors detected in source code")
                self._collect_errors(tree.root_node, context)

        except Exception as e:
            context.errors.append(f"Parse error: {e}")

        return context

    def _node_to_parsed(
        self, node: Node, source: str, parent_name: str | None = None
    ) -> ParsedNode:
        """Convert a tree-sitter Node to a ParsedNode.

        Args:
            node: tree-sitter Node
            source: Original source code
            parent_name: Name of the parent node (if any)

        Returns:
            ParsedNode representation
        """
        # Get the text for this node
        text = source[node.start_byte : node.end_byte]

        # Map tree-sitter type to our NodeType
        node_type = self._map_node_type(node.type)

        # Check for async keyword in children (even unnamed ones)
        has_async = any(child.type == "async" for child in node.children)

        parsed = ParsedNode(
            node_type=node_type,
            text=text,
            start_line=node.start_point[0] + 1,  # 1-indexed
            end_line=node.end_point[0] + 1,
            start_col=node.start_point[1],
            end_col=node.end_point[1],
            parent_name=parent_name,
            metadata={
                "tree_sitter_type": node.type,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "is_named": node.is_named,
                "child_count": node.child_count,
                "is_async": has_async,
            },
        )

        # Extract name if this is a named declaration
        name = self._extract_name(node, source)
        if name:
            parsed.name = name

        # Recursively convert children (only named nodes to reduce noise)
        for child in node.named_children:
            child_parsed = self._node_to_parsed(child, source, parsed.name)
            parsed.children.append(child_parsed)

        return parsed

    def _map_node_type(self, ts_type: str) -> NodeType | str:
        """Map a tree-sitter node type to our NodeType enum."""
        # Common mappings across languages
        type_map: dict[str, NodeType] = {
            # Python
            "module": NodeType.MODULE,
            "class_definition": NodeType.CLASS,
            "function_definition": NodeType.FUNCTION,
            "async_function_definition": NodeType.FUNCTION,
            "decorated_definition": NodeType.DECORATOR,
            "import_statement": NodeType.IMPORT,
            "import_from_statement": NodeType.IMPORT,
            "assignment": NodeType.ASSIGNMENT,
            "expression_statement": NodeType.ASSIGNMENT,
            "return_statement": NodeType.RETURN,
            "if_statement": NodeType.IF,
            "for_statement": NodeType.FOR,
            "while_statement": NodeType.WHILE,
            "try_statement": NodeType.TRY,
            "except_clause": NodeType.CATCH,
            "raise_statement": NodeType.THROW,
            "call": NodeType.CALL,
            "identifier": NodeType.IDENTIFIER,
            "comment": NodeType.COMMENT,
            "decorator": NodeType.DECORATOR,
            "lambda": NodeType.LAMBDA,
            "await": NodeType.AWAIT,
            # TypeScript/JavaScript
            "program": NodeType.MODULE,
            "class_declaration": NodeType.CLASS,
            "function_declaration": NodeType.FUNCTION,
            "arrow_function": NodeType.LAMBDA,
            "method_definition": NodeType.METHOD,
            "variable_declaration": NodeType.VARIABLE,
            "lexical_declaration": NodeType.VARIABLE,
            "const_declaration": NodeType.CONSTANT,
            "import_declaration": NodeType.IMPORT,
            "export_statement": NodeType.EXPORT,
            "export_declaration": NodeType.EXPORT,
            "interface_declaration": NodeType.INTERFACE,
            "type_alias_declaration": NodeType.TYPE_ALIAS,
            "enum_declaration": NodeType.ENUM,
            "call_expression": NodeType.CALL,
            "member_expression": NodeType.MEMBER_ACCESS,
            "subscript_expression": NodeType.INDEX_ACCESS,
            "await_expression": NodeType.AWAIT,
            # Rust
            "source_file": NodeType.MODULE,
            "struct_item": NodeType.CLASS,
            "impl_item": NodeType.CLASS,
            "function_item": NodeType.FUNCTION,
            "mod_item": NodeType.MODULE,
            "use_declaration": NodeType.IMPORT,
            "enum_item": NodeType.ENUM,
            "trait_item": NodeType.INTERFACE,
            "type_item": NodeType.TYPE_ALIAS,
            # Go
            "source_file": NodeType.MODULE,
            "type_declaration": NodeType.CLASS,
            "function_declaration": NodeType.FUNCTION,
            "method_declaration": NodeType.METHOD,
            "import_declaration": NodeType.IMPORT,
            # Java
            "compilation_unit": NodeType.MODULE,
            "class_declaration": NodeType.CLASS,
            "interface_declaration": NodeType.INTERFACE,
            "method_declaration": NodeType.METHOD,
            "constructor_declaration": NodeType.FUNCTION,
            "import_declaration": NodeType.IMPORT,
            "enum_declaration": NodeType.ENUM,
        }

        return type_map.get(ts_type, ts_type)

    def _extract_name(self, node: Node, source: str) -> str | None:
        """Extract the name from a declaration node."""
        # Try common name child types
        name_types = ["identifier", "name", "type_identifier", "property_identifier"]

        for name_type in name_types:
            name_node = node.child_by_field_name("name")
            if name_node:
                return source[name_node.start_byte : name_node.end_byte]

        # Try first identifier child
        for child in node.children:
            if child.type in name_types:
                return source[child.start_byte : child.end_byte]

        return None

    def _collect_errors(self, node: Node, context: ASTContext) -> None:
        """Collect all error nodes from the tree."""
        if node.type == "ERROR":
            line = node.start_point[0] + 1
            col = node.start_point[1]
            context.errors.append(f"Syntax error at line {line}, column {col}")

        for child in node.children:
            self._collect_errors(child, context)

    def traverse(
        self,
        tree: Tree,
        source: str,
        callback: Callable[[Node, str], bool | None],
    ) -> None:
        """Traverse the AST and call callback for each node.

        Args:
            tree: tree-sitter Tree to traverse
            source: Original source code
            callback: Function called for each node. Return False to stop traversal
                     of children, True/None to continue.
        """

        def _traverse(node: Node) -> None:
            result = callback(node, source)
            if result is not False:
                for child in node.children:
                    _traverse(child)

        _traverse(tree.root_node)

    def walk(self, tree: Tree) -> Iterator[Node]:
        """Walk the tree and yield each node.

        Args:
            tree: tree-sitter Tree to walk

        Yields:
            Each node in depth-first order
        """
        cursor = tree.walk()

        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    def find_nodes(
        self,
        tree: Tree,
        node_types: list[str] | None = None,
        predicate: Callable[[Node], bool] | None = None,
    ) -> list[Node]:
        """Find all nodes matching criteria.

        Args:
            tree: tree-sitter Tree to search
            node_types: List of node types to match (or None for all)
            predicate: Additional filter function

        Returns:
            List of matching nodes
        """
        results = []
        for node in self.walk(tree):
            if node_types and node.type not in node_types:
                continue
            if predicate and not predicate(node):
                continue
            results.append(node)
        return results


class TreeSitterQuery:
    """Wrapper for tree-sitter queries."""

    def __init__(self, parser: TreeSitterParser, query_string: str):
        """Create a query for the given language.

        Args:
            parser: TreeSitterParser instance
            query_string: S-expression query pattern

        Raises:
            ValueError: If query is invalid
        """
        self._parser = parser
        self._query_string = query_string

        try:
            self._query = Query(parser.language, query_string)
        except Exception as e:
            raise ValueError(f"Invalid query: {e}") from e

    @property
    def pattern_count(self) -> int:
        """Number of patterns in the query."""
        return self._query.pattern_count

    @property
    def capture_names(self) -> list[str]:
        """Names of all captures in the query."""
        # Use capture_name(index) for each capture in the query
        names = []
        for i in range(self._query.capture_count):
            names.append(self._query.capture_name(i))
        return names

    def execute(self, tree: Tree, source: str) -> list[QueryMatch]:
        """Execute the query against a tree.

        Args:
            tree: Parsed tree to query
            source: Original source code

        Returns:
            List of QueryMatch results
        """
        matches: list[QueryMatch] = []

        # Use QueryCursor for query execution
        cursor = QueryCursor(self._query)
        raw_matches = cursor.matches(tree.root_node)

        for pattern_idx, capture_dict in raw_matches:
            match = QueryMatch(pattern_index=pattern_idx)

            # capture_dict is {capture_name: [nodes]}
            for capture_name, nodes in capture_dict.items():
                for node in nodes:
                    # Convert Node to ParsedNode
                    parsed = self._parser._node_to_parsed(node, source)
                    capture = QueryCapture(
                        name=capture_name,
                        node=parsed,
                        pattern_index=pattern_idx,
                    )
                    match.captures.append(capture)

            matches.append(match)

        return matches

    def find_all(
        self, tree: Tree, source: str, capture_name: str | None = None
    ) -> list[ParsedNode]:
        """Find all nodes matching the query.

        Args:
            tree: Parsed tree to query
            source: Original source code
            capture_name: Optional name to filter captures

        Returns:
            List of matching ParsedNode objects
        """
        matches = self.execute(tree, source)
        nodes = []

        for match in matches:
            for capture in match.captures:
                if capture_name is None or capture.name == capture_name:
                    nodes.append(capture.node)

        return nodes


def get_supported_languages() -> list[str]:
    """Get list of supported languages."""
    return list(LANGUAGE_MAP.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    return language.lower() in LANGUAGE_MAP
