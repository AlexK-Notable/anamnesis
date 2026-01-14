"""
Parsing Infrastructure - Tree-sitter based code parsing.

This module provides:
- TreeSitterParser: Low-level tree-sitter wrapper
- LanguageParser: High-level language-specific parsing
- AST traversal and query utilities
"""

from anamnesis.parsing.ast_types import (
    ASTContext,
    NodeType,
    ParsedNode,
    QueryCapture,
    QueryMatch,
)
from anamnesis.parsing.language_parsers import (
    LanguageParser,
    get_parser_for_language,
    get_supported_languages,
)
from anamnesis.parsing.tree_sitter_wrapper import (
    TreeSitterParser,
    TreeSitterQuery,
)

__all__ = [
    # AST Types
    "ASTContext",
    "NodeType",
    "ParsedNode",
    "QueryCapture",
    "QueryMatch",
    # Tree-sitter wrapper
    "TreeSitterParser",
    "TreeSitterQuery",
    # Language parsers
    "LanguageParser",
    "get_parser_for_language",
    "get_supported_languages",
]
