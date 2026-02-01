"""Unified output types for the extraction pipeline.

These types are the single source of truth for extraction results across
all backends (tree-sitter, regex, and future LSP). Converters in
converters.py map these to existing consumer types (SemanticConcept,
DetectedPattern, etc.) for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SymbolKind(StrEnum):
    """Unified symbol kinds across all backends.

    Adopted from extractors.symbol_extractor.SymbolKind as the most
    complete enum (18 values vs ConceptType's 11).
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


class PatternKind(StrEnum):
    """Unified pattern kinds across all backends.

    Superset of both PatternType (intelligence) and PatternKind (extractors).
    """

    # Design patterns
    SINGLETON = "singleton"
    FACTORY = "factory"
    ABSTRACT_FACTORY = "abstract_factory"
    BUILDER = "builder"
    PROTOTYPE = "prototype"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    PROXY = "proxy"
    COMPOSITE = "composite"
    COMMAND = "command"
    ITERATOR = "iterator"
    STATE = "state"
    TEMPLATE_METHOD = "template_method"
    VISITOR = "visitor"

    # Architectural
    DEPENDENCY_INJECTION = "dependency_injection"
    REPOSITORY = "repository"
    SERVICE = "service"
    CONTROLLER = "controller"
    DTO = "dto"
    ENTITY = "entity"
    VALUE_OBJECT = "value_object"
    AGGREGATE = "aggregate"

    # Language-specific
    CONTEXT_MANAGER = "context_manager"
    DATACLASS = "dataclass"
    PROPERTY_PATTERN = "property"
    ASYNC_PATTERN = "async_pattern"
    MIXIN = "mixin"
    ENUM_PATTERN = "enum"

    # Naming conventions (regex-only)
    CAMEL_CASE_FUNCTION = "camelCase_function_naming"
    PASCAL_CASE_CLASS = "PascalCase_class_naming"
    SNAKE_CASE_VARIABLE = "snake_case_naming"
    SCREAMING_SNAKE_CASE_CONST = "SCREAMING_SNAKE_CASE_constant"

    # Structural (regex-only)
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    CLEAN_ARCHITECTURE = "clean_architecture"

    # Code organization (regex-only)
    TESTING = "testing"
    API_DESIGN = "api_design"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    CONFIGURATION = "configuration"

    # Anti-patterns
    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    CALLBACK_HELL = "callback_hell"

    # Custom
    CUSTOM = "custom"


class ImportKind(StrEnum):
    """Unified import kinds."""

    IMPORT = "import"
    FROM_IMPORT = "from_import"
    IMPORT_ALIAS = "import_alias"
    STAR_IMPORT = "star_import"
    RELATIVE = "relative"
    DYNAMIC = "dynamic"
    CONDITIONAL = "conditional"
    TYPE_ONLY = "type_only"
    LAZY = "lazy"


class ConfidenceTier:
    """Standardized confidence tiers reflecting extraction quality.

    Used as constants, not enforced on existing code until Phase 4.
    """

    REGEX_FALLBACK = 0.5
    GENERIC_TREE_SITTER = 0.6
    PARTIAL_PARSE = 0.7
    FULL_PARSE_BASIC = 0.8
    FULL_PARSE_RICH = 0.9
    LSP_BACKED = 0.95


@dataclass
class UnifiedSymbol:
    """A symbol extracted from source code. Single type for all backends."""

    # Identity
    name: str
    kind: SymbolKind | str
    file_path: str

    # Location
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    # Confidence
    confidence: float = ConfidenceTier.FULL_PARSE_BASIC

    # Hierarchy
    parent_name: str | None = None
    qualified_name: str | None = None
    children: list[UnifiedSymbol] = field(default_factory=list)

    # Code metadata
    signature: str | None = None
    docstring: str | None = None
    visibility: str = "public"
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    is_exported: bool = False
    decorators: list[str] = field(default_factory=list)

    # Type info
    return_type: str | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)

    # Relationships (populated by cross-file analysis or LSP)
    references: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Provenance
    language: str = ""
    backend: str = ""  # "tree_sitter", "regex", "lsp"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": str(self.kind),
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": self.confidence,
            "backend": self.backend,
        }
        if self.start_col:
            result["start_col"] = self.start_col
        if self.end_col:
            result["end_col"] = self.end_col
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


@dataclass
class UnifiedPattern:
    """A pattern detected in source code. Single type for all backends."""

    kind: PatternKind | str
    name: str
    description: str
    confidence: float

    file_path: str | None = None
    start_line: int = 0
    end_line: int = 0
    code_snippet: str | None = None
    frequency: int = 1
    evidence: list[dict[str, Any]] = field(default_factory=list)
    related_symbols: list[str] = field(default_factory=list)

    # Provenance
    language: str = ""
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "kind": str(self.kind),
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code_snippet": self.code_snippet,
            "frequency": self.frequency,
            "evidence": self.evidence,
            "backend": self.backend,
        }


@dataclass
class UnifiedImport:
    """An import statement extracted from source code."""

    module: str
    kind: ImportKind | str
    file_path: str

    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_relative: bool = False
    relative_level: int = 0
    is_type_only: bool = False
    is_stdlib: bool = False
    is_third_party: bool = False
    is_local: bool = False

    # Provenance
    language: str = ""
    backend: str = ""
    confidence: float = ConfidenceTier.FULL_PARSE_RICH


@dataclass
class DetectedFramework:
    """A framework detected in the codebase."""

    name: str
    language: str
    confidence: float
    evidence_files: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction result from a single file."""

    file_path: str
    language: str
    symbols: list[UnifiedSymbol] = field(default_factory=list)
    patterns: list[UnifiedPattern] = field(default_factory=list)
    imports: list[UnifiedImport] = field(default_factory=list)
    frameworks: list[DetectedFramework] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Quality metadata
    backend_used: str = ""
    confidence: float = ConfidenceTier.FULL_PARSE_BASIC
    parse_had_errors: bool = False
