"""
Pattern Extractor - Detect common code patterns in source code.

This module identifies design patterns and common code patterns:
- Creational: Singleton, Factory, Builder, Prototype
- Structural: Adapter, Decorator, Facade, Proxy
- Behavioral: Observer, Strategy, Command, Iterator
- Architectural: Repository, Service, Controller, DTO
- Python-specific: Context Manager, Dataclass, Property
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from anamnesis.parsing.ast_types import ASTContext, NodeType, ParsedNode


class PatternKind(StrEnum):
    """Kind of detected pattern."""

    # Creational Patterns
    SINGLETON = "singleton"
    FACTORY = "factory"
    ABSTRACT_FACTORY = "abstract_factory"
    BUILDER = "builder"
    PROTOTYPE = "prototype"

    # Structural Patterns
    ADAPTER = "adapter"
    DECORATOR = "decorator"
    FACADE = "facade"
    PROXY = "proxy"
    COMPOSITE = "composite"

    # Behavioral Patterns
    OBSERVER = "observer"
    STRATEGY = "strategy"
    COMMAND = "command"
    ITERATOR = "iterator"
    STATE = "state"
    TEMPLATE_METHOD = "template_method"
    VISITOR = "visitor"

    # Architectural Patterns
    REPOSITORY = "repository"
    SERVICE = "service"
    CONTROLLER = "controller"
    DTO = "dto"
    ENTITY = "entity"
    VALUE_OBJECT = "value_object"
    AGGREGATE = "aggregate"

    # Language-Specific
    CONTEXT_MANAGER = "context_manager"
    DATACLASS = "dataclass"
    PROPERTY_PATTERN = "property"
    DEPENDENCY_INJECTION = "dependency_injection"
    ASYNC_PATTERN = "async_pattern"
    MIXIN = "mixin"
    ENUM_PATTERN = "enum"

    # Anti-patterns (for detection)
    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    CALLBACK_HELL = "callback_hell"

    # Unknown/Custom
    CUSTOM = "custom"


@dataclass
class PatternEvidence:
    """Evidence supporting a pattern detection."""

    description: str
    location: tuple[int, int]  # (start_line, end_line)
    code_snippet: str = ""
    confidence_contribution: float = 0.0


@dataclass
class DetectedPattern:
    """A code pattern detected in source code."""

    # Identity
    kind: PatternKind | str
    name: str  # Specific name (e.g., "UserFactory" for Factory pattern)
    file_path: str

    # Location
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    # Detection details
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: list[PatternEvidence] = field(default_factory=list)

    # Context
    class_name: str | None = None  # If pattern is in a class
    method_names: list[str] = field(default_factory=list)  # Related methods

    # Related symbols
    related_symbols: list[str] = field(default_factory=list)

    # Metadata
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_evidence(
        self,
        description: str,
        location: tuple[int, int],
        code_snippet: str = "",
        confidence: float = 0.1,
    ) -> None:
        """Add evidence for this pattern detection."""
        self.evidence.append(
            PatternEvidence(
                description=description,
                location=location,
                code_snippet=code_snippet,
                confidence_contribution=confidence,
            )
        )
        self.confidence = min(1.0, self.confidence + confidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": self.confidence,
        }

        if self.evidence:
            result["evidence"] = [
                {
                    "description": e.description,
                    "location": list(e.location),
                    "confidence": e.confidence_contribution,
                }
                for e in self.evidence
            ]

        if self.class_name:
            result["class_name"] = self.class_name
        if self.method_names:
            result["method_names"] = self.method_names
        if self.related_symbols:
            result["related_symbols"] = self.related_symbols
        if self.language:
            result["language"] = self.language

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectedPattern":
        """Create from dictionary."""
        kind_val = data.get("kind", "custom")
        try:
            kind = PatternKind(kind_val)
        except ValueError:
            kind = kind_val

        evidence = []
        for e in data.get("evidence", []):
            evidence.append(
                PatternEvidence(
                    description=e["description"],
                    location=tuple(e["location"]),
                    confidence_contribution=e.get("confidence", 0.0),
                )
            )

        return cls(
            kind=kind,
            name=data["name"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_col=data.get("start_col", 0),
            end_col=data.get("end_col", 0),
            confidence=data.get("confidence", 0.0),
            evidence=evidence,
            class_name=data.get("class_name"),
            method_names=data.get("method_names", []),
            related_symbols=data.get("related_symbols", []),
            language=data.get("language", ""),
            metadata=data.get("metadata", {}),
        )


class PatternExtractor:
    """Extracts code patterns from parsed AST contexts."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        detect_antipatterns: bool = False,
    ):
        """Initialize pattern extractor.

        Args:
            min_confidence: Minimum confidence threshold for reporting patterns
            detect_antipatterns: Whether to detect anti-patterns
        """
        self.min_confidence = min_confidence
        self.detect_antipatterns = detect_antipatterns

    def extract(self, context: ASTContext) -> list[DetectedPattern]:
        """Extract all patterns from an AST context.

        Args:
            context: Parsed AST context

        Returns:
            List of detected patterns
        """
        if not context.is_valid or context.root is None:
            return []

        patterns: list[DetectedPattern] = []

        # Extract patterns from each class/module
        self._extract_patterns_recursive(
            context.root,
            context.file_path,
            context.language,
            context.source_code,
            patterns,
        )

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]

        return patterns

    def extract_from_file(
        self, file_path: str, source: str, language: str
    ) -> list[DetectedPattern]:
        """Extract patterns from source code.

        Args:
            file_path: Path to the file
            source: Source code
            language: Programming language

        Returns:
            List of detected patterns
        """
        from anamnesis.parsing.tree_sitter_wrapper import TreeSitterParser

        try:
            parser = TreeSitterParser(language)
            context = parser.parse_to_context(source, file_path)
            return self.extract(context)
        except ValueError:
            return []

    def _get_class_methods(self, node: ParsedNode) -> list[ParsedNode]:
        """Extract method nodes from a class, handling tree-sitter AST structure.

        In tree-sitter, class methods are inside the 'block' child node,
        and decorated methods are wrapped in 'decorated_definition' nodes.
        """
        methods: list[ParsedNode] = []

        # Find the block child containing the class body
        block_children: list[ParsedNode] = []
        for child in node.children:
            ts_type = child.metadata.get("tree_sitter_type", "")
            if ts_type == "block":
                block_children = child.children
                break

        # If no block found, try direct children (for some node types)
        if not block_children:
            block_children = node.children

        # Extract methods from block children
        for child in block_children:
            ts_type = child.metadata.get("tree_sitter_type", "")

            # Direct function/method definition
            if ts_type in ("function_definition", "method_definition"):
                methods.append(child)
            # Decorated definition wraps the actual function
            elif ts_type == "decorated_definition":
                # Find the inner function/method
                for inner in child.children:
                    inner_ts = inner.metadata.get("tree_sitter_type", "")
                    if inner_ts in ("function_definition", "async_function_definition"):
                        methods.append(inner)
                        break
            # Also check by NodeType
            elif isinstance(child.node_type, NodeType) and child.node_type == NodeType.METHOD:
                methods.append(child)
            elif isinstance(child.node_type, NodeType) and child.node_type == NodeType.FUNCTION:
                methods.append(child)

        return methods

    def _extract_patterns_recursive(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        patterns: list[DetectedPattern],
        decorators: list[str] | None = None,
    ) -> None:
        """Recursively extract patterns from nodes."""
        ts_type = node.metadata.get("tree_sitter_type", "")
        node_type = (
            node.node_type.value
            if isinstance(node.node_type, NodeType)
            else node.node_type
        )

        # Handle decorated definitions specially (classes and functions with decorators)
        if ts_type == "decorated_definition":
            collected_decorators: list[str] = []
            inner_node = None

            for child in node.children:
                child_ts = child.metadata.get("tree_sitter_type", "")
                if child_ts == "decorator":
                    # Collect the decorator text (e.g., "@dataclass")
                    collected_decorators.append(child.text.strip())
                elif child_ts in ("class_definition", "function_definition",
                                  "async_function_definition"):
                    inner_node = child

            if inner_node:
                # Process the inner node with collected decorators
                self._extract_patterns_recursive(
                    inner_node, file_path, language, source, patterns,
                    decorators=collected_decorators
                )
            return

        # Check for class patterns
        if node_type == "class":
            self._detect_class_patterns(
                node, file_path, language, source, patterns, decorators
            )

        # Check for function patterns
        elif node_type in ("function", "method"):
            self._detect_function_patterns(
                node, file_path, language, source, patterns, decorators
            )

        # Check for decorator-based patterns (from node or passed decorators)
        effective_decorators = decorators or node.decorators
        if effective_decorators:
            self._detect_decorator_patterns(
                node, file_path, language, patterns, effective_decorators
            )

        # Recurse
        for child in node.children:
            self._extract_patterns_recursive(
                child, file_path, language, source, patterns
            )

    def _detect_class_patterns(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        patterns: list[DetectedPattern],
        decorators: list[str] | None = None,
    ) -> None:
        """Detect patterns in a class definition."""
        if not node.name:
            return

        class_name = node.name
        methods = self._get_class_methods(node)

        method_names = [m.name for m in methods if m.name]

        # Singleton pattern detection
        singleton = self._detect_singleton(node, class_name, methods, file_path, language)
        if singleton:
            patterns.append(singleton)

        # Factory pattern detection
        factory = self._detect_factory(node, class_name, methods, file_path, language)
        if factory:
            patterns.append(factory)

        # Repository pattern detection
        repository = self._detect_repository(node, class_name, methods, file_path, language)
        if repository:
            patterns.append(repository)

        # Service pattern detection
        service = self._detect_service(node, class_name, methods, file_path, language)
        if service:
            patterns.append(service)

        # Context Manager detection (Python)
        if language == "python":
            ctx_mgr = self._detect_context_manager(
                node, class_name, method_names, file_path
            )
            if ctx_mgr:
                patterns.append(ctx_mgr)

        # Dataclass detection
        if language == "python":
            dataclass = self._detect_dataclass(node, class_name, file_path, decorators)
            if dataclass:
                patterns.append(dataclass)

        # Anti-patterns
        if self.detect_antipatterns:
            god_class = self._detect_god_class(
                node, class_name, methods, file_path, language
            )
            if god_class:
                patterns.append(god_class)

    def _detect_function_patterns(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        patterns: list[DetectedPattern],
        decorators: list[str] | None = None,
    ) -> None:
        """Detect patterns in a function definition."""
        if not node.name:
            return

        # Async pattern detection
        if node.is_async or node.metadata.get("is_async"):
            async_pattern = DetectedPattern(
                kind=PatternKind.ASYNC_PATTERN,
                name=node.name,
                file_path=file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                language=language,
            )
            async_pattern.add_evidence(
                "Async function definition",
                (node.start_line, node.end_line),
                confidence=0.9,
            )
            patterns.append(async_pattern)

        # Long method anti-pattern
        if self.detect_antipatterns:
            line_count = node.end_line - node.start_line + 1
            if line_count > 50:
                long_method = DetectedPattern(
                    kind=PatternKind.LONG_METHOD,
                    name=node.name,
                    file_path=file_path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    language=language,
                )
                long_method.add_evidence(
                    f"Method has {line_count} lines (threshold: 50)",
                    (node.start_line, node.end_line),
                    confidence=0.7 if line_count < 100 else 0.9,
                )
                patterns.append(long_method)

    def _detect_decorator_patterns(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        patterns: list[DetectedPattern],
        decorators: list[str] | None = None,
    ) -> None:
        """Detect patterns based on decorators."""
        # Use passed decorators or fall back to node.decorators
        decorator_list = decorators or node.decorators
        if not decorator_list:
            return

        for decorator in decorator_list:
            # Property pattern
            if decorator in ("@property", "property"):
                pattern = DetectedPattern(
                    kind=PatternKind.PROPERTY_PATTERN,
                    name=node.name or "property",
                    file_path=file_path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    language=language,
                )
                pattern.add_evidence(
                    "Property decorator detected",
                    (node.start_line, node.end_line),
                    confidence=1.0,
                )
                patterns.append(pattern)

            # Dataclass from decorator
            elif decorator in ("@dataclass", "@dataclasses.dataclass", "dataclass"):
                pattern = DetectedPattern(
                    kind=PatternKind.DATACLASS,
                    name=node.name or "dataclass",
                    file_path=file_path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    language=language,
                )
                pattern.add_evidence(
                    "Dataclass decorator",
                    (node.start_line, node.end_line),
                    confidence=1.0,
                )
                patterns.append(pattern)

    def _detect_singleton(
        self,
        node: ParsedNode,
        class_name: str,
        methods: list[ParsedNode],
        file_path: str,
        language: str,
    ) -> DetectedPattern | None:
        """Detect Singleton pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.SINGLETON,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language=language,
        )

        method_names = [m.name for m in methods if m.name]

        # Check for instance attribute
        text_lower = node.text.lower()
        if "_instance" in text_lower or "instance" in method_names:
            pattern.add_evidence(
                "Has _instance attribute",
                (node.start_line, node.end_line),
                confidence=0.3,
            )

        # Check for get_instance method
        if "get_instance" in method_names or "getinstance" in [m.lower() for m in method_names]:
            pattern.add_evidence(
                "Has get_instance method",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        # Check for __new__ method (Python)
        if language == "python" and "__new__" in method_names:
            pattern.add_evidence(
                "Overrides __new__ method",
                (node.start_line, node.end_line),
                confidence=0.3,
            )

        # Name-based heuristic
        if class_name.endswith("Singleton") or "singleton" in class_name.lower():
            pattern.add_evidence(
                "Class name suggests Singleton",
                (node.start_line, node.end_line),
                confidence=0.2,
            )

        return pattern if pattern.confidence >= self.min_confidence else None

    def _detect_factory(
        self,
        node: ParsedNode,
        class_name: str,
        methods: list[ParsedNode],
        file_path: str,
        language: str,
    ) -> DetectedPattern | None:
        """Detect Factory pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.FACTORY,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language=language,
        )

        method_names = [m.name.lower() for m in methods if m.name]

        # Check for create methods
        create_methods = [m for m in method_names if "create" in m]
        if create_methods:
            pattern.add_evidence(
                f"Has create methods: {create_methods}",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        # Check for make/build methods
        make_methods = [m for m in method_names if "make" in m or "build" in m]
        if make_methods:
            pattern.add_evidence(
                f"Has make/build methods: {make_methods}",
                (node.start_line, node.end_line),
                confidence=0.3,
            )

        # Name-based heuristic
        if "factory" in class_name.lower():
            pattern.add_evidence(
                "Class name contains 'Factory'",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        return pattern if pattern.confidence >= self.min_confidence else None

    def _detect_repository(
        self,
        node: ParsedNode,
        class_name: str,
        methods: list[ParsedNode],
        file_path: str,
        language: str,
    ) -> DetectedPattern | None:
        """Detect Repository pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.REPOSITORY,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language=language,
        )

        method_names = [m.name.lower() for m in methods if m.name]

        # CRUD method indicators
        crud_methods = {"get", "find", "save", "update", "delete", "create", "add", "remove"}
        found_crud = [m for m in method_names if any(c in m for c in crud_methods)]

        if len(found_crud) >= 2:
            pattern.add_evidence(
                f"Has CRUD methods: {found_crud[:5]}",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        # Name-based heuristic
        if "repository" in class_name.lower() or "repo" in class_name.lower():
            pattern.add_evidence(
                "Class name suggests Repository",
                (node.start_line, node.end_line),
                confidence=0.5,
            )

        # DAO pattern (related)
        if "dao" in class_name.lower():
            pattern.add_evidence(
                "Class name suggests DAO (Data Access Object)",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        return pattern if pattern.confidence >= self.min_confidence else None

    def _detect_service(
        self,
        node: ParsedNode,
        class_name: str,
        methods: list[ParsedNode],
        file_path: str,
        language: str,
    ) -> DetectedPattern | None:
        """Detect Service pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.SERVICE,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language=language,
        )

        # Name-based heuristic
        if "service" in class_name.lower():
            pattern.add_evidence(
                "Class name contains 'Service'",
                (node.start_line, node.end_line),
                confidence=0.6,
            )

        # Check for typical service methods
        method_names = [m.name.lower() for m in methods if m.name]
        service_methods = {"execute", "process", "handle", "run", "perform"}
        found = [m for m in method_names if any(s in m for s in service_methods)]

        if found:
            pattern.add_evidence(
                f"Has service-like methods: {found[:5]}",
                (node.start_line, node.end_line),
                confidence=0.3,
            )

        return pattern if pattern.confidence >= self.min_confidence else None

    def _detect_context_manager(
        self,
        node: ParsedNode,
        class_name: str,
        method_names: list[str],
        file_path: str,
    ) -> DetectedPattern | None:
        """Detect Python Context Manager pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.CONTEXT_MANAGER,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language="python",
        )

        # Check for __enter__ and __exit__
        if "__enter__" in method_names:
            pattern.add_evidence(
                "Has __enter__ method",
                (node.start_line, node.end_line),
                confidence=0.5,
            )

        if "__exit__" in method_names:
            pattern.add_evidence(
                "Has __exit__ method",
                (node.start_line, node.end_line),
                confidence=0.5,
            )

        # Check for @contextmanager decorator
        if node.decorators:
            for dec in node.decorators:
                if "contextmanager" in dec.lower():
                    pattern.add_evidence(
                        "Has @contextmanager decorator",
                        (node.start_line, node.end_line),
                        confidence=1.0,
                    )

        return pattern if pattern.confidence >= self.min_confidence else None

    def _detect_dataclass(
        self,
        node: ParsedNode,
        class_name: str,
        file_path: str,
        decorators: list[str] | None = None,
    ) -> DetectedPattern | None:
        """Detect Python Dataclass pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.DATACLASS,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language="python",
        )

        # Check for @dataclass decorator (use passed decorators or node.decorators)
        decorator_list = decorators or node.decorators
        if decorator_list:
            for dec in decorator_list:
                if "dataclass" in dec.lower():
                    pattern.add_evidence(
                        "Has @dataclass decorator",
                        (node.start_line, node.end_line),
                        confidence=1.0,
                    )
                    return pattern

        return None

    def _detect_god_class(
        self,
        node: ParsedNode,
        class_name: str,
        methods: list[ParsedNode],
        file_path: str,
        language: str,
    ) -> DetectedPattern | None:
        """Detect God Class anti-pattern."""
        pattern = DetectedPattern(
            kind=PatternKind.GOD_CLASS,
            name=class_name,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            class_name=class_name,
            language=language,
        )

        # Check method count
        if len(methods) > 20:
            pattern.add_evidence(
                f"Has {len(methods)} methods (threshold: 20)",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        # Check line count
        line_count = node.end_line - node.start_line + 1
        if line_count > 500:
            pattern.add_evidence(
                f"Has {line_count} lines (threshold: 500)",
                (node.start_line, node.end_line),
                confidence=0.4,
            )

        return pattern if pattern.confidence >= self.min_confidence else None


def extract_patterns_from_source(
    source: str,
    language: str,
    file_path: str = "<string>",
    min_confidence: float = 0.5,
) -> list[DetectedPattern]:
    """Convenience function to extract patterns from source code.

    Args:
        source: Source code
        language: Programming language
        file_path: Path to the file (for context)
        min_confidence: Minimum confidence for pattern detection

    Returns:
        List of detected patterns
    """
    extractor = PatternExtractor(min_confidence=min_confidence)
    return extractor.extract_from_file(file_path, source, language)
