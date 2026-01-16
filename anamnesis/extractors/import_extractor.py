"""
Import Extractor - Extract import statements and dependencies from code.

This module provides import extraction for building dependency graphs:
- Module imports (import x, from x import y)
- Relative imports
- Re-exports
- Lazy/dynamic imports where detectable
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from anamnesis.parsing.ast_types import ASTContext, NodeType, ParsedNode


class ImportKind(StrEnum):
    """Kind of import statement."""

    # Standard imports
    IMPORT = "import"  # import x
    FROM_IMPORT = "from_import"  # from x import y
    IMPORT_ALIAS = "import_alias"  # import x as y
    STAR_IMPORT = "star_import"  # from x import *

    # Special cases
    RELATIVE = "relative"  # from . import x, from .. import y
    DYNAMIC = "dynamic"  # __import__, importlib.import_module
    CONDITIONAL = "conditional"  # Import inside if/try block
    TYPE_ONLY = "type_only"  # TYPE_CHECKING imports
    LAZY = "lazy"  # Lazy import patterns


@dataclass
class ImportedName:
    """A single imported name from an import statement."""

    name: str  # Original name
    alias: str | None = None  # Alias if renamed (import x as y)
    is_type_only: bool = False  # True if only used for type hints

    @property
    def local_name(self) -> str:
        """The name used locally in the file."""
        return self.alias or self.name


@dataclass
class ExtractedImport:
    """An import statement extracted from source code."""

    # Identity
    module: str  # Module being imported from
    kind: ImportKind
    file_path: str

    # Location
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    # Import details
    names: list[ImportedName] = field(default_factory=list)  # Specific names imported
    is_relative: bool = False
    relative_level: int = 0  # Number of dots for relative imports

    # Context
    is_conditional: bool = False  # Inside if/try block
    is_type_only: bool = False  # Inside TYPE_CHECKING block
    is_lazy: bool = False  # Lazy import pattern

    # Resolved info (populated later)
    resolved_path: str | None = None  # Resolved file path
    is_stdlib: bool = False  # Is standard library
    is_third_party: bool = False  # Is third-party package
    is_local: bool = False  # Is local module

    # Raw
    raw_text: str = ""

    # Metadata
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def imported_names(self) -> list[str]:
        """Get list of imported name strings."""
        return [n.name for n in self.names]

    @property
    def local_names(self) -> list[str]:
        """Get list of local names (accounting for aliases)."""
        return [n.local_name for n in self.names]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "module": self.module,
            "kind": self.kind.value if isinstance(self.kind, ImportKind) else self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

        if self.names:
            result["names"] = [
                {"name": n.name, "alias": n.alias, "is_type_only": n.is_type_only}
                for n in self.names
                if n.alias or n.is_type_only  # Only include if non-default
            ] or [n.name for n in self.names]  # Simplified if no extras

        if self.is_relative:
            result["is_relative"] = True
            result["relative_level"] = self.relative_level

        if self.is_conditional:
            result["is_conditional"] = True
        if self.is_type_only:
            result["is_type_only"] = True
        if self.is_lazy:
            result["is_lazy"] = True

        if self.resolved_path:
            result["resolved_path"] = self.resolved_path
        if self.is_stdlib:
            result["is_stdlib"] = True
        if self.is_third_party:
            result["is_third_party"] = True
        if self.is_local:
            result["is_local"] = True

        if self.language:
            result["language"] = self.language

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedImport":
        """Create from dictionary."""
        kind_val = data.get("kind", "import")
        try:
            kind = ImportKind(kind_val)
        except ValueError:
            kind = ImportKind.IMPORT

        # Parse names
        names_data = data.get("names", [])
        names = []
        for n in names_data:
            if isinstance(n, str):
                names.append(ImportedName(name=n))
            else:
                names.append(
                    ImportedName(
                        name=n["name"],
                        alias=n.get("alias"),
                        is_type_only=n.get("is_type_only", False),
                    )
                )

        return cls(
            module=data["module"],
            kind=kind,
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_col=data.get("start_col", 0),
            end_col=data.get("end_col", 0),
            names=names,
            is_relative=data.get("is_relative", False),
            relative_level=data.get("relative_level", 0),
            is_conditional=data.get("is_conditional", False),
            is_type_only=data.get("is_type_only", False),
            is_lazy=data.get("is_lazy", False),
            resolved_path=data.get("resolved_path"),
            is_stdlib=data.get("is_stdlib", False),
            is_third_party=data.get("is_third_party", False),
            is_local=data.get("is_local", False),
            language=data.get("language", ""),
            metadata=data.get("metadata", {}),
        )


class ImportExtractor:
    """Extracts import statements from parsed AST contexts."""

    # Python standard library modules (subset for detection)
    PYTHON_STDLIB = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "concurrent", "configparser", "contextlib", "copy", "csv",
        "dataclasses", "datetime", "decimal", "enum", "functools",
        "glob", "hashlib", "heapq", "http", "importlib", "inspect",
        "io", "itertools", "json", "logging", "math", "multiprocessing",
        "operator", "os", "pathlib", "pickle", "platform", "pprint",
        "queue", "random", "re", "shutil", "signal", "socket", "sqlite3",
        "ssl", "string", "struct", "subprocess", "sys", "tempfile",
        "textwrap", "threading", "time", "traceback", "typing", "unittest",
        "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
    }

    def __init__(
        self,
        resolve_paths: bool = False,
        project_root: str | None = None,
    ):
        """Initialize import extractor.

        Args:
            resolve_paths: Whether to resolve import paths to files
            project_root: Root of the project for resolving relative imports
        """
        self.resolve_paths = resolve_paths
        self.project_root = Path(project_root) if project_root else None

    def extract(self, context: ASTContext) -> list[ExtractedImport]:
        """Extract all imports from an AST context.

        Args:
            context: Parsed AST context

        Returns:
            List of extracted imports
        """
        if not context.is_valid or context.root is None:
            return []

        imports: list[ExtractedImport] = []
        self._extract_from_node(
            context.root,
            context.file_path,
            context.language,
            context.source_code,
            imports,
            in_type_checking=False,
            in_conditional=False,
        )

        # Resolve paths if requested
        if self.resolve_paths:
            for imp in imports:
                self._resolve_import(imp, context.file_path)

        return imports

    def extract_from_file(
        self, file_path: str, source: str, language: str
    ) -> list[ExtractedImport]:
        """Extract imports from source code.

        Args:
            file_path: Path to the file
            source: Source code
            language: Programming language

        Returns:
            List of extracted imports
        """
        from anamnesis.parsing.tree_sitter_wrapper import TreeSitterParser

        try:
            parser = TreeSitterParser(language)
            context = parser.parse_to_context(source, file_path)
            return self.extract(context)
        except ValueError:
            return []

    def _extract_from_node(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        imports: list[ExtractedImport],
        in_type_checking: bool,
        in_conditional: bool,
    ) -> None:
        """Recursively extract imports from a parsed node."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        # Check for TYPE_CHECKING block
        if ts_type == "if_statement":
            # Check if this is TYPE_CHECKING
            if "TYPE_CHECKING" in node.text:
                in_type_checking = True
            else:
                in_conditional = True

        # Check for try block
        if ts_type in ("try_statement", "try_except_statement"):
            in_conditional = True

        # Check if this is an import node
        if node.node_type == NodeType.IMPORT or ts_type in (
            "import_statement",
            "import_from_statement",
            "import_declaration",
            "use_declaration",
        ):
            extracted = self._node_to_import(
                node, file_path, language, source, in_type_checking, in_conditional
            )
            if extracted:
                imports.append(extracted)

        # Recurse into children
        for child in node.children:
            self._extract_from_node(
                child,
                file_path,
                language,
                source,
                imports,
                in_type_checking,
                in_conditional,
            )

    def _node_to_import(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        is_type_only: bool,
        is_conditional: bool,
    ) -> ExtractedImport | None:
        """Convert a parsed node to an extracted import."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if language == "python":
            return self._extract_python_import(
                node, file_path, source, is_type_only, is_conditional
            )
        elif language in ("typescript", "javascript"):
            return self._extract_js_import(
                node, file_path, language, source, is_type_only, is_conditional
            )
        elif language == "go":
            return self._extract_go_import(node, file_path, source)
        elif language == "rust":
            return self._extract_rust_import(node, file_path, source)

        return None

    def _extract_python_import(
        self,
        node: ParsedNode,
        file_path: str,
        source: str,
        is_type_only: bool,
        is_conditional: bool,
    ) -> ExtractedImport | None:
        """Extract Python import."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type == "import_statement":
            # import x, import x as y
            names = []
            module = ""

            for child in node.children:
                child_type = child.metadata.get("tree_sitter_type", "")
                if child_type == "dotted_name":
                    module = child.text
                    names.append(ImportedName(name=child.text))
                elif child_type == "aliased_import":
                    # import x as y
                    original = ""
                    alias = None
                    for sub in child.children:
                        sub_type = sub.metadata.get("tree_sitter_type", "")
                        if sub_type == "dotted_name":
                            original = sub.text
                        elif sub_type == "identifier":
                            alias = sub.text
                    if original:
                        module = original
                        names.append(ImportedName(name=original, alias=alias))

            if module:
                return ExtractedImport(
                    module=module,
                    kind=ImportKind.IMPORT_ALIAS if any(n.alias for n in names) else ImportKind.IMPORT,
                    file_path=file_path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    start_col=node.start_col,
                    end_col=node.end_col,
                    names=names,
                    is_type_only=is_type_only,
                    is_conditional=is_conditional,
                    is_stdlib=self._is_stdlib_python(module),
                    raw_text=node.text,
                    language="python",
                )

        elif ts_type == "import_from_statement":
            # from x import y, from . import x
            module = ""
            names = []
            is_relative = False
            relative_level = 0
            is_star = False

            # Check for relative_import node (AST-based, more reliable than text parsing)
            for child in node.children:
                child_type = child.metadata.get("tree_sitter_type", "")
                if child_type == "relative_import":
                    is_relative = True
                    # Count dots
                    for c in child.text:
                        if c == ".":
                            relative_level += 1
                        else:
                            break
                    # Extract module part after dots
                    module_part = child.text.lstrip(".")
                    if module_part:
                        module = module_part

            # In ParsedNode, keywords are filtered out.
            # First dotted_name is the module, subsequent ones are imported names.
            # wildcard_import indicates star import.
            found_module = False
            for child in node.children:
                child_type = child.metadata.get("tree_sitter_type", "")

                if child_type == "wildcard_import":
                    is_star = True
                elif child_type == "dotted_name":
                    if not found_module and not module:
                        # First dotted_name is the module
                        module = child.text
                        found_module = True
                    else:
                        # Subsequent dotted_names are imported names
                        names.append(ImportedName(name=child.text))
                elif child_type == "identifier" and found_module:
                    names.append(ImportedName(name=child.text))
                elif child_type == "aliased_import":
                    original = ""
                    alias = None
                    for sub in child.children:
                        sub_type = sub.metadata.get("tree_sitter_type", "")
                        if sub_type in ("dotted_name", "identifier"):
                            if not original:
                                original = sub.text
                            else:
                                alias = sub.text
                    if original:
                        names.append(ImportedName(name=original, alias=alias))

            kind = ImportKind.STAR_IMPORT if is_star else ImportKind.FROM_IMPORT
            if is_relative:
                kind = ImportKind.RELATIVE

            return ExtractedImport(
                module=module,
                kind=kind,
                file_path=file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                start_col=node.start_col,
                end_col=node.end_col,
                names=names,
                is_relative=is_relative,
                relative_level=relative_level,
                is_type_only=is_type_only,
                is_conditional=is_conditional,
                is_stdlib=self._is_stdlib_python(module) if module else False,
                raw_text=node.text,
                language="python",
            )

        return None

    def _extract_js_import(
        self,
        node: ParsedNode,
        file_path: str,
        language: str,
        source: str,
        is_type_only: bool,
        is_conditional: bool,
    ) -> ExtractedImport | None:
        """Extract JavaScript/TypeScript import."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type not in ("import_statement", "import_declaration"):
            return None

        module = ""
        names = []
        is_type_import = is_type_only

        for child in node.children:
            child_type = child.metadata.get("tree_sitter_type", "")

            # Check for type import
            if child.text == "type":
                is_type_import = True

            # Module specifier (the string path)
            if child_type == "string":
                module = child.text.strip("'\"")

            # Named imports
            elif child_type in ("import_clause", "named_imports", "import_specifier"):
                for sub in child.children:
                    sub_type = sub.metadata.get("tree_sitter_type", "")
                    if sub_type == "identifier":
                        names.append(ImportedName(name=sub.text))
                    elif sub_type == "import_specifier":
                        original = ""
                        alias = None
                        for spec in sub.children:
                            spec_type = spec.metadata.get("tree_sitter_type", "")
                            if spec_type == "identifier":
                                if not original:
                                    original = spec.text
                                else:
                                    alias = spec.text
                        if original:
                            names.append(ImportedName(name=original, alias=alias))

            # Default import
            elif child_type == "identifier":
                names.append(ImportedName(name=child.text))

        if module:
            # Determine if relative
            is_relative = module.startswith(".")
            relative_level = 0
            if is_relative:
                for c in module:
                    if c == ".":
                        relative_level += 1
                    elif c != "/":
                        break

            return ExtractedImport(
                module=module,
                kind=ImportKind.RELATIVE if is_relative else ImportKind.FROM_IMPORT,
                file_path=file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                start_col=node.start_col,
                end_col=node.end_col,
                names=names,
                is_relative=is_relative,
                relative_level=relative_level,
                is_type_only=is_type_import,
                is_conditional=is_conditional,
                raw_text=node.text,
                language=language,
            )

        return None

    def _extract_go_import(
        self,
        node: ParsedNode,
        file_path: str,
        source: str,
    ) -> ExtractedImport | None:
        """Extract Go import."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type != "import_declaration":
            return None

        # Go imports can be single or grouped
        imports = []

        for child in node.children:
            child_type = child.metadata.get("tree_sitter_type", "")

            if child_type == "import_spec":
                module = ""
                alias = None

                for sub in child.children:
                    sub_type = sub.metadata.get("tree_sitter_type", "")
                    if sub_type == "interpreted_string_literal":
                        module = sub.text.strip('"')
                    elif sub_type in ("package_identifier", "identifier", "blank_identifier"):
                        alias = sub.text

                if module:
                    imports.append(ExtractedImport(
                        module=module,
                        kind=ImportKind.IMPORT_ALIAS if alias else ImportKind.IMPORT,
                        file_path=file_path,
                        start_line=child.start_line,
                        end_line=child.end_line,
                        names=[ImportedName(name=module.split("/")[-1], alias=alias)],
                        raw_text=child.text,
                        language="go",
                    ))

            elif child_type == "import_spec_list":
                # Grouped imports
                for spec in child.children:
                    spec_type = spec.metadata.get("tree_sitter_type", "")
                    if spec_type == "import_spec":
                        module = ""
                        alias = None

                        for sub in spec.children:
                            sub_type = sub.metadata.get("tree_sitter_type", "")
                            if sub_type == "interpreted_string_literal":
                                module = sub.text.strip('"')
                            elif sub_type in ("package_identifier", "identifier", "blank_identifier"):
                                alias = sub.text

                        if module:
                            imports.append(ExtractedImport(
                                module=module,
                                kind=ImportKind.IMPORT_ALIAS if alias else ImportKind.IMPORT,
                                file_path=file_path,
                                start_line=spec.start_line,
                                end_line=spec.end_line,
                                names=[ImportedName(name=module.split("/")[-1], alias=alias)],
                                raw_text=spec.text,
                                language="go",
                            ))

        # Return first import or None
        # Note: For grouped imports, caller should handle multiple
        return imports[0] if imports else None

    def _extract_rust_import(
        self,
        node: ParsedNode,
        file_path: str,
        source: str,
    ) -> ExtractedImport | None:
        """Extract Rust use statement."""
        ts_type = node.metadata.get("tree_sitter_type", "")

        if ts_type != "use_declaration":
            return None

        # Parse use path
        module = ""
        names = []

        for child in node.children:
            child_type = child.metadata.get("tree_sitter_type", "")
            if child_type in ("use_tree", "scoped_identifier", "identifier"):
                # Extract the full path
                text = child.text
                if "::" in text:
                    parts = text.split("::")
                    module = "::".join(parts[:-1])
                    names.append(ImportedName(name=parts[-1]))
                else:
                    module = text
                    names.append(ImportedName(name=text))

        if module or names:
            return ExtractedImport(
                module=module,
                kind=ImportKind.FROM_IMPORT,
                file_path=file_path,
                start_line=node.start_line,
                end_line=node.end_line,
                names=names,
                raw_text=node.text,
                language="rust",
            )

        return None

    def _is_stdlib_python(self, module: str) -> bool:
        """Check if a Python module is from the standard library."""
        if not module:
            return False
        root = module.split(".")[0]
        return root in self.PYTHON_STDLIB

    def _resolve_import(self, imp: ExtractedImport, file_path: str) -> None:
        """Resolve an import to its source file."""
        if not self.project_root:
            return

        # Determine import type
        if imp.is_stdlib:
            return

        # For relative imports
        if imp.is_relative and imp.language == "python":
            current_dir = Path(file_path).parent
            # Go up relative_level directories
            target_dir = current_dir
            for _ in range(imp.relative_level - 1):  # -1 because first dot is current
                target_dir = target_dir.parent

            # Resolve module path
            if imp.module:
                module_path = target_dir / imp.module.replace(".", "/")
            else:
                module_path = target_dir

            # Check for file or package
            py_file = module_path.with_suffix(".py")
            init_file = module_path / "__init__.py"

            if py_file.exists():
                imp.resolved_path = str(py_file)
                imp.is_local = True
            elif init_file.exists():
                imp.resolved_path = str(init_file)
                imp.is_local = True


def extract_imports_from_source(
    source: str,
    language: str,
    file_path: str = "<string>",
) -> list[ExtractedImport]:
    """Convenience function to extract imports from source code.

    Args:
        source: Source code
        language: Programming language
        file_path: Path to the file (for context)

    Returns:
        List of extracted imports
    """
    extractor = ImportExtractor()
    return extractor.extract_from_file(file_path, source, language)
