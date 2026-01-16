"""Dependency graph builder and analyzer.

This module builds and analyzes dependency graphs from extracted imports,
enabling detection of circular dependencies, coupling analysis, and
understanding of module relationships.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anamnesis.extractors.import_extractor import ExtractedImport


class DependencyType(str, Enum):
    """Types of dependencies between modules."""

    DIRECT = "direct"  # import foo
    FROM_IMPORT = "from_import"  # from foo import bar
    RELATIVE = "relative"  # from . import bar
    STAR = "star"  # from foo import *
    TYPE_ONLY = "type_only"  # TYPE_CHECKING imports
    DYNAMIC = "dynamic"  # importlib, __import__


@dataclass
class DependencyNode:
    """Represents a node (module/file) in the dependency graph."""

    path: str
    name: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    is_external: bool = False
    is_stdlib: bool = False
    language: str = "python"
    node_type: str = "module"
    metadata: dict = field(default_factory=dict)

    @property
    def import_count(self) -> int:
        """Number of modules this node imports."""
        return len(self.imports)

    @property
    def imported_by_count(self) -> int:
        """Number of modules that import this node."""
        return len(self.imported_by)

    @property
    def coupling_score(self) -> float:
        """Calculate coupling score (higher = more coupled)."""
        # Afferent + Efferent coupling
        return self.import_count + self.imported_by_count

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "imports": self.imports,
            "imported_by": self.imported_by,
            "is_external": self.is_external,
            "is_stdlib": self.is_stdlib,
            "language": self.language,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "import_count": self.import_count,
            "imported_by_count": self.imported_by_count,
            "coupling_score": self.coupling_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DependencyNode:
        """Deserialize from dictionary."""
        return cls(
            path=data["path"],
            name=data["name"],
            imports=data.get("imports", []),
            imported_by=data.get("imported_by", []),
            is_external=data.get("is_external", False),
            is_stdlib=data.get("is_stdlib", False),
            language=data.get("language", "python"),
            node_type=data.get("node_type", "module"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DependencyEdge:
    """Represents an edge (dependency) in the graph."""

    source: str  # importing module
    target: str  # imported module
    dependency_type: DependencyType = DependencyType.DIRECT
    names: list[str] = field(default_factory=list)  # specific imports
    is_type_only: bool = False
    line_number: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "dependency_type": self.dependency_type.value,
            "names": self.names,
            "is_type_only": self.is_type_only,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DependencyEdge:
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            dependency_type=DependencyType(data.get("dependency_type", "direct")),
            names=data.get("names", []),
            is_type_only=data.get("is_type_only", False),
            line_number=data.get("line_number", 0),
        )


@dataclass
class CircularDependency:
    """Represents a detected circular dependency."""

    cycle: list[str]  # modules in the cycle
    severity: str = "warning"  # warning, error

    @property
    def length(self) -> int:
        """Length of the cycle."""
        return len(self.cycle)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "cycle": self.cycle,
            "severity": self.severity,
            "length": self.length,
        }


@dataclass
class DependencyMetrics:
    """Aggregated dependency metrics for a codebase."""

    total_modules: int = 0
    total_edges: int = 0
    external_dependencies: int = 0
    stdlib_dependencies: int = 0
    circular_dependencies: int = 0
    max_depth: int = 0
    avg_imports_per_module: float = 0.0
    avg_imported_by: float = 0.0
    instability: float = 0.0  # Ce / (Ca + Ce)
    abstractness: float = 0.0
    most_imported: list[tuple[str, int]] = field(default_factory=list)
    most_importing: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_modules": self.total_modules,
            "total_edges": self.total_edges,
            "external_dependencies": self.external_dependencies,
            "stdlib_dependencies": self.stdlib_dependencies,
            "circular_dependencies": self.circular_dependencies,
            "max_depth": self.max_depth,
            "avg_imports_per_module": self.avg_imports_per_module,
            "avg_imported_by": self.avg_imported_by,
            "instability": self.instability,
            "abstractness": self.abstractness,
            "most_imported": self.most_imported,
            "most_importing": self.most_importing,
        }


class DependencyGraph:
    """Builds and analyzes dependency graphs from extracted imports."""

    def __init__(
        self,
        project_root: str | None = None,
        include_external: bool = True,
        include_stdlib: bool = False,
    ) -> None:
        """Initialize dependency graph builder.

        Args:
            project_root: Root path for resolving relative imports.
            include_external: Include external (third-party) dependencies.
            include_stdlib: Include standard library dependencies.
        """
        self.project_root = Path(project_root) if project_root else None
        self.include_external = include_external
        self.include_stdlib = include_stdlib

        self.nodes: dict[str, DependencyNode] = {}
        self.edges: list[DependencyEdge] = []
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._reverse_adjacency: dict[str, set[str]] = defaultdict(set)

    def add_node(
        self,
        name: str,
        node_type: str = "unknown",
        file_path: str = "",
        metadata: dict | None = None,
    ) -> DependencyNode:
        """Add a node to the graph.

        Args:
            name: Unique name for the node.
            node_type: Type of the node (e.g., 'class', 'function', 'module').
            file_path: Path to the file containing this node.
            metadata: Optional additional metadata.

        Returns:
            The created or existing DependencyNode.
        """
        if name not in self.nodes:
            self.nodes[name] = DependencyNode(
                path=file_path,
                name=name,
                language="unknown",
                metadata=metadata or {},
            )
            self.nodes[name].node_type = node_type
        return self.nodes[name]

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "unknown",
    ) -> DependencyEdge | None:
        """Add an edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
            edge_type: Type of the relationship.

        Returns:
            The created DependencyEdge or None if nodes don't exist.
        """
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)

        # Convert string edge_type to DependencyType enum
        try:
            dep_type = DependencyType(edge_type)
        except ValueError:
            dep_type = DependencyType.DIRECT

        edge = DependencyEdge(
            source=source,
            target=target,
            dependency_type=dep_type,
        )
        self.edges.append(edge)
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)
        return edge

    def build_from_imports(
        self,
        imports: list[ExtractedImport],
        file_modules: dict[str, str] | None = None,
    ) -> None:
        """Build dependency graph from extracted imports.

        Args:
            imports: List of extracted imports from ImportExtractor.
            file_modules: Optional mapping of file paths to module names.
        """
        file_modules = file_modules or {}

        # Group imports by file
        imports_by_file: dict[str, list[ExtractedImport]] = defaultdict(list)
        for imp in imports:
            imports_by_file[imp.file_path].append(imp)

        # Create nodes for each file
        for file_path, file_imports in imports_by_file.items():
            module_name = file_modules.get(file_path, self._path_to_module(file_path))
            if module_name not in self.nodes:
                self.nodes[module_name] = DependencyNode(
                    path=file_path,
                    name=module_name,
                    language=file_imports[0].language if file_imports else "python",
                )

            # Process each import
            for imp in file_imports:
                self._process_import(module_name, imp)

        # Build reverse adjacency
        for source, targets in self._adjacency.items():
            for target in targets:
                self._reverse_adjacency[target].add(source)

        # Update imported_by lists
        for module_name, node in self.nodes.items():
            node.imported_by = list(self._reverse_adjacency[module_name])

    def _process_import(self, source_module: str, imp: ExtractedImport) -> None:
        """Process a single import and add to graph."""
        from anamnesis.extractors.import_extractor import ImportKind

        # Skip if we're excluding stdlib/external
        if imp.is_stdlib and not self.include_stdlib:
            return
        if not imp.is_stdlib and not imp.is_relative and not self.include_external:
            # Check if it's a local module
            if not self._is_local_module(imp.module):
                return

        target_module = imp.module
        if imp.is_relative:
            target_module = self._resolve_relative_import(
                source_module, imp.module, imp.relative_level
            )

        # Determine dependency type
        dep_type = DependencyType.DIRECT
        if imp.kind == ImportKind.FROM_IMPORT:
            dep_type = DependencyType.FROM_IMPORT
        elif imp.kind == ImportKind.RELATIVE:
            dep_type = DependencyType.RELATIVE
        elif imp.kind == ImportKind.STAR_IMPORT:
            dep_type = DependencyType.STAR
        elif imp.kind == ImportKind.TYPE_ONLY:
            dep_type = DependencyType.TYPE_ONLY
        elif imp.kind == ImportKind.DYNAMIC:
            dep_type = DependencyType.DYNAMIC

        # Create target node if doesn't exist
        if target_module not in self.nodes:
            self.nodes[target_module] = DependencyNode(
                path="",  # Unknown for external
                name=target_module,
                is_external=not imp.is_relative and not self._is_local_module(target_module),
                is_stdlib=imp.is_stdlib,
            )

        # Create edge
        edge = DependencyEdge(
            source=source_module,
            target=target_module,
            dependency_type=dep_type,
            names=[n.name for n in imp.names] if imp.names else [],
            is_type_only=imp.kind == ImportKind.TYPE_ONLY,
            line_number=imp.start_line,
        )
        self.edges.append(edge)

        # Update adjacency
        self._adjacency[source_module].add(target_module)
        self.nodes[source_module].imports.append(target_module)

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        path = Path(file_path)

        if self.project_root:
            try:
                path = path.relative_to(self.project_root)
            except ValueError:
                pass

        # Remove extension and convert to module path
        module = str(path.with_suffix("")).replace("/", ".").replace("\\", ".")

        # Handle __init__.py
        if module.endswith(".__init__"):
            module = module[:-9]

        return module

    def _resolve_relative_import(
        self, source_module: str, target: str, level: int
    ) -> str:
        """Resolve relative import to absolute module path."""
        parts = source_module.split(".")

        # Go up 'level' directories
        if level > len(parts):
            return target or source_module

        base_parts = parts[: -level] if level > 0 else parts

        if target:
            return ".".join(base_parts + [target])
        return ".".join(base_parts)

    def _is_local_module(self, module: str) -> bool:
        """Check if module is a local (project) module."""
        if not self.project_root:
            return False

        # Check if module path exists in project
        module_path = self.project_root / module.replace(".", "/")
        return (
            module_path.exists()
            or module_path.with_suffix(".py").exists()
            or (module_path / "__init__.py").exists()
        )

    def find_circular_dependencies(
        self, max_depth: int = 500
    ) -> list[CircularDependency]:
        """Find all circular dependencies in the graph.

        Args:
            max_depth: Maximum recursion depth to prevent stack overflow.

        Returns:
            List of detected circular dependencies.
        """
        cycles: list[CircularDependency] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str, depth: int = 0) -> None:
            if depth >= max_depth:
                return  # Prevent stack overflow

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, depth + 1)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(
                        CircularDependency(
                            cycle=cycle,
                            severity="error" if len(cycle) <= 3 else "warning",
                        )
                    )

            path.pop()
            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node, 0)

        return cycles

    def get_dependency_depth(self, module: str, max_depth: int = 500) -> int:
        """Get maximum dependency depth from a module.

        Args:
            module: Module name to analyze.
            max_depth: Maximum recursion depth to prevent stack overflow.

        Returns:
            Maximum dependency depth, capped at max_depth.
        """
        if module not in self.nodes:
            return 0

        visited: set[str] = set()

        def get_depth(node: str, current_depth: int) -> int:
            if current_depth >= max_depth:
                return max_depth  # Prevent stack overflow

            if node in visited:
                return current_depth
            visited.add(node)

            result = current_depth
            for neighbor in self._adjacency.get(node, set()):
                result = max(result, get_depth(neighbor, current_depth + 1))

            return result

        return get_depth(module, 0)

    def get_dependents(self, module: str) -> list[str]:
        """Get all modules that depend on the given module (directly or indirectly)."""
        if module not in self.nodes:
            return []

        dependents: set[str] = set()
        queue = list(self._reverse_adjacency.get(module, set()))

        while queue:
            dependent = queue.pop(0)
            if dependent not in dependents:
                dependents.add(dependent)
                queue.extend(self._reverse_adjacency.get(dependent, set()))

        return list(dependents)

    def get_dependencies(self, module: str) -> list[str]:
        """Get all dependencies of a module (directly or indirectly)."""
        if module not in self.nodes:
            return []

        dependencies: set[str] = set()
        queue = list(self._adjacency.get(module, set()))

        while queue:
            dep = queue.pop(0)
            if dep not in dependencies:
                dependencies.add(dep)
                queue.extend(self._adjacency.get(dep, set()))

        return list(dependencies)

    def calculate_metrics(self) -> DependencyMetrics:
        """Calculate aggregated dependency metrics."""
        if not self.nodes:
            return DependencyMetrics()

        internal_nodes = [n for n in self.nodes.values() if not n.is_external]
        external_count = sum(1 for n in self.nodes.values() if n.is_external)
        stdlib_count = sum(1 for n in self.nodes.values() if n.is_stdlib)

        # Calculate max depth
        max_depth = 0
        for module in self.nodes:
            depth = self.get_dependency_depth(module)
            max_depth = max(max_depth, depth)

        # Calculate averages
        total_imports = sum(n.import_count for n in internal_nodes)
        total_imported_by = sum(n.imported_by_count for n in internal_nodes)
        avg_imports = total_imports / len(internal_nodes) if internal_nodes else 0
        avg_imported_by = total_imported_by / len(internal_nodes) if internal_nodes else 0

        # Calculate instability (Ce / (Ca + Ce))
        total_efferent = total_imports  # Outgoing
        total_afferent = total_imported_by  # Incoming
        instability = (
            total_efferent / (total_afferent + total_efferent)
            if (total_afferent + total_efferent) > 0
            else 0
        )

        # Find most imported/importing modules
        sorted_by_imported = sorted(
            internal_nodes, key=lambda n: n.imported_by_count, reverse=True
        )
        sorted_by_importing = sorted(
            internal_nodes, key=lambda n: n.import_count, reverse=True
        )

        circular = self.find_circular_dependencies()

        return DependencyMetrics(
            total_modules=len(internal_nodes),
            total_edges=len(self.edges),
            external_dependencies=external_count,
            stdlib_dependencies=stdlib_count,
            circular_dependencies=len(circular),
            max_depth=max_depth,
            avg_imports_per_module=avg_imports,
            avg_imported_by=avg_imported_by,
            instability=instability,
            most_imported=[
                (n.name, n.imported_by_count) for n in sorted_by_imported[:5]
            ],
            most_importing=[
                (n.name, n.import_count) for n in sorted_by_importing[:5]
            ],
        )

    def to_dict(self) -> dict:
        """Serialize entire graph to dictionary."""
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metrics": self.calculate_metrics().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary with nodes and edges.

        Returns:
            Reconstructed DependencyGraph instance.
        """
        graph = cls()

        # Restore nodes
        nodes_data = data.get("nodes", {})
        if isinstance(nodes_data, dict):
            for name, node_dict in nodes_data.items():
                graph.nodes[name] = DependencyNode.from_dict(node_dict)
        elif isinstance(nodes_data, list):
            # Also support list format for flexibility
            for node_dict in nodes_data:
                node = DependencyNode.from_dict(node_dict)
                graph.nodes[node.name] = node

        # Restore edges
        for edge_dict in data.get("edges", []):
            graph.edges.append(DependencyEdge.from_dict(edge_dict))

        # Rebuild adjacency structures
        for edge in graph.edges:
            graph._adjacency[edge.source].add(edge.target)
            graph._reverse_adjacency[edge.target].add(edge.source)

        return graph

    def to_mermaid(self, max_nodes: int = 20) -> str:
        """Generate Mermaid diagram representation.

        Args:
            max_nodes: Maximum number of nodes to include.
        """
        lines = ["graph TD"]

        # Get most connected nodes
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.coupling_score,
            reverse=True,
        )[:max_nodes]
        included_nodes = {n.name for n in sorted_nodes}

        # Add edges
        for edge in self.edges:
            if edge.source in included_nodes and edge.target in included_nodes:
                source = edge.source.replace(".", "_")
                target = edge.target.replace(".", "_")
                lines.append(f"    {source} --> {target}")

        return "\n".join(lines)


def build_dependency_graph(
    imports: list[ExtractedImport],
    project_root: str | None = None,
    include_external: bool = True,
    include_stdlib: bool = False,
) -> DependencyGraph:
    """Convenience function to build a dependency graph.

    Args:
        imports: List of extracted imports.
        project_root: Root path for the project.
        include_external: Include external dependencies.
        include_stdlib: Include stdlib dependencies.

    Returns:
        Built dependency graph.
    """
    graph = DependencyGraph(
        project_root=project_root,
        include_external=include_external,
        include_stdlib=include_stdlib,
    )
    graph.build_from_imports(imports)
    return graph
