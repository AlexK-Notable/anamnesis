"""
Configuration types for semantic analysis.

Contains file filtering logic and analysis configuration.
Ported from Rust config.rs
"""

from dataclasses import dataclass, field
from pathlib import Path


# Default ignored directories
IGNORED_DIRECTORIES: frozenset[str] = frozenset([
    "node_modules",
    ".git",
    "target",
    "dist",
    "build",
    "out",
    "output",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".vitepress",
    "_site",
    "public",
    "static",
    "assets",
    "__pycache__",
    ".pytest_cache",
    "coverage",
    ".coverage",
    "htmlcov",
    "vendor",
    "bin",
    "obj",
    "Debug",
    "Release",
    ".venv",
    "venv",
    "env",
    ".env",
    "tmp",
    "temp",
    ".tmp",
    "cache",
    ".cache",
    "logs",
    ".logs",
    "lib-cov",
    "nyc_output",
    ".nyc_output",
    "bower_components",
    "jspm_packages",
])

# Default ignored file patterns
IGNORED_FILE_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    ".map",
)

IGNORED_FILE_NAMES: frozenset[str] = frozenset([
    "package-lock.json",
    "yarn.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "Pipfile.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "uv.lock",
])

# Default supported extensions
DEFAULT_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset([
    "ts", "tsx", "js", "jsx", "rs", "py", "go", "java",
    "cpp", "c", "cs", "svelte", "sql", "cc", "cxx", "h", "hpp",
])

# Language detection mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "rs": "rust",
    "py": "python",
    "sql": "sql",
    "go": "go",
    "java": "java",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "svelte": "svelte",
}


@dataclass
class AnalysisConfig:
    """Configuration for file analysis."""

    max_file_size: int = 1_048_576  # 1MB
    max_files: int = 1000
    supported_extensions: frozenset[str] = field(
        default_factory=lambda: DEFAULT_SUPPORTED_EXTENSIONS
    )
    ignored_directories: frozenset[str] = field(
        default_factory=lambda: IGNORED_DIRECTORIES
    )
    ignored_file_names: frozenset[str] = field(
        default_factory=lambda: IGNORED_FILE_NAMES
    )
    ignored_file_suffixes: tuple[str, ...] = IGNORED_FILE_SUFFIXES

    def should_analyze_file(self, file_path: str | Path) -> bool:
        """Check if a file should be analyzed based on configuration rules."""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        path_str = str(path)

        # Skip common non-source directories and build artifacts
        if self._is_ignored_directory(path_str):
            return False

        # Skip common generated/minified file patterns
        file_name = path.name
        if self._is_ignored_file(file_name):
            return False

        # Check file size - skip very large files to prevent hanging
        if path.exists():
            try:
                if path.stat().st_size > self.max_file_size:
                    return False
            except OSError:
                pass

        # Check if file extension is supported
        extension = path.suffix.lstrip(".").lower()
        return extension in self.supported_extensions

    def _is_ignored_directory(self, path_str: str) -> bool:
        """Check if a directory should be ignored."""
        return any(ignored in path_str for ignored in self.ignored_directories)

    def _is_ignored_file(self, file_name: str) -> bool:
        """Check if a file should be ignored based on its name."""
        # Check for ignored suffixes
        if any(file_name.endswith(suffix) for suffix in self.ignored_file_suffixes):
            return True

        # Check for hidden files (starting with .)
        if file_name.startswith("."):
            return True

        # Check for ignored file names
        return file_name in self.ignored_file_names

    def detect_language_from_path(self, file_path: str | Path) -> str:
        """Detect programming language from file path."""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        extension = path.suffix.lstrip(".").lower()
        return EXTENSION_TO_LANGUAGE.get(extension, "generic")


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    path: str = ".anamnesis/data.db"
    vector_dimensions: int = 384
    enable_wal: bool = True
    max_connections: int = 5


@dataclass
class CacheConfig:
    """Configuration for caching."""

    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    enable_persistence: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    rotation: str = "10 MB"
    retention: str = "1 week"
    enable_correlation_id: bool = True


@dataclass
class WatcherConfig:
    """Configuration for file watching."""

    debounce_ms: int = 100
    include_content: bool = False
    persistent: bool = True
    patterns: list[str] = field(default_factory=lambda: ["**/*"])
    ignored: list[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
        "**/build/**",
        "**/__pycache__/**",
    ])


@dataclass
class EngineConfig:
    """Configuration for analysis engines."""

    enable_semantic: bool = True
    enable_pattern: bool = True
    enable_vector: bool = True
    semantic_confidence_threshold: float = 0.5
    pattern_confidence_threshold: float = 0.6
    max_concepts_per_file: int = 100
    max_patterns_per_file: int = 50


@dataclass
class AnamnesisConfig:
    """Complete configuration for Anamnesis."""

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    version: str = "0.1.0"
