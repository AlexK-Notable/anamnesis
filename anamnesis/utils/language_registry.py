"""
Language detection and registry for code analysis.

Provides:
- Extension-to-language mapping
- Language-to-extensions reverse mapping
- Language detection from file paths
- Language metadata and categorization

Consolidated from patterns found across the TypeScript In-Memoria codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ============================================================================
# Language Categories
# ============================================================================


class LanguageCategory(Enum):
    """Categories for programming languages."""

    SYSTEMS = "systems"  # Low-level, compiled (C, C++, Rust)
    WEB_FRONTEND = "web_frontend"  # Browser-focused (JavaScript, TypeScript)
    WEB_BACKEND = "web_backend"  # Server-side web (PHP, Ruby)
    GENERAL = "general"  # Multi-purpose (Python, Java, Go)
    FUNCTIONAL = "functional"  # Functional paradigm (Haskell, Clojure, Elm)
    MOBILE = "mobile"  # Mobile development (Swift, Kotlin, Dart)
    DATA_SCIENCE = "data_science"  # Data/ML focused (R, Julia)
    SCRIPTING = "scripting"  # Scripting languages (Lua, Perl, Shell)
    MARKUP = "markup"  # Markup/config (HTML, CSS, YAML, JSON)
    QUERY = "query"  # Query languages (SQL, GraphQL)
    OTHER = "other"


# ============================================================================
# Language Metadata
# ============================================================================


@dataclass
class LanguageInfo:
    """Metadata about a programming language."""

    name: str  # Canonical name (lowercase)
    display_name: str  # Human-readable name
    extensions: list[str]  # File extensions (without dot)
    category: LanguageCategory
    typed: bool = False  # Statically typed
    compiled: bool = False  # Compiled (vs interpreted)
    comment_styles: list[str] = field(default_factory=list)  # "//" or "#" or "--"
    file_patterns: list[str] = field(default_factory=list)  # Glob patterns
    aliases: list[str] = field(default_factory=list)  # Alternative names


# ============================================================================
# Language Registry
# ============================================================================


# Comprehensive language definitions
LANGUAGES: dict[str, LanguageInfo] = {
    # Systems languages
    "c": LanguageInfo(
        name="c",
        display_name="C",
        extensions=["c", "h"],
        category=LanguageCategory.SYSTEMS,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.c", "*.h"],
    ),
    "cpp": LanguageInfo(
        name="cpp",
        display_name="C++",
        extensions=["cpp", "cc", "cxx", "hpp", "hxx", "h++"],
        category=LanguageCategory.SYSTEMS,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.cpp", "*.cc", "*.cxx", "*.hpp"],
        aliases=["c++", "cplusplus"],
    ),
    "rust": LanguageInfo(
        name="rust",
        display_name="Rust",
        extensions=["rs"],
        category=LanguageCategory.SYSTEMS,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.rs"],
    ),
    # Web frontend
    "javascript": LanguageInfo(
        name="javascript",
        display_name="JavaScript",
        extensions=["js", "jsx", "mjs", "cjs"],
        category=LanguageCategory.WEB_FRONTEND,
        typed=False,
        compiled=False,
        comment_styles=["//", "/*"],
        file_patterns=["*.js", "*.jsx", "*.mjs"],
        aliases=["js", "ecmascript"],
    ),
    "typescript": LanguageInfo(
        name="typescript",
        display_name="TypeScript",
        extensions=["ts", "tsx", "mts", "cts"],
        category=LanguageCategory.WEB_FRONTEND,
        typed=True,
        compiled=False,  # Transpiled
        comment_styles=["//", "/*"],
        file_patterns=["*.ts", "*.tsx", "*.mts"],
        aliases=["ts"],
    ),
    "vue": LanguageInfo(
        name="vue",
        display_name="Vue",
        extensions=["vue"],
        category=LanguageCategory.WEB_FRONTEND,
        typed=False,
        compiled=False,
        comment_styles=["//", "/*", "<!--"],
        file_patterns=["*.vue"],
    ),
    "svelte": LanguageInfo(
        name="svelte",
        display_name="Svelte",
        extensions=["svelte"],
        category=LanguageCategory.WEB_FRONTEND,
        typed=False,
        compiled=False,
        comment_styles=["//", "/*", "<!--"],
        file_patterns=["*.svelte"],
    ),
    # Web backend
    "php": LanguageInfo(
        name="php",
        display_name="PHP",
        extensions=["php", "phtml", "php3", "php4", "php5", "phps"],
        category=LanguageCategory.WEB_BACKEND,
        typed=False,
        compiled=False,
        comment_styles=["//", "#", "/*"],
        file_patterns=["*.php"],
    ),
    "ruby": LanguageInfo(
        name="ruby",
        display_name="Ruby",
        extensions=["rb", "rake", "gemspec"],
        category=LanguageCategory.WEB_BACKEND,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.rb", "Rakefile", "Gemfile"],
        aliases=["rb"],
    ),
    # General purpose
    "python": LanguageInfo(
        name="python",
        display_name="Python",
        extensions=["py", "pyw", "pyi", "pyx"],
        category=LanguageCategory.GENERAL,
        typed=False,  # Optional type hints
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.py"],
        aliases=["py", "python3"],
    ),
    "java": LanguageInfo(
        name="java",
        display_name="Java",
        extensions=["java"],
        category=LanguageCategory.GENERAL,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.java"],
    ),
    "go": LanguageInfo(
        name="go",
        display_name="Go",
        extensions=["go"],
        category=LanguageCategory.GENERAL,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.go"],
        aliases=["golang"],
    ),
    "csharp": LanguageInfo(
        name="csharp",
        display_name="C#",
        extensions=["cs", "csx"],
        category=LanguageCategory.GENERAL,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.cs"],
        aliases=["c#", "cs"],
    ),
    # Functional languages
    "haskell": LanguageInfo(
        name="haskell",
        display_name="Haskell",
        extensions=["hs", "lhs"],
        category=LanguageCategory.FUNCTIONAL,
        typed=True,
        compiled=True,
        comment_styles=["--", "{-"],
        file_patterns=["*.hs"],
        aliases=["hs"],
    ),
    "clojure": LanguageInfo(
        name="clojure",
        display_name="Clojure",
        extensions=["clj", "cljs", "cljc", "edn"],
        category=LanguageCategory.FUNCTIONAL,
        typed=False,
        compiled=False,
        comment_styles=[";"],
        file_patterns=["*.clj", "*.cljs"],
        aliases=["clj"],
    ),
    "elm": LanguageInfo(
        name="elm",
        display_name="Elm",
        extensions=["elm"],
        category=LanguageCategory.FUNCTIONAL,
        typed=True,
        compiled=True,
        comment_styles=["--", "{-"],
        file_patterns=["*.elm"],
    ),
    "ocaml": LanguageInfo(
        name="ocaml",
        display_name="OCaml",
        extensions=["ml", "mli"],
        category=LanguageCategory.FUNCTIONAL,
        typed=True,
        compiled=True,
        comment_styles=["(*"],
        file_patterns=["*.ml", "*.mli"],
        aliases=["ml"],
    ),
    "fsharp": LanguageInfo(
        name="fsharp",
        display_name="F#",
        extensions=["fs", "fsi", "fsx"],
        category=LanguageCategory.FUNCTIONAL,
        typed=True,
        compiled=True,
        comment_styles=["//", "(*"],
        file_patterns=["*.fs", "*.fsx"],
        aliases=["f#"],
    ),
    "scala": LanguageInfo(
        name="scala",
        display_name="Scala",
        extensions=["scala", "sc"],
        category=LanguageCategory.FUNCTIONAL,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.scala"],
    ),
    # Mobile
    "swift": LanguageInfo(
        name="swift",
        display_name="Swift",
        extensions=["swift"],
        category=LanguageCategory.MOBILE,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.swift"],
    ),
    "kotlin": LanguageInfo(
        name="kotlin",
        display_name="Kotlin",
        extensions=["kt", "kts"],
        category=LanguageCategory.MOBILE,
        typed=True,
        compiled=True,
        comment_styles=["//", "/*"],
        file_patterns=["*.kt", "*.kts"],
        aliases=["kt"],
    ),
    "dart": LanguageInfo(
        name="dart",
        display_name="Dart",
        extensions=["dart"],
        category=LanguageCategory.MOBILE,
        typed=True,
        compiled=False,  # JIT/AOT
        comment_styles=["//", "/*"],
        file_patterns=["*.dart"],
    ),
    # Data science
    "r": LanguageInfo(
        name="r",
        display_name="R",
        extensions=["r", "R", "rmd", "Rmd"],
        category=LanguageCategory.DATA_SCIENCE,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.r", "*.R", "*.rmd"],
    ),
    "julia": LanguageInfo(
        name="julia",
        display_name="Julia",
        extensions=["jl"],
        category=LanguageCategory.DATA_SCIENCE,
        typed=False,  # Optional types
        compiled=False,  # JIT compiled
        comment_styles=["#", "#="],
        file_patterns=["*.jl"],
        aliases=["jl"],
    ),
    # Scripting
    "lua": LanguageInfo(
        name="lua",
        display_name="Lua",
        extensions=["lua"],
        category=LanguageCategory.SCRIPTING,
        typed=False,
        compiled=False,
        comment_styles=["--", "--[["],
        file_patterns=["*.lua"],
    ),
    "perl": LanguageInfo(
        name="perl",
        display_name="Perl",
        extensions=["pl", "pm", "t"],
        category=LanguageCategory.SCRIPTING,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.pl", "*.pm"],
        aliases=["pl"],
    ),
    "shell": LanguageInfo(
        name="shell",
        display_name="Shell",
        extensions=["sh", "bash", "zsh", "fish"],
        category=LanguageCategory.SCRIPTING,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.sh", "*.bash"],
        aliases=["bash", "sh", "zsh"],
    ),
    "powershell": LanguageInfo(
        name="powershell",
        display_name="PowerShell",
        extensions=["ps1", "psm1", "psd1"],
        category=LanguageCategory.SCRIPTING,
        typed=False,
        compiled=False,
        comment_styles=["#", "<#"],
        file_patterns=["*.ps1"],
        aliases=["ps1"],
    ),
    # Markup / Config
    "html": LanguageInfo(
        name="html",
        display_name="HTML",
        extensions=["html", "htm", "xhtml"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["<!--"],
        file_patterns=["*.html", "*.htm"],
    ),
    "css": LanguageInfo(
        name="css",
        display_name="CSS",
        extensions=["css", "scss", "sass", "less"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["/*", "//"],
        file_patterns=["*.css", "*.scss"],
    ),
    "yaml": LanguageInfo(
        name="yaml",
        display_name="YAML",
        extensions=["yaml", "yml"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.yaml", "*.yml"],
        aliases=["yml"],
    ),
    "json": LanguageInfo(
        name="json",
        display_name="JSON",
        extensions=["json", "jsonc", "json5"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=[],  # JSON doesn't support comments
        file_patterns=["*.json"],
    ),
    "xml": LanguageInfo(
        name="xml",
        display_name="XML",
        extensions=["xml", "xsd", "xsl", "xslt", "svg"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["<!--"],
        file_patterns=["*.xml"],
    ),
    "toml": LanguageInfo(
        name="toml",
        display_name="TOML",
        extensions=["toml"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.toml"],
    ),
    "markdown": LanguageInfo(
        name="markdown",
        display_name="Markdown",
        extensions=["md", "markdown", "mdown", "mkd"],
        category=LanguageCategory.MARKUP,
        typed=False,
        compiled=False,
        comment_styles=["<!--"],
        file_patterns=["*.md", "*.markdown"],
        aliases=["md"],
    ),
    # Query languages
    "sql": LanguageInfo(
        name="sql",
        display_name="SQL",
        extensions=["sql"],
        category=LanguageCategory.QUERY,
        typed=False,
        compiled=False,
        comment_styles=["--", "/*"],
        file_patterns=["*.sql"],
    ),
    "graphql": LanguageInfo(
        name="graphql",
        display_name="GraphQL",
        extensions=["graphql", "gql"],
        category=LanguageCategory.QUERY,
        typed=True,
        compiled=False,
        comment_styles=["#"],
        file_patterns=["*.graphql", "*.gql"],
        aliases=["gql"],
    ),
}


# ============================================================================
# Extension Mappings
# ============================================================================


def _build_extension_map() -> dict[str, str]:
    """Build extension-to-language mapping from LANGUAGES registry."""
    mapping: dict[str, str] = {}
    for lang_name, lang_info in LANGUAGES.items():
        for ext in lang_info.extensions:
            # Store without dot
            ext_clean = ext.lstrip(".")
            mapping[ext_clean] = lang_name
            # Also store with dot for convenience
            mapping[f".{ext_clean}"] = lang_name
    return mapping


def _build_alias_map() -> dict[str, str]:
    """Build alias-to-language mapping."""
    mapping: dict[str, str] = {}
    for lang_name, lang_info in LANGUAGES.items():
        for alias in lang_info.aliases:
            mapping[alias.lower()] = lang_name
    return mapping


# Pre-built lookup tables for performance
EXTENSION_TO_LANGUAGE = _build_extension_map()
ALIAS_TO_LANGUAGE = _build_alias_map()


# ============================================================================
# Language Detection Functions
# ============================================================================


def detect_language(file_path: str) -> str:
    """
    Detect programming language from file path.

    Args:
        file_path: Path to file (can be relative or absolute)

    Returns:
        Language name (lowercase) or "unknown"
    """
    ext = Path(file_path).suffix.lower()
    if ext:
        # Try with dot first
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]
        # Try without dot
        ext_no_dot = ext.lstrip(".")
        if ext_no_dot in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext_no_dot]
    return "unknown"


def detect_language_from_extension(extension: str) -> str:
    """
    Detect programming language from file extension.

    Args:
        extension: File extension (with or without dot)

    Returns:
        Language name (lowercase) or "unknown"
    """
    ext = extension.lower().lstrip(".")
    return EXTENSION_TO_LANGUAGE.get(ext, "unknown")


def get_language_info(language: str) -> LanguageInfo | None:
    """
    Get language metadata by name or alias.

    Args:
        language: Language name or alias

    Returns:
        LanguageInfo or None if not found
    """
    lang = language.lower()

    # Direct lookup
    if lang in LANGUAGES:
        return LANGUAGES[lang]

    # Try alias lookup
    if lang in ALIAS_TO_LANGUAGE:
        return LANGUAGES[ALIAS_TO_LANGUAGE[lang]]

    return None


def get_extensions_for_language(language: str) -> list[str]:
    """
    Get file extensions for a language.

    Args:
        language: Language name or alias

    Returns:
        List of extensions (without dots) or empty list
    """
    info = get_language_info(language)
    return info.extensions if info else []


def get_file_patterns_for_language(language: str) -> list[str]:
    """
    Get glob patterns for a language.

    Args:
        language: Language name or alias

    Returns:
        List of glob patterns or empty list
    """
    info = get_language_info(language)
    return info.file_patterns if info else []


def normalize_language_name(language: str) -> str:
    """
    Normalize a language name to its canonical form.

    Args:
        language: Language name or alias

    Returns:
        Canonical language name or original if not found
    """
    lang = language.lower()

    if lang in LANGUAGES:
        return lang

    if lang in ALIAS_TO_LANGUAGE:
        return ALIAS_TO_LANGUAGE[lang]

    return language


# ============================================================================
# Language Queries
# ============================================================================


def get_languages_by_category(category: LanguageCategory) -> list[str]:
    """
    Get all languages in a category.

    Args:
        category: Language category

    Returns:
        List of language names
    """
    return [
        name for name, info in LANGUAGES.items() if info.category == category
    ]


def get_typed_languages() -> list[str]:
    """Get all statically typed languages."""
    return [name for name, info in LANGUAGES.items() if info.typed]


def get_compiled_languages() -> list[str]:
    """Get all compiled languages."""
    return [name for name, info in LANGUAGES.items() if info.compiled]


def get_all_languages() -> list[str]:
    """Get all registered language names."""
    return list(LANGUAGES.keys())


def get_all_extensions() -> list[str]:
    """Get all registered file extensions (without dots)."""
    extensions: set[str] = set()
    for info in LANGUAGES.values():
        extensions.update(info.extensions)
    return sorted(extensions)


def get_code_extensions(*, with_dot: bool = True) -> frozenset[str]:
    """Get extensions for programming languages (not markup/config/query).

    Returns extensions for languages in categories: SYSTEMS, WEB_FRONTEND,
    WEB_BACKEND, GENERAL, FUNCTIONAL, MOBILE, DATA_SCIENCE, SCRIPTING.
    Excludes MARKUP, QUERY, and OTHER.

    Args:
        with_dot: If True (default), return extensions with leading dot.

    Returns:
        Frozen set of extensions, e.g. {".py", ".ts", ...} or {"py", "ts", ...}.
    """
    _NON_CODE_CATEGORIES = {
        LanguageCategory.MARKUP,
        LanguageCategory.QUERY,
        LanguageCategory.OTHER,
    }
    exts: set[str] = set()
    for info in LANGUAGES.values():
        if info.category not in _NON_CODE_CATEGORIES:
            for ext in info.extensions:
                exts.add(f".{ext}" if with_dot else ext)
    return frozenset(exts)


def is_code_file(file_path: str) -> bool:
    """
    Check if a file is a recognized code file.

    Args:
        file_path: Path to check

    Returns:
        True if file has a recognized code extension
    """
    return detect_language(file_path) != "unknown"


def get_comment_styles(language: str) -> list[str]:
    """
    Get comment styles for a language.

    Args:
        language: Language name or alias

    Returns:
        List of comment prefixes (e.g., ["//", "/*"])
    """
    info = get_language_info(language)
    return info.comment_styles if info else []


# ============================================================================
# Watch Pattern Generation
# ============================================================================


def get_watch_patterns_for_languages(languages: list[str]) -> list[str]:
    """
    Generate glob patterns for watching files of specified languages.

    Args:
        languages: List of language names

    Returns:
        Combined list of glob patterns
    """
    patterns: set[str] = set()
    for lang in languages:
        patterns.update(get_file_patterns_for_language(lang))
    return sorted(patterns)


def get_default_watch_patterns() -> list[str]:
    """
    Get default patterns for watching common code files.

    Returns commonly used source code patterns.
    """
    common_languages = [
        "typescript",
        "javascript",
        "python",
        "rust",
        "go",
        "java",
        "cpp",
        "c",
        "csharp",
        "ruby",
        "php",
    ]
    return get_watch_patterns_for_languages(common_languages)


# ============================================================================
# Ignore Pattern Helpers
# ============================================================================


# Common directories to ignore during file scanning
_SCAN_IGNORE_DIRS = frozenset({
    "node_modules",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "vendor",
    "Pods",
    ".gradle",
    ".idea",
    ".vscode",
    ".vs",
    "bin",
    "obj",
})


# Common files to ignore
DEFAULT_IGNORE_FILES = frozenset({
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.o",
    "*.obj",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.min.js",
    "*.min.css",
    "*.map",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
})


def should_ignore_path(path: str) -> bool:
    """
    Check if a path should be ignored during scanning.

    Args:
        path: File or directory path

    Returns:
        True if path should be ignored
    """
    path_obj = Path(path)

    # Check if any parent directory is in ignore list
    for part in path_obj.parts:
        if part in _SCAN_IGNORE_DIRS:
            return True

    # Check filename
    name = path_obj.name
    if name in DEFAULT_IGNORE_FILES:
        return True

    # Check patterns
    for pattern in DEFAULT_IGNORE_FILES:
        if "*" in pattern:
            # Simple glob match
            import fnmatch

            if fnmatch.fnmatch(name, pattern):
                return True

    return False
