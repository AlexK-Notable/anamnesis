"""
Phase 1 Tests: Language Registry

Tests for language detection and metadata registry including:
- Language detection from file paths
- Extension to language mapping
- Language metadata and categories
- Watch patterns and ignore patterns
"""

import pytest

from anamnesis.constants import DEFAULT_IGNORE_DIRS
from anamnesis.utils import (
    DEFAULT_IGNORE_FILES,
    EXTENSION_TO_LANGUAGE,
    LanguageCategory,
    detect_language,
    detect_language_from_extension,
    get_all_extensions,
    get_all_languages,
    get_comment_styles,
    get_compiled_languages,
    get_default_watch_patterns,
    get_extensions_for_language,
    get_language_info,
    get_languages_by_category,
    get_typed_languages,
    get_watch_patterns_for_languages,
    is_code_file,
    normalize_language_name,
    should_ignore_path,
)


class TestLanguageDetection:
    """Tests for language detection from file paths."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("file.ts", "typescript"),
            ("file.tsx", "typescript"),  # tsx is part of typescript
            ("file.js", "javascript"),
            ("file.jsx", "javascript"),  # jsx is part of javascript
            ("file.py", "python"),
            ("file.rs", "rust"),
            ("file.go", "go"),
            ("file.java", "java"),
            ("file.c", "c"),
            ("file.cpp", "cpp"),
            ("file.h", "c"),
            ("file.hpp", "cpp"),
            ("file.cs", "csharp"),
            ("file.rb", "ruby"),
            ("file.php", "php"),
            ("file.swift", "swift"),
            ("file.kt", "kotlin"),
            ("file.scala", "scala"),
            ("file.sql", "sql"),
            ("file.html", "html"),
            ("file.css", "css"),
            ("file.scss", "css"),  # scss is part of css extensions
            ("file.json", "json"),
            ("file.yaml", "yaml"),
            ("file.yml", "yaml"),
            ("file.xml", "xml"),
            ("file.md", "markdown"),
            ("file.sh", "shell"),
            ("file.bash", "shell"),
            ("file.zsh", "shell"),
        ],
    )
    def test_detect_common_languages(self, path: str, expected: str):
        """Detects common programming languages."""
        assert detect_language(path) == expected

    def test_detect_from_full_path(self):
        """Detects language from full file path."""
        assert detect_language("/home/user/project/src/main.py") == "python"
        assert detect_language("C:\\Users\\project\\src\\app.ts") == "typescript"

    def test_detect_unknown_extension(self):
        """Returns 'unknown' for unrecognized extensions."""
        assert detect_language("file.xyz") == "unknown"
        assert detect_language("file.unknown") == "unknown"

    def test_detect_no_extension(self):
        """Returns 'unknown' for files without extension."""
        assert detect_language("Makefile") == "unknown"
        assert detect_language("README") == "unknown"

    def test_detect_case_insensitive(self):
        """Extension detection is case-insensitive."""
        assert detect_language("file.PY") == "python"
        assert detect_language("file.TS") == "typescript"
        assert detect_language("file.Js") == "javascript"


class TestExtensionMapping:
    """Tests for extension to language mapping."""

    def test_extension_to_language_coverage(self):
        """Extension map has reasonable coverage."""
        # Should have at least 50 extensions mapped
        assert len(EXTENSION_TO_LANGUAGE) >= 50

    def test_common_extensions_present(self):
        """Common extensions are mapped."""
        required = [".py", ".ts", ".js", ".rs", ".go", ".java", ".c", ".cpp"]
        for ext in required:
            assert ext in EXTENSION_TO_LANGUAGE

    def test_detect_from_extension_with_dot(self):
        """Detects language from extension with dot."""
        assert detect_language_from_extension(".py") == "python"
        assert detect_language_from_extension(".ts") == "typescript"

    def test_detect_from_extension_without_dot(self):
        """Detects language from extension without dot."""
        assert detect_language_from_extension("py") == "python"
        assert detect_language_from_extension("ts") == "typescript"


class TestLanguageInfo:
    """Tests for LanguageInfo metadata."""

    def test_language_info_structure(self):
        """LanguageInfo has required fields."""
        info = get_language_info("python")
        assert info is not None
        assert info.name == "python"
        assert info.display_name == "Python"
        assert ".py" in [f".{ext}" for ext in info.extensions]
        assert info.category == LanguageCategory.GENERAL

    def test_get_language_info_unknown(self):
        """Returns None for unknown languages."""
        assert get_language_info("nonexistent") is None

    def test_language_categories(self):
        """Languages have appropriate categories."""
        # General purpose
        assert get_language_info("python").category == LanguageCategory.GENERAL
        assert get_language_info("java").category == LanguageCategory.GENERAL

        # Markup (HTML, CSS, JSON, YAML are all in MARKUP category)
        assert get_language_info("html").category == LanguageCategory.MARKUP
        assert get_language_info("css").category == LanguageCategory.MARKUP
        assert get_language_info("json").category == LanguageCategory.MARKUP
        assert get_language_info("yaml").category == LanguageCategory.MARKUP

        # Systems
        assert get_language_info("c").category == LanguageCategory.SYSTEMS

    def test_typed_languages(self):
        """Typed languages are correctly marked."""
        typed = get_typed_languages()  # Returns list[str]

        assert "typescript" in typed
        assert "java" in typed
        assert "rust" in typed
        assert "go" in typed

        # Python and JavaScript are dynamically typed
        assert "python" not in typed
        assert "javascript" not in typed

    def test_compiled_languages(self):
        """Compiled languages are correctly marked."""
        compiled = get_compiled_languages()  # Returns list[str]

        assert "rust" in compiled
        assert "go" in compiled
        assert "c" in compiled
        assert "java" in compiled

        # Interpreted languages
        assert "python" not in compiled
        assert "javascript" not in compiled


class TestCommentStyles:
    """Tests for comment style metadata."""

    def test_get_comment_styles_c_style(self):
        """Gets C-style comments."""
        styles = get_comment_styles("javascript")
        assert "//" in styles
        assert "/* */" in styles or "/*" in styles

    def test_get_comment_styles_python(self):
        """Gets Python comments."""
        styles = get_comment_styles("python")
        assert "#" in styles

    def test_get_comment_styles_unknown(self):
        """Returns empty for unknown languages."""
        styles = get_comment_styles("nonexistent")
        assert styles == []


class TestFilePatterns:
    """Tests for file patterns and extensions."""

    def test_get_extensions_for_language(self):
        """Gets extensions for a language."""
        exts = get_extensions_for_language("python")
        assert "py" in exts
        assert "pyw" in exts

    def test_get_extensions_unknown_language(self):
        """Returns empty for unknown language."""
        exts = get_extensions_for_language("nonexistent")
        assert exts == []

    def test_get_all_extensions(self):
        """Gets all registered extensions."""
        all_exts = get_all_extensions()
        assert len(all_exts) >= 50
        assert "py" in all_exts
        assert "ts" in all_exts
        assert "js" in all_exts

    def test_get_all_languages(self):
        """Gets all registered languages."""
        all_langs = get_all_languages()
        assert len(all_langs) >= 25
        required = ["python", "typescript", "javascript", "rust", "go"]
        for lang in required:
            assert lang in all_langs


class TestWatchPatterns:
    """Tests for file watch patterns."""

    def test_get_default_watch_patterns(self):
        """Gets default watch patterns."""
        patterns = get_default_watch_patterns()
        assert len(patterns) > 0
        # Should include common extensions
        assert any("*.py" in p for p in patterns)
        assert any("*.ts" in p for p in patterns)

    def test_get_watch_patterns_for_languages(self):
        """Gets watch patterns for specific languages."""
        patterns = get_watch_patterns_for_languages(["python", "typescript"])
        assert any("py" in p for p in patterns)
        assert any("ts" in p for p in patterns)


class TestIgnorePatterns:
    """Tests for ignore patterns."""

    def test_default_ignore_dirs(self):
        """Default ignore dirs include common patterns."""
        assert "node_modules" in DEFAULT_IGNORE_DIRS
        assert ".git" in DEFAULT_IGNORE_DIRS
        assert "__pycache__" in DEFAULT_IGNORE_DIRS
        assert "venv" in DEFAULT_IGNORE_DIRS

    def test_default_ignore_files(self):
        """Default ignore files include common patterns."""
        assert ".DS_Store" in DEFAULT_IGNORE_FILES

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("node_modules/package/index.js", True),
            (".git/config", True),
            ("__pycache__/module.pyc", True),
            ("venv/lib/python3.9/site.py", True),
            ("src/main.py", False),
            ("lib/utils.ts", False),
        ],
    )
    def test_should_ignore_path(self, path: str, expected: bool):
        """Correctly identifies paths to ignore."""
        assert should_ignore_path(path) == expected


class TestCodeFileDetection:
    """Tests for code file detection."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("main.py", True),
            ("app.ts", True),
            ("index.js", True),
            ("lib.rs", True),
            ("main.go", True),
            ("README.md", True),  # Markdown is considered code
            ("config.json", True),  # JSON is considered code
            ("image.png", False),
            ("document.pdf", False),
            ("archive.zip", False),
        ],
    )
    def test_is_code_file(self, path: str, expected: bool):
        """Identifies code files correctly."""
        assert is_code_file(path) == expected


class TestLanguagesByCategory:
    """Tests for filtering languages by category."""

    def test_get_general_purpose_languages(self):
        """Gets general purpose languages."""
        langs = get_languages_by_category(LanguageCategory.GENERAL)  # Returns list[str]
        assert "python" in langs
        assert "java" in langs

    def test_get_markup_languages(self):
        """Gets markup/config languages."""
        langs = get_languages_by_category(LanguageCategory.MARKUP)  # Returns list[str]
        assert "html" in langs
        assert "css" in langs
        assert "json" in langs
        assert "yaml" in langs

    def test_get_systems_languages(self):
        """Gets systems languages."""
        langs = get_languages_by_category(LanguageCategory.SYSTEMS)  # Returns list[str]
        assert "c" in langs
        assert "rust" in langs


class TestLanguageNameNormalization:
    """Tests for language name normalization."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("Python", "python"),
            ("TYPESCRIPT", "typescript"),
            ("JavaScript", "javascript"),
            ("c++", "cpp"),
            ("C++", "cpp"),
            ("c#", "csharp"),
            ("C#", "csharp"),
        ],
    )
    def test_normalize_language_name(self, input_name: str, expected: str):
        """Normalizes language names correctly."""
        assert normalize_language_name(input_name) == expected
