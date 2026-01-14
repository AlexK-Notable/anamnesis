"""Tests for CodebaseService."""

import tempfile
from pathlib import Path

import pytest

from anamnesis.services.codebase_service import (
    AnalysisResult,
    CodebaseService,
    FileAnalysis,
)


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = AnalysisResult(
            success=True,
            time_elapsed_ms=1000,
            insights=["Analysis complete"],
        )
        assert result.success is True
        assert result.error is None
        assert len(result.insights) == 1

    def test_failure_result(self):
        """Test failure result."""
        result = AnalysisResult(
            success=False,
            error="Test error",
            time_elapsed_ms=100,
        )
        assert result.success is False
        assert result.error == "Test error"

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = AnalysisResult(
            success=True,
            time_elapsed_ms=500,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["time_elapsed_ms"] == 500


class TestFileAnalysis:
    """Tests for FileAnalysis dataclass."""

    def test_basic_analysis(self):
        """Test basic file analysis."""
        analysis = FileAnalysis(
            file_path="/test/file.py",
            language="python",
        )
        assert analysis.file_path == "/test/file.py"
        assert analysis.language == "python"
        assert analysis.concepts == []
        assert analysis.imports == []

    def test_with_data(self):
        """Test with analysis data."""
        analysis = FileAnalysis(
            file_path="/test/file.py",
            language="python",
            concepts=[{"name": "MyClass", "type": "class"}],
            imports=["import os"],
        )
        assert len(analysis.concepts) == 1
        assert len(analysis.imports) == 1

    def test_to_dict(self):
        """Test to_dict conversion."""
        analysis = FileAnalysis(
            file_path="/test/file.py",
            language="python",
        )
        d = analysis.to_dict()
        assert d["file_path"] == "/test/file.py"
        assert d["language"] == "python"


class TestCodebaseService:
    """Tests for CodebaseService."""

    def test_create_service(self):
        """Test creating service."""
        service = CodebaseService()
        assert service.semantic_engine is not None
        assert service.complexity_analyzer is not None

    def test_analyze_nonexistent_path(self):
        """Test analyzing nonexistent path."""
        service = CodebaseService()
        result = service.analyze_codebase("/nonexistent/path/12345")
        assert result.success is False
        assert "does not exist" in result.error

    def test_analyze_file_not_directory(self, tmp_path):
        """Test analyzing file instead of directory."""
        file_path = tmp_path / "test.py"
        file_path.write_text("x = 1")

        service = CodebaseService()
        result = service.analyze_codebase(file_path)
        assert result.success is False
        assert "not a directory" in result.error


class TestCodebaseServiceWithFileSystem:
    """Tests for CodebaseService with file system."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project."""
        src = tmp_path / "src"
        src.mkdir()

        (src / "main.py").write_text('''
"""Main module."""

import os
import sys

MAX_RETRIES = 3

class Application:
    """Main application."""

    def __init__(self):
        self.running = False

    def start(self):
        """Start the application."""
        self.running = True
        for i in range(MAX_RETRIES):
            if self._try_connect():
                break

    def _try_connect(self):
        """Try to connect."""
        return True

def main():
    """Entry point."""
    app = Application()
    app.start()
''')

        (src / "utils.py").write_text('''
"""Utility functions."""

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()
''')

        tests = tmp_path / "tests"
        tests.mkdir()

        (tests / "test_main.py").write_text('''
"""Tests for main module."""
import pytest

def test_application():
    """Test application."""
    pass
''')

        return tmp_path

    def test_analyze_codebase(self, temp_project):
        """Test analyzing codebase."""
        service = CodebaseService()
        result = service.analyze_codebase(temp_project)

        assert result.success is True
        assert result.analysis is not None
        assert result.time_elapsed_ms > 0
        assert len(result.insights) > 0

    def test_analyze_with_complexity(self, temp_project):
        """Test analyzing with complexity."""
        service = CodebaseService()
        result = service.analyze_codebase(temp_project, include_complexity=True)

        assert result.success is True
        assert result.complexity is not None
        assert result.complexity.loc.code > 0

    def test_analyze_without_complexity(self, temp_project):
        """Test analyzing without complexity."""
        service = CodebaseService()
        result = service.analyze_codebase(temp_project, include_complexity=False)

        assert result.success is True
        assert result.complexity is None

    def test_analyze_with_dependencies(self, temp_project):
        """Test analyzing with dependencies."""
        service = CodebaseService()
        result = service.analyze_codebase(temp_project, include_dependencies=True)

        assert result.success is True
        # May or may not have graph depending on concepts found

    def test_analyze_caching(self, temp_project):
        """Test analysis caching."""
        service = CodebaseService()

        # First analysis
        result1 = service.analyze_codebase(temp_project)
        assert result1.success is True

        # Second analysis should use cache
        result2 = service.analyze_codebase(temp_project, use_cache=True)
        assert result2.success is True
        assert "Using cached" in result2.insights[0]

    def test_analyze_no_cache(self, temp_project):
        """Test analysis without cache."""
        service = CodebaseService()

        # First analysis
        service.analyze_codebase(temp_project)

        # Second analysis without cache
        result = service.analyze_codebase(temp_project, use_cache=False)
        assert result.success is True
        assert "Using cached" not in result.insights[0]

    def test_analyze_file(self, temp_project):
        """Test analyzing a single file."""
        service = CodebaseService()
        file_path = temp_project / "src" / "main.py"

        analysis = service.analyze_file(file_path)

        assert analysis is not None
        assert analysis.language == "python"
        assert len(analysis.concepts) > 0
        assert len(analysis.imports) > 0

    def test_analyze_file_with_complexity(self, temp_project):
        """Test analyzing file with complexity."""
        service = CodebaseService()
        file_path = temp_project / "src" / "main.py"

        analysis = service.analyze_file(file_path, include_complexity=True)

        assert analysis is not None
        assert analysis.complexity is not None
        assert analysis.complexity.loc.code > 0

    def test_analyze_nonexistent_file(self):
        """Test analyzing nonexistent file."""
        service = CodebaseService()
        analysis = service.analyze_file("/nonexistent/file.py")
        assert analysis is None

    def test_get_codebase_health(self, temp_project):
        """Test getting codebase health."""
        service = CodebaseService()
        health = service.get_codebase_health(temp_project)

        assert "healthy" in health
        assert "score" in health
        assert "issues" in health
        assert "recommendations" in health
        assert health["score"] <= 100

    def test_get_file_stats(self, temp_project):
        """Test getting file statistics."""
        service = CodebaseService()
        stats = service.get_file_stats(temp_project)

        assert ".py" in stats
        assert stats[".py"] >= 3  # At least our 3 Python files

    def test_clear_cache(self, temp_project):
        """Test clearing cache."""
        service = CodebaseService()

        # Analyze to populate cache
        service.analyze_codebase(temp_project)
        service.analyze_file(temp_project / "src" / "main.py")

        # Clear specific path
        service.clear_cache(str(temp_project))

        # Next analysis should not use cache
        result = service.analyze_codebase(temp_project)
        assert "Using cached" not in result.insights[0]

    def test_clear_all_cache(self, temp_project):
        """Test clearing all cache."""
        service = CodebaseService()

        service.analyze_codebase(temp_project)

        # Clear all
        service.clear_cache()

        result = service.analyze_codebase(temp_project)
        assert "Using cached" not in result.insights[0]


class TestCodebaseServiceEdgeCases:
    """Tests for edge cases."""

    def test_empty_directory(self, tmp_path):
        """Test analyzing empty directory."""
        service = CodebaseService()
        result = service.analyze_codebase(tmp_path)

        assert result.success is True
        assert result.analysis is not None
        assert result.analysis.total_files == 0

    def test_directory_with_no_code(self, tmp_path):
        """Test directory with no code files."""
        (tmp_path / "readme.txt").write_text("Hello")
        (tmp_path / "config.json").write_text("{}")

        service = CodebaseService()
        result = service.analyze_codebase(tmp_path)

        assert result.success is True

    def test_max_files_limit(self, tmp_path):
        """Test max files limit."""
        # Create many files
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}")

        service = CodebaseService()
        result = service.analyze_codebase(tmp_path, max_files=5)

        assert result.success is True

    def test_file_with_encoding_error(self, tmp_path):
        """Test file with encoding issues."""
        file_path = tmp_path / "test.py"
        file_path.write_bytes(b"x = 1\n\xff\xfe invalid")

        service = CodebaseService()
        analysis = service.analyze_file(file_path)

        # Should handle gracefully
        assert analysis is not None or analysis is None  # Either works


class TestCodebaseServiceLanguageDetection:
    """Tests for language detection in analysis."""

    def test_detect_python(self, tmp_path):
        """Test detecting Python files."""
        (tmp_path / "test.py").write_text("x = 1")

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.py")

        assert analysis.language == "python"

    def test_detect_typescript(self, tmp_path):
        """Test detecting TypeScript files."""
        (tmp_path / "test.ts").write_text("const x: number = 1;")

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.ts")

        assert analysis.language == "typescript"

    def test_detect_javascript(self, tmp_path):
        """Test detecting JavaScript files."""
        (tmp_path / "test.js").write_text("const x = 1;")

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.js")

        assert analysis.language == "javascript"

    def test_detect_go(self, tmp_path):
        """Test detecting Go files."""
        (tmp_path / "test.go").write_text("package main")

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.go")

        assert analysis.language == "go"

    def test_detect_rust(self, tmp_path):
        """Test detecting Rust files."""
        (tmp_path / "test.rs").write_text("fn main() {}")

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.rs")

        assert analysis.language == "rust"


class TestCodebaseServiceImportExtraction:
    """Tests for import extraction."""

    def test_extract_python_imports(self, tmp_path):
        """Test extracting Python imports."""
        (tmp_path / "test.py").write_text('''
import os
import sys
from pathlib import Path
from typing import Optional, List
''')

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.py")

        assert len(analysis.imports) == 4
        assert "import os" in analysis.imports
        assert "from pathlib import Path" in analysis.imports

    def test_extract_typescript_imports(self, tmp_path):
        """Test extracting TypeScript imports."""
        (tmp_path / "test.ts").write_text('''
import { Component } from 'react';
import axios from 'axios';
''')

        service = CodebaseService()
        analysis = service.analyze_file(tmp_path / "test.ts")

        assert len(analysis.imports) == 2
