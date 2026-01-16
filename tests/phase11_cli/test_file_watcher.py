"""
Phase 11 Tests: File Watcher

Tests for the file watcher including:
- File change detection
- Pattern matching and filtering
- Debouncing behavior
- Language detection
- Hash-based change detection
"""

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anamnesis.watchers.file_watcher import (
    AnamnesisEventHandler,
    FileChange,
    FileWatcher,
    WatcherOptions,
)


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_file_change_creation(self):
        """FileChange can be created with all fields."""
        change = FileChange(
            type="add",
            path="/test/file.py",
            content="print('hello')",
            hash="abc123",
            language="python",
        )
        assert change.type == "add"
        assert change.path == "/test/file.py"
        assert change.content == "print('hello')"
        assert change.hash == "abc123"
        assert change.language == "python"

    def test_file_change_defaults(self):
        """FileChange has sensible defaults."""
        change = FileChange(type="change", path="/test/file.py")
        assert change.stats is None
        assert change.content is None
        assert change.hash is None
        assert change.language is None
        assert change.timestamp is not None

    def test_file_change_to_dict(self):
        """FileChange can be converted to dict."""
        change = FileChange(
            type="add",
            path="/test/file.py",
            content="print('hello')",
            hash="abc123",
            language="python",
        )
        d = change.to_dict()
        assert d["type"] == "add"
        assert d["path"] == "/test/file.py"
        assert d["content"] == "print('hello')"
        assert d["hash"] == "abc123"
        assert d["language"] == "python"
        assert "timestamp" in d

    def test_file_change_to_dict_minimal(self):
        """to_dict excludes None values."""
        change = FileChange(type="unlink", path="/test/file.py")
        d = change.to_dict()
        assert "content" not in d
        assert "hash" not in d
        assert "language" not in d
        assert "stats" not in d


class TestWatcherOptions:
    """Tests for WatcherOptions dataclass."""

    def test_default_options(self):
        """WatcherOptions has sensible defaults."""
        opts = WatcherOptions()
        assert opts.patterns == ["**/*.py"]
        assert opts.debounce_ms == 500
        assert opts.include_content is True
        assert opts.recursive is True
        assert "**/node_modules/**" in opts.ignored
        assert "**/.git/**" in opts.ignored
        assert "**/__pycache__/**" in opts.ignored

    def test_custom_options(self):
        """WatcherOptions accepts custom values."""
        opts = WatcherOptions(
            patterns=["**/*.ts", "**/*.tsx"],
            debounce_ms=1000,
            include_content=False,
            recursive=False,
        )
        assert opts.patterns == ["**/*.ts", "**/*.tsx"]
        assert opts.debounce_ms == 1000
        assert opts.include_content is False
        assert opts.recursive is False


class TestFileWatcherInit:
    """Tests for FileWatcher initialization."""

    def test_watcher_creation(self):
        """FileWatcher can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher.path == Path(tmpdir).resolve()
            assert not watcher.is_watching()

    def test_watcher_with_custom_patterns(self):
        """FileWatcher accepts custom patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(
                path=tmpdir,
                patterns=["**/*.ts"],
                debounce_ms=100,
            )
            assert watcher.options.patterns == ["**/*.ts"]
            assert watcher.options.debounce_ms == 100

    def test_watcher_path_resolution(self):
        """FileWatcher resolves path to absolute."""
        watcher = FileWatcher(path=".")
        assert watcher.path.is_absolute()


class TestFileWatcherLanguageDetection:
    """Tests for language detection."""

    def test_python_detection(self):
        """Detects Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.py") == "python"

    def test_typescript_detection(self):
        """Detects TypeScript files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.ts") == "typescript"
            assert watcher._detect_language("test.tsx") == "typescript"

    def test_javascript_detection(self):
        """Detects JavaScript files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.js") == "javascript"
            assert watcher._detect_language("test.jsx") == "javascript"

    def test_rust_detection(self):
        """Detects Rust files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.rs") == "rust"

    def test_go_detection(self):
        """Detects Go files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.go") == "go"

    def test_unknown_extension(self):
        """Unknown extensions return 'unknown'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.xyz") == "unknown"

    def test_case_insensitive(self):
        """Language detection is case insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._detect_language("test.PY") == "python"
            assert watcher._detect_language("test.Ts") == "typescript"


class TestFileWatcherTextFileDetection:
    """Tests for text file detection."""

    def test_code_files_are_text(self):
        """Code files are detected as text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._is_text_file("test.py") is True
            assert watcher._is_text_file("test.ts") is True
            assert watcher._is_text_file("test.js") is True
            assert watcher._is_text_file("test.rs") is True
            assert watcher._is_text_file("test.go") is True

    def test_config_files_are_text(self):
        """Config files are detected as text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._is_text_file("config.json") is True
            assert watcher._is_text_file("config.yaml") is True
            assert watcher._is_text_file("config.toml") is True
            assert watcher._is_text_file(".gitignore") is True

    def test_binary_files_not_text(self):
        """Binary files are not detected as text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._is_text_file("image.png") is False
            assert watcher._is_text_file("data.bin") is False
            assert watcher._is_text_file("archive.zip") is False


class TestFileWatcherPatternMatching:
    """Tests for pattern matching."""

    def test_matches_python_pattern(self):
        """Matches Python pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py"])
            assert watcher._matches_patterns("test.py") is True
            assert watcher._matches_patterns("src/test.py") is True

    def test_matches_multiple_patterns(self):
        """Matches multiple patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py", "**/*.ts"])
            assert watcher._matches_patterns("test.py") is True
            assert watcher._matches_patterns("test.ts") is True
            assert watcher._matches_patterns("test.rs") is False

    def test_no_match_wrong_extension(self):
        """Doesn't match wrong extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py"])
            assert watcher._matches_patterns("test.ts") is False


class TestFileWatcherIgnorePatterns:
    """Tests for ignore patterns."""

    def test_ignores_node_modules(self):
        """Ignores node_modules directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._should_ignore("node_modules/test.py") is True
            assert watcher._should_ignore("src/node_modules/test.py") is True

    def test_ignores_git(self):
        """Ignores .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._should_ignore(".git/config") is True
            assert watcher._should_ignore("project/.git/HEAD") is True

    def test_ignores_pycache(self):
        """Ignores __pycache__ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._should_ignore("__pycache__/test.pyc") is True
            assert watcher._should_ignore("src/__pycache__/module.pyc") is True

    def test_does_not_ignore_normal_files(self):
        """Doesn't ignore normal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._should_ignore("src/main.py") is False
            assert watcher._should_ignore("test.ts") is False


class TestFileWatcherHashCalculation:
    """Tests for hash calculation."""

    def test_hash_calculation(self):
        """Hash is calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            content = "print('hello')"
            expected = hashlib.sha256(content.encode()).hexdigest()
            assert watcher._calculate_hash(content) == expected

    def test_hash_different_for_different_content(self):
        """Different content produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            hash1 = watcher._calculate_hash("content1")
            hash2 = watcher._calculate_hash("content2")
            assert hash1 != hash2

    def test_has_content_changed_true(self):
        """Detects content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            watcher._file_hashes["test.py"] = "old_hash"
            assert watcher._has_content_changed("test.py", "new_hash") is True

    def test_has_content_changed_false(self):
        """Detects unchanged content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            watcher._file_hashes["test.py"] = "same_hash"
            assert watcher._has_content_changed("test.py", "same_hash") is False

    def test_has_content_changed_new_file(self):
        """New files are always changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert watcher._has_content_changed("new_file.py", "some_hash") is True


class TestFileWatcherStartStop:
    """Tests for start/stop functionality."""

    def test_start_stop(self):
        """Watcher can be started and stopped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            assert not watcher.is_watching()

            watcher.start()
            assert watcher.is_watching()

            watcher.stop()
            assert not watcher.is_watching()

    def test_double_start(self):
        """Double start is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            watcher.start()
            watcher.start()  # Should not raise
            assert watcher.is_watching()
            watcher.stop()

    def test_double_stop(self):
        """Double stop is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            watcher.start()
            watcher.stop()
            watcher.stop()  # Should not raise
            assert not watcher.is_watching()

    def test_on_ready_callback(self):
        """on_ready callback is called when started."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            ready_called = False

            def on_ready():
                nonlocal ready_called
                ready_called = True

            watcher.on_ready = on_ready
            watcher.start()

            assert ready_called
            watcher.stop()


class TestFileWatcherBuildChange:
    """Tests for building FileChange objects."""

    def test_build_change_for_add(self):
        """Builds change for file addition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("print('hello')")

            watcher = FileWatcher(path=tmpdir)
            change = watcher._build_change("add", str(test_file))

            assert change.type == "add"
            assert change.path == str(test_file)
            assert change.language == "python"
            assert change.content == "print('hello')"
            assert change.hash is not None
            assert change.stats is not None
            assert change.stats["size"] > 0

    def test_build_change_for_unlink(self):
        """Builds change for file deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir)
            # File doesn't exist anymore
            change = watcher._build_change("unlink", "/tmp/nonexistent.py")

            assert change.type == "unlink"
            assert change.stats is None  # File doesn't exist


class TestAnamnesisEventHandler:
    """Tests for event handler."""

    def test_event_handler_on_created(self):
        """Handler processes file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py"])
            handler = AnamnesisEventHandler(watcher, watcher.options)

            # Mock the _handle_event method
            watcher._handle_event = MagicMock()

            # Simulate file created event
            from watchdog.events import FileCreatedEvent

            event = FileCreatedEvent(str(Path(tmpdir) / "test.py"))
            handler.on_created(event)

            watcher._handle_event.assert_called_once()
            call_args = watcher._handle_event.call_args[0]
            assert call_args[0] == "add"

    def test_event_handler_on_modified(self):
        """Handler processes file modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py"])
            handler = AnamnesisEventHandler(watcher, watcher.options)

            watcher._handle_event = MagicMock()

            from watchdog.events import FileModifiedEvent

            event = FileModifiedEvent(str(Path(tmpdir) / "test.py"))
            handler.on_modified(event)

            watcher._handle_event.assert_called_once()
            call_args = watcher._handle_event.call_args[0]
            assert call_args[0] == "change"

    def test_event_handler_on_deleted(self):
        """Handler processes file deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(path=tmpdir, patterns=["**/*.py"])
            handler = AnamnesisEventHandler(watcher, watcher.options)

            watcher._handle_event = MagicMock()

            from watchdog.events import FileDeletedEvent

            event = FileDeletedEvent(str(Path(tmpdir) / "test.py"))
            handler.on_deleted(event)

            watcher._handle_event.assert_called_once()
            call_args = watcher._handle_event.call_args[0]
            assert call_args[0] == "unlink"
