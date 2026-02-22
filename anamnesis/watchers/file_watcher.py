"""File watcher implementation using watchdog."""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from anamnesis.constants import utcnow
from anamnesis.utils.language_registry import EXTENSION_TO_LANGUAGE

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


@dataclass
class FileChange:
    """Represents a file change event."""

    type: str  # 'add', 'change', 'unlink', 'addDir', 'unlinkDir'
    path: str
    stats: dict[str, Any] | None = None
    content: str | None = None
    hash: str | None = None
    language: str | None = None
    timestamp: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.stats:
            result["stats"] = self.stats
        if self.content:
            result["content"] = self.content
        if self.hash:
            result["hash"] = self.hash
        if self.language:
            result["language"] = self.language
        return result


@dataclass
class WatcherOptions:
    """Options for the file watcher."""

    patterns: list[str] = field(default_factory=lambda: ["**/*.py"])
    ignored: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/target/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/*.log",
            "**/*.pyc",
        ]
    )
    debounce_ms: int = 500
    include_content: bool = True
    recursive: bool = True


class AnamnesisEventHandler(FileSystemEventHandler):
    """Event handler for file system events."""

    def __init__(
        self,
        watcher: "FileWatcher",
        options: WatcherOptions,
    ):
        """Initialize event handler.

        Args:
            watcher: Parent FileWatcher instance
            options: Watcher options
        """
        super().__init__()
        self.watcher = watcher
        self.options = options

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file/directory creation."""
        if isinstance(event, FileCreatedEvent):
            self.watcher._handle_event("add", event.src_path)
        elif isinstance(event, DirCreatedEvent):
            self.watcher._handle_event("addDir", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if isinstance(event, FileModifiedEvent):
            self.watcher._handle_event("change", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file/directory deletion."""
        if isinstance(event, FileDeletedEvent):
            self.watcher._handle_event("unlink", event.src_path)
        elif isinstance(event, DirDeletedEvent):
            self.watcher._handle_event("unlinkDir", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file/directory move."""
        if hasattr(event, "dest_path"):
            # Treat as delete + add
            self.watcher._handle_event("unlink", event.src_path)
            self.watcher._handle_event("add", event.dest_path)


class FileWatcher:
    """File watcher for detecting code changes."""

    # Language detection — delegates to canonical registry
    LANGUAGE_MAP: dict[str, str] = {
        k: v for k, v in EXTENSION_TO_LANGUAGE.items() if k.startswith(".")
    }

    # Text file extensions
    TEXT_EXTENSIONS: set[str] = {
        ".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go", ".java",
        ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cs", ".php",
        ".rb", ".swift", ".kt", ".scala", ".clj", ".hs", ".ml",
        ".fs", ".elm", ".dart", ".r", ".jl", ".lua", ".pl",
        ".sh", ".ps1", ".sql", ".html", ".css", ".scss", ".sass",
        ".less", ".json", ".xml", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".conf", ".md", ".rst", ".tex", ".txt",
        ".log",
    }

    # Dotfile names that are text files (no extension)
    TEXT_DOTFILES: set[str] = {
        ".gitignore", ".dockerignore", ".editorconfig", ".env",
        ".eslintrc", ".prettierrc", ".babelrc", ".npmrc",
        ".gitattributes", ".mailmap",
    }

    def __init__(
        self,
        path: str,
        patterns: list[str] | None = None,
        ignored: list[str] | None = None,
        debounce_ms: int = 500,
        include_content: bool = True,
        recursive: bool = True,
    ):
        """Initialize file watcher.

        Args:
            path: Directory to watch
            patterns: Glob patterns for files to watch
            ignored: Glob patterns for files to ignore
            debounce_ms: Debounce delay in milliseconds
            include_content: Include file content in changes
            recursive: Watch subdirectories recursively
        """
        self.path = Path(path).resolve()
        self.options = WatcherOptions(
            patterns=patterns or ["**/*.py"],
            ignored=ignored or WatcherOptions().ignored,
            debounce_ms=debounce_ms,
            include_content=include_content,
            recursive=recursive,
        )

        self._observer: Observer | None = None
        self._event_handler: AnamnesisEventHandler | None = None
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._file_hashes: dict[str, str] = {}
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()

        # Callbacks
        self.on_change: Callable[[dict], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.on_ready: Callable[[], None] | None = None

    def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        self._stop_event.clear()
        self._event_handler = AnamnesisEventHandler(self, self.options)
        self._observer = Observer()
        self._observer.schedule(
            self._event_handler,
            str(self.path),
            recursive=self.options.recursive,
        )
        self._observer.start()
        self._running = True

        if self.on_ready:
            self.on_ready()

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            return

        self._stop_event.set()

        # Clear debounce timers
        with self._lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        self._running = False

    def wait(self) -> None:
        """Wait for the watcher to stop (blocking)."""
        try:
            while self._running and not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def is_watching(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def _handle_event(self, event_type: str, file_path: str) -> None:
        """Handle a file system event with debouncing.

        Args:
            event_type: Type of event ('add', 'change', 'unlink', etc.)
            file_path: Path to the file
        """
        # Check if file should be ignored
        if self._should_ignore(file_path):
            return

        # Check if file matches patterns
        if not self._matches_patterns(file_path):
            return

        # Cancel existing timer for this path
        with self._lock:
            if file_path in self._debounce_timers:
                self._debounce_timers[file_path].cancel()

            # Set new timer
            timer = threading.Timer(
                self.options.debounce_ms / 1000,
                self._process_event,
                args=[event_type, file_path],
            )
            self._debounce_timers[file_path] = timer
            timer.start()

    def _process_event(self, event_type: str, file_path: str) -> None:
        """Process a debounced event.

        Args:
            event_type: Type of event
            file_path: Path to the file
        """
        with self._lock:
            self._debounce_timers.pop(file_path, None)

        try:
            change = self._build_change(event_type, file_path)

            # Skip if content hasn't changed (for modify events)
            if event_type == "change":
                if change.hash and not self._has_content_changed(file_path, change.hash):
                    return

            # Update hash
            if change.hash and event_type != "unlink":
                self._file_hashes[file_path] = change.hash
            elif event_type == "unlink":
                self._file_hashes.pop(file_path, None)

            # Invoke callback
            if self.on_change:
                self.on_change(change.to_dict())

        except Exception as e:
            if self.on_error:
                self.on_error(e)

    def _build_change(self, event_type: str, file_path: str) -> FileChange:
        """Build a FileChange object.

        Args:
            event_type: Type of event
            file_path: Path to the file

        Returns:
            FileChange object
        """
        change = FileChange(type=event_type, path=file_path)
        path_obj = Path(file_path)

        # Add stats if file exists
        if path_obj.exists():
            stat = path_obj.stat()
            change.stats = {
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "is_directory": path_obj.is_dir(),
            }

        # Add content and language for files
        if event_type not in ("unlinkDir", "addDir") and path_obj.exists():
            if not path_obj.is_dir():
                change.language = self._detect_language(file_path)

                if self.options.include_content and self._is_text_file(file_path):
                    try:
                        content = path_obj.read_text(encoding="utf-8")
                        change.content = content
                        change.hash = self._calculate_hash(content)
                    except (OSError, UnicodeDecodeError):
                        pass

        return change

    def _should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored.

        Args:
            file_path: Path to check

        Returns:
            True if file should be ignored
        """
        path_obj = Path(file_path)

        for pattern in self.options.ignored:
            # Convert glob pattern to check
            pattern_parts = pattern.replace("**", "").replace("*", "").strip("/")
            if pattern_parts in str(path_obj):
                return True

        return False

    def _matches_patterns(self, file_path: str) -> bool:
        """Check if file matches watch patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file matches patterns
        """
        path_obj = Path(file_path)

        # Always include directories
        if path_obj.is_dir():
            return True

        for pattern in self.options.patterns:
            # Extract extension from pattern
            if "*." in pattern:
                ext = pattern.split("*.")[-1]
                if file_path.endswith(f".{ext}"):
                    return True
            elif path_obj.match(pattern):
                return True

        return False

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language identifier
        """
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(ext, "unknown")

    def _is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file.

        Args:
            file_path: Path to file

        Returns:
            True if text file
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        name = path.name.lower()

        # Check extension first
        if ext in self.TEXT_EXTENSIONS:
            return True

        # Check for known text dotfiles
        if name in self.TEXT_DOTFILES:
            return True

        return False

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content.

        Args:
            content: Content to hash

        Returns:
            Hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def _has_content_changed(self, file_path: str, new_hash: str) -> bool:
        """Check if file content has changed.

        Args:
            file_path: Path to file
            new_hash: New content hash

        Returns:
            True if content changed
        """
        old_hash = self._file_hashes.get(file_path)
        return old_hash != new_hash
