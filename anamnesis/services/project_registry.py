"""Project registry for multi-project service isolation.

Manages per-project service contexts, ensuring that learning data,
sessions, memories, and search indices don't leak between projects.

Before this refactoring, services were global singletons that accumulated
state across project switches — a latent cross-contamination bug.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from anamnesis.constants import utcnow
from anamnesis.services.memory_service import MemoryService
from anamnesis.utils.logger import logger

_DEFAULT_PERSIST_PATH: Path = Path.home() / ".anamnesis" / "projects.json"

if TYPE_CHECKING:
    from anamnesis.lsp.manager import LspManager
    from anamnesis.services.codebase_service import CodebaseService
    from anamnesis.services.intelligence_service import IntelligenceService
    from anamnesis.services.learning_service import LearningService
    from anamnesis.services.session_manager import SessionManager
    from anamnesis.services.symbol_service import SymbolService
    from anamnesis.search.service import SearchService


@dataclass
class ProjectContext:
    """All services and state scoped to a single project.

    Each project gets its own service instances, preventing
    cross-project data contamination. Services are lazily
    initialized on first access.
    """

    path: str
    activated_at: Optional[datetime] = None
    _learning_service: Optional["LearningService"] = field(default=None, repr=False)
    _intelligence_service: Optional["IntelligenceService"] = field(
        default=None, repr=False
    )
    _codebase_service: Optional["CodebaseService"] = field(default=None, repr=False)
    _session_manager: Optional["SessionManager"] = field(default=None, repr=False)
    _memory_service: Optional["MemoryService"] = field(default=None, repr=False)
    _search_service: Optional["SearchService"] = field(default=None, repr=False)
    _semantic_initialized: bool = field(default=False, repr=False)
    _lsp_manager: Optional["LspManager"] = field(default=None, repr=False)
    _symbol_service: Optional["SymbolService"] = field(default=None, repr=False)
    _init_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        """Short project name (directory basename)."""
        return Path(self.path).name

    def get_learning_service(self) -> "LearningService":
        """Get or create the learning service for this project."""
        if self._learning_service is not None:
            return self._learning_service
        with self._init_lock:
            if self._learning_service is None:
                from anamnesis.services.learning_service import LearningService

                self._learning_service = LearningService()
            return self._learning_service

    def get_intelligence_service(self) -> "IntelligenceService":
        """Get or create the intelligence service for this project."""
        if self._intelligence_service is not None:
            return self._intelligence_service
        with self._init_lock:
            if self._intelligence_service is None:
                from anamnesis.services.intelligence_service import IntelligenceService

                self._intelligence_service = IntelligenceService()
            return self._intelligence_service

    def get_codebase_service(self) -> "CodebaseService":
        """Get or create the codebase service for this project."""
        if self._codebase_service is not None:
            return self._codebase_service
        with self._init_lock:
            if self._codebase_service is None:
                from anamnesis.services.codebase_service import CodebaseService

                self._codebase_service = CodebaseService()
            return self._codebase_service

    def get_session_manager(self) -> "SessionManager":
        """Get or create the session manager for this project.

        Each project gets its own in-memory SQLite backend,
        isolating sessions per project.
        """
        if self._session_manager is not None:
            return self._session_manager
        with self._init_lock:
            if self._session_manager is None:
                from anamnesis.services.session_manager import SessionManager
                from anamnesis.storage.sync_backend import SyncSQLiteBackend

                backend = SyncSQLiteBackend(":memory:")
                backend.connect()
                self._session_manager = SessionManager(backend)
            return self._session_manager

    def get_memory_service(self) -> MemoryService:
        """Get or create the memory service for this project."""
        if self._memory_service is not None:
            return self._memory_service
        with self._init_lock:
            if self._memory_service is None:
                self._memory_service = MemoryService(self.path)
            return self._memory_service

    def get_search_service(self) -> "SearchService":
        """Get or create the search service for this project.

        Creates a synchronous search service (text + pattern backends).
        For semantic search, use ensure_semantic_search().
        """
        if self._search_service is not None:
            return self._search_service
        with self._init_lock:
            if self._search_service is None:
                from anamnesis.search.service import SearchService

                self._search_service = SearchService.create_sync(self.path)
            return self._search_service

    def get_lsp_manager(self) -> "LspManager":
        """Get or create the LSP manager for this project.

        Lazily creates an LspManager that handles language server
        lifecycle for this project.
        """
        if self._lsp_manager is not None:
            return self._lsp_manager
        with self._init_lock:
            if self._lsp_manager is None:
                from anamnesis.lsp.manager import LspManager

                self._lsp_manager = LspManager(self.path)
            return self._lsp_manager

    def get_symbol_service(self) -> "SymbolService":
        """Get or create the symbol service for this project.

        Provides a high-level facade over SymbolRetriever (navigation)
        and CodeEditor (mutations), both backed by the project's LSP
        manager.
        """
        if self._symbol_service is not None:
            return self._symbol_service
        with self._init_lock:
            if self._symbol_service is None:
                from anamnesis.services.symbol_service import SymbolService

                self._symbol_service = SymbolService(
                    self.path,
                    lsp_manager=self.get_lsp_manager(),
                    intelligence_service=self._intelligence_service,
                )
            return self._symbol_service

    def shutdown_lsp(self) -> None:
        """Shut down all LSP servers for this project.

        Called during project deactivation to prevent zombie processes.
        """
        if self._lsp_manager is not None:
            self._lsp_manager.stop_all()
            self._lsp_manager = None
        self._symbol_service = None

    def cleanup(self) -> None:
        """Release all resources held by this project context.

        Shuts down LSP servers, closes session backends, and clears
        service references. Called during server shutdown or project
        deactivation. Each step is wrapped individually so a failure
        in one does not prevent cleanup of others.
        """
        # LSP servers (sends shutdown request, kills processes)
        try:
            self.shutdown_lsp()
        except Exception:
            logger.warning("Error shutting down LSP for %s", self.path, exc_info=True)

        # Session backend (stops background thread, closes DB)
        try:
            if self._session_manager is not None:
                self._session_manager._backend.close()
        except Exception:
            logger.warning(
                "Error closing session backend for %s", self.path, exc_info=True
            )

        # Clear all service references
        self._learning_service = None
        self._intelligence_service = None
        self._codebase_service = None
        self._session_manager = None
        self._memory_service = None
        self._search_service = None
        self._semantic_initialized = False

    async def ensure_semantic_search(self) -> bool:
        """Initialize semantic search backend lazily.

        Called on first semantic search request. Creates the full async
        SearchService with semantic backend (embeddings + Qdrant).

        Returns:
            True if semantic search is now available.
        """
        from anamnesis.search.service import SearchService

        if self._semantic_initialized:
            return (
                self._search_service is not None
                and self._search_service.is_semantic_available()
            )

        try:
            self._search_service = await SearchService.create(
                self.path,
                enable_semantic=True,
            )
            self._semantic_initialized = True
            logger.info(
                f"Semantic search initialized for project '{self.name}'"
            )
            return self._search_service.is_semantic_available()
        except Exception as e:
            logger.warning(
                f"Semantic search init failed for '{self.name}': {e}"
            )
            if self._search_service is None:
                self._search_service = SearchService.create_sync(self.path)
            self._semantic_initialized = True
            return False

    @property
    def semantic_initialized(self) -> bool:
        """Whether semantic search initialization has been attempted."""
        return self._semantic_initialized

    def reset_search(self) -> None:
        """Reset search service (e.g., after re-learning)."""
        self._search_service = None
        self._semantic_initialized = False

    def to_dict(self) -> dict:
        """Serialize project context to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "activated_at": (
                self.activated_at.isoformat() if self.activated_at else None
            ),
            "services": {
                "learning": self._learning_service is not None,
                "intelligence": self._intelligence_service is not None,
                "codebase": self._codebase_service is not None,
                "session_manager": self._session_manager is not None,
                "memory": self._memory_service is not None,
                "search": self._search_service is not None,
                "semantic_search": self._semantic_initialized,
                "lsp": self._lsp_manager is not None,
                "symbol_service": self._symbol_service is not None,
            },
        }


class ProjectRegistry:
    """Registry managing multiple project contexts.

    Provides project activation, switching, and lookup. The active
    project determines which services are used for MCP tool calls.

    Single-project backward compatibility: if no project is explicitly
    activated, the first access auto-activates the current working
    directory.

    Usage:
        registry = ProjectRegistry()

        # Explicit activation
        ctx = registry.activate("/path/to/project")
        service = ctx.get_learning_service()

        # Or via convenience (auto-activates cwd if needed)
        ctx = registry.get_active()
    """

    def __init__(
        self,
        persist_path: Optional[Union[str, Path]] = _DEFAULT_PERSIST_PATH,
    ) -> None:
        self._projects: dict[str, ProjectContext] = {}
        self._active_path: Optional[str] = None
        self._lock = threading.Lock()
        self._persist_path: Optional[Path] = (
            Path(persist_path) if persist_path is not None else None
        )
        self._load()

    @property
    def active_path(self) -> Optional[str]:
        """Get the active project path."""
        return self._active_path

    @property
    def active_project(self) -> Optional[ProjectContext]:
        """Get the active project context, or None if none activated."""
        if self._active_path is not None:
            return self._projects.get(self._active_path)
        return None

    # -----------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------

    def _load(self) -> None:
        """Load known projects from the persist file.

        Pre-registers projects with ``activated_at=None`` (known but not
        active).  Silently skips directories that no longer exist and
        gracefully handles missing files, corrupt JSON, and permission
        errors.
        """
        if self._persist_path is None:
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load project registry from %s: %s", self._persist_path, exc)
            return

        projects = data.get("projects", {})
        for project_path, _meta in projects.items():
            resolved = str(Path(project_path).resolve())
            if not Path(resolved).is_dir():
                logger.debug("Skipping persisted project (directory missing): %s", resolved)
                continue
            if resolved not in self._projects:
                self._projects[resolved] = ProjectContext(path=resolved)
                logger.info("Restored known project: %s", resolved)

    def _save(self) -> None:
        """Persist known projects to disk using atomic write.

        Creates the ``~/.anamnesis/`` directory on first save.
        Writes to a temporary file then renames for crash safety.
        Gracefully handles write/permission errors.
        """
        if self._persist_path is None:
            return
        payload: dict[str, object] = {
            "version": 1,
            "projects": {
                p.path: {
                    "last_activated_at": (
                        p.activated_at.isoformat() if p.activated_at else None
                    ),
                }
                for p in self._projects.values()
            },
        }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".tmp",
                dir=self._persist_path.parent,
                delete=False,
                encoding="utf-8",
            )
            try:
                json.dump(payload, fd, indent=2)
                fd.flush()
                os.fsync(fd.fileno())
                fd.close()
                os.rename(fd.name, self._persist_path)
            except BaseException:
                fd.close()
                try:
                    os.unlink(fd.name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("Failed to save project registry to %s: %s", self._persist_path, exc)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def activate(self, path: str) -> ProjectContext:
        """Activate a project by path.

        Creates a new ProjectContext if this is the first activation.
        If already known, returns the existing context (preserving
        all cached service state). Thread-safe.

        Args:
            path: Project root directory path.

        Returns:
            The activated ProjectContext.

        Raises:
            ValueError: If path does not exist or is not a directory.
        """
        resolved = str(Path(path).resolve())

        resolved_path = Path(resolved)
        if not resolved_path.exists():
            raise ValueError(f"Project path does not exist: {resolved}")
        if not resolved_path.is_dir():
            raise ValueError(f"Project path is not a directory: {resolved}")

        with self._lock:
            if resolved not in self._projects:
                self._projects[resolved] = ProjectContext(path=resolved)
                logger.info(f"New project registered: {resolved}")

            ctx = self._projects[resolved]
            ctx.activated_at = utcnow()
            self._active_path = resolved

            logger.info(f"Project activated: {ctx.name} ({resolved})")
            self._save()
            return ctx

    def get_active(self) -> ProjectContext:
        """Get the active project context.

        If no project has been explicitly activated, auto-activates
        the current working directory for backward compatibility.
        Thread-safe.

        Returns:
            The active ProjectContext.
        """
        with self._lock:
            if self._active_path is None:
                # Release lock before calling activate (which re-acquires it)
                pass
            else:
                return self._projects[self._active_path]
        # Outside the lock — activate will acquire it
        cwd = os.getcwd()
        return self.activate(cwd)

    def get_project(self, path: str) -> Optional[ProjectContext]:
        """Get a specific project context by path.

        Args:
            path: Project root directory path.

        Returns:
            ProjectContext if known, None otherwise.
        """
        resolved = str(Path(path).resolve())
        return self._projects.get(resolved)

    def list_projects(self) -> list[ProjectContext]:
        """List all known project contexts.

        Returns:
            List of ProjectContexts, sorted by activation time (newest first).
        """
        projects = list(self._projects.values())
        projects.sort(
            key=lambda p: p.activated_at or datetime.min,
            reverse=True,
        )
        return projects

    def deactivate(self, path: str) -> bool:
        """Remove a project from the registry.

        This discards all cached services for the project. Thread-safe.

        Args:
            path: Project root directory path.

        Returns:
            True if removed, False if not found.
        """
        resolved = str(Path(path).resolve())
        with self._lock:
            if resolved not in self._projects:
                return False

            # Clean up all project resources
            self._projects[resolved].cleanup()
            del self._projects[resolved]
            if self._active_path == resolved:
                self._active_path = None
            logger.info(f"Project deactivated: {resolved}")
            self._save()
            return True

    def reset(self) -> None:
        """Clear all projects and state. Used for testing.

        Cleans up each project's resources (LSP servers, background
        threads) before dropping references to prevent resource leaks.
        """
        for ctx in self._projects.values():
            try:
                ctx.cleanup()
            except Exception:
                logger.debug("Cleanup failed during reset for %s", ctx.path, exc_info=True)
                pass
        self._projects.clear()
        self._active_path = None

    def to_dict(self) -> dict:
        """Serialize registry state."""
        return {
            "active_path": self._active_path,
            "active_project": (
                self.active_project.name if self.active_project else None
            ),
            "projects": [p.to_dict() for p in self.list_projects()],
            "project_count": len(self._projects),
        }
