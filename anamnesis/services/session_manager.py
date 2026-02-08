"""Session manager service for work session lifecycle management.

Provides high-level operations for managing work sessions and recording
decisions during development work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from anamnesis.constants import utcnow
from anamnesis.services.type_converters import generate_id
from anamnesis.storage.schema import ProjectDecision, WorkSession

# ---------------------------------------------------------------------------
# Conversion helpers — single source of truth for schema → info mapping
# ---------------------------------------------------------------------------


def _session_to_info(s: WorkSession, decision_count: int = 0) -> "SessionInfo":
    """Convert a WorkSession schema object to a SessionInfo."""
    return SessionInfo(
        session_id=s.id,
        name=s.name,
        feature=s.feature,
        files=s.files,
        tasks=s.tasks,
        is_active=s.is_active,
        started_at=s.started_at,
        updated_at=s.updated_at,
        ended_at=s.ended_at,
        decision_count=decision_count,
    )


def _decision_to_info(d: ProjectDecision) -> "DecisionInfo":
    """Convert a ProjectDecision schema object to a DecisionInfo."""
    return DecisionInfo(
        decision_id=d.id,
        decision=d.decision,
        context=d.context,
        rationale=d.rationale,
        session_id=d.session_id,
        related_files=d.related_files,
        tags=d.tags,
        created_at=d.created_at,
    )

if TYPE_CHECKING:
    from anamnesis.storage.sync_backend import SyncSQLiteBackend


@dataclass
class SessionInfo:
    """Information about a work session."""

    session_id: str
    name: str
    feature: str
    files: list[str]
    tasks: list[str]
    is_active: bool
    started_at: datetime
    updated_at: datetime
    ended_at: Optional[datetime] = None
    decision_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "feature": self.feature,
            "files": self.files,
            "tasks": self.tasks,
            "is_active": self.is_active,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "decision_count": self.decision_count,
        }


@dataclass
class DecisionInfo:
    """Information about a project decision."""

    decision_id: str
    decision: str
    context: str
    rationale: str
    session_id: str
    related_files: list[str]
    tags: list[str]
    created_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "decision": self.decision,
            "context": self.context,
            "rationale": self.rationale,
            "session_id": self.session_id,
            "related_files": self.related_files,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }


class SessionManager:
    """Manager for work sessions and project decisions.

    Provides lifecycle management for work sessions:
    - Start/end sessions
    - Track files and tasks being worked on
    - Record project decisions
    - Link decisions to sessions

    Usage:
        backend = SyncSQLiteBackend("path/to/db.sqlite")
        backend.connect()
        manager = SessionManager(backend)

        # Start a session
        session = manager.start_session(
            name="Add user authentication",
            feature="authentication",
            files=["/src/auth.py"],
        )

        # Record decisions during work
        manager.record_decision(
            session_id=session.session_id,
            decision="Use JWT for token-based auth",
            rationale="Better for stateless API design",
        )

        # End the session
        manager.end_session(session.session_id)
    """

    def __init__(self, backend: "SyncSQLiteBackend"):
        """Initialize session manager.

        Args:
            backend: Storage backend for persistence
        """
        self._backend = backend
        self._active_session_id: Optional[str] = None

    @property
    def backend(self) -> "SyncSQLiteBackend":
        """Get the storage backend."""
        return self._backend

    @property
    def active_session_id(self) -> Optional[str]:
        """Get the currently active session ID (most recently started)."""
        return self._active_session_id

    def start_session(
        self,
        name: str = "",
        feature: str = "",
        files: Optional[list[str]] = None,
        tasks: Optional[list[str]] = None,
        notes: str = "",
        metadata: Optional[dict] = None,
    ) -> SessionInfo:
        """Start a new work session.

        Args:
            name: Name or description of the session
            feature: Feature being worked on
            files: Initial list of files being worked on
            tasks: Initial list of tasks
            notes: Session notes
            metadata: Additional metadata

        Returns:
            SessionInfo with the new session details
        """
        session_id = generate_id("session")
        now = utcnow()

        session = WorkSession(
            id=session_id,
            name=name,
            feature=feature,
            files=files or [],
            tasks=tasks or [],
            notes=notes,
            metadata=metadata or {},
            started_at=now,
            updated_at=now,
            ended_at=None,
        )

        self._backend.save_work_session(session)
        self._active_session_id = session_id

        return SessionInfo(
            session_id=session_id,
            name=name,
            feature=feature,
            files=files or [],
            tasks=tasks or [],
            is_active=True,
            started_at=now,
            updated_at=now,
            ended_at=None,
            decision_count=0,
        )

    def end_session(self, session_id: Optional[str] = None) -> bool:
        """End a work session.

        Args:
            session_id: Session ID to end. If None, ends the active session.

        Returns:
            True if session was ended, False if not found
        """
        target_id = session_id or self._active_session_id
        if not target_id:
            return False

        success = self._backend.end_work_session(target_id)
        if success and target_id == self._active_session_id:
            self._active_session_id = None

        return success

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get a session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionInfo or None if not found
        """
        session = self._backend.get_work_session(session_id)
        if not session:
            return None

        counts = self._backend.get_decision_counts_by_sessions([session_id])
        return _session_to_info(session, decision_count=counts.get(session_id, 0))

    def get_active_sessions(self) -> list[SessionInfo]:
        """Get all active (non-ended) sessions.

        Returns:
            List of active SessionInfo objects
        """
        sessions = self._backend.get_active_sessions()
        if not sessions:
            return []
        counts = self._backend.get_decision_counts_by_sessions(
            [s.id for s in sessions]
        )
        return [
            _session_to_info(s, decision_count=counts.get(s.id, 0))
            for s in sessions
        ]

    def get_recent_sessions(self, limit: int = 10) -> list[SessionInfo]:
        """Get recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of recent SessionInfo objects
        """
        sessions = self._backend.get_recent_sessions(limit)
        if not sessions:
            return []
        counts = self._backend.get_decision_counts_by_sessions(
            [s.id for s in sessions]
        )
        return [
            _session_to_info(s, decision_count=counts.get(s.id, 0))
            for s in sessions
        ]

    def update_session(
        self,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None,
        tasks: Optional[list[str]] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[SessionInfo]:
        """Update an existing session.

        Args:
            session_id: Session ID to update. If None, updates the active session.
            files: New files list (replaces existing)
            tasks: New tasks list (replaces existing)
            notes: New notes
            metadata: Additional metadata (merged with existing)

        Returns:
            Updated SessionInfo or None if not found
        """
        target_id = session_id or self._active_session_id
        if not target_id:
            return None

        session = self._backend.get_work_session(target_id)
        if not session:
            return None

        # Update fields
        if files is not None:
            session.files = files
        if tasks is not None:
            session.tasks = tasks
        if notes is not None:
            session.notes = notes
        if metadata is not None:
            session.metadata.update(metadata)
        session.updated_at = utcnow()

        self._backend.save_work_session(session)

        return self.get_session(target_id)

    def add_file_to_session(
        self,
        file_path: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add a file to a session's working files.

        Args:
            file_path: File path to add
            session_id: Session ID. If None, uses active session.

        Returns:
            True if added, False if session not found
        """
        target_id = session_id or self._active_session_id
        if not target_id:
            return False

        session = self._backend.get_work_session(target_id)
        if not session:
            return False

        if file_path not in session.files:
            session.files.append(file_path)
            session.updated_at = utcnow()
            self._backend.save_work_session(session)

        return True

    def add_task_to_session(
        self,
        task: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add a task to a session.

        Args:
            task: Task description to add
            session_id: Session ID. If None, uses active session.

        Returns:
            True if added, False if session not found
        """
        target_id = session_id or self._active_session_id
        if not target_id:
            return False

        session = self._backend.get_work_session(target_id)
        if not session:
            return False

        if task not in session.tasks:
            session.tasks.append(task)
            session.updated_at = utcnow()
            self._backend.save_work_session(session)

        return True

    def record_decision(
        self,
        decision: str,
        context: str = "",
        rationale: str = "",
        session_id: Optional[str] = None,
        related_files: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> DecisionInfo:
        """Record a project decision.

        Args:
            decision: The decision made
            context: Context for the decision
            rationale: Why this decision was made
            session_id: Session to link to. If None, uses active session.
            related_files: Files related to the decision
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            DecisionInfo with the recorded decision
        """
        decision_id = generate_id("decision")
        target_session = session_id or self._active_session_id or ""
        now = utcnow()

        project_decision = ProjectDecision(
            id=decision_id,
            decision=decision,
            context=context,
            rationale=rationale,
            session_id=target_session,
            related_files=related_files or [],
            tags=tags or [],
            metadata=metadata or {},
            created_at=now,
        )

        self._backend.save_project_decision(project_decision)

        return DecisionInfo(
            decision_id=decision_id,
            decision=decision,
            context=context,
            rationale=rationale,
            session_id=target_session,
            related_files=related_files or [],
            tags=tags or [],
            created_at=now,
        )

    def get_decision(self, decision_id: str) -> Optional[DecisionInfo]:
        """Get a decision by ID.

        Args:
            decision_id: Decision ID to retrieve

        Returns:
            DecisionInfo or None if not found
        """
        decision = self._backend.get_project_decision(decision_id)
        if not decision:
            return None

        return _decision_to_info(decision)

    def get_decisions_by_session(self, session_id: str) -> list[DecisionInfo]:
        """Get all decisions for a session.

        Args:
            session_id: Session ID to get decisions for

        Returns:
            List of DecisionInfo objects
        """
        decisions = self._backend.get_decisions_by_session(session_id)
        return [_decision_to_info(d) for d in decisions]

    def get_recent_decisions(self, limit: int = 10) -> list[DecisionInfo]:
        """Get recent decisions.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of recent DecisionInfo objects
        """
        decisions = self._backend.get_recent_decisions(limit)
        return [_decision_to_info(d) for d in decisions]
