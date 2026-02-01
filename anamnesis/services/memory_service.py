"""Memory service for persistent project-scoped notes.

Provides CRUD operations for markdown-based memory files stored in
`.anamnesis/memories/` within the project root. These memories persist
across server restarts and are human-readable as plain markdown files.

Inspired by Serena's MemoriesManager but designed to integrate with
Anamnesis's intelligence pipeline (future: embedding-based search).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from anamnesis.utils.logger import logger


@dataclass
class MemoryInfo:
    """Information about a project memory."""

    name: str
    content: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    size_bytes: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "size_bytes": self.size_bytes,
        }


@dataclass
class MemoryListEntry:
    """Summary entry for memory listing (without full content)."""

    name: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    size_bytes: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "size_bytes": self.size_bytes,
        }


# Valid memory name: alphanumeric, hyphens, underscores, dots (no path separators)
_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_MAX_NAME_LENGTH = 200
_MAX_CONTENT_SIZE = 1_000_000  # 1MB


class MemoryService:
    """Manager for persistent project-scoped memory files.

    Stores markdown notes in `.anamnesis/memories/` within the project root.
    Memory names are sanitized to prevent path traversal.

    Usage:
        service = MemoryService("/path/to/project")
        service.write_memory("architecture-notes", "# Architecture\\n...")
        content = service.read_memory("architecture-notes")
        memories = service.list_memories()
        service.delete_memory("architecture-notes")
    """

    MEMORIES_DIR = ".anamnesis/memories"

    def __init__(self, project_path: str):
        """Initialize memory service.

        Args:
            project_path: Root path of the project.
        """
        self._project_path = Path(project_path).resolve()
        self._memories_dir = self._project_path / self.MEMORIES_DIR

    @property
    def project_path(self) -> Path:
        """Get the project root path."""
        return self._project_path

    @property
    def memories_dir(self) -> Path:
        """Get the memories directory path."""
        return self._memories_dir

    def _validate_name(self, name: str) -> str:
        """Validate and normalize a memory name.

        Args:
            name: Memory name to validate.

        Returns:
            Normalized name (without .md extension).

        Raises:
            ValueError: If name is invalid.
        """
        if not name or not name.strip():
            raise ValueError("Memory name cannot be empty")

        # Strip .md extension if provided
        clean = name.strip()
        if clean.endswith(".md"):
            clean = clean[:-3]

        if not clean:
            raise ValueError("Memory name cannot be just '.md'")

        if len(clean) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Memory name too long: {len(clean)} chars (max {_MAX_NAME_LENGTH})"
            )

        if not _VALID_NAME_PATTERN.match(clean):
            raise ValueError(
                f"Invalid memory name '{clean}'. "
                "Use only letters, numbers, hyphens, underscores, and dots. "
                "Must start with a letter or number."
            )

        return clean

    def _get_memory_path(self, name: str) -> Path:
        """Get the file path for a memory.

        Args:
            name: Validated memory name (without .md).

        Returns:
            Absolute path to the memory file.
        """
        path = (self._memories_dir / f"{name}.md").resolve()
        # Verify the path is within the memories directory (prevent traversal)
        if not str(path).startswith(str(self._memories_dir.resolve())):
            raise ValueError(f"Memory name resolves outside memories directory: {name}")
        return path

    def _ensure_dir(self) -> None:
        """Create the memories directory if it doesn't exist."""
        self._memories_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(self, name: str, content: str) -> MemoryInfo:
        """Write or overwrite a memory file.

        Args:
            name: Memory name (used as filename, .md appended automatically).
            content: Markdown content to write.

        Returns:
            MemoryInfo with the written memory details.

        Raises:
            ValueError: If name is invalid or content too large.
        """
        clean_name = self._validate_name(name)

        if len(content.encode("utf-8")) > _MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content too large: {len(content.encode('utf-8'))} bytes "
                f"(max {_MAX_CONTENT_SIZE})"
            )

        self._ensure_dir()
        path = self._get_memory_path(clean_name)
        path.write_text(content, encoding="utf-8")

        stat = path.stat()
        now = datetime.fromtimestamp(stat.st_mtime)

        logger.info(f"Memory written: {clean_name} ({stat.st_size} bytes)")

        return MemoryInfo(
            name=clean_name,
            content=content,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=now,
            size_bytes=stat.st_size,
        )

    def read_memory(self, name: str) -> Optional[MemoryInfo]:
        """Read a memory file.

        Args:
            name: Memory name to read.

        Returns:
            MemoryInfo with content, or None if not found.

        Raises:
            ValueError: If name is invalid.
        """
        clean_name = self._validate_name(name)
        path = self._get_memory_path(clean_name)

        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        stat = path.stat()

        return MemoryInfo(
            name=clean_name,
            content=content,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            size_bytes=stat.st_size,
        )

    def list_memories(self) -> list[MemoryListEntry]:
        """List all memory files.

        Returns:
            List of MemoryListEntry objects sorted by update time (newest first).
        """
        if not self._memories_dir.exists():
            return []

        entries = []
        for path in sorted(self._memories_dir.glob("*.md")):
            if path.is_file():
                stat = path.stat()
                entries.append(
                    MemoryListEntry(
                        name=path.stem,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        updated_at=datetime.fromtimestamp(stat.st_mtime),
                        size_bytes=stat.st_size,
                    )
                )

        # Sort by updated_at descending (newest first)
        entries.sort(key=lambda e: e.updated_at or datetime.min, reverse=True)
        return entries

    def delete_memory(self, name: str) -> bool:
        """Delete a memory file.

        Args:
            name: Memory name to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If name is invalid.
        """
        clean_name = self._validate_name(name)
        path = self._get_memory_path(clean_name)

        if not path.exists():
            return False

        path.unlink()
        logger.info(f"Memory deleted: {clean_name}")
        return True

    def edit_memory(
        self,
        name: str,
        old_text: str,
        new_text: str,
    ) -> Optional[MemoryInfo]:
        """Edit a memory by replacing text.

        Args:
            name: Memory name to edit.
            old_text: Text to find and replace.
            new_text: Replacement text.

        Returns:
            Updated MemoryInfo, or None if memory not found.

        Raises:
            ValueError: If name is invalid or old_text not found in content.
        """
        clean_name = self._validate_name(name)
        path = self._get_memory_path(clean_name)

        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")

        if old_text not in content:
            raise ValueError(
                f"Text to replace not found in memory '{clean_name}'"
            )

        new_content = content.replace(old_text, new_text, 1)

        if len(new_content.encode("utf-8")) > _MAX_CONTENT_SIZE:
            raise ValueError(
                f"Edited content too large: {len(new_content.encode('utf-8'))} bytes "
                f"(max {_MAX_CONTENT_SIZE})"
            )

        path.write_text(new_content, encoding="utf-8")

        stat = path.stat()
        logger.info(f"Memory edited: {clean_name}")

        return MemoryInfo(
            name=clean_name,
            content=new_content,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            size_bytes=stat.st_size,
        )
