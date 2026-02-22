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
from datetime import datetime, timezone
from pathlib import Path

from anamnesis.utils.logger import logger


@dataclass
class MemoryInfo:
    """Information about a project memory."""

    name: str
    content: str
    created_at: datetime | None
    updated_at: datetime | None
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
    created_at: datetime | None
    updated_at: datetime | None
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
        self._index: MemoryIndex | None = None

    def _get_index(self) -> MemoryIndex:
        """Get or create the memory index, indexing existing memories."""
        if self._index is None:
            self._index = MemoryIndex()
            for entry in self.list_memories():
                mem = self.read_memory(entry.name)
                if mem:
                    self._index.index(entry.name, mem.content)
        return self._index

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
        now = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        logger.info("Memory written: %s (%s bytes)", clean_name, stat.st_size)

        # Update search index
        if self._index is not None:
            self._index.index(clean_name, content)

        return MemoryInfo(
            name=clean_name,
            content=content,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=now,
            size_bytes=stat.st_size,
        )

    def read_memory(self, name: str) -> MemoryInfo | None:
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
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
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
                        created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        size_bytes=stat.st_size,
                    )
                )

        # Sort by updated_at descending (newest first)
        entries.sort(key=lambda e: e.updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
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
        # Update search index
        if self._index is not None:
            self._index.remove(clean_name)
        logger.info("Memory deleted: %s", clean_name)
        return True

    def edit_memory(
        self,
        name: str,
        old_text: str,
        new_text: str,
    ) -> MemoryInfo | None:
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

        # Update search index
        if self._index is not None:
            self._index.index(clean_name, new_content)

        stat = path.stat()
        logger.info("Memory edited: %s", clean_name)

        return MemoryInfo(
            name=clean_name,
            content=new_content,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            size_bytes=stat.st_size,
        )

    def search_memories(
        self, query: str, limit: int = 5
    ) -> list[dict]:
        """Search memories by semantic similarity or substring matching.

        Uses sentence-transformers embeddings when available, falls back
        to substring matching on memory names and content.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.

        Returns:
            List of dicts with name, score, snippet.
        """
        memories = self.list_memories()
        if not memories:
            return []

        # Try semantic search first
        index = self._get_index()
        semantic_results = index.search(query, limit=limit)

        if semantic_results:
            results = []
            for name, score in semantic_results:
                mem = self.read_memory(name)
                snippet = mem.content[:200] if mem else ""
                results.append({"name": name, "score": round(score, 3), "snippet": snippet})
            return results

        # Fallback: substring matching on name + content
        query_lower = query.lower()
        scored = []
        for entry in memories:
            mem = self.read_memory(entry.name)
            if not mem:
                continue
            score = 0.0
            if query_lower in entry.name.lower():
                score += 0.8
            if query_lower in mem.content.lower():
                score += 0.5
            if score > 0:
                scored.append({"name": entry.name, "score": round(score, 3), "snippet": mem.content[:200]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


class MemoryIndex:
    """Lightweight embedding index for memory search.

    Uses sentence-transformers all-MiniLM-L6-v2 for semantic search.
    Gracefully falls back to substring matching if the model is unavailable.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._embeddings: dict[str, "np.ndarray"] = {}

    def _load_model(self) -> bool:
        if self._model_loaded:
            return self._model is not None
        self._model_loaded = True
        try:
            from anamnesis.utils.model_registry import get_shared_sentence_transformer
            self._model = get_shared_sentence_transformer()
            return True
        except Exception:
            logger.debug("Sentence-transformers model load failed", exc_info=True)
            return False

    def index(self, name: str, content: str) -> None:
        if not self._load_model() or self._model is None:
            return
        embedding = self._model.encode(content[:2000], convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings[name] = embedding

    def remove(self, name: str) -> None:
        self._embeddings.pop(name, None)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        if not self._load_model() or self._model is None or not self._embeddings:
            return []
        import numpy as np
        query_emb = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = []
        for name, emb in self._embeddings.items():
            sim = float(np.dot(query_emb, emb))
            scores.append((name, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]