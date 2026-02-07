"""CLI service context -- mirrors _shared.py's registry pattern for CLI commands.

All CLI commands that need services should use cli_project_scope(path)
instead of instantiating services directly. This provides:
- Project isolation via ProjectRegistry
- Lazy service initialization with thread safety
- Proper cleanup on command exit
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path

from anamnesis.services.project_registry import ProjectContext, ProjectRegistry
from anamnesis.utils.logger import logger

_registry = ProjectRegistry()


def get_project_context(path: str) -> ProjectContext:
    """Activate and return a project context for a CLI command."""
    resolved = str(Path(path).resolve())
    return _registry.activate(resolved)


@contextlib.contextmanager
def cli_project_scope(path: str) -> Generator[ProjectContext, None, None]:
    """Context manager providing a ProjectContext with cleanup on exit."""
    ctx = get_project_context(path)
    try:
        yield ctx
    finally:
        try:
            ctx.cleanup()
        except Exception:
            logger.debug("Error during CLI context cleanup", exc_info=True)
