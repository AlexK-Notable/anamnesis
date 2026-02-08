"""Shared infrastructure for the Anamnesis MCP server.

This module contains the FastMCP instance, project registry, service accessors,
error handling decorator, metacognition prompts, and utility helpers shared
across all tool modules. Extracted from server.py for better modularity.
"""

import atexit
import functools
import inspect
import re
import time
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from anamnesis.search.service import SearchService
from anamnesis.services import (
    CodebaseService,
    IntelligenceService,
    LearningService,
    MemoryService,
    SessionManager,
)
from anamnesis.services.project_registry import ProjectContext, ProjectRegistry
from anamnesis.utils.error_classifier import classify_error
from anamnesis.utils.logger import logger
from anamnesis.utils.toon_encoder import ToonEncoder, is_structurally_toon_eligible

# =============================================================================
# Server Lifecycle
# =============================================================================


def _cleanup_all_projects() -> None:
    """Clean up resources for all registered projects.

    Called during server shutdown and registered as an atexit fallback.
    Each project's cleanup is independent — a failure in one does not
    prevent cleanup of others.
    """
    for ctx in _registry.list_projects():
        try:
            ctx.cleanup()
        except Exception:
            logger.warning(
                "Error cleaning up project %s", ctx.path, exc_info=True
            )
    try:
        _registry._save()
    except Exception:
        logger.warning("Error saving registry on shutdown", exc_info=True)


@asynccontextmanager
async def _server_lifespan(app):
    """FastMCP lifespan context manager for startup/shutdown.

    Runs cleanup logic when the server shuts down, while the event
    loop is still alive. An atexit handler provides a fallback for
    cases where the lifespan exit is not reached (e.g. SIGKILL).
    """
    yield {}
    _cleanup_all_projects()


# Register atexit fallback (runs after event loop shutdown, sync only)
atexit.register(_cleanup_all_projects)


# =============================================================================
# FastMCP Server Instance
# =============================================================================

mcp = FastMCP(
    "anamnesis",
    lifespan=_server_lifespan,
    instructions="""Anamnesis - Codebase Intelligence Server

This server provides tools for understanding and navigating codebases through
intelligent analysis, pattern recognition, and semantic understanding.

Key capabilities:
- Learn from codebases to build intelligence database
- Search for code symbols and understand relationships
- Get pattern recommendations based on learned conventions
- Predict coding approaches for implementation tasks
- Track developer profiles and coding styles
- Contribute and retrieve AI-discovered insights
- Store and retrieve persistent project memories
- Reflect on your work with metacognition tools

Start with `auto_learn_if_needed` to initialize intelligence, then use
other tools to query and interact with the learned knowledge.

Use `write_memory` and `read_memory` to persist project knowledge across
sessions. Use `reflect` to pause and think before, during, and after
complex tasks.

For multi-project workflows, use `activate_project(path)` to switch between
projects. Each project gets isolated services, preventing cross-project
data contamination.
""",
)

# =============================================================================
# Module-level Globals
# =============================================================================

_registry = ProjectRegistry()
_server_start_time: float = time.time()
_toon_encoder = ToonEncoder()

# =============================================================================
# Registry-based Service Access
# =============================================================================
# All service access goes through the project registry. Each project gets
# isolated services, preventing cross-project data contamination.


def _get_active_context() -> ProjectContext:
    """Get the active project context (auto-activates cwd if needed)."""
    return _registry.get_active()


def _get_search_service() -> SearchService:
    """Get search service for the active project."""
    return _get_active_context().get_search_service()


async def _ensure_semantic_search() -> bool:
    """Initialize semantic search for the active project."""
    return await _get_active_context().ensure_semantic_search()


def _get_learning_service() -> LearningService:
    """Get learning service for the active project."""
    return _get_active_context().get_learning_service()


def _get_intelligence_service() -> IntelligenceService:
    """Get intelligence service for the active project."""
    return _get_active_context().get_intelligence_service()


def _get_codebase_service() -> CodebaseService:
    """Get codebase service for the active project."""
    return _get_active_context().get_codebase_service()


def _get_session_manager() -> SessionManager:
    """Get session manager for the active project."""
    return _get_active_context().get_session_manager()


def _get_symbol_service():
    """Get symbol service for the active project."""
    return _get_active_context().get_symbol_service()


def _get_memory_service() -> MemoryService:
    """Get memory service for the active project."""
    return _get_active_context().get_memory_service()


def _get_current_path() -> str:
    """Get current working path (active project path)."""
    return _get_active_context().path


def _set_current_path(path: str) -> None:
    """Set current working path by activating a project.

    This replaces the old global path mutation with project activation.
    All services are now project-scoped, so switching projects
    automatically uses isolated service instances.
    """
    _registry.activate(path)


# =============================================================================
# Metacognition Prompts
# =============================================================================

_THINK_COLLECTED_PROMPT = """\
Think about the collected information and whether it is sufficient and relevant.

Consider:
1. **Completeness**: Do you have enough information to proceed with the task?
   - Have you identified all relevant files and symbols?
   - Are there dependencies or relationships you haven't explored?
   - Is there context missing that could change your approach?

2. **Relevance**: Is the information you've gathered actually useful?
   - Does it directly relate to the task at hand?
   - Are you going down a rabbit hole that won't help?
   - Should you refocus on a different aspect?

3. **Confidence**: How confident are you in your understanding?
   - Are there assumptions you're making that should be verified?
   - Could the code behave differently than you expect?
   - Have you considered edge cases?

Take a moment to assess before proceeding.
"""

_THINK_TASK_ADHERENCE_PROMPT = """\
Think about the task at hand and whether you are still on track.

Consider:
1. **Original goal**: What was the user's actual request?
   - Are you still working toward that goal?
   - Have you drifted into tangential work?
   - Is your current approach aligned with what was asked?

2. **Scope**: Are you doing too much or too little?
   - Are you over-engineering the solution?
   - Are you adding unnecessary features or abstractions?
   - Have you missed any requirements?

3. **Progress**: What have you accomplished so far?
   - What concrete steps remain?
   - Are there blockers you need to address?
   - Should you ask for clarification before continuing?

Refocus on the core task if you've drifted.
"""

_THINK_DONE_PROMPT = """\
Think about whether you are truly done with the task.

Consider:
1. **Completeness**: Have you addressed everything the user asked for?
   - Review the original request point by point
   - Have you tested or verified your changes?
   - Are there any loose ends or TODO items?

2. **Quality**: Is the work at an acceptable standard?
   - Is the code clean and well-structured?
   - Have you handled error cases?
   - Would you be confident in this code going to production?

3. **Communication**: Have you explained what you did?
   - Does the user understand the changes and why?
   - Are there caveats or limitations they should know about?
   - Is there follow-up work they should consider?

If not fully done, identify what remains. If done, summarize the outcome.
"""

_REFLECT_PROMPTS = {
    "collected_information": _THINK_COLLECTED_PROMPT,
    "task_adherence": _THINK_TASK_ADHERENCE_PROMPT,
    "whether_done": _THINK_DONE_PROMPT,
}

# =============================================================================
# Utility Helpers
# =============================================================================


def _format_blueprint_as_memory(
    blueprint: dict,
    symbol_data: dict[str, list[dict[str, str]]] | None = None,
) -> str:
    """Format a project blueprint dict as a readable markdown memory.

    Called during auto-onboarding to generate a project-overview memory
    from the learned blueprint. Optionally enriched with top-level
    symbols extracted via tree-sitter during learning.

    Args:
        blueprint: Dict from IntelligenceService.get_project_blueprint().
        symbol_data: Optional mapping of file paths to lists of symbol dicts,
            each with ``name`` and ``kind`` keys (e.g. "Class", "Function").

    Returns:
        Markdown string suitable for MemoryService.write_memory().
    """
    lines = ["# Project Overview", "", "Auto-generated from codebase analysis.", ""]

    tech = blueprint.get("tech_stack", [])
    if tech:
        lines.append("## Tech Stack")
        for t in tech:
            lines.append(f"- {t}")
        lines.append("")

    arch = blueprint.get("architecture", "")
    if arch:
        lines.append("## Architecture")
        lines.append(f"{arch}")
        lines.append("")

    entries = blueprint.get("entry_points", {})
    if entries:
        lines.append("## Entry Points")
        for etype, epath in entries.items():
            lines.append(f"- **{etype}**: `{epath}`")
        lines.append("")

    dirs = blueprint.get("key_directories", {})
    if dirs:
        lines.append("## Key Directories")
        for dpath, dtype in dirs.items():
            lines.append(f"- `{dpath}/` — {dtype}")
        lines.append("")

    features = blueprint.get("feature_map", {})
    if features:
        lines.append("## Features")
        for feature, files in features.items():
            file_list = ", ".join(f"`{f}`" for f in files[:5])
            lines.append(f"- **{feature}**: {file_list}")
        lines.append("")

    if symbol_data:
        lines.append("## Key Symbols")
        for file_path, symbols in sorted(symbol_data.items()):
            if not symbols:
                continue
            lines.append(f"### `{file_path}`")
            for sym in symbols:
                kind = sym.get("kind", "Symbol")
                name = sym.get("name", "?")
                lines.append(f"- {kind}: **{name}**")
            lines.append("")

    return "\n".join(lines)


def _collect_key_symbols(
    blueprint: dict,
    project_path: str,
    max_files: int = 10,
    max_symbols_per_file: int = 20,
) -> dict[str, list[dict[str, str]]] | None:
    """Collect top-level symbols from key project files via tree-sitter.

    Uses the fast tree-sitter backend (no LSP startup required) to extract
    classes and functions from entry point files identified in the blueprint.
    Returns None on any failure — callers should treat this as optional
    enrichment.

    Args:
        blueprint: Project blueprint with ``entry_points`` and ``feature_map``.
        project_path: Absolute path to the project root.
        max_files: Maximum number of files to scan.
        max_symbols_per_file: Maximum symbols to include per file.

    Returns:
        Dict mapping relative file paths to lists of ``{name, kind}`` dicts,
        or None if extraction fails or no symbols found.
    """
    import os

    from anamnesis.extraction.backends import get_shared_tree_sitter
    from anamnesis.extraction.types import SymbolKind
    from anamnesis.lsp.utils import safe_join
    from anamnesis.utils.language_registry import detect_language

    # Gather candidate files from entry points and feature map
    candidates: list[str] = []
    for _etype, epath in blueprint.get("entry_points", {}).items():
        if epath and isinstance(epath, str):
            candidates.append(epath)
    for _feature, files in blueprint.get("feature_map", {}).items():
        if isinstance(files, list):
            candidates.extend(f for f in files if isinstance(f, str))

    # Deduplicate and limit
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    unique = unique[:max_files]

    if not unique:
        return None

    backend = get_shared_tree_sitter()
    symbol_data: dict[str, list[dict[str, str]]] = {}
    _TOP_LEVEL_KINDS = {SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.INTERFACE}

    for rel_path in unique:
        try:
            abs_path = safe_join(project_path, rel_path)
        except ValueError:
            logger.debug("Path traversal blocked in _collect_key_symbols: %s", rel_path)
            continue
        if not os.path.isfile(abs_path):
            continue

        lang = detect_language(rel_path)
        if lang == "unknown" or not backend.supports_language(lang):
            continue

        try:
            with open(abs_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            result = backend.extract_all(content, rel_path, lang)
        except Exception:
            logger.debug("Symbol extraction failed for %s", rel_path, exc_info=True)
            continue

        file_symbols: list[dict[str, str]] = []
        for sym in result.symbols:
            if sym.kind in _TOP_LEVEL_KINDS:
                file_symbols.append({
                    "name": sym.name,
                    "kind": str(sym.kind),
                })
            if len(file_symbols) >= max_symbols_per_file:
                break

        if file_symbols:
            symbol_data[rel_path] = file_symbols

    return symbol_data if symbol_data else None


def _categorize_references(references: list[dict]) -> dict[str, list[dict]]:
    """Categorize symbol references by file type.

    Thin wrapper — delegates to SymbolService.categorize_references().
    """
    from anamnesis.services.symbol_service import SymbolService

    return SymbolService.categorize_references(references)


def _check_names_against_convention(
    names: list[str],
    expected: str,
    symbol_kind: str,
) -> list[dict]:
    """Check symbol names against an expected naming convention.

    Thin wrapper — delegates to SymbolService.check_names_against_convention().
    """
    from anamnesis.services.symbol_service import SymbolService

    return SymbolService.check_names_against_convention(names, expected, symbol_kind)


# =============================================================================
# Error Handling Helpers & Decorator
# =============================================================================


_RE_UNIX_PATH = re.compile(
    r"(?:/(?:home|tmp|var|etc|usr|opt|root|Users|Windows"
    r"|srv|mnt|media|run|data|proc|sys|snap|nix)[^\s'\",:;)\}\]]*)"
)
_RE_WIN_PATH = re.compile(r"(?:[A-Z]:\\[^\s'\",:;)\}\]]+)")
_RE_FILE_URI = re.compile(r"file://[^\s\"']+")
_RE_DOTDOT = re.compile(r"(?:\.\./){2,}[^\s\"']*")


def _sanitize_error_message(error_msg: str) -> str:
    """Remove sensitive details from error messages returned to clients."""
    sanitized = _RE_UNIX_PATH.sub(
        lambda m: '.../' + m.group(0).rsplit('/', 1)[-1] if '/' in m.group(0) else m.group(0),
        error_msg,
    )
    sanitized = _RE_WIN_PATH.sub(
        lambda m: '...\\' + m.group(0).rsplit('\\', 1)[-1] if '\\' in m.group(0) else m.group(0),
        sanitized,
    )
    sanitized = _RE_FILE_URI.sub("<file-uri>", sanitized)
    sanitized = _RE_DOTDOT.sub("<redacted-path>", sanitized)
    return sanitized


def _failure_response(
    error_message: str,
    error_code: str = "client_error",
    is_retryable: bool = False,
    **extra: object,
) -> dict:
    """Build a standard failure response dict.

    All MCP tool failures should use this shape so LLM consumers see a
    single, predictable structure with consistent error classification
    fields: ``{"success": False, "error": "...", "error_code": "...",
    "is_retryable": ...}``.

    Args:
        error_message: Human-readable error description.
        error_code: Error classification (default ``"client_error"``).
            Use ``"system_error"`` for infrastructure failures.
        is_retryable: Whether the consumer should retry the operation
            (default ``False``).
        **extra: Additional keys merged into the response (e.g. ``results=[]``).
    """
    return {
        "success": False,
        "error": error_message,
        "error_code": error_code,
        "is_retryable": is_retryable,
        **extra,
    }


def _with_error_handling(operation_name: str, toon_auto: bool = True):
    """Decorator for MCP tool implementations with error handling and TOON auto-encoding.

    Catches exceptions and returns standardized
    ``{"success": False, "error": "..."}`` error responses.
    Supports both sync and async tool implementations.

    When toon_auto is True (default), successful dict responses are checked
    for structural TOON eligibility. Eligible responses (flat uniform arrays
    with ≥5 elements, no nested arrays) are encoded to TOON format for
    ~25-40% token savings. The encoded string is returned directly — FastMCP
    wraps it as TextContent. Error responses always stay as JSON dicts for
    maximum debuggability.

    If TOON encoding fails for any reason, the original dict is returned
    silently (no error propagated).

    Args:
        operation_name: Name of the operation for logging and error context.
        toon_auto: Enable automatic TOON encoding for eligible responses.
            Set to False for tools where JSON output is required.
    """

    def _apply_toon(result):
        """Apply TOON encoding to eligible success dicts."""
        if toon_auto and isinstance(result, dict) and result.get("success"):
            try:
                if is_structurally_toon_eligible(result):
                    return _toon_encoder.encode(result)
            except Exception:
                pass  # Silent fallback to dict/JSON on any encoding error
        return result

    def _handle_exception(e: Exception):
        classification = classify_error(e, {"operation": operation_name})
        logger.error(
            f"Error in operation '{operation_name}': {e}",
            extra={
                "operation": operation_name,
                "category": classification.category.value,
                "error_type": type(e).__name__,
            },
        )
        return _failure_response(
            _sanitize_error_message(str(e)),
            error_code=classification.category.value,
            is_retryable=classification.is_retryable,
        )

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    return _apply_toon(result)
                except Exception as e:
                    return _handle_exception(e)

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    return _apply_toon(result)
                except Exception as e:
                    return _handle_exception(e)

            return wrapper

    return decorator
