"""Shared infrastructure for the Anamnesis MCP server.

This module contains the FastMCP instance, project registry, service accessors,
error handling decorator, metacognition prompts, and utility helpers shared
across all tool modules. Extracted from server.py for better modularity.
"""

import functools
import inspect
import re
import time

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
from anamnesis.utils.circuit_breaker import CircuitBreakerError
from anamnesis.utils.error_classifier import classify_error
from anamnesis.utils.logger import logger
from anamnesis.utils.toon_encoder import ToonEncoder, is_structurally_toon_eligible

# =============================================================================
# FastMCP Server Instance
# =============================================================================

mcp = FastMCP(
    "anamnesis",
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


def _format_blueprint_as_memory(blueprint: dict) -> str:
    """Format a project blueprint dict as a readable markdown memory.

    Called during auto-onboarding to generate a project-overview memory
    from the learned blueprint. Handles missing/empty fields gracefully.

    Args:
        blueprint: Dict from IntelligenceService.get_project_blueprint().

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

    return "\n".join(lines)


def _categorize_references(references: list[dict]) -> dict[str, list[dict]]:
    """Categorize symbol references by file type for intelligent filtering.

    Groups references into source, test, config, and other categories
    based on file path heuristics. Each reference retains its original
    data and gains a 'category' field.

    Args:
        references: List of reference dicts with at least a 'file' key.

    Returns:
        Dict mapping category names to lists of references.
    """
    if not references:
        return {}

    categories: dict[str, list[dict]] = {}

    for ref in references:
        file_path = ref.get("file", ref.get("relative_path", "")).lower()

        if any(t in file_path for t in ("test", "spec", "fixture", "conftest")):
            cat = "test"
        elif any(c in file_path for c in ("config", "settings", "env", ".cfg", ".ini", ".toml", ".yaml", ".yml")):
            cat = "config"
        elif any(s in file_path for s in ("src/", "lib/", "app/", "anamnesis/", "pkg/")):
            cat = "source"
        elif file_path.endswith(".py") or file_path.endswith(".ts") or file_path.endswith(".rs"):
            cat = "source"
        else:
            cat = "other"

        ref_with_cat = {**ref, "category": cat}
        categories.setdefault(cat, []).append(ref_with_cat)

    return categories


_RE_UPPER_CASE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_RE_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_RE_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
_RE_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_RE_FLAT_CASE = re.compile(r"^[a-z][a-z0-9]*$")


def _detect_naming_style(name: str) -> str:
    """Detect the naming convention of a single identifier.

    Args:
        name: The identifier name to analyze.

    Returns:
        One of: snake_case, PascalCase, camelCase, UPPER_CASE, flat_case, kebab-case, mixed
    """
    if not name or name.startswith("_"):
        name = name.lstrip("_")
    if not name:
        return "unknown"

    if _RE_UPPER_CASE.match(name) and "_" in name:
        return "UPPER_CASE"
    if _RE_PASCAL_CASE.match(name):
        return "PascalCase"
    if _RE_SNAKE_CASE.match(name):
        return "snake_case"
    if _RE_CAMEL_CASE.match(name) and any(c.isupper() for c in name):
        return "camelCase"
    if _RE_FLAT_CASE.match(name):
        return "flat_case"
    if "-" in name:
        return "kebab-case"
    return "mixed"


def _check_names_against_convention(
    names: list[str],
    expected: str,
    symbol_kind: str,
) -> list[dict]:
    """Check a list of symbol names against an expected naming convention.

    Args:
        names: List of identifier names to check.
        expected: Expected convention (snake_case, PascalCase, etc.).
        symbol_kind: Kind of symbol for context (function, class, etc.).

    Returns:
        List of violation dicts with name, expected, actual, symbol_kind.
    """
    violations = []
    for name in names:
        # Skip private/dunder names
        clean = name.lstrip("_")
        if not clean or clean.startswith("__"):
            continue
        actual = _detect_naming_style(name)
        if actual != expected and actual != "unknown":
            # flat_case is compatible with snake_case for single-word names
            if expected == "snake_case" and actual == "flat_case":
                continue
            violations.append({
                "name": name,
                "expected": expected,
                "actual": actual,
                "symbol_kind": symbol_kind,
            })
    return violations


# =============================================================================
# Error Handling Helpers & Decorator
# =============================================================================


_RE_UNIX_PATH = re.compile(r"(?:/(?:home|tmp|var|etc|usr|opt|root|Users|Windows)[^\s'\",:;)\}\]]*)")
_RE_WIN_PATH = re.compile(r"(?:[A-Z]:\\[^\s'\",:;)\}\]]+)")


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
    return sanitized


def _failure_response(error_message: str, **extra: object) -> dict:
    """Build a standard failure response dict.

    All MCP tool failures should use this shape so LLM consumers see a
    single, predictable structure: ``{"success": False, "error": "..."}``.

    Args:
        error_message: Human-readable error description.
        **extra: Additional keys merged into the response (e.g. ``results=[]``).
    """
    return {"success": False, "error": error_message, **extra}


def _with_error_handling(operation_name: str, toon_auto: bool = True):
    """Decorator for MCP tool implementations with error handling and TOON auto-encoding.

    Catches CircuitBreakerError and other exceptions, returning
    standardized ``{"success": False, "error": "..."}`` error responses.
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

    def _handle_circuit_breaker(e: CircuitBreakerError):
        logger.error(
            f"Circuit breaker open for operation '{operation_name}'",
            extra={
                "operation": operation_name,
                "circuit_state": "OPEN",
                "details": str(e.details) if e.details else None,
            },
        )
        return _failure_response(
            str(e),
            error_code="circuit_breaker",
            is_retryable=True,
        )

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
                except CircuitBreakerError as e:
                    return _handle_circuit_breaker(e)
                except Exception as e:
                    return _handle_exception(e)

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    return _apply_toon(result)
                except CircuitBreakerError as e:
                    return _handle_circuit_breaker(e)
                except Exception as e:
                    return _handle_exception(e)

            return wrapper

    return decorator
