"""MCP Server implementation for Anamnesis - Codebase Intelligence.

This module provides the FastMCP server with all intelligence, automation,
and monitoring tools for AI-assisted codebase understanding.
"""

import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from anamnesis.services import (
    CodebaseService,
    IntelligenceService,
    LearningOptions,
    LearningService,
    MemoryService,
    SessionManager,
)
from anamnesis.services.project_registry import ProjectContext, ProjectRegistry
from anamnesis.interfaces.search import SearchQuery, SearchType
from anamnesis.search.service import SearchService
from anamnesis.utils.circuit_breaker import CircuitBreakerError
from anamnesis.utils.error_classifier import classify_error
from anamnesis.utils.logger import logger
from anamnesis.utils.response_wrapper import ResponseWrapper, wrap_async_operation

# Create the MCP server instance
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
sessions. Use the `think_about_*` tools to reflect before, during, and
after complex tasks.

For multi-project workflows, use `activate_project` to switch between
projects. Each project gets isolated services, preventing cross-project
data contamination.
""",
)

# Project registry (replaces individual service globals)
# All per-project state is managed through ProjectContext instances.
_registry = ProjectRegistry()
_server_start_time: float = time.time()  # Track server uptime


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
   - Are there loose ends that need tying up?

2. **Quality**: Is the work at an acceptable standard?
   - Does the code follow project conventions?
   - Are there obvious bugs or edge cases?
   - Would you be confident in this code going to production?

3. **Communication**: Have you explained what you did?
   - Does the user understand the changes and why?
   - Are there caveats or limitations they should know about?
   - Is there follow-up work they should consider?

If not fully done, identify what remains. If done, summarize the outcome.
"""


def _with_error_handling(operation_name: str):
    """Decorator for MCP tool implementations with error handling.

    Catches CircuitBreakerError and other exceptions, returning
    ResponseWrapper-formatted error responses.

    Args:
        operation_name: Name of the operation for logging and error context.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CircuitBreakerError as e:
                logger.error(
                    f"Circuit breaker open for operation '{operation_name}'",
                    extra={
                        "operation": operation_name,
                        "circuit_state": "OPEN",
                        "details": str(e.details) if e.details else None,
                    },
                )
                return ResponseWrapper.failure_result(
                    e, operation=operation_name
                ).to_dict()
            except Exception as e:
                classification = classify_error(e, {"operation": operation_name})
                logger.error(
                    f"Error in operation '{operation_name}': {e}",
                    extra={
                        "operation": operation_name,
                        "category": classification.category.value,
                        "error_type": type(e).__name__,
                    },
                )
                return ResponseWrapper.failure_result(
                    e, operation=operation_name
                ).to_dict()

        return wrapper

    return decorator


# =============================================================================
# Intelligence Tool Implementations (testable functions)
# =============================================================================


@_with_error_handling("learn_codebase_intelligence")
def _learn_codebase_intelligence_impl(
    path: str,
    force: bool = False,
    max_files: int = 1000,
) -> dict:
    """Implementation for learn_codebase_intelligence tool."""
    # Activate path first so services belong to the correct project context
    _set_current_path(path)
    learning_service = _get_learning_service()
    intelligence_service = _get_intelligence_service()

    options = LearningOptions(
        force=force,
        max_files=max_files,
    )

    result = learning_service.learn_from_codebase(path, options)

    # Transfer learned data to intelligence service
    if result.success:
        learned_data = learning_service.get_learned_data(path)
        if learned_data:
            concepts = learned_data.get("concepts", [])
            patterns = learned_data.get("patterns", [])
            intelligence_service.load_concepts(concepts)
            intelligence_service.load_patterns(patterns)

    return result.to_dict()


@_with_error_handling("get_semantic_insights")
def _get_semantic_insights_impl(
    query: Optional[str] = None,
    concept_type: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """Implementation for get_semantic_insights tool."""
    intelligence_service = _get_intelligence_service()

    insights, total = intelligence_service.get_semantic_insights(
        query=query,
        concept_type=concept_type,
        limit=limit,
    )

    return {
        "insights": [i.to_dict() for i in insights],
        "total": total,
        "query": query,
        "concept_type": concept_type,
    }


@_with_error_handling("get_pattern_recommendations")
def _get_pattern_recommendations_impl(
    problem_description: str,
    current_file: Optional[str] = None,
    include_related_files: bool = False,
) -> dict:
    """Implementation for get_pattern_recommendations tool."""
    intelligence_service = _get_intelligence_service()

    recommendations, reasoning, related_files = intelligence_service.get_pattern_recommendations(
        problem_description=problem_description,
        current_file=current_file,
        include_related_files=include_related_files,
    )

    return {
        "recommendations": recommendations,
        "reasoning": reasoning,
        "related_files": related_files if include_related_files else [],
        "problem_description": problem_description,
    }


@_with_error_handling("predict_coding_approach")
def _predict_coding_approach_impl(
    problem_description: str,
    include_file_routing: bool = True,
) -> dict:
    """Implementation for predict_coding_approach tool."""
    intelligence_service = _get_intelligence_service()

    prediction = intelligence_service.predict_coding_approach(
        problem_description=problem_description,
    )

    result = prediction.to_dict()
    result["include_file_routing"] = include_file_routing

    return result


@_with_error_handling("get_developer_profile")
def _get_developer_profile_impl(
    include_recent_activity: bool = False,
    include_work_context: bool = False,
) -> dict:
    """Implementation for get_developer_profile tool."""
    intelligence_service = _get_intelligence_service()

    profile = intelligence_service.get_developer_profile(
        include_recent_activity=include_recent_activity,
        include_work_context=include_work_context,
        project_path=_get_current_path(),
    )

    return profile.to_dict()


@_with_error_handling("contribute_insights")
def _contribute_insights_impl(
    insight_type: str,
    content: dict,
    confidence: float,
    source_agent: str,
    session_update: Optional[dict] = None,
) -> dict:
    """Implementation for contribute_insights tool."""
    intelligence_service = _get_intelligence_service()

    success, insight_id, message = intelligence_service.contribute_insight(
        insight_type=insight_type,
        content=content,
        confidence=confidence,
        source_agent=source_agent,
        session_update=session_update,
    )

    return {
        "success": success,
        "insight_id": insight_id,
        "message": message,
    }


@_with_error_handling("get_project_blueprint")
def _get_project_blueprint_impl(
    path: Optional[str] = None,
    include_feature_map: bool = True,
) -> dict:
    """Implementation for get_project_blueprint tool."""
    intelligence_service = _get_intelligence_service()

    blueprint = intelligence_service.get_project_blueprint(
        path=path or _get_current_path(),
        include_feature_map=include_feature_map,
    )

    return blueprint


# =============================================================================
# Automation Tool Implementations
# =============================================================================


@_with_error_handling("auto_learn_if_needed")
def _auto_learn_if_needed_impl(
    path: Optional[str] = None,
    force: bool = False,
    include_progress: bool = True,
    include_setup_steps: bool = False,
    skip_learning: bool = False,
) -> dict:
    """Implementation for auto_learn_if_needed tool."""
    path = path or os.getcwd()
    resolved_path = str(Path(path).resolve())
    _set_current_path(resolved_path)

    learning_service = _get_learning_service()
    intelligence_service = _get_intelligence_service()

    # Check if learning is needed
    has_existing = learning_service.has_intelligence(resolved_path)

    if has_existing and not force:
        learned_data = learning_service.get_learned_data(resolved_path)
        concepts_count = len(learned_data.get("concepts", [])) if learned_data else 0
        patterns_count = len(learned_data.get("patterns", [])) if learned_data else 0

        return {
            "status": "already_learned",
            "message": "Intelligence data already exists for this codebase",
            "path": resolved_path,
            "concepts_count": concepts_count,
            "patterns_count": patterns_count,
            "action_taken": "none",
            "include_progress": include_progress,
        }

    if skip_learning:
        return {
            "status": "skipped",
            "message": "Learning skipped as requested",
            "path": resolved_path,
            "action_taken": "none",
        }

    # Perform learning
    options = LearningOptions(
        force=force,
        max_files=1000,
    )

    result = learning_service.learn_from_codebase(resolved_path, options)

    # Transfer to intelligence service
    if result.success:
        learned_data = learning_service.get_learned_data(resolved_path)
        if learned_data:
            intelligence_service.load_concepts(learned_data.get("concepts", []))
            intelligence_service.load_patterns(learned_data.get("patterns", []))

    response = {
        "status": "learned" if result.success else "failed",
        "message": "Successfully learned from codebase" if result.success else result.error,
        "path": resolved_path,
        "action_taken": "learn",
        "concepts_learned": result.concepts_learned,
        "patterns_learned": result.patterns_learned,
        "features_learned": result.features_learned,
        "time_elapsed_ms": result.time_elapsed_ms,
    }

    if include_progress:
        response["insights"] = result.insights

    if include_setup_steps:
        response["setup_steps"] = [
            "1. Analyzed codebase structure",
            "2. Extracted semantic concepts",
            "3. Discovered coding patterns",
            "4. Analyzed relationships",
            "5. Synthesized intelligence",
            "6. Built feature map",
            "7. Generated project blueprint",
        ]

    return response


# =============================================================================
# Monitoring Tool Implementations
# =============================================================================


@_with_error_handling("get_system_status")
def _get_system_status_impl(
    include_metrics: bool = True,
    include_diagnostics: bool = False,
) -> dict:
    """Implementation for get_system_status tool."""
    learning_service = _get_learning_service()
    current_path = _get_current_path()

    # Get learning status
    has_intelligence = learning_service.has_intelligence(current_path)
    learned_data = learning_service.get_learned_data(current_path) if has_intelligence else None

    concepts_count = len(learned_data.get("concepts", [])) if learned_data else 0
    patterns_count = len(learned_data.get("patterns", [])) if learned_data else 0
    learned_at = learned_data.get("learned_at") if learned_data else None

    status = {
        "status": "healthy",
        "current_path": current_path,
        "intelligence": {
            "has_data": has_intelligence,
            "concepts_count": concepts_count,
            "patterns_count": patterns_count,
            "learned_at": learned_at.isoformat() if learned_at else None,
        },
        "services": {
            "learning_service": "active",
            "intelligence_service": "active",
            "codebase_service": "active",
        },
    }

    if include_metrics:
        # Collect actual runtime metrics
        gc_stats = gc.get_stats()
        collected_total = sum(s.get("collected", 0) for s in gc_stats)

        # Get process memory info (platform-independent approximation)
        try:
            import resource
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Convert to MB (Linux returns KB, macOS returns bytes)
            mem_mb = mem_usage / 1024 if sys.platform != "darwin" else mem_usage / (1024 * 1024)
        except (ImportError, AttributeError):
            mem_mb = 0  # Windows doesn't have resource module

        status["metrics"] = {
            "memory_mb": round(mem_mb, 1),
            "python_objects": len(gc.get_objects()),
            "gc_collections": collected_total,
            "uptime_seconds": round(time.time() - _server_start_time, 1) if _server_start_time else 0,
        }

    if include_diagnostics:
        status["diagnostics"] = {
            "last_check": datetime.now().isoformat(),
            "issues": [],
        }

    return status


@_with_error_handling("get_intelligence_metrics")
def _get_intelligence_metrics_impl(
    include_breakdown: bool = True,
) -> dict:
    """Implementation for get_intelligence_metrics tool."""
    learning_service = _get_learning_service()
    current_path = _get_current_path()

    has_intelligence = learning_service.has_intelligence(current_path)
    learned_data = learning_service.get_learned_data(current_path) if has_intelligence else None

    concepts = learned_data.get("concepts", []) if learned_data else []
    patterns = learned_data.get("patterns", []) if learned_data else []

    metrics = {
        "total_concepts": len(concepts),
        "total_patterns": len(patterns),
        "has_intelligence": has_intelligence,
        "current_path": current_path,
    }

    if include_breakdown and learned_data:
        # Concept breakdown by type
        concept_types: dict[str, int] = {}
        for concept in concepts:
            ctype = concept.concept_type.value
            concept_types[ctype] = concept_types.get(ctype, 0) + 1

        # Pattern breakdown by type
        pattern_types: dict[str, int] = {}
        for pattern in patterns:
            ptype = pattern.pattern_type
            ptype_str = ptype.value if hasattr(ptype, "value") else str(ptype)
            pattern_types[ptype_str] = pattern_types.get(ptype_str, 0) + 1

        # Confidence distribution
        concept_confidences = [c.confidence for c in concepts]
        pattern_confidences = [p.confidence for p in patterns]

        metrics["breakdown"] = {
            "concepts_by_type": concept_types,
            "patterns_by_type": pattern_types,
            "avg_concept_confidence": sum(concept_confidences) / len(concept_confidences) if concept_confidences else 0,
            "avg_pattern_confidence": sum(pattern_confidences) / len(pattern_confidences) if pattern_confidences else 0,
        }

    return metrics


@_with_error_handling("get_performance_status")
def _get_performance_status_impl(
    run_benchmark: bool = False,
) -> dict:
    """Implementation for get_performance_status tool."""
    learning_service = _get_learning_service()
    current_path = _get_current_path()

    status = {
        "status": "healthy",
        "current_path": current_path,
        "metrics": {
            "learning_service": "operational",
            "intelligence_service": "operational",
            "codebase_service": "operational",
        },
    }

    if run_benchmark:
        # Simple benchmark
        start = datetime.now()
        learning_service.has_intelligence(current_path)
        elapsed = (datetime.now() - start).total_seconds() * 1000

        status["benchmark"] = {
            "intelligence_check_ms": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    return status


@_with_error_handling("health_check")
def _health_check_impl(
    path: Optional[str] = None,
) -> dict:
    """Implementation for health_check tool."""
    path = path or os.getcwd()
    resolved_path = Path(path).resolve()

    issues: list[str] = []
    checks: dict[str, bool] = {}

    # Check path exists
    checks["path_exists"] = resolved_path.exists()
    if not checks["path_exists"]:
        issues.append(f"Path does not exist: {resolved_path}")

    # Check path is directory
    checks["is_directory"] = resolved_path.is_dir() if checks["path_exists"] else False
    if checks["path_exists"] and not checks["is_directory"]:
        issues.append(f"Path is not a directory: {resolved_path}")

    # Check services are available
    try:
        _get_learning_service()
        checks["learning_service"] = True
    except Exception as e:
        checks["learning_service"] = False
        issues.append(f"Learning service error: {e}")

    try:
        _get_intelligence_service()
        checks["intelligence_service"] = True
    except Exception as e:
        checks["intelligence_service"] = False
        issues.append(f"Intelligence service error: {e}")

    try:
        _get_codebase_service()
        checks["codebase_service"] = True
    except Exception as e:
        checks["codebase_service"] = False
        issues.append(f"Codebase service error: {e}")

    # Check for existing intelligence
    learning_service = _get_learning_service()
    checks["has_intelligence"] = learning_service.has_intelligence(str(resolved_path))

    return {
        "healthy": len(issues) == 0,
        "path": str(resolved_path),
        "checks": checks,
        "issues": issues,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Search Tool Implementations
# =============================================================================


@_with_error_handling("search_codebase")
def _search_codebase_impl(
    query: str,
    search_type: str = "text",
    limit: int = 50,
    language: Optional[str] = None,
) -> dict:
    """Implementation for search_codebase tool.

    Routes to appropriate search backend based on search_type:
    - text: Simple substring matching (fast, always available)
    - pattern: Regex and AST structural patterns
    - semantic: Vector similarity search (requires indexing)
    """
    import asyncio

    current_path = _get_current_path()

    # Map string to SearchType enum
    type_map = {
        "text": SearchType.TEXT,
        "pattern": SearchType.PATTERN,
        "semantic": SearchType.SEMANTIC,
    }
    search_type_enum = type_map.get(search_type.lower(), SearchType.TEXT)

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Initialize semantic search lazily if requested
    if search_type_enum == SearchType.SEMANTIC:
        semantic_available = loop.run_until_complete(_ensure_semantic_search())
        if not semantic_available:
            logger.warning("Semantic search not available, falling back to text search")
            # Will fall back via SearchService logic

    # Get search service (may have been upgraded to async version)
    search_service = _get_search_service()

    # Build search query
    search_query = SearchQuery(
        query=query,
        search_type=search_type_enum,
        limit=limit,
        language=language,
    )

    # Execute search
    try:
        results = loop.run_until_complete(search_service.search(search_query))

        # Convert to response format
        return {
            "results": [
                {
                    "file": r.file_path,
                    "matches": r.matches,
                    "score": r.score,
                }
                for r in results
            ],
            "query": query,
            "search_type": search_type,
            "total": len(results),
            "path": current_path,
            "available_types": [t.value for t in search_service.get_available_backends()],
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        # Return empty results on error
        return {
            "results": [],
            "query": query,
            "search_type": search_type,
            "total": 0,
            "path": current_path,
            "error": str(e),
        }


@_with_error_handling("analyze_codebase")
def _analyze_codebase_impl(
    path: Optional[str] = None,
    include_file_content: bool = False,
) -> dict:
    """Implementation for analyze_codebase tool."""
    codebase_service = _get_codebase_service()
    path = path or _get_current_path()

    analysis_result = codebase_service.analyze_codebase(
        path=path,
        include_complexity=True,
        include_dependencies=True,
    )

    return {
        "path": path,
        "analysis": analysis_result.to_dict() if hasattr(analysis_result, "to_dict") else analysis_result,
        "include_file_content": include_file_content,
    }


# =============================================================================
# Session Tool Implementations
# =============================================================================


@_with_error_handling("start_session")
def _start_session_impl(
    name: str = "",
    feature: str = "",
    files: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
) -> dict:
    """Implementation for start_session tool."""
    session_manager = _get_session_manager()

    session = session_manager.start_session(
        name=name,
        feature=feature,
        files=files or [],
        tasks=tasks or [],
    )

    return {
        "success": True,
        "session": session.to_dict(),
        "message": f"Session '{session.session_id}' started",
    }


@_with_error_handling("end_session")
def _end_session_impl(
    session_id: Optional[str] = None,
) -> dict:
    """Implementation for end_session tool."""
    session_manager = _get_session_manager()

    target_id = session_id or session_manager.active_session_id
    if not target_id:
        return {
            "success": False,
            "message": "No active session to end",
        }

    success = session_manager.end_session(target_id)

    if success:
        ended_session = session_manager.get_session(target_id)
        return {
            "success": True,
            "session": ended_session.to_dict() if ended_session else None,
            "message": f"Session '{target_id}' ended",
        }
    else:
        return {
            "success": False,
            "message": f"Session '{target_id}' not found",
        }


@_with_error_handling("record_decision")
def _record_decision_impl(
    decision: str,
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Implementation for record_decision tool."""
    session_manager = _get_session_manager()

    decision_info = session_manager.record_decision(
        decision=decision,
        context=context,
        rationale=rationale,
        session_id=session_id,
        related_files=related_files,
        tags=tags,
    )

    return {
        "success": True,
        "decision": decision_info.to_dict(),
        "message": f"Decision '{decision_info.decision_id}' recorded",
    }


@_with_error_handling("get_session")
def _get_session_impl(
    session_id: Optional[str] = None,
) -> dict:
    """Implementation for get_session tool."""
    session_manager = _get_session_manager()

    target_id = session_id or session_manager.active_session_id
    if not target_id:
        return {
            "success": False,
            "session": None,
            "message": "No active session",
        }

    session = session_manager.get_session(target_id)
    if session:
        return {
            "success": True,
            "session": session.to_dict(),
        }
    else:
        return {
            "success": False,
            "session": None,
            "message": f"Session '{target_id}' not found",
        }


@_with_error_handling("list_sessions")
def _list_sessions_impl(
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """Implementation for list_sessions tool."""
    session_manager = _get_session_manager()

    if active_only:
        sessions = session_manager.get_active_sessions()
    else:
        sessions = session_manager.get_recent_sessions(limit=limit)

    return {
        "success": True,
        "sessions": [s.to_dict() for s in sessions],
        "count": len(sessions),
        "active_session_id": session_manager.active_session_id,
    }


@_with_error_handling("get_decisions")
def _get_decisions_impl(
    session_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Implementation for get_decisions tool."""
    session_manager = _get_session_manager()

    if session_id:
        decisions = session_manager.get_decisions_by_session(session_id)
    else:
        decisions = session_manager.get_recent_decisions(limit=limit)

    return {
        "success": True,
        "decisions": [d.to_dict() for d in decisions],
        "count": len(decisions),
    }


# =============================================================================
# Project Management Tool Implementations
# =============================================================================


@_with_error_handling("activate_project")
def _activate_project_impl(path: str) -> dict:
    """Implementation for activate_project tool."""
    ctx = _registry.activate(path)
    return {
        "success": True,
        "project": ctx.to_dict(),
    }


@_with_error_handling("get_current_config")
def _get_current_config_impl() -> dict:
    """Implementation for get_current_config tool."""
    return {
        "success": True,
        "registry": _registry.to_dict(),
    }


@_with_error_handling("list_projects")
def _list_projects_impl() -> dict:
    """Implementation for list_projects tool."""
    projects = _registry.list_projects()
    return {
        "success": True,
        "projects": [p.to_dict() for p in projects],
        "count": len(projects),
        "active_path": _registry.active_path,
    }


# =============================================================================
# Metacognition Tool Implementations
# =============================================================================


def _think_about_collected_information_impl() -> dict:
    """Implementation for think_about_collected_information tool."""
    return {
        "success": True,
        "prompt": _THINK_COLLECTED_PROMPT,
    }


def _think_about_task_adherence_impl() -> dict:
    """Implementation for think_about_task_adherence tool."""
    return {
        "success": True,
        "prompt": _THINK_TASK_ADHERENCE_PROMPT,
    }


def _think_about_whether_you_are_done_impl() -> dict:
    """Implementation for think_about_whether_you_are_done tool."""
    return {
        "success": True,
        "prompt": _THINK_DONE_PROMPT,
    }


# =============================================================================
# Memory Tool Implementations
# =============================================================================


@_with_error_handling("write_memory")
def _write_memory_impl(
    memory_file_name: str,
    content: str,
) -> dict:
    """Implementation for write_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.write_memory(memory_file_name, content)
    return {
        "success": True,
        "memory": result.to_dict(),
    }


@_with_error_handling("read_memory")
def _read_memory_impl(
    memory_file_name: str,
) -> dict:
    """Implementation for read_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.read_memory(memory_file_name)
    if result is None:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "memory": result.to_dict(),
    }


@_with_error_handling("list_memories")
def _list_memories_impl() -> dict:
    """Implementation for list_memories tool."""
    memory_service = _get_memory_service()
    memories = memory_service.list_memories()
    return {
        "success": True,
        "memories": [m.to_dict() for m in memories],
        "count": len(memories),
    }


@_with_error_handling("delete_memory")
def _delete_memory_impl(
    memory_file_name: str,
) -> dict:
    """Implementation for delete_memory tool."""
    memory_service = _get_memory_service()
    deleted = memory_service.delete_memory(memory_file_name)
    if not deleted:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "deleted": memory_file_name,
    }


@_with_error_handling("edit_memory")
def _edit_memory_impl(
    memory_file_name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Implementation for edit_memory tool."""
    memory_service = _get_memory_service()
    result = memory_service.edit_memory(memory_file_name, old_text, new_text)
    if result is None:
        return {
            "success": False,
            "error": f"Memory '{memory_file_name}' not found",
        }
    return {
        "success": True,
        "memory": result.to_dict(),
    }


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def learn_codebase_intelligence(
    path: str,
    force: bool = False,
    max_files: int = 1000,
) -> dict:
    """Build intelligence database from codebase (one-time setup, ~30-60s).

    Required before using predict_coding_approach, get_project_blueprint, or
    get_pattern_recommendations. Re-run with force=True if codebase has
    significant changes. Most users should use auto_learn_if_needed instead.

    Args:
        path: Path to the codebase to learn from
        force: Force re-learning even if codebase was previously analyzed
        max_files: Maximum number of files to analyze (default 1000)

    Returns:
        Learning result with concepts, patterns, and insights discovered
    """
    return _learn_codebase_intelligence_impl(path, force, max_files)


@mcp.tool
def get_semantic_insights(
    query: Optional[str] = None,
    concept_type: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """Search for code-level symbols by name and see relationships.

    Use this to find where a specific function/class is defined, how it's
    used, or what it depends on. Searches actual code identifiers (e.g.,
    "DatabaseConnection", "processRequest"), NOT business concepts.

    Args:
        query: Code identifier to search for (matches function/class/variable names)
        concept_type: Filter by concept type (class, function, interface, variable)
        limit: Maximum number of insights to return (default 50)

    Returns:
        List of semantic insights with relationships and usage patterns
    """
    return _get_semantic_insights_impl(query, concept_type, limit)


@mcp.tool
def get_pattern_recommendations(
    problem_description: str,
    current_file: Optional[str] = None,
    include_related_files: bool = False,
) -> dict:
    """Get coding pattern recommendations learned from this codebase.

    Use this when implementing new features to follow existing patterns
    (e.g., "create a new service class", "add API endpoint"). Returns
    patterns like Factory, Singleton, DependencyInjection with confidence
    scores and actual examples from your code.

    Args:
        problem_description: What you want to implement (e.g., "create a new service")
        current_file: Current file being worked on (optional)
        include_related_files: Include suggestions for related files

    Returns:
        Pattern recommendations with examples and related files
    """
    return _get_pattern_recommendations_impl(problem_description, current_file, include_related_files)


@mcp.tool
def predict_coding_approach(
    problem_description: str,
    include_file_routing: bool = True,
) -> dict:
    """Find which files to modify for a task using intelligent file routing.

    Use this when asked "where should I...", "what files...", or "how do I
    add/implement..." to route directly to relevant files without exploration.

    Args:
        problem_description: What the user wants to add/modify/implement
        include_file_routing: Include smart file routing (default True)

    Returns:
        Coding approach prediction with target files and reasoning
    """
    return _predict_coding_approach_impl(problem_description, include_file_routing)


@mcp.tool
def get_developer_profile(
    include_recent_activity: bool = False,
    include_work_context: bool = False,
) -> dict:
    """Get patterns and conventions learned from this codebase's code style.

    Shows frequently-used patterns (DI, Factory, etc.), naming conventions,
    and architectural preferences. Use this to understand "how we do things
    here" before writing new code.

    Args:
        include_recent_activity: Include recent coding activity patterns
        include_work_context: Include current work session context

    Returns:
        Developer profile with coding style and preferences
    """
    return _get_developer_profile_impl(include_recent_activity, include_work_context)


@mcp.tool
def contribute_insights(
    insight_type: str,
    content: dict,
    confidence: float,
    source_agent: str,
    session_update: Optional[dict] = None,
) -> dict:
    """Save AI-discovered insights back to Anamnesis for future reference.

    Use this when you discover a recurring pattern, potential bug, or
    refactoring opportunity that other agents/sessions should know about.

    Args:
        insight_type: Type of insight (bug_pattern, optimization, refactor_suggestion, best_practice)
        content: The insight details as a structured object
        confidence: Confidence score (0.0 to 1.0)
        source_agent: Identifier of the AI agent contributing

    Returns:
        Result with insight_id and success status
    """
    return _contribute_insights_impl(insight_type, content, confidence, source_agent, session_update)


@mcp.tool
def get_project_blueprint(
    path: Optional[str] = None,
    include_feature_map: bool = True,
) -> dict:
    """Get instant project blueprint - eliminates cold start exploration.

    Provides tech stack, entry points, key directories, and architecture
    overview for quick project understanding.

    Args:
        path: Path to the project (defaults to current working directory)
        include_feature_map: Include feature-to-file mapping

    Returns:
        Project blueprint with tech stack and architecture
    """
    return _get_project_blueprint_impl(path, include_feature_map)


@mcp.tool
def auto_learn_if_needed(
    path: Optional[str] = None,
    force: bool = False,
    include_progress: bool = True,
    include_setup_steps: bool = False,
    skip_learning: bool = False,
) -> dict:
    """Automatically learn from codebase if intelligence data is missing or stale.

    Call this first before using other Anamnesis tools - it's a no-op if data
    already exists. Includes project setup and verification. Perfect for
    seamless agent integration.

    Args:
        path: Path to the codebase directory (defaults to current working directory)
        force: Force re-learning even if data exists
        include_progress: Include detailed progress information
        include_setup_steps: Include detailed setup verification steps
        skip_learning: Skip the learning phase for faster setup

    Returns:
        Status with learning results or existing data information
    """
    return _auto_learn_if_needed_impl(path, force, include_progress, include_setup_steps, skip_learning)


@mcp.tool
def get_system_status(
    include_metrics: bool = True,
    include_diagnostics: bool = False,
) -> dict:
    """Get comprehensive system status including intelligence data and health.

    Args:
        include_metrics: Include detailed performance metrics
        include_diagnostics: Include system diagnostics

    Returns:
        System status with health indicators
    """
    return _get_system_status_impl(include_metrics, include_diagnostics)


@mcp.tool
def get_intelligence_metrics(
    include_breakdown: bool = True,
) -> dict:
    """Get detailed metrics about the intelligence database and learning state.

    Args:
        include_breakdown: Include detailed breakdown by type and confidence

    Returns:
        Intelligence metrics with database statistics
    """
    return _get_intelligence_metrics_impl(include_breakdown)


@mcp.tool
def get_performance_status(
    run_benchmark: bool = False,
) -> dict:
    """Get performance metrics including database size and query times.

    Args:
        run_benchmark: Run a quick performance benchmark

    Returns:
        Performance status with metrics
    """
    return _get_performance_status_impl(run_benchmark)


@mcp.tool
def health_check(
    path: Optional[str] = None,
) -> dict:
    """Verify Anamnesis setup and configuration for a project.

    Checks database accessibility, project structure, and service health.

    Args:
        path: Project path to check (defaults to current directory)

    Returns:
        Health check results with any issues found
    """
    return _health_check_impl(path)


@mcp.tool
def search_codebase(
    query: str,
    search_type: str = "text",
    limit: int = 50,
    language: Optional[str] = None,
) -> dict:
    """Search for code by text matching or patterns.

    Use "text" type for finding specific strings/keywords in code.
    Use "pattern" type for regex/AST patterns.

    Args:
        query: Search query - literal text string or regex pattern
        search_type: Type of search ("text", "pattern", "semantic")
        limit: Maximum number of results (default 50)
        language: Filter results by programming language

    Returns:
        Search results with file paths and matched content
    """
    return _search_codebase_impl(query, search_type, limit, language)


@mcp.tool
def analyze_codebase(
    path: Optional[str] = None,
    include_file_content: bool = False,
) -> dict:
    """One-time analysis of a specific file or directory.

    Returns AST structure, complexity metrics, and detected patterns.
    For project-wide understanding, use get_project_blueprint instead.

    Args:
        path: Path to file or directory to analyze
        include_file_content: Include full file content in response

    Returns:
        Analysis results with structure and metrics
    """
    return _analyze_codebase_impl(path, include_file_content)


# =============================================================================
# Session Tool Registrations
# =============================================================================


@mcp.tool
def start_session(
    name: str = "",
    feature: str = "",
    files: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
) -> dict:
    """Start a new work session to track development context.

    Use this to begin tracking a focused piece of work. Sessions help
    organize decisions, files, and tasks related to a specific feature
    or bug fix.

    Args:
        name: Name or description of the session
        feature: Feature being worked on (e.g., "authentication", "search")
        files: Initial list of files being worked on
        tasks: Initial list of tasks to complete

    Returns:
        Session info with session_id and status
    """
    return _start_session_impl(name, feature, files, tasks)


@mcp.tool
def end_session(
    session_id: Optional[str] = None,
) -> dict:
    """End a work session.

    Marks the session as completed and records the end time.
    If no session_id is provided, ends the currently active session.

    Args:
        session_id: Session ID to end (optional, defaults to active session)

    Returns:
        Result with ended session info
    """
    return _end_session_impl(session_id)


@mcp.tool
def record_decision(
    decision: str,
    context: str = "",
    rationale: str = "",
    session_id: Optional[str] = None,
    related_files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Record a project decision for future reference.

    Use this to capture important decisions made during development,
    including the reasoning and context. Decisions can be linked to
    sessions or recorded independently.

    Args:
        decision: The decision made (e.g., "Use JWT for authentication")
        context: Context for the decision (e.g., "API design discussion")
        rationale: Why this decision was made
        session_id: Session to link to (optional, defaults to active session)
        related_files: Files related to the decision
        tags: Tags for categorization (e.g., ["security", "api"])

    Returns:
        Decision info with decision_id
    """
    return _record_decision_impl(decision, context, rationale, session_id, related_files, tags)


@mcp.tool
def get_session(
    session_id: Optional[str] = None,
) -> dict:
    """Get information about a work session.

    Retrieves session details including files, tasks, and decision count.
    If no session_id is provided, returns the currently active session.

    Args:
        session_id: Session ID to retrieve (optional, defaults to active session)

    Returns:
        Session info or error if not found
    """
    return _get_session_impl(session_id)


@mcp.tool
def list_sessions(
    active_only: bool = False,
    limit: int = 10,
) -> dict:
    """List work sessions.

    Get a list of recent sessions or only active (non-ended) sessions.

    Args:
        active_only: Only return active sessions (default False)
        limit: Maximum number of sessions to return (default 10)

    Returns:
        List of sessions with count and active session ID
    """
    return _list_sessions_impl(active_only, limit)


@mcp.tool
def get_decisions(
    session_id: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Get project decisions.

    Retrieve decisions for a specific session or recent decisions across
    all sessions.

    Args:
        session_id: Filter by session ID (optional)
        limit: Maximum number of decisions to return (default 10)

    Returns:
        List of decisions with count
    """
    return _get_decisions_impl(session_id, limit)


# =============================================================================
# Project Management Tool Registrations
# =============================================================================


@mcp.tool
def activate_project(
    path: str,
) -> dict:
    """Activate a project by path for multi-project workflows.

    Switches the active project context. All subsequent tool calls will
    use services scoped to this project. Previously activated projects
    retain their cached state (learning data, sessions, etc.).

    If no project is explicitly activated, the current working directory
    is used automatically.

    Args:
        path: Path to the project root directory

    Returns:
        Project context with active services status
    """
    return _activate_project_impl(path)


@mcp.tool
def get_current_config() -> dict:
    """Get the current project configuration and registry state.

    Shows the active project, all known projects, and which services
    are initialized for each.

    Returns:
        Registry state with project details
    """
    return _get_current_config_impl()


@mcp.tool
def list_projects() -> dict:
    """List all known projects in the registry.

    Shows projects that have been activated during this server session,
    sorted by most recently activated first.

    Returns:
        List of projects with their service status
    """
    return _list_projects_impl()


# =============================================================================
# Metacognition Tool Registrations
# =============================================================================


@mcp.tool
def think_about_collected_information() -> dict:
    """Think about the collected information and whether it is sufficient and relevant.

    This tool should be called after completing a sequence of search and
    exploration steps (find_symbol, search_codebase, etc.) to reflect on
    whether you have enough information to proceed.

    Returns:
        A reflective prompt to guide your thinking
    """
    return _think_about_collected_information_impl()


@mcp.tool
def think_about_task_adherence() -> dict:
    """Think about the task at hand and whether you are still on track.

    Especially important during long conversations with significant back
    and forth. Call this before making code changes to ensure you haven't
    drifted from the original goal.

    Returns:
        A reflective prompt to guide your thinking
    """
    return _think_about_task_adherence_impl()


@mcp.tool
def think_about_whether_you_are_done() -> dict:
    """Think about whether you are truly done with the task.

    Call this when you believe you've completed the user's request.
    Helps verify completeness, quality, and communication before
    declaring the work finished.

    Returns:
        A reflective prompt to guide your thinking
    """
    return _think_about_whether_you_are_done_impl()


# =============================================================================
# Memory Tool Registrations
# =============================================================================


@mcp.tool
def write_memory(
    memory_file_name: str,
    content: str,
) -> dict:
    """Write information about this project that can be useful for future tasks.

    Stores a markdown file in `.anamnesis/memories/` within the project root.
    The memory name should be meaningful and descriptive.

    Args:
        memory_file_name: Name for the memory (e.g., "architecture-decisions",
            "api-patterns"). Letters, numbers, hyphens, underscores, dots only.
        content: The content to write (markdown format recommended)

    Returns:
        Result with the written memory details
    """
    return _write_memory_impl(memory_file_name, content)


@mcp.tool
def read_memory(
    memory_file_name: str,
) -> dict:
    """Read the content of a memory file.

    Use this to retrieve previously stored project knowledge. Only read
    memories that are relevant to the current task.

    Args:
        memory_file_name: Name of the memory to read

    Returns:
        Memory content and metadata, or error if not found
    """
    return _read_memory_impl(memory_file_name)


@mcp.tool
def list_memories() -> dict:
    """List available memories for this project.

    Returns names and metadata for all stored memories. Use this to
    discover what project knowledge is available before reading specific
    memories.

    Returns:
        List of memory entries with names, sizes, and timestamps
    """
    return _list_memories_impl()


@mcp.tool
def delete_memory(
    memory_file_name: str,
) -> dict:
    """Delete a memory file.

    Remove a memory that is no longer relevant or correct.

    Args:
        memory_file_name: Name of the memory to delete

    Returns:
        Success status
    """
    return _delete_memory_impl(memory_file_name)


@mcp.tool
def edit_memory(
    memory_file_name: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Edit an existing memory by replacing text.

    Use this to update specific parts of a memory without rewriting
    the entire content.

    Args:
        memory_file_name: Name of the memory to edit
        old_text: The exact text to find and replace
        new_text: The replacement text

    Returns:
        Updated memory content and metadata
    """
    return _edit_memory_impl(memory_file_name, old_text, new_text)


# =============================================================================
# Server Factory
# =============================================================================


def create_server() -> FastMCP:
    """Create and return the configured MCP server instance.

    Returns:
        Configured FastMCP server with all tools registered
    """
    return mcp


# Allow running directly
if __name__ == "__main__":
    mcp.run()
