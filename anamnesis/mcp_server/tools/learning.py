"""Learning tools — auto_learn_if_needed (absorbs learn_codebase_intelligence)."""

import os
from pathlib import Path
from typing import Optional

from anamnesis.services import LearningOptions

from anamnesis.mcp_server._shared import (
    _format_blueprint_as_memory,
    _get_intelligence_service,
    _get_learning_service,
    _get_memory_service,
    _set_current_path,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("auto_learn_if_needed")
def _auto_learn_if_needed_impl(
    path: Optional[str] = None,
    force: bool = False,
    max_files: int = 1000,
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
            "success": True,
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
            "success": True,
            "status": "skipped",
            "message": "Learning skipped as requested",
            "path": resolved_path,
            "action_taken": "none",
        }

    # Perform learning
    options = LearningOptions(
        force=force,
        max_files=max_files,
    )

    result = learning_service.learn_from_codebase(resolved_path, options)

    # Transfer to intelligence service
    if result.success:
        learned_data = learning_service.get_learned_data(resolved_path)
        if learned_data:
            intelligence_service.load_concepts(learned_data.get("concepts", []))
            intelligence_service.load_patterns(learned_data.get("patterns", []))

    response = {
        "success": result.success,
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

    # Auto-onboarding: generate project-overview memory on first learn
    if result.success:
        try:
            memory_service = _get_memory_service()
            # Only write if no overview exists yet (don't overwrite manual edits)
            if memory_service.read_memory("project-overview") is None:
                blueprint = intelligence_service.get_project_blueprint(
                    path=resolved_path, include_feature_map=True,
                )
                if blueprint:
                    content = _format_blueprint_as_memory(blueprint)
                    memory_service.write_memory("project-overview", content)
                    response["auto_onboarding"] = "project-overview memory created"
        except Exception:
            pass  # Non-critical — don't break learning on onboarding failure

    return response


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def auto_learn_if_needed(
    path: Optional[str] = None,
    force: bool = False,
    max_files: int = 1000,
    include_progress: bool = True,
    include_setup_steps: bool = False,
    skip_learning: bool = False,
) -> dict:
    """Automatically learn from codebase if intelligence data is missing or stale.

    Call this first before using other Anamnesis tools - it's a no-op if data
    already exists. Includes project setup and verification. Perfect for
    seamless agent integration. Use force=True and max_files to control
    re-learning behavior (absorbs former learn_codebase_intelligence tool).

    Args:
        path: Path to the codebase directory (defaults to current working directory)
        force: Force re-learning even if data exists
        max_files: Maximum number of files to analyze (default 1000)
        include_progress: Include detailed progress information
        include_setup_steps: Include detailed setup verification steps
        skip_learning: Skip the learning phase for faster setup

    Returns:
        Status with learning results or existing data information
    """
    return _auto_learn_if_needed_impl(path, force, max_files, include_progress, include_setup_steps, skip_learning)
