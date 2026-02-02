"""Intelligence tools — semantic insights, patterns, profiles, blueprints."""

from typing import Optional

from anamnesis.mcp_server._shared import (
    _get_current_path,
    _get_intelligence_service,
    _with_error_handling,
    mcp,
)


# =============================================================================
# Implementations
# =============================================================================


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
# MCP Tool Registrations
# =============================================================================


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
