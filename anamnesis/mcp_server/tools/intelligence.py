"""Intelligence tools — concepts, coding guidance, profiles, project analysis."""

from pathlib import Path
from typing import Literal

from anamnesis.utils.security import (
    MAX_NAME_LENGTH,
    MAX_QUERY_LENGTH,
    clamp_integer,
    validate_string_length,
)

from anamnesis.mcp_server._shared import (
    _failure_response,
    _get_codebase_service,
    _get_current_path,
    _get_intelligence_service,
    _success_response,
    _with_error_handling,
    mcp,
)

_VALID_INSIGHT_TYPES = frozenset(
    {"bug_pattern", "optimization", "refactor_suggestion", "best_practice"}
)


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("manage_concepts")
def _manage_concepts_impl(
    action: str = "query",
    query: str | None = None,
    concept_type: str | None = None,
    limit: int = 50,
    insight_type: str | None = None,
    content: dict | None = None,
    confidence: float | None = None,
    source_agent: str | None = None,
    session_update: dict | None = None,
) -> dict:
    """Implementation for manage_concepts tool.

    action="query": query learned concepts.
    action="contribute": contribute an AI-discovered insight.
    """
    if action == "query":
        if query is not None:
            validate_string_length(query, "query", max_length=MAX_QUERY_LENGTH)
        limit = clamp_integer(limit, "limit", 1, 500)
        intelligence_service = _get_intelligence_service()

        insights, total = intelligence_service.get_semantic_insights(
            query=query,
            concept_type=concept_type,
            limit=limit,
        )

        return _success_response(
            [i.to_dict() for i in insights],
            total=total,
            query=query,
            concept_type=concept_type,
        )
    elif action == "contribute":
        if (
            not insight_type
            or content is None
            or confidence is None
            or not source_agent
        ):
            return _failure_response(
                "action='contribute' requires insight_type, content, "
                "confidence, and source_agent"
            )
        validate_string_length(
            source_agent, "source_agent", min_length=1, max_length=MAX_NAME_LENGTH
        )
        confidence = max(0.0, min(confidence, 1.0))

        if insight_type not in _VALID_INSIGHT_TYPES:
            return _failure_response(
                f"Unknown insight_type '{insight_type}'. "
                f"Choose from: {', '.join(sorted(_VALID_INSIGHT_TYPES))}"
            )
        intelligence_service = _get_intelligence_service()

        success, insight_id, message = intelligence_service.contribute_insight(
            insight_type=insight_type,
            content=content,
            confidence=confidence,
            source_agent=source_agent,
            session_update=session_update,
        )

        if not success:
            return _failure_response(message, insight_id=insight_id)
        return _success_response({"insight_id": insight_id}, message=message)
    else:
        return _failure_response(
            f"Unknown action '{action}'. Choose from: query, contribute"
        )


@_with_error_handling("get_coding_guidance")
def _get_coding_guidance_impl(
    problem_description: str,
    relative_path: str | None = None,
    include_patterns: bool = True,
    include_file_routing: bool = True,
    include_related_files: bool = False,
) -> dict:
    """Implementation for get_coding_guidance tool.

    Combines pattern recommendations and file routing into a single response.
    """
    validate_string_length(
        problem_description,
        "problem_description",
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
    )
    intelligence_service = _get_intelligence_service()
    data: dict = {}

    if include_patterns:
        recommendations, reasoning, related_files = (
            intelligence_service.get_pattern_recommendations(
                problem_description=problem_description,
                current_file=relative_path,
                include_related_files=include_related_files,
            )
        )
        data["recommendations"] = recommendations
        data["reasoning"] = reasoning
        data["related_files"] = related_files if include_related_files else []

    if include_file_routing:
        prediction = intelligence_service.predict_coding_approach(
            problem_description=problem_description,
        )
        data["prediction"] = prediction.to_dict()

    return _success_response(data, problem_description=problem_description)


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

    return _success_response(profile.to_dict())


@_with_error_handling("analyze_project")
def _analyze_project_impl(
    path: str | None = None,
    scope: str = "project",
    include_feature_map: bool = True,
    include_file_content: bool = False,
) -> dict:
    """Implementation for analyze_project tool.

    scope="project": returns project blueprint.
    scope="file": returns AST structure, complexity metrics, detected patterns.
    """
    if scope == "project":
        intelligence_service = _get_intelligence_service()
        blueprint = intelligence_service.get_project_blueprint(
            path=path or _get_current_path(),
            include_feature_map=include_feature_map,
        )
        return _success_response(
            {"blueprint": dict(blueprint) if blueprint else {}},
        )
    elif scope == "file":
        codebase_service = _get_codebase_service()
        path = path or _get_current_path()

        analysis_result = codebase_service.analyze_codebase(
            path=path,
            include_complexity=True,
            include_dependencies=True,
        )

        data: dict = {
            "analysis": (
                analysis_result.to_dict()
                if hasattr(analysis_result, "to_dict")
                else analysis_result
            ),
        }

        file_contents = getattr(analysis_result, "file_contents", None)
        if include_file_content and file_contents is not None:
            data["file_contents"] = file_contents
        elif include_file_content:
            target = Path(path).resolve()
            project_root = _get_current_path()
            fc = codebase_service.read_file_content(
                str(target), project_root, max_size=50000
            )
            if fc is not None:
                data["file_contents"] = {str(target): fc}
            else:
                data["file_contents"] = {}

        return _success_response(data, path=path)
    else:
        return _failure_response(
            f"Unknown scope '{scope}'. Choose from: project, file"
        )


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def manage_concepts(
    action: Literal["query", "contribute"] = "query",
    query: str | None = None,
    concept_type: str | None = None,
    limit: int = 50,
    insight_type: str | None = None,
    content: dict | None = None,
    confidence: float | None = None,
    source_agent: str | None = None,
    session_update: dict | None = None,
) -> dict:
    """Query learned concepts or contribute AI-discovered insights.

    Use action="query" to search concepts (functions, classes, variables)
    extracted during auto_learn_if_needed. Unlike search_codebase which
    searches raw file contents, this queries the structured concept index.

    Use action="contribute" to save insights (bug patterns, optimizations,
    refactoring suggestions) back to Anamnesis for future reference.

    Args:
        action: "query" to search concepts, "contribute" to save insights
        query: Code identifier to search for (query mode)
        concept_type: Filter by type: class, function, interface, variable (query mode)
        limit: Maximum results (default 50, query mode)
        insight_type: bug_pattern, optimization, refactor_suggestion, best_practice (contribute mode)
        content: Insight details as structured object (contribute mode)
        confidence: Confidence score 0.0-1.0 (contribute mode)
        source_agent: AI agent identifier (contribute mode)
        session_update: Optional session context (contribute mode)

    Returns:
        Concept list (query) or insight_id confirmation (contribute)
    """
    return _manage_concepts_impl(
        action,
        query,
        concept_type,
        limit,
        insight_type,
        content,
        confidence,
        source_agent,
        session_update,
    )


@mcp.tool
def get_coding_guidance(
    problem_description: str,
    relative_path: str | None = None,
    include_patterns: bool = True,
    include_file_routing: bool = True,
    include_related_files: bool = False,
) -> dict:
    """Get coding pattern recommendations and file routing for a task.

    Combines two intelligence capabilities:
    - Pattern recommendations: existing patterns (Factory, Singleton, DI)
      with confidence scores and code examples from your codebase
    - File routing: which files to modify, predicted approach and reasoning

    Use this when implementing new features or asked "where should I..."
    or "how do I add/implement..." questions.

    Args:
        problem_description: What you want to implement
        relative_path: Current file being worked on (optional)
        include_patterns: Include pattern recommendations (default True)
        include_file_routing: Include smart file routing (default True)
        include_related_files: Include suggestions for related files

    Returns:
        Pattern recommendations, file routing predictions, and reasoning
    """
    return _get_coding_guidance_impl(
        problem_description,
        relative_path,
        include_patterns,
        include_file_routing,
        include_related_files,
    )


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
    return _get_developer_profile_impl(
        include_recent_activity, include_work_context
    )


@mcp.tool
def analyze_project(
    path: str | None = None,
    scope: Literal["project", "file"] = "project",
    include_feature_map: bool = True,
    include_file_content: bool = False,
) -> dict:
    """Analyze a project or file for structure, complexity, and patterns.

    Use scope="project" for an instant project blueprint that eliminates
    cold-start exploration: tech stack, entry points, key directories,
    and architecture overview.

    Use scope="file" for one-time analysis of a specific file or directory:
    AST structure, complexity metrics, and detected patterns.

    Args:
        path: Path to project or file (defaults to current working directory)
        scope: "project" for blueprint overview, "file" for detailed analysis
        include_feature_map: Include feature-to-file mapping (project scope)
        include_file_content: Include full file content (file scope)

    Returns:
        Project blueprint (project scope) or analysis results (file scope)
    """
    return _analyze_project_impl(
        path, scope, include_feature_map, include_file_content
    )
