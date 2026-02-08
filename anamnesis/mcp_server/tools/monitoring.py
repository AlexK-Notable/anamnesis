"""Monitoring tools — unified system status dashboard."""

import gc
import sys
import time
from pathlib import Path
from typing import Optional

from anamnesis.constants import utcnow
from anamnesis.mcp_server._shared import (
    _get_codebase_service,
    _get_current_path,
    _get_intelligence_service,
    _get_learning_service,
    _sanitize_error_message,
    _server_start_time,
    _with_error_handling,
    mcp,
)
from anamnesis.utils.helpers import enum_value


# =============================================================================
# Implementations
# =============================================================================


@_with_error_handling("get_system_status")
def _get_system_status_impl(
    sections: str = "summary,metrics",
    path: Optional[str] = None,
    include_breakdown: bool = True,
    run_benchmark: bool = False,
) -> dict:
    """Consolidated monitoring dashboard (was 4 tools: get_system_status,
    get_intelligence_metrics, get_performance_status, health_check).

    Sections: summary, metrics, intelligence, performance, health (comma-separated or 'all').
    """
    requested = {s.strip() for s in sections.split(",")} if sections != "all" else {
        "summary", "metrics", "intelligence", "performance", "health"
    }

    learning_service = _get_learning_service()
    current_path = path or _get_current_path()
    has_intel = learning_service.has_intelligence(current_path)
    learned_data = learning_service.get_learned_data(current_path) if has_intel else None

    result: dict = {"success": True, "current_path": current_path}

    # --- summary section ---
    if "summary" in requested:
        concepts_count = len(learned_data.get("concepts", [])) if learned_data else 0
        patterns_count = len(learned_data.get("patterns", [])) if learned_data else 0
        learned_at = learned_data.get("learned_at") if learned_data else None
        result["summary"] = {
            "status": "healthy",
            "intelligence": {
                "has_data": has_intel,
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

    # --- metrics section (runtime metrics) ---
    if "metrics" in requested:
        gc_stats = gc.get_stats()
        collected_total = sum(s.get("collected", 0) for s in gc_stats)
        try:
            import resource
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_mb = mem_usage / 1024 if sys.platform != "darwin" else mem_usage / (1024 * 1024)
        except (ImportError, AttributeError):
            mem_mb = 0
        from anamnesis.utils.model_registry import get_model_cache_stats
        result["metrics"] = {
            "memory_mb": round(mem_mb, 1),
            "python_objects": len(gc.get_objects()),
            "gc_collections": collected_total,
            "uptime_seconds": round(time.time() - _server_start_time, 1) if _server_start_time else 0,
            "model_cache": get_model_cache_stats(),
        }

    # --- intelligence section (detailed breakdown) ---
    if "intelligence" in requested:
        concepts = learned_data.get("concepts", []) if learned_data else []
        patterns = learned_data.get("patterns", []) if learned_data else []
        intel_data: dict = {
            "total_concepts": len(concepts),
            "total_patterns": len(patterns),
            "has_intelligence": has_intel,
        }
        if include_breakdown and learned_data:
            concept_types: dict[str, int] = {}
            for concept in concepts:
                ctype = concept.concept_type.value
                concept_types[ctype] = concept_types.get(ctype, 0) + 1
            pattern_types: dict[str, int] = {}
            for pattern in patterns:
                ptype = pattern.pattern_type
                ptype_str = enum_value(ptype)
                pattern_types[ptype_str] = pattern_types.get(ptype_str, 0) + 1
            concept_confidences = [c.confidence for c in concepts]
            pattern_confidences = [p.confidence for p in patterns]
            intel_data["breakdown"] = {
                "concepts_by_type": concept_types,
                "patterns_by_type": pattern_types,
                "avg_concept_confidence": sum(concept_confidences) / len(concept_confidences) if concept_confidences else 0,
                "avg_pattern_confidence": sum(pattern_confidences) / len(pattern_confidences) if pattern_confidences else 0,
            }
        result["intelligence"] = intel_data

    # --- performance section ---
    if "performance" in requested:
        perf: dict = {
            "services": {
                "learning_service": "operational",
                "intelligence_service": "operational",
                "codebase_service": "operational",
            },
        }
        if run_benchmark:
            start = utcnow()
            learning_service.has_intelligence(current_path)
            elapsed = (utcnow() - start).total_seconds() * 1000
            perf["benchmark"] = {
                "intelligence_check_ms": elapsed,
                "timestamp": utcnow().isoformat(),
            }
        result["performance"] = perf

    # --- health section ---
    if "health" in requested:
        resolved_path = Path(current_path).resolve()
        issues: list[str] = []
        checks: dict[str, bool] = {}
        checks["path_exists"] = resolved_path.exists()
        if not checks["path_exists"]:
            issues.append(_sanitize_error_message(f"Path does not exist: {resolved_path}"))
        checks["is_directory"] = resolved_path.is_dir() if checks["path_exists"] else False
        if checks["path_exists"] and not checks["is_directory"]:
            issues.append(_sanitize_error_message(f"Path is not a directory: {resolved_path}"))
        try:
            _get_learning_service()
            checks["learning_service"] = True
        except Exception as e:
            checks["learning_service"] = False
            issues.append(f"Learning service error: {_sanitize_error_message(str(e))}")
        try:
            _get_intelligence_service()
            checks["intelligence_service"] = True
        except Exception as e:
            checks["intelligence_service"] = False
            issues.append(f"Intelligence service error: {_sanitize_error_message(str(e))}")
        try:
            _get_codebase_service()
            checks["codebase_service"] = True
        except Exception as e:
            checks["codebase_service"] = False
            issues.append(f"Codebase service error: {_sanitize_error_message(str(e))}")
        checks["has_intelligence"] = has_intel
        result["health"] = {
            "healthy": len(issues) == 0,
            "checks": checks,
            "issues": issues,
            "timestamp": utcnow().isoformat(),
        }

    return result


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool
def get_system_status(
    sections: str = "summary,metrics",
    path: Optional[str] = None,
    include_breakdown: bool = True,
    run_benchmark: bool = False,
) -> dict:
    """Get comprehensive system status including intelligence data, performance, and health.

    This is a unified monitoring dashboard. Use the `sections` parameter to
    select which information to include.

    Args:
        sections: Comma-separated sections to include:
            - "summary": Note counts, service status, intelligence overview
            - "metrics": Runtime metrics (memory, GC, uptime)
            - "intelligence": Detailed concept/pattern breakdown
            - "performance": Service health, optional benchmark
            - "health": Path validation, service checks, issue list
            - "all": Include all sections (default: "summary,metrics")
        path: Project path (defaults to current directory)
        include_breakdown: Include concept/pattern type breakdown in intelligence section
        run_benchmark: Run a quick performance benchmark in performance section

    Returns:
        System status with requested sections
    """
    return _get_system_status_impl(sections, path, include_breakdown, run_benchmark)
