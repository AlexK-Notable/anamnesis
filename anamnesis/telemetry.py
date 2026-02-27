"""Tool usage telemetry logging.

Writes structured JSONL records for every MCP tool invocation — tool name,
truncated args, duration, success/failure, correlation ID, and project path.
Output goes to ``.anamnesis/tool_usage.jsonl`` under the active project root.

Disable entirely with ``ANAMNESIS_TELEMETRY=false`` env var.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from anamnesis.utils.logger import get_correlation_id, logger

# Max chars per argument value in the log entry.
_DEFAULT_MAX_ARG_LEN = 200

# Argument names whose values are especially likely to be large/sensitive.
_TRUNCATED_ARG_NAMES = frozenset({"content", "old_text", "new_text", "source", "body"})


def _telemetry_enabled() -> bool:
    """Check whether telemetry is enabled (default: true)."""
    return os.environ.get("ANAMNESIS_TELEMETRY", "true").lower() not in ("false", "0", "no")


def _truncate_args(args: dict, max_len: int = _DEFAULT_MAX_ARG_LEN) -> dict:
    """Return a shallow copy of *args* with string values truncated.

    Values whose key is in ``_TRUNCATED_ARG_NAMES`` get a ``[truncated]``
    annotation when shortened.  Other string values are simply clipped.
    """
    out: dict = {}
    for key, value in args.items():
        if not isinstance(value, str):
            out[key] = value
            continue
        if len(value) <= max_len:
            out[key] = value
            continue
        snippet = value[:max_len]
        if key in _TRUNCATED_ARG_NAMES:
            out[key] = snippet + "... [truncated]"
        else:
            out[key] = snippet + "..."
    return out


class ToolUsageLogger:
    """Append-only JSONL writer for tool invocation records."""

    def __init__(self, log_dir: Path) -> None:
        self._log_path = log_dir / "tool_usage.jsonl"

    def log_call(
        self,
        tool: str,
        args: dict,
        duration_ms: float,
        success: bool,
        error: str | None = None,
        project: str | None = None,
    ) -> None:
        """Write one tool invocation record to the JSONL file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": get_correlation_id(),
            "tool": tool,
            "args": _truncate_args(args),
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "error": error,
            "project": project,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write telemetry record for %s", tool, exc_info=True)


# ── Factory ──────────────────────────────────────────────────────────────

# Cache one logger per project path to avoid reopening / re-resolving.
_loggers: dict[str, ToolUsageLogger] = {}


def get_telemetry_logger(project_path: Path) -> ToolUsageLogger:
    """Return (or create) a ``ToolUsageLogger`` for *project_path*."""
    key = str(project_path)
    cached = _loggers.get(key)
    if cached is not None:
        return cached
    tl = ToolUsageLogger(project_path / ".anamnesis")
    _loggers[key] = tl
    return tl


def log_tool_call(
    tool: str,
    args: dict,
    duration_ms: float,
    success: bool,
    error: str | None = None,
    project_path: str | None = None,
) -> None:
    """Convenience wrapper: log a tool call if telemetry is enabled."""
    if not _telemetry_enabled():
        return
    if project_path is None:
        return  # No project active — nowhere to write.
    tl = get_telemetry_logger(Path(project_path))
    tl.log_call(tool, args, duration_ms, success, error, project_path)
