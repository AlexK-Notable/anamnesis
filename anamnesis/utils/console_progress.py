"""
Console progress tracking and rendering utilities.

Provides multi-phase progress tracking with:
- Weighted phases for overall progress calculation
- ETA estimation based on processing rate
- Console progress bar rendering
- MCP-compatible ASCII mode

Ported from TypeScript progress-tracker.ts and console-progress.ts
"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable


# ============================================================================
# Progress Types
# ============================================================================


@dataclass
class ProgressUpdate:
    """Progress update event data."""

    phase: str
    current: int
    total: int
    percentage: float
    start_time: float
    elapsed: float
    eta: str | None = None
    message: str | None = None
    rate: float | None = None  # items per second


@dataclass
class ProgressPhase:
    """A phase within the progress tracker."""

    name: str
    weight: float  # Relative weight for overall progress
    current: int
    total: int
    started: bool = False


# Alias for backwards compatibility
PhaseProgress = ProgressPhase

ProgressCallback = Callable[[ProgressUpdate], None]


# ============================================================================
# Progress Tracker
# ============================================================================


class ProgressTracker:
    """
    Multi-phase progress tracker with weighted overall progress.

    Supports adding multiple phases with different weights,
    tracking individual and overall progress, and ETA estimation.

    Example:
        tracker = ProgressTracker()
        tracker.add_phase("discovery", total=100, weight=1)
        tracker.add_phase("analysis", total=500, weight=3)

        tracker.start_phase("discovery")
        for i in range(100):
            tracker.increment_progress("discovery")

        tracker.start_phase("analysis")
        for i in range(500):
            tracker.increment_progress("analysis")
    """

    def __init__(self) -> None:
        """Initialize the progress tracker."""
        self._phases: dict[str, ProgressPhase] = {}
        self._start_time = time.time()
        self._current_phase: str | None = None
        self._total_weight: float = 0.0
        self._callbacks: dict[str, list[ProgressCallback]] = {
            "progress": [],
            "phase_start": [],
            "complete": [],
            "overall": [],
        }
        self._lock = threading.RLock()  # RLock for nested acquisitions

    def on(self, event: str, callback: ProgressCallback) -> None:
        """
        Register an event callback.

        Events:
            - "progress": Called on any progress update
            - "phase_start": Called when a phase starts
            - "complete": Called when all phases complete
            - "overall": Called with overall progress updates
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def add_phase(
        self,
        name: str,
        total: int,
        weight: float = 1.0,
    ) -> None:
        """
        Add a progress phase.

        Args:
            name: Phase identifier
            total: Total items in this phase
            weight: Relative weight for overall progress calculation
        """
        with self._lock:
            self._phases[name] = ProgressPhase(
                name=name,
                weight=weight,
                current=0,
                total=total,
                started=False,
            )
            self._total_weight += weight

    def start_phase(self, phase_name: str) -> None:
        """
        Start a phase.

        Args:
            phase_name: Name of the phase to start

        Raises:
            ValueError: If phase doesn't exist
        """
        with self._lock:
            if phase_name not in self._phases:
                raise ValueError(
                    f"Phase {phase_name} not found. Add it first with add_phase()"
                )

            phase = self._phases[phase_name]
            phase.started = True
            self._current_phase = phase_name

        self._emit("phase_start", self._create_update(phase_name))
        self._emit_progress(phase_name, 0)

    def update_progress(
        self,
        phase_name: str,
        current: int,
        message: str | None = None,
    ) -> None:
        """
        Update progress for a phase.

        Args:
            phase_name: Phase to update
            current: Current progress value
            message: Optional status message
        """
        with self._lock:
            phase = self._phases.get(phase_name)
            if not phase:
                raise ValueError(f"Phase {phase_name} not found")

            old_current = phase.current
            phase.current = min(current, phase.total)

            # Only emit if there's an actual change
            if old_current != phase.current:
                self._emit_progress(phase_name, phase.current, message)

    def increment_progress(
        self,
        phase_name: str,
        increment: int = 1,
        message: str | None = None,
    ) -> None:
        """
        Increment progress for a phase.

        Args:
            phase_name: Phase to increment
            increment: Amount to increment (default: 1)
            message: Optional status message
        """
        with self._lock:
            phase = self._phases.get(phase_name)
            if not phase:
                raise ValueError(f"Phase {phase_name} not found")

            phase.current = min(phase.current + increment, phase.total)

        self._emit_progress(phase_name, phase.current, message)

    def _emit_progress(
        self,
        phase_name: str,
        current: int,
        message: str | None = None,
    ) -> None:
        """Emit progress update events."""
        update = self._create_update(phase_name, message)
        self._emit("progress", update)
        self._emit(f"progress:{phase_name}", update)

        # Emit overall progress
        overall = self._create_overall_update()
        self._emit("overall", overall)

    def _create_update(
        self,
        phase_name: str,
        message: str | None = None,
    ) -> ProgressUpdate:
        """Create a progress update object."""
        with self._lock:
            phase = self._phases[phase_name]
            elapsed = time.time() - self._start_time
            percentage = (phase.current / phase.total * 100) if phase.total > 0 else 0

            # Calculate ETA and rate
            eta: str | None = None
            rate: float | None = None

            if phase.current > 0 and elapsed > 1.0:
                rate = phase.current / elapsed
                if rate > 0:
                    remaining = phase.total - phase.current
                    eta_seconds = remaining / rate
                    eta = self._format_eta(eta_seconds)

            return ProgressUpdate(
                phase=phase_name,
                current=phase.current,
                total=phase.total,
                percentage=percentage,
                eta=eta,
                message=message,
                start_time=self._start_time,
                elapsed=elapsed,
                rate=rate,
            )

    def _create_overall_update(self) -> ProgressUpdate:
        """Create overall progress update."""
        with self._lock:
            elapsed = time.time() - self._start_time
            percentage = self._calculate_overall_progress()

            return ProgressUpdate(
                phase="overall",
                current=int(percentage),
                total=100,
                percentage=percentage,
                start_time=self._start_time,
                elapsed=elapsed,
            )

    def _calculate_overall_progress(self) -> float:
        """Calculate weighted overall progress percentage."""
        if self._total_weight == 0:
            return 0.0

        weighted_progress = 0.0
        for phase in self._phases.values():
            phase_progress = (phase.current / phase.total) if phase.total > 0 else 0
            weighted_progress += phase_progress * phase.weight

        return (weighted_progress / self._total_weight) * 100

    def _format_eta(self, seconds: float) -> str:
        """Format ETA as human-readable string."""
        if seconds < 1:
            return "less than 1s"

        seconds = int(seconds)
        minutes = seconds // 60
        hours = minutes // 60

        if hours > 0:
            return f"{hours}h {minutes % 60}m"
        elif minutes > 0:
            return f"{minutes}m {seconds % 60}s"
        else:
            return f"{seconds}s"

    def _emit(self, event: str, update: ProgressUpdate) -> None:
        """Emit an event to all registered callbacks."""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(update)
            except Exception:
                pass  # Don't let callback errors break progress

    def get_progress(self, phase_name: str | None = None) -> ProgressUpdate | None:
        """
        Get current progress.

        Args:
            phase_name: Specific phase or None for overall progress

        Returns:
            Progress update or None if phase not found
        """
        with self._lock:
            if phase_name:
                if phase_name not in self._phases:
                    return None
                return self._create_update(phase_name)
            else:
                return self._create_overall_update()

    def complete(self, phase_name: str | None = None) -> None:
        """
        Mark phase(s) as complete.

        Args:
            phase_name: Specific phase or None for all phases
        """
        with self._lock:
            if phase_name:
                phase = self._phases.get(phase_name)
                if phase:
                    phase.current = phase.total
                    self._emit_progress(phase_name, phase.total, "Completed")

                    # Check if all phases complete
                    all_complete = all(
                        p.current == p.total for p in self._phases.values()
                    )
                    if all_complete:
                        self._emit("complete", self._create_overall_update())
            else:
                # Complete all phases
                for name, phase in self._phases.items():
                    phase.current = phase.total
                    self._emit_progress(name, phase.total, "Completed")
                self._emit("complete", self._create_overall_update())

    def reset(self) -> None:
        """Reset all progress."""
        with self._lock:
            for phase in self._phases.values():
                phase.current = 0
                phase.started = False
            self._start_time = time.time()
            self._current_phase = None

    def render_progress_bar(
        self,
        phase_name: str,
        width: int = 40,
    ) -> str:
        """
        Render a progress bar for a phase.

        Args:
            phase_name: Phase to render
            width: Bar width in characters

        Returns:
            Formatted progress bar string
        """
        with self._lock:
            phase = self._phases.get(phase_name)
            if not phase:
                return ""

            percentage = (phase.current / phase.total) if phase.total > 0 else 0
            filled = int(percentage * width)
            empty = width - filled

            # Use ASCII in MCP mode, Unicode otherwise
            is_mcp_mode = os.environ.get("MCP_SERVER", "").lower() == "true"
            if is_mcp_mode:
                bar = "=" * filled + "-" * empty
            else:
                bar = "█" * filled + "░" * empty

            percent_str = f"{percentage * 100:5.1f}%"
            count_str = f"({phase.current}/{phase.total})"

            # Format phase name
            formatted_name = phase_name.replace("_", " ").title().ljust(20)

            # Get emoji
            emoji = self._get_phase_emoji(phase_name)

            # Build line
            base = f"{emoji} {formatted_name} [{bar}] {percent_str} {count_str}"

            # Add completion checkmark
            if phase.current >= phase.total:
                if not is_mcp_mode:
                    # Green color in terminal mode
                    return f"\033[32m{base}\033[0m ✓"
                return f"{base} [DONE]"

            return base

    def _get_phase_emoji(self, phase_name: str) -> str:
        """Get emoji for a phase based on its name."""
        name_lower = phase_name.lower()
        if "semantic" in name_lower:
            return "🧠"
        if "pattern" in name_lower:
            return "🔍"
        if "discovery" in name_lower:
            return "🔎"
        if "indexing" in name_lower:
            return "📇"
        if "analysis" in name_lower:
            return "📊"
        return "⚙️"

    def get_console_status(self) -> list[str]:
        """
        Get console-friendly status lines.

        Returns:
            List of formatted status lines
        """
        lines: list[str] = []

        # Overall progress line
        overall = self.get_progress()
        if overall:
            elapsed = self._format_elapsed(overall.elapsed)
            eta = self._estimate_overall_eta()
            eta_str = f" | ETA: {eta}" if eta else ""
            lines.append(
                f"⏱️  Overall: {overall.percentage:.1f}% | Time: {elapsed}{eta_str}"
            )

        # Phase progress bars
        with self._lock:
            for name in self._phases:
                lines.append(self.render_progress_bar(name))

        return lines

    def _format_elapsed(self, elapsed: float) -> str:
        """Format elapsed time."""
        seconds = int(elapsed)
        minutes = seconds // 60

        if minutes > 0:
            return f"{minutes}m {seconds % 60}s"
        return f"{seconds}s"

    def _estimate_overall_eta(self) -> str | None:
        """Estimate overall ETA based on progress rate."""
        overall = self.get_progress()
        if not overall or overall.percentage == 0 or overall.percentage >= 100:
            return None

        elapsed = overall.elapsed
        remaining = (elapsed / overall.percentage) * (100 - overall.percentage)

        if remaining < 1:
            return None

        return self._format_eta(remaining)

    @property
    def phases(self) -> dict[str, ProgressPhase]:
        """Get all phases (for inspection)."""
        return dict(self._phases)


# ============================================================================
# Console Progress Renderer
# ============================================================================


class ConsoleProgressRenderer:
    """
    Console renderer for progress tracking with live updates.

    Handles terminal cursor manipulation to provide smooth updates
    without flooding the console with new lines.
    """

    def __init__(self, tracker: ProgressTracker) -> None:
        """
        Initialize the renderer.

        Args:
            tracker: ProgressTracker to render
        """
        self._tracker = tracker
        self._is_active = False
        self._render_timer: threading.Timer | None = None
        self._last_output = ""
        self._has_rendered = False
        self._is_completed = False
        self._has_shown_completion = False
        self._lock = threading.RLock()  # RLock for nested acquisitions

        # Set up completion listener
        tracker.on("complete", self._on_complete)

    def _on_complete(self, update: ProgressUpdate) -> None:
        """Handle completion event."""
        with self._lock:
            self._is_completed = True
            if self._is_active:
                self._render()

    def start(self) -> None:
        """Start the progress renderer."""
        with self._lock:
            self._is_active = True
            self._has_rendered = False
            self._is_completed = False
            self._has_shown_completion = False

        # Initial render
        self._render()

        # Start periodic updates
        self._schedule_render()

    def _schedule_render(self) -> None:
        """Schedule the next render."""
        if self._is_active:
            self._render_timer = threading.Timer(0.5, self._periodic_render)
            self._render_timer.daemon = True
            self._render_timer.start()

    def _periodic_render(self) -> None:
        """Periodic render callback."""
        if self._is_active:
            self._render()
            self._schedule_render()

    def stop(self) -> None:
        """Stop the progress renderer."""
        # Cancel timer
        if self._render_timer:
            self._render_timer.cancel()
            self._render_timer = None

        # Final render
        self._render_final()

        with self._lock:
            self._is_active = False

    def _render(self) -> None:
        """Render current progress to console."""
        with self._lock:
            if not self._is_active:
                return

        # Get status lines
        all_lines = self._tracker.get_console_status()
        if not all_lines:
            return

        # Filter unstarted phases unless completed
        if self._is_completed:
            lines = all_lines
        else:
            lines = [
                line for i, line in enumerate(all_lines)
                if i == 0 or "0.0% (0/" not in line or "✓" in line or "[DONE]" in line
            ]

        if len(lines) <= 1:
            return  # Only overall line, don't render yet

        # Build output
        separator = "━" * 60
        output = f"\n{separator}\n" + "\n".join(lines) + f"\n{separator}"

        # Check if output changed
        with self._lock:
            if output == self._last_output:
                return

            # Clear previous output
            if self._has_rendered:
                line_count = self._last_output.count("\n") + 1
                # Move cursor up and clear
                sys.stderr.write(f"\033[{line_count}A\033[0J")

            sys.stderr.write(output + "\n")
            sys.stderr.flush()
            self._last_output = output
            self._has_rendered = True

    def _render_final(self) -> None:
        """Render final state with completion message."""
        with self._lock:
            if not self._has_rendered:
                return

            self._is_completed = True

        self._render()

        with self._lock:
            if not self._has_shown_completion:
                sys.stderr.write("✅ Complete!\n\n")
                sys.stderr.flush()
                self._has_shown_completion = True

    def get_progress_data(self) -> dict:
        """
        Get progress data as structured dictionary.

        Useful for MCP server responses.

        Returns:
            Progress data dictionary
        """
        overall = self._tracker.get_progress()
        phases = []

        for name, phase in self._tracker.phases.items():
            phase_progress = self._tracker.get_progress(name)
            if phase_progress:
                phases.append({
                    "name": name,
                    "current": phase_progress.current,
                    "total": phase_progress.total,
                    "percentage": phase_progress.percentage,
                    "eta": phase_progress.eta,
                })

        return {
            "overall": overall.percentage if overall else 0,
            "phases": phases,
            "elapsed": overall.elapsed if overall else 0,
            "is_complete": overall.percentage >= 100 if overall else False,
        }


# ============================================================================
# Simple Progress Bar Helper
# ============================================================================


def render_simple_bar(
    current: int,
    total: int,
    width: int = 40,
    message: str | None = None,
) -> str:
    """
    Render a simple progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Bar width in characters
        message: Optional message prefix

    Returns:
        Formatted progress bar string
    """
    percentage = (current / total) if total > 0 else 0
    filled = int(percentage * width)
    empty = width - filled

    # Use ASCII in MCP mode
    is_mcp_mode = os.environ.get("MCP_SERVER", "").lower() == "true"
    if is_mcp_mode:
        bar = "=" * filled + "-" * empty
    else:
        bar = "█" * filled + "░" * empty

    percent_str = f"{percentage * 100:.1f}%"
    base_text = f"[{bar}] {percent_str} ({current}/{total})"

    return f"{message}: {base_text}" if message else base_text
