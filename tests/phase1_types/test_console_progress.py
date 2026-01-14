"""
Phase 1 Tests: Console Progress

Tests for progress tracking and rendering including:
- Progress phase management
- Weighted overall progress calculation
- ETA estimation
- Progress bar rendering
- Event callbacks
"""

import time

import pytest

from anamnesis.utils import (
    ConsoleProgressRenderer,
    PhaseProgress,
    ProgressPhase,
    ProgressTracker,
)


class TestProgressPhase:
    """Tests for ProgressPhase dataclass."""

    def test_phase_creation(self):
        """Phase can be created with all fields."""
        phase = ProgressPhase(
            name="analysis",
            weight=2.0,
            current=50,
            total=100,
            started=True,
        )
        assert phase.name == "analysis"
        assert phase.weight == 2.0
        assert phase.current == 50
        assert phase.total == 100
        assert phase.started is True

    def test_phase_defaults(self):
        """Phase has sensible defaults."""
        phase = ProgressPhase(
            name="test",
            weight=1.0,
            current=0,
            total=10,
        )
        assert phase.started is False


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_add_phase(self):
        """Can add phases."""
        tracker = ProgressTracker()
        tracker.add_phase("phase1", total=100, weight=1.0)
        tracker.add_phase("phase2", total=200, weight=2.0)

        assert len(tracker.phases) == 2
        assert "phase1" in tracker.phases
        assert "phase2" in tracker.phases

    def test_start_phase(self):
        """Can start a phase."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")

        assert tracker.phases["test"].started is True

    def test_start_nonexistent_phase(self):
        """Starting nonexistent phase raises error."""
        tracker = ProgressTracker()
        with pytest.raises(ValueError, match="not found"):
            tracker.start_phase("nonexistent")

    def test_update_progress(self):
        """Can update phase progress."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)

        assert tracker.phases["test"].current == 50

    def test_update_progress_capped(self):
        """Progress is capped at total."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 150)

        assert tracker.phases["test"].current == 100

    def test_increment_progress(self):
        """Can increment phase progress."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")

        tracker.increment_progress("test", 10)
        assert tracker.phases["test"].current == 10

        tracker.increment_progress("test", 5)
        assert tracker.phases["test"].current == 15

    def test_overall_progress_single_phase(self):
        """Overall progress calculated correctly for single phase."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100, weight=1.0)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)

        progress = tracker.get_progress()
        assert progress is not None
        assert progress.percentage == 50.0

    def test_overall_progress_weighted_phases(self):
        """Overall progress calculated correctly with weighted phases."""
        tracker = ProgressTracker()
        tracker.add_phase("light", total=100, weight=1.0)
        tracker.add_phase("heavy", total=100, weight=3.0)
        tracker.start_phase("light")
        tracker.start_phase("heavy")

        # Complete light phase (weight 1)
        tracker.update_progress("light", 100)
        # Heavy phase at 0%

        # Overall: (1.0 * 100% + 3.0 * 0%) / 4.0 = 25%
        progress = tracker.get_progress()
        assert progress is not None
        assert progress.percentage == 25.0

    def test_overall_progress_both_phases(self):
        """Overall progress with both phases at 50%."""
        tracker = ProgressTracker()
        tracker.add_phase("phase1", total=100, weight=1.0)
        tracker.add_phase("phase2", total=100, weight=1.0)
        tracker.start_phase("phase1")
        tracker.start_phase("phase2")

        tracker.update_progress("phase1", 50)
        tracker.update_progress("phase2", 50)

        progress = tracker.get_progress()
        assert progress is not None
        assert progress.percentage == 50.0

    def test_get_phase_progress(self):
        """Can get progress for specific phase."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 75)

        progress = tracker.get_progress("test")
        assert progress is not None
        assert progress.current == 75
        assert progress.total == 100
        assert progress.percentage == 75.0

    def test_progress_callback(self):
        """Progress callbacks are called."""
        tracker = ProgressTracker()
        updates = []

        def on_progress(update):
            updates.append(update)

        tracker.on("progress", on_progress)
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)

        assert len(updates) > 0
        assert updates[-1].current == 50

    def test_complete_phase(self):
        """Can mark phase as complete."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.complete("test")

        assert tracker.phases["test"].current == 100

    def test_complete_all_phases(self):
        """Can mark all phases as complete."""
        tracker = ProgressTracker()
        tracker.add_phase("phase1", total=100)
        tracker.add_phase("phase2", total=200)
        tracker.complete()

        assert tracker.phases["phase1"].current == 100
        assert tracker.phases["phase2"].current == 200

    def test_reset(self):
        """Can reset all progress."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)
        tracker.reset()

        assert tracker.phases["test"].current == 0
        assert tracker.phases["test"].started is False


class TestProgressBarRendering:
    """Tests for progress bar rendering."""

    def test_render_progress_bar_empty(self):
        """Renders empty progress bar."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")

        bar = tracker.render_progress_bar("test", width=20)
        assert "[" in bar
        assert "]" in bar
        assert "0.0%" in bar

    def test_render_progress_bar_half(self):
        """Renders half-filled progress bar."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)

        bar = tracker.render_progress_bar("test", width=20)
        assert "50.0%" in bar

    def test_render_progress_bar_full(self):
        """Renders full progress bar."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 100)

        bar = tracker.render_progress_bar("test", width=20)
        assert "100.0%" in bar
        # Should have completion marker
        assert "✓" in bar or "[DONE]" in bar

    def test_render_nonexistent_phase(self):
        """Rendering nonexistent phase returns empty string."""
        tracker = ProgressTracker()
        bar = tracker.render_progress_bar("nonexistent")
        assert bar == ""

    def test_get_console_status(self):
        """Gets console status lines."""
        tracker = ProgressTracker()
        tracker.add_phase("discovery", total=100)
        tracker.add_phase("analysis", total=200)
        tracker.start_phase("discovery")
        tracker.start_phase("analysis")
        tracker.update_progress("discovery", 50)
        tracker.update_progress("analysis", 100)

        lines = tracker.get_console_status()
        assert len(lines) >= 2  # Overall + at least one phase


class TestETAEstimation:
    """Tests for ETA estimation."""

    def test_eta_format_seconds(self):
        """ETA formats seconds correctly."""
        tracker = ProgressTracker()
        eta = tracker._format_eta(45)
        assert "45s" in eta

    def test_eta_format_minutes(self):
        """ETA formats minutes correctly."""
        tracker = ProgressTracker()
        eta = tracker._format_eta(90)
        assert "1m" in eta and "30s" in eta

    def test_eta_format_hours(self):
        """ETA formats hours correctly."""
        tracker = ProgressTracker()
        eta = tracker._format_eta(3700)
        assert "1h" in eta

    def test_eta_format_less_than_second(self):
        """ETA formats sub-second correctly."""
        tracker = ProgressTracker()
        eta = tracker._format_eta(0.5)
        assert "less than 1s" in eta


class TestConsoleProgressRenderer:
    """Tests for ConsoleProgressRenderer class."""

    def test_renderer_creation(self):
        """Renderer can be created."""
        tracker = ProgressTracker()
        renderer = ConsoleProgressRenderer(tracker)
        assert renderer is not None

    def test_get_progress_data(self):
        """Gets progress data as dictionary."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        tracker.start_phase("test")
        tracker.update_progress("test", 50)

        renderer = ConsoleProgressRenderer(tracker)
        data = renderer.get_progress_data()

        assert "overall" in data
        assert "phases" in data
        assert "elapsed" in data
        assert "is_complete" in data
        assert data["overall"] == 50.0
        assert len(data["phases"]) == 1

    def test_renderer_start_stop(self):
        """Renderer can start and stop."""
        tracker = ProgressTracker()
        tracker.add_phase("test", total=100)
        renderer = ConsoleProgressRenderer(tracker)

        renderer.start()
        # Brief delay to allow render
        time.sleep(0.1)
        renderer.stop()

        # Should not raise any errors


class TestPhaseProgress:
    """Tests for PhaseProgress type alias (if used)."""

    def test_progress_phase_is_phased(self):
        """ProgressPhase is the expected type."""
        # PhaseProgress is imported as an alias
        # This test verifies the import works
        phase = PhaseProgress(
            name="test",
            weight=1.0,
            current=0,
            total=100,
        )
        assert isinstance(phase, ProgressPhase)
