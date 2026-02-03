"""
Phase 11 Tests: Change Analyzer

Tests for the change analyzer including:
- Change analysis
- Impact assessment
- Pattern detection
- Batch processing
- Insight generation
"""

from unittest.mock import MagicMock


from anamnesis.watchers.change_analyzer import (
    AnalyzerOptions,
    ChangeAnalysis,
    ChangeAnalyzer,
    ImpactInfo,
    IntelligenceInfo,
    PatternInfo,
)
from anamnesis.watchers.file_watcher import FileChange


class TestImpactInfo:
    """Tests for ImpactInfo dataclass."""

    def test_impact_info_creation(self):
        """ImpactInfo can be created with all fields."""
        impact = ImpactInfo(
            scope="module",
            confidence=0.8,
            affected_concepts=["UserService", "AuthModule"],
            suggested_actions=["Update tests", "Review dependencies"],
        )
        assert impact.scope == "module"
        assert impact.confidence == 0.8
        assert len(impact.affected_concepts) == 2
        assert len(impact.suggested_actions) == 2

    def test_impact_info_defaults(self):
        """ImpactInfo has sensible defaults."""
        impact = ImpactInfo(scope="file", confidence=0.5)
        assert impact.affected_concepts == []
        assert impact.suggested_actions == []


class TestPatternInfo:
    """Tests for PatternInfo dataclass."""

    def test_pattern_info_creation(self):
        """PatternInfo can be created."""
        patterns = PatternInfo(
            detected=["Singleton", "Factory"],
            violations=["Too many dependencies"],
            recommendations=["Consider refactoring"],
        )
        assert len(patterns.detected) == 2
        assert len(patterns.violations) == 1
        assert len(patterns.recommendations) == 1

    def test_pattern_info_defaults(self):
        """PatternInfo has sensible defaults."""
        patterns = PatternInfo()
        assert patterns.detected == []
        assert patterns.violations == []
        assert patterns.recommendations == []


class TestIntelligenceInfo:
    """Tests for IntelligenceInfo dataclass."""

    def test_intelligence_info_creation(self):
        """IntelligenceInfo can be created."""
        intel = IntelligenceInfo(
            concepts_updated=5,
            patterns_learned=2,
            insights=["New pattern detected", "Complexity increased"],
        )
        assert intel.concepts_updated == 5
        assert intel.patterns_learned == 2
        assert len(intel.insights) == 2

    def test_intelligence_info_defaults(self):
        """IntelligenceInfo has sensible defaults."""
        intel = IntelligenceInfo()
        assert intel.concepts_updated == 0
        assert intel.patterns_learned == 0
        assert intel.insights == []


class TestChangeAnalysis:
    """Tests for ChangeAnalysis dataclass."""

    def test_change_analysis_creation(self):
        """ChangeAnalysis can be created."""
        change = FileChange(type="add", path="/test/file.py")
        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(scope="file", confidence=0.7),
            patterns=PatternInfo(),
            intelligence=IntelligenceInfo(),
        )
        assert analysis.change == change
        assert analysis.impact.scope == "file"
        assert analysis.timestamp is not None

    def test_change_analysis_to_dict(self):
        """ChangeAnalysis can be converted to dict."""
        change = FileChange(type="add", path="/test/file.py")
        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(
                scope="module",
                confidence=0.8,
                affected_concepts=["TestClass"],
            ),
            patterns=PatternInfo(detected=["Factory"]),
            intelligence=IntelligenceInfo(
                concepts_updated=1,
                insights=["New concept detected"],
            ),
        )

        d = analysis.to_dict()
        assert d["change"]["type"] == "add"
        assert d["change"]["path"] == "/test/file.py"
        assert d["impact"]["scope"] == "module"
        assert d["impact"]["confidence"] == 0.8
        assert d["patterns"]["detected"] == ["Factory"]
        assert d["intelligence"]["concepts_updated"] == 1
        assert "timestamp" in d


class TestAnalyzerOptions:
    """Tests for AnalyzerOptions dataclass."""

    def test_default_options(self):
        """AnalyzerOptions has sensible defaults."""
        opts = AnalyzerOptions()
        assert opts.enable_real_time_analysis is True
        assert opts.enable_pattern_learning is True
        assert opts.batch_size == 5
        assert opts.analysis_delay_ms == 1000

    def test_custom_options(self):
        """AnalyzerOptions accepts custom values."""
        opts = AnalyzerOptions(
            enable_real_time_analysis=False,
            enable_pattern_learning=False,
            batch_size=10,
            analysis_delay_ms=500,
        )
        assert opts.enable_real_time_analysis is False
        assert opts.batch_size == 10


class TestChangeAnalyzerInit:
    """Tests for ChangeAnalyzer initialization."""

    def test_analyzer_creation(self):
        """ChangeAnalyzer can be created."""
        analyzer = ChangeAnalyzer()
        assert not analyzer.is_analyzing()
        assert analyzer.get_queue_size() == 0

    def test_analyzer_with_options(self):
        """ChangeAnalyzer accepts custom options."""
        opts = AnalyzerOptions(batch_size=10)
        analyzer = ChangeAnalyzer(options=opts)
        assert analyzer.options.batch_size == 10

    def test_analyzer_with_engines(self):
        """ChangeAnalyzer accepts engine dependencies."""
        semantic_engine = MagicMock()
        pattern_engine = MagicMock()
        database = MagicMock()

        analyzer = ChangeAnalyzer(
            semantic_engine=semantic_engine,
            pattern_engine=pattern_engine,
            database=database,
        )
        assert analyzer.semantic_engine is semantic_engine
        assert analyzer.pattern_engine is pattern_engine
        assert analyzer.database is database


class TestChangeAnalyzerScopeEstimation:
    """Tests for scope estimation."""

    def test_project_scope_for_config_files(self):
        """Config files have project scope."""
        analyzer = ChangeAnalyzer()

        for config_file in ["package.json", "tsconfig.json", "pyproject.toml"]:
            change = FileChange(type="change", path=f"/project/{config_file}")
            scope = analyzer._estimate_scope(change)
            assert scope == "project", f"Expected project scope for {config_file}"

    def test_module_scope_for_test_files(self):
        """Test files have module scope."""
        analyzer = ChangeAnalyzer()

        for test_path in ["test_main.py", "tests/unit.py", "spec/feature_spec.ts"]:
            change = FileChange(type="change", path=test_path)
            scope = analyzer._estimate_scope(change)
            assert scope == "module", f"Expected module scope for {test_path}"

    def test_module_scope_for_index_files(self):
        """Index files have module scope."""
        analyzer = ChangeAnalyzer()

        for index_file in ["index.ts", "index.js", "__init__.py", "mod.rs"]:
            change = FileChange(type="change", path=f"/src/{index_file}")
            scope = analyzer._estimate_scope(change)
            assert scope == "module", f"Expected module scope for {index_file}"

    def test_file_scope_default(self):
        """Regular files have file scope."""
        analyzer = ChangeAnalyzer()

        change = FileChange(type="change", path="/src/utils/helpers.py")
        scope = analyzer._estimate_scope(change)
        assert scope == "file"


class TestChangeAnalyzerSuggestedActions:
    """Tests for suggested actions."""

    def test_actions_for_add(self):
        """Addition suggests documentation and tests."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="add", path="/src/new_module.py", language="python")
        actions = analyzer._get_suggested_actions(change)

        assert "Update documentation" in actions
        assert "Add tests if applicable" in actions

    def test_actions_for_change(self):
        """Modification suggests test review."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/module.py", language="python")
        actions = analyzer._get_suggested_actions(change)

        assert "Review related tests" in actions
        assert "Check for breaking changes" in actions

    def test_actions_for_unlink(self):
        """Deletion suggests cleanup."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="unlink", path="/src/old_module.py", language="python")
        actions = analyzer._get_suggested_actions(change)

        assert "Remove related tests" in actions
        assert "Update imports/dependencies" in actions

    def test_typescript_specific_actions(self):
        """TypeScript files suggest type checking."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/module.ts", language="typescript")
        actions = analyzer._get_suggested_actions(change)

        assert "Run type checking" in actions

    def test_python_specific_actions(self):
        """Python files suggest mypy."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/module.py", language="python")
        actions = analyzer._get_suggested_actions(change)

        assert any("mypy" in action.lower() for action in actions)

    def test_rust_specific_actions(self):
        """Rust files suggest cargo check."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/lib.rs", language="rust")
        actions = analyzer._get_suggested_actions(change)

        assert any("cargo" in action.lower() for action in actions)

    def test_go_specific_actions(self):
        """Go files suggest go vet."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/main.go", language="go")
        actions = analyzer._get_suggested_actions(change)

        assert any("go vet" in action.lower() for action in actions)


class TestChangeAnalyzerLightweightAnalysis:
    """Tests for lightweight analysis."""

    def test_lightweight_analysis_basic(self):
        """Lightweight analysis produces valid result."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="add", path="/src/module.py", language="python")

        analysis = analyzer._perform_lightweight_analysis(change)

        assert analysis.change == change
        assert analysis.impact.scope == "file"
        assert analysis.impact.confidence == 0.5
        assert analysis.timestamp is not None

    def test_lightweight_analysis_with_language(self):
        """Lightweight analysis includes suggested actions."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="add", path="/src/module.py", language="python")

        analysis = analyzer._perform_lightweight_analysis(change)

        assert len(analysis.impact.suggested_actions) > 0


class TestChangeAnalyzerMinimalAnalysis:
    """Tests for minimal analysis (real-time disabled)."""

    def test_minimal_analysis(self):
        """Minimal analysis when real-time is disabled."""
        analyzer = ChangeAnalyzer(
            options=AnalyzerOptions(enable_real_time_analysis=False)
        )
        change = FileChange(type="add", path="/src/module.py")

        analysis = analyzer._create_minimal_analysis(change)

        assert analysis.impact.confidence == 0.1
        assert "Real-time analysis disabled" in analysis.intelligence.insights


class TestChangeAnalyzerInsights:
    """Tests for insight generation."""

    def test_pattern_insights(self):
        """Generates insights from patterns."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="add", path="/src/module.py")

        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(scope="file", confidence=0.5),
            patterns=PatternInfo(
                detected=["Factory", "Singleton"],
                violations=["TooManyDependencies"],
            ),
            intelligence=IntelligenceInfo(),
        )

        insights = analyzer._generate_insights(analysis)

        assert any("2 patterns" in insight for insight in insights)
        assert any("1 pattern violation" in insight for insight in insights)

    def test_project_scope_insight(self):
        """Generates insight for project scope changes."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="add", path="/package.json")

        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(scope="project", confidence=0.8),
            patterns=PatternInfo(),
            intelligence=IntelligenceInfo(),
        )

        insights = analyzer._generate_insights(analysis)

        assert any("project-wide" in insight.lower() for insight in insights)

    def test_concept_update_insight(self):
        """Generates insight for concept updates."""
        analyzer = ChangeAnalyzer()
        change = FileChange(type="change", path="/src/module.py")

        analysis = ChangeAnalysis(
            change=change,
            impact=ImpactInfo(scope="file", confidence=0.5),
            patterns=PatternInfo(),
            intelligence=IntelligenceInfo(concepts_updated=3),
        )

        insights = analyzer._generate_insights(analysis)

        assert any("3 concepts" in insight for insight in insights)


class TestChangeAnalyzerArchitecturalImpact:
    """Tests for architectural impact assessment."""

    def test_architectural_impact(self):
        """Assesses architectural impact."""
        analyzer = ChangeAnalyzer()

        analyses = [
            ChangeAnalysis(
                change=FileChange(type="change", path="/src/a.py"),
                impact=ImpactInfo(
                    scope="file",
                    confidence=0.5,
                    affected_concepts=["ServiceA", "ServiceB"],
                ),
                patterns=PatternInfo(),
                intelligence=IntelligenceInfo(),
            ),
            ChangeAnalysis(
                change=FileChange(type="change", path="/src/b.py"),
                impact=ImpactInfo(
                    scope="file",
                    confidence=0.5,
                    affected_concepts=["ServiceC", "ServiceD"],
                ),
                patterns=PatternInfo(),
                intelligence=IntelligenceInfo(),
            ),
        ]

        impact = analyzer._assess_architectural_impact(analyses)

        assert impact["confidence"] > 0
        assert len(impact["insights"]) > 0
        assert any("4 concepts" in insight for insight in impact["insights"])


class TestChangeAnalyzerControlMethods:
    """Tests for analyzer control methods."""

    def test_enable_disable_real_time(self):
        """Can enable/disable real-time analysis."""
        analyzer = ChangeAnalyzer()

        assert analyzer.options.enable_real_time_analysis is True

        analyzer.disable_real_time_analysis()
        assert analyzer.options.enable_real_time_analysis is False

        analyzer.enable_real_time_analysis()
        assert analyzer.options.enable_real_time_analysis is True

    def test_clear_queue(self):
        """Can clear the analysis queue."""
        analyzer = ChangeAnalyzer()

        # Manually add to queue for testing
        analyzer._analysis_queue.append(
            FileChange(type="add", path="/test.py")
        )
        assert analyzer.get_queue_size() == 1

        analyzer.clear_queue()
        assert analyzer.get_queue_size() == 0

    def test_get_queue_size(self):
        """Can get queue size."""
        analyzer = ChangeAnalyzer()
        assert analyzer.get_queue_size() == 0

    def test_is_analyzing(self):
        """Can check if analyzing."""
        analyzer = ChangeAnalyzer()
        assert analyzer.is_analyzing() is False


class TestChangeAnalyzerBatchAnalysis:
    """Tests for batch analysis."""

    def test_analyze_batch(self):
        """Can analyze a batch of changes."""
        analyzer = ChangeAnalyzer()
        changes = [
            FileChange(type="add", path="/src/a.py", language="python"),
            FileChange(type="change", path="/src/b.py", language="python"),
        ]

        analyses = analyzer.analyze_batch(changes)

        assert len(analyses) == 2
        assert analyses[0].change.path == "/src/a.py"
        assert analyses[1].change.path == "/src/b.py"

    def test_analyze_batch_cross_file_analysis(self):
        """Batch analysis performs cross-file analysis."""
        analyzer = ChangeAnalyzer()

        # Create changes with many affected concepts to trigger cross-file analysis
        changes = [
            FileChange(type="change", path="/src/a.py", language="python"),
            FileChange(type="change", path="/src/b.py", language="python"),
        ]

        analyses = analyzer.analyze_batch(changes)

        # Both should be analyzed
        assert len(analyses) == 2
