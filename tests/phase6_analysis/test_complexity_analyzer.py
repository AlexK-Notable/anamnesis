"""Tests for complexity analyzer module."""

import pytest

from anamnesis.analysis.complexity_analyzer import (
    CognitiveComplexity,
    ComplexityAnalyzer,
    ComplexityLevel,
    ComplexityResult,
    CyclomaticComplexity,
    FileComplexity,
    HalsteadMetrics,
    LinesOfCode,
    MaintainabilityIndex,
)


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_low_complexity(self):
        """Test low complexity level."""
        assert ComplexityLevel.LOW == "low"

    def test_moderate_complexity(self):
        """Test moderate complexity level."""
        assert ComplexityLevel.MODERATE == "moderate"

    def test_high_complexity(self):
        """Test high complexity level."""
        assert ComplexityLevel.HIGH == "high"

    def test_very_high_complexity(self):
        """Test very high complexity level."""
        assert ComplexityLevel.VERY_HIGH == "very_high"


class TestCyclomaticComplexity:
    """Tests for CyclomaticComplexity dataclass."""

    def test_basic_creation(self):
        """Test creating basic cyclomatic complexity."""
        cc = CyclomaticComplexity(value=5)
        assert cc.value == 5
        assert cc.decision_points == 0

    def test_low_complexity_level(self):
        """Test low complexity level assignment (1-10)."""
        cc = CyclomaticComplexity(value=5)
        assert cc.level == ComplexityLevel.LOW

    def test_moderate_complexity_level(self):
        """Test moderate complexity level assignment (11-20)."""
        cc = CyclomaticComplexity(value=15)
        assert cc.level == ComplexityLevel.MODERATE

    def test_high_complexity_level(self):
        """Test high complexity level assignment (21-50)."""
        cc = CyclomaticComplexity(value=30)
        assert cc.level == ComplexityLevel.HIGH

    def test_very_high_complexity_level(self):
        """Test very high complexity level assignment (50+)."""
        cc = CyclomaticComplexity(value=60)
        assert cc.level == ComplexityLevel.VERY_HIGH

    def test_with_decision_points(self):
        """Test with decision points."""
        cc = CyclomaticComplexity(value=10, decision_points=9)
        assert cc.value == 10
        assert cc.decision_points == 9

    def test_to_dict(self):
        """Test serialization to dict."""
        cc = CyclomaticComplexity(value=15, decision_points=14)
        data = cc.to_dict()
        assert data["value"] == 15
        assert data["decision_points"] == 14
        assert data["level"] == "moderate"


class TestCognitiveComplexity:
    """Tests for CognitiveComplexity dataclass."""

    def test_basic_creation(self):
        """Test creating basic cognitive complexity."""
        cc = CognitiveComplexity(value=3)
        assert cc.value == 3

    def test_low_complexity_level(self):
        """Test low complexity level (0-5)."""
        cc = CognitiveComplexity(value=3)
        assert cc.level == ComplexityLevel.LOW

    def test_moderate_complexity_level(self):
        """Test moderate complexity level (6-15)."""
        cc = CognitiveComplexity(value=10)
        assert cc.level == ComplexityLevel.MODERATE

    def test_high_complexity_level(self):
        """Test high complexity level (16-25)."""
        cc = CognitiveComplexity(value=20)
        assert cc.level == ComplexityLevel.HIGH

    def test_very_high_complexity_level(self):
        """Test very high complexity level (>25)."""
        cc = CognitiveComplexity(value=30)
        assert cc.level == ComplexityLevel.VERY_HIGH

    def test_with_components(self):
        """Test with nesting penalty and structural complexity."""
        cc = CognitiveComplexity(
            value=12,
            nesting_penalty=5,
            structural_complexity=7,
        )
        assert cc.nesting_penalty == 5
        assert cc.structural_complexity == 7

    def test_to_dict(self):
        """Test serialization to dict."""
        cc = CognitiveComplexity(
            value=20,
            nesting_penalty=8,
            structural_complexity=10,
        )
        data = cc.to_dict()
        assert data["value"] == 20
        assert data["nesting_penalty"] == 8
        assert data["structural_complexity"] == 10
        assert data["level"] == "high"  # 20 is in 16-25 range = HIGH


class TestHalsteadMetrics:
    """Tests for HalsteadMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating basic Halstead metrics."""
        h = HalsteadMetrics(n1=5, n2=8, N1=10, N2=15)
        assert h.n1 == 5  # Distinct operators
        assert h.n2 == 8  # Distinct operands
        assert h.N1 == 10  # Total operators
        assert h.N2 == 15  # Total operands

    def test_vocabulary(self):
        """Test vocabulary calculation (n = n1 + n2)."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        assert h.vocabulary == 30

    def test_length(self):
        """Test length calculation (N = N1 + N2)."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        assert h.length == 150

    def test_volume(self):
        """Test volume calculation (V = N * log2(n))."""
        h = HalsteadMetrics(n1=10, n2=10, N1=50, N2=50)
        # N = 100, n = 20, V = 100 * log2(20) ≈ 432
        assert h.volume > 0

    def test_difficulty(self):
        """Test difficulty calculation."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        # D = (n1/2) * (N2/n2) = 5 * 5 = 25
        assert h.difficulty == 25.0

    def test_effort(self):
        """Test effort calculation (E = D * V)."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        assert h.effort == h.difficulty * h.volume

    def test_time_to_program(self):
        """Test time calculation (T = E / 18)."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        assert h.time_to_program == h.effort / 18

    def test_bugs_delivered(self):
        """Test bug estimation (B = V / 3000)."""
        h = HalsteadMetrics(n1=10, n2=20, N1=50, N2=100)
        assert h.bugs_delivered == h.volume / 3000

    def test_zero_vocabulary(self):
        """Test handling zero vocabulary."""
        h = HalsteadMetrics(n1=0, n2=0, N1=0, N2=0)
        assert h.vocabulary == 0
        assert h.volume == 0

    def test_zero_operands(self):
        """Test handling zero operands."""
        h = HalsteadMetrics(n1=5, n2=0, N1=10, N2=0)
        assert h.difficulty == 0
        assert h.calculated_length == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        h = HalsteadMetrics(n1=5, n2=10, N1=20, N2=40)
        data = h.to_dict()
        assert data["distinct_operators"] == 5
        assert data["distinct_operands"] == 10
        assert data["total_operators"] == 20
        assert data["total_operands"] == 40
        assert "volume" in data
        assert "difficulty" in data
        assert "time_to_program" in data
        assert "bugs_delivered" in data


class TestLinesOfCode:
    """Tests for LinesOfCode dataclass."""

    def test_basic_creation(self):
        """Test creating basic LOC metrics."""
        loc = LinesOfCode(
            total=100,
            code=70,
            comments=20,
            blank=10,
        )
        assert loc.total == 100
        assert loc.code == 70
        assert loc.comments == 20
        assert loc.blank == 10

    def test_logical_lines(self):
        """Test logical lines tracking."""
        loc = LinesOfCode(
            total=100,
            code=70,
            logical=50,  # Statements, not physical lines
        )
        assert loc.logical == 50

    def test_comment_ratio(self):
        """Test comment ratio calculation (comments / code)."""
        loc = LinesOfCode(total=100, code=80, comments=20, blank=0)
        assert loc.comment_ratio == 0.25  # 20 / 80

    def test_comment_ratio_zero_code(self):
        """Test handling zero code lines."""
        loc = LinesOfCode(total=10, code=0, comments=10, blank=0)
        assert loc.comment_ratio == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        loc = LinesOfCode(total=50, code=40, comments=5, blank=5)
        data = loc.to_dict()
        assert data["total"] == 50
        assert data["code"] == 40
        assert data["comments"] == 5
        assert data["blank"] == 5
        assert "comment_ratio" in data


class TestMaintainabilityIndex:
    """Tests for MaintainabilityIndex dataclass."""

    def test_highly_maintainable(self):
        """Test highly maintainable interpretation (>= 85)."""
        mi = MaintainabilityIndex(value=90.0)
        assert mi.value == 90.0
        assert mi.interpretation == "highly maintainable"

    def test_moderately_maintainable(self):
        """Test moderately maintainable interpretation (65-84)."""
        mi = MaintainabilityIndex(value=70.0)
        assert mi.value == 70.0
        assert mi.interpretation == "moderately maintainable"

    def test_difficult_to_maintain(self):
        """Test difficult to maintain interpretation (20-64)."""
        mi = MaintainabilityIndex(value=50.0)
        assert mi.value == 50.0
        assert mi.interpretation == "difficult to maintain"

    def test_unmaintainable(self):
        """Test unmaintainable interpretation (< 20)."""
        mi = MaintainabilityIndex(value=10.0)
        assert mi.value == 10.0
        assert mi.interpretation == "unmaintainable"

    def test_to_dict(self):
        """Test serialization to dict."""
        mi = MaintainabilityIndex(value=80.0)
        data = mi.to_dict()
        assert data["value"] == 80.0
        assert "interpretation" in data


class TestComplexityResult:
    """Tests for ComplexityResult dataclass."""

    def test_basic_creation(self):
        """Test creating basic result."""
        result = ComplexityResult(
            name="test_function",
            file_path="/test.py",
            start_line=1,
            end_line=10,
        )
        assert result.name == "test_function"
        assert result.file_path == "/test.py"
        assert result.start_line == 1
        assert result.end_line == 10

    def test_default_metrics(self):
        """Test default metric values."""
        result = ComplexityResult(
            name="func",
            file_path="/file.py",
            start_line=1,
            end_line=5,
        )
        assert result.cyclomatic.value == 1
        assert result.cognitive.value == 0
        assert result.halstead.n1 == 0
        assert result.loc.total == 0
        assert result.maintainability.value == 100.0

    def test_with_metrics(self):
        """Test result with all metrics."""
        result = ComplexityResult(
            name="complex_function",
            file_path="/test.py",
            start_line=1,
            end_line=50,
            cyclomatic=CyclomaticComplexity(value=15),
            cognitive=CognitiveComplexity(value=20),
            halstead=HalsteadMetrics(n1=20, n2=30, N1=100, N2=150),
            loc=LinesOfCode(total=50, code=40, comments=5, blank=5),
        )
        assert result.cyclomatic.value == 15
        assert result.cognitive.value == 20
        assert result.halstead is not None
        assert result.loc.total == 50

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ComplexityResult(
            name="func",
            file_path="/test.py",
            start_line=1,
            end_line=20,
            cyclomatic=CyclomaticComplexity(value=5),
        )
        data = result.to_dict()
        assert data["name"] == "func"
        assert data["file_path"] == "/test.py"
        assert data["start_line"] == 1
        assert data["end_line"] == 20
        assert "cyclomatic" in data
        assert data["cyclomatic"]["value"] == 5


class TestFileComplexity:
    """Tests for FileComplexity dataclass."""

    def test_basic_creation(self):
        """Test creating basic file complexity."""
        fc = FileComplexity(file_path="/test.py")
        assert fc.file_path == "/test.py"
        assert fc.total_cyclomatic == 0
        assert fc.total_cognitive == 0

    def test_with_aggregates(self):
        """Test with aggregate values."""
        fc = FileComplexity(
            file_path="/test.py",
            total_cyclomatic=50,
            total_cognitive=30,
            avg_cyclomatic=10.0,
            avg_cognitive=6.0,
            max_cyclomatic=20,
            max_cognitive=15,
            function_count=5,
            class_count=2,
        )
        assert fc.total_cyclomatic == 50
        assert fc.total_cognitive == 30
        assert fc.avg_cyclomatic == 10.0
        assert fc.avg_cognitive == 6.0
        assert fc.max_cyclomatic == 20
        assert fc.max_cognitive == 15
        assert fc.function_count == 5
        assert fc.class_count == 2

    def test_hotspots(self):
        """Test hotspots list."""
        fc = FileComplexity(
            file_path="/file.py",
            hotspots=["complex_function", "another_complex"],
        )
        assert len(fc.hotspots) == 2
        assert "complex_function" in fc.hotspots

    def test_to_dict(self):
        """Test serialization to dict."""
        fc = FileComplexity(
            file_path="/test.py",
            total_cyclomatic=25,
            function_count=3,
            hotspots=["bad_func"],
        )
        data = fc.to_dict()
        assert data["file_path"] == "/test.py"
        assert data["total_cyclomatic"] == 25
        assert data["function_count"] == 3
        assert data["hotspots"] == ["bad_func"]
        assert "loc" in data
        assert "maintainability" in data


class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer class."""

    def test_default_language(self):
        """Test default language is Python."""
        analyzer = ComplexityAnalyzer()
        assert analyzer.language == "python"

    def test_with_language(self):
        """Test with specific language."""
        analyzer = ComplexityAnalyzer(language="typescript")
        assert analyzer.language == "typescript"

    def test_simple_function(self):
        """Test analyzing simple function."""
        source = '''def simple():
    return 42
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "simple")

        assert result.name == "simple"
        assert result.cyclomatic.value >= 1

    def test_function_with_if(self):
        """Test function with if statement."""
        source = '''def check(x):
    if x > 0:
        return "positive"
    return "non-positive"
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "check")

        # Cyclomatic: 1 base + 1 for if = 2
        assert result.cyclomatic.value >= 2

    def test_function_with_multiple_branches(self):
        """Test function with multiple branches."""
        source = '''def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "classify")

        # Multiple branches increase complexity
        assert result.cyclomatic.value >= 3

    def test_function_with_loop(self):
        """Test function with loop."""
        source = '''def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "sum_list")

        # Loop adds to cyclomatic complexity
        assert result.cyclomatic.value >= 2

    def test_function_with_try_except(self):
        """Test function with try/except."""
        source = '''def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "safe_divide")

        # Exception handling adds complexity
        assert result.cyclomatic.value >= 2

    def test_nested_complexity(self):
        """Test nested structures increase cognitive complexity."""
        source = '''def nested(items):
    for item in items:
        if item > 0:
            for sub in item:
                if sub < 0:
                    print(sub)
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "nested")

        # Nesting should increase cognitive complexity
        assert result.cognitive.value > 0
        assert result.cognitive.nesting_penalty > 0

    def test_halstead_metrics(self):
        """Test Halstead metrics calculation."""
        source = '''def calculate(a, b, c):
    x = a + b
    y = b * c
    z = x - y
    return z
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "calculate")

        assert result.halstead is not None
        assert result.halstead.vocabulary > 0
        assert result.halstead.volume > 0

    def test_lines_of_code(self):
        """Test LOC calculation."""
        source = '''def example():
    # This is a comment
    x = 1

    # Another comment
    y = 2

    return x + y
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "example")

        assert result.loc is not None
        assert result.loc.total >= 8
        assert result.loc.comments >= 2
        assert result.loc.blank >= 2

    def test_maintainability_index(self):
        """Test maintainability index calculation."""
        source = '''def simple_func():
    return 1
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "simple_func")

        assert result.maintainability is not None
        # Simple function should have maintainability value
        assert result.maintainability.value >= 0

    def test_analyze_file_no_symbols(self):
        """Test analyzing file without symbols."""
        source = '''x = 1
y = 2
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_file(source, "/test.py")

        assert result.file_path == "/test.py"
        assert result.function_count == 0

    def test_complexity_levels(self):
        """Test complexity level assignment."""
        analyzer = ComplexityAnalyzer()

        # Low complexity
        simple = "def f(): return 1"
        result = analyzer.analyze_source(simple, "/test.py", "f")
        assert result.cyclomatic.level == ComplexityLevel.LOW

    def test_high_complexity_detection(self):
        """Test detecting high complexity code."""
        # Generate code with many branches
        branches = "\n".join([f"    elif x == {i}: return {i}" for i in range(30)])
        source = f'''def complex_switch(x):
    if x == 0:
        return 0
{branches}
    return -1
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "complex_switch")

        # Should be high or very high complexity
        assert result.cyclomatic.level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)

    def test_empty_function(self):
        """Test analyzing empty function."""
        source = '''def empty():
    pass
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "empty")

        assert result.cyclomatic.value >= 1  # Base complexity

    def test_typescript_analyzer(self):
        """Test TypeScript analyzer."""
        source = '''function greet(name: string): string {
    if (name) {
        return "Hello, " + name;
    }
    return "Hello";
}
'''
        analyzer = ComplexityAnalyzer(language="typescript")
        result = analyzer.analyze_source(source)

        assert result.cyclomatic.value >= 1

    def test_go_analyzer(self):
        """Test Go analyzer."""
        source = '''func check(x int) string {
    if x > 0 {
        return "positive"
    }
    return "non-positive"
}
'''
        analyzer = ComplexityAnalyzer(language="go")
        result = analyzer.analyze_source(source)

        assert result.cyclomatic.value >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_source(self):
        """Test analyzing empty source."""
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source("", "/test.py", "empty")

        assert result.cyclomatic.value >= 1
        assert result.loc.total >= 1  # At least one line (empty)

    def test_only_comments(self):
        """Test source with only comments."""
        source = '''# Comment 1
# Comment 2
# Comment 3
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "comments")

        assert result.loc.comments >= 3
        assert result.loc.code == 0

    def test_boolean_operators(self):
        """Test boolean operators increase complexity."""
        source = '''def check(a, b, c):
    if a and b or c:
        return True
    return False
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "check")

        # 'and' and 'or' should add to complexity
        assert result.cyclomatic.value >= 3

    def test_lambda_not_counted_separately(self):
        """Test lambdas in the same analysis unit."""
        source = '''def process(items):
    return list(filter(lambda x: x > 0, items))
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "process")

        # Lambda adds minimal complexity
        assert result.cyclomatic.value >= 1

    def test_comprehension_complexity(self):
        """Test list comprehensions with conditions."""
        source = '''def filter_positive(items):
    return [x for x in items if x > 0]
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "filter_positive")

        # Comprehension with condition adds complexity
        assert result.cyclomatic.value >= 2

    def test_multiline_string(self):
        """Test with multiline strings (docstrings)."""
        source = '''def func():
    """
    This is a docstring.
    It spans multiple lines.
    """
    return 1
'''
        analyzer = ComplexityAnalyzer()
        result = analyzer.analyze_source(source, "/test.py", "func")

        assert result.loc.comments > 0
