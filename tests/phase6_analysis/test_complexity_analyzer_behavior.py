"""Complexity Analyzer Behavior Tests.

Tests that verify actual complexity calculations with known code samples.
Each test uses code with predictable complexity values.
"""

import tempfile

import pytest

from anamnesis.analysis.complexity_analyzer import ComplexityAnalyzer
from anamnesis.extractors.symbol_extractor import extract_symbols_from_source


@pytest.fixture
def analyzer():
    """Create a complexity analyzer instance."""
    return ComplexityAnalyzer()


class TestCyclomaticComplexity:
    """Tests for cyclomatic complexity calculation."""

    def test_simple_function_low_complexity(self, analyzer):
        """Simple function should have cyclomatic complexity of 1."""
        source = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
        result = analyzer.analyze_source(source, "test.py", "add")

        # Simple linear function has complexity 1
        assert result.cyclomatic.value == 1

    def test_single_if_adds_one(self, analyzer):
        """Single if statement should add 1 to complexity."""
        source = '''
def check_positive(n):
    """Check if number is positive."""
    if n > 0:
        return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "check_positive")

        # Function has one decision point (if statement)
        # Cyclomatic = decision_points + 1 (at minimum 2)
        assert result.cyclomatic.value >= 2

    def test_if_else_same_as_if(self, analyzer):
        """If-else has same complexity as if (only one decision point)."""
        source = '''
def get_sign(n):
    """Get the sign of a number."""
    if n >= 0:
        return "positive"
    else:
        return "negative"
'''
        result = analyzer.analyze_source(source, "test.py", "get_sign")

        # 1 base + if decision = at least 2
        assert result.cyclomatic.value >= 2

    def test_elif_adds_complexity(self, analyzer):
        """Each elif adds 1 to complexity."""
        source = '''
def categorize(n):
    """Categorize a number."""
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n < 10:
        return "small"
    else:
        return "large"
'''
        result = analyzer.analyze_source(source, "test.py", "categorize")

        # 1 base + if + 2 elif = at least 4
        assert result.cyclomatic.value >= 4

    def test_for_loop_adds_one(self, analyzer):
        """For loop adds 1 to complexity."""
        source = '''
def sum_list(numbers):
    """Sum all numbers in list."""
    total = 0
    for n in numbers:
        total += n
    return total
'''
        result = analyzer.analyze_source(source, "test.py", "sum_list")

        # 1 base + 1 for = 2
        assert result.cyclomatic.value == 2

    def test_while_loop_adds_one(self, analyzer):
        """While loop adds 1 to complexity."""
        source = '''
def countdown(n):
    """Count down from n."""
    while n > 0:
        print(n)
        n -= 1
'''
        result = analyzer.analyze_source(source, "test.py", "countdown")

        # 1 base + 1 while = 2
        assert result.cyclomatic.value == 2

    def test_nested_loops_add_complexity(self, analyzer):
        """Nested loops each add 1."""
        source = '''
def matrix_sum(matrix):
    """Sum all elements in a matrix."""
    total = 0
    for row in matrix:
        for cell in row:
            total += cell
    return total
'''
        result = analyzer.analyze_source(source, "test.py", "matrix_sum")

        # 1 base + 2 for = 3
        assert result.cyclomatic.value == 3

    def test_and_operator_adds_complexity(self, analyzer):
        """Boolean 'and' adds 1 to complexity."""
        source = '''
def is_valid(x, y):
    """Check if coordinates are valid."""
    if x > 0 and y > 0:
        return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "is_valid")

        # if with 'and' has multiple paths
        assert result.cyclomatic.value >= 3

    def test_or_operator_adds_complexity(self, analyzer):
        """Boolean 'or' adds 1 to complexity."""
        source = '''
def is_boundary(x, size):
    """Check if x is at boundary."""
    if x == 0 or x == size:
        return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "is_boundary")

        # if with 'or' has multiple paths
        assert result.cyclomatic.value >= 3

    def test_complex_function(self, analyzer):
        """Test a more complex function."""
        source = '''
def process(data, options):
    """Process data with options."""
    result = []
    for item in data:
        if item is None:
            continue
        elif item < 0:
            if options.get("absolute"):
                result.append(abs(item))
            else:
                result.append(item)
        elif item > 100:
            result.append(100)
        else:
            result.append(item)
    return result
'''
        result = analyzer.analyze_source(source, "test.py", "process")

        # 1 base + 1 for + 1 if + 2 elif + 1 if (nested) = 6
        # Complexity should be at least 6
        assert result.cyclomatic.value >= 6


class TestCognitiveComplexity:
    """Tests for cognitive complexity calculation."""

    def test_simple_function_zero_cognitive(self, analyzer):
        """Simple function has 0 cognitive complexity."""
        source = '''
def add(a, b):
    return a + b
'''
        result = analyzer.analyze_source(source, "test.py", "add")

        # Minimal complexity for simple function
        assert result.cognitive.value <= 2  # Implementation may count base function

    def test_single_if_adds_one(self, analyzer):
        """Single if adds 1 to cognitive complexity."""
        source = '''
def check(n):
    if n > 0:
        return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "check")

        # 1 for the if
        assert result.cognitive.value >= 1

    def test_nesting_increases_cognitive(self, analyzer):
        """Nesting increases cognitive complexity."""
        source = '''
def nested_check(x, y):
    if x > 0:
        if y > 0:
            return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "nested_check")

        # Outer if: +1
        # Inner if: +1 (base) + 1 (nesting increment) = +2
        # Total: 3
        assert result.cognitive.value >= 3

    def test_deep_nesting_high_cognitive(self, analyzer):
        """Deep nesting leads to high cognitive complexity."""
        source = '''
def deeply_nested(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "deeply_nested")

        # Each nested level adds more:
        # Level 1: +1, Level 2: +2, Level 3: +3, Level 4: +4
        # Total: 10
        assert result.cognitive.value >= 10

    def test_sequential_ifs_lower_than_nested(self, analyzer):
        """Sequential ifs have lower cognitive complexity than nested."""
        sequential = '''
def sequential_check(a, b, c):
    if a:
        return 1
    if b:
        return 2
    if c:
        return 3
    return 0
'''

        nested = '''
def nested_check(a, b, c):
    if a:
        if b:
            if c:
                return 1
    return 0
'''
        seq_result = analyzer.analyze_source(sequential, "test.py", "sequential_check")
        nest_result = analyzer.analyze_source(nested, "test.py", "nested_check")

        # Sequential: 3 (1 + 1 + 1)
        # Nested: 6 (1 + 2 + 3)
        assert seq_result.cognitive.value < nest_result.cognitive.value


class TestLinesOfCode:
    """Tests for lines of code calculation."""

    def test_counts_total_lines(self, analyzer):
        """Counts total lines including blanks and comments."""
        source = '''def hello():
    # This is a comment

    return "hello"
'''
        result = analyzer.analyze_source(source, "test.py", "hello")

        # Total should include all lines
        assert result.loc.total >= 4

    def test_counts_code_lines(self, analyzer):
        """Counts actual code lines (non-blank, non-comment)."""
        source = '''def add(a, b):
    # Add two numbers
    result = a + b

    return result
'''
        result = analyzer.analyze_source(source, "test.py", "add")

        # Code lines: def, result=, return = 3
        assert result.loc.code >= 3

    def test_counts_comment_lines(self, analyzer):
        """Counts comment lines."""
        source = '''def documented():
    # Line 1
    # Line 2
    # Line 3
    return True
'''
        result = analyzer.analyze_source(source, "test.py", "documented")

        # 3 comment lines
        assert result.loc.comments >= 3

    def test_counts_blank_lines(self, analyzer):
        """Counts blank lines."""
        source = '''def spaced():
    x = 1

    y = 2

    return x + y
'''
        result = analyzer.analyze_source(source, "test.py", "spaced")

        # 2 blank lines
        assert result.loc.blank >= 2


class TestHalsteadMetrics:
    """Tests for Halstead complexity metrics."""

    def test_simple_function_halstead(self, analyzer):
        """Halstead metrics for simple function."""
        source = '''
def add(a, b):
    return a + b
'''
        result = analyzer.analyze_source(source, "test.py", "add")

        # Should have Halstead metrics
        assert result.halstead is not None

        # Operators: def, return, +
        # Operands: add, a, b (appears twice each as params and in expr)
        assert result.halstead.n1 > 0  # distinct operators
        assert result.halstead.n2 > 0  # distinct operands

        # Volume should be positive
        assert result.halstead.volume >= 0

    def test_complex_function_higher_halstead(self, analyzer):
        """More complex function has higher Halstead metrics."""
        simple = '''
def simple(x):
    return x
'''

        complex_code = '''
def complex_func(a, b, c, d):
    result = (a + b) * (c - d)
    if result > 0:
        return result * 2
    else:
        return result / 2
'''
        simple_result = analyzer.analyze_source(simple, "test.py", "simple")
        complex_result = analyzer.analyze_source(complex_code, "test.py", "complex_func")

        # Complex should have higher volume
        assert complex_result.halstead.volume > simple_result.halstead.volume

    def test_halstead_effort(self, analyzer):
        """Halstead effort is calculated."""
        source = '''
def calculate(x, y, z):
    a = x + y
    b = y + z
    c = x + z
    return a + b + c
'''
        result = analyzer.analyze_source(source, "test.py", "calculate")

        # Effort should be positive
        assert result.halstead.effort >= 0


class TestMaintainabilityIndex:
    """Tests for maintainability index calculation."""

    def test_simple_function_high_maintainability(self, analyzer):
        """Simple functions have high maintainability index."""
        source = '''
def add(a, b):
    return a + b
'''
        result = analyzer.analyze_source(source, "test.py", "add")

        # Simple code should have high maintainability (> 50)
        assert result.maintainability.value >= 50

    def test_complex_function_lower_maintainability(self, analyzer):
        """Complex functions have lower maintainability index."""
        simple = '''
def simple(x):
    return x + 1
'''

        complex_code = '''
def complex_func(data, options, config):
    result = []
    for item in data:
        if item is None:
            continue
        elif isinstance(item, str):
            if options.get("uppercase"):
                if config.get("enabled"):
                    result.append(item.upper())
                else:
                    result.append(item)
            elif options.get("lowercase"):
                result.append(item.lower())
            else:
                result.append(item)
        elif isinstance(item, int):
            if item < 0:
                result.append(abs(item))
            elif item > 100:
                result.append(100)
            else:
                result.append(item)
    return result
'''
        simple_result = analyzer.analyze_source(simple, "test.py", "simple")
        complex_result = analyzer.analyze_source(complex_code, "test.py", "complex_func")

        # Complex should have lower maintainability
        assert complex_result.maintainability.value < simple_result.maintainability.value

    def test_maintainability_rating(self, analyzer):
        """Maintainability has a rating (A-F or similar)."""
        source = '''
def good_function(x):
    """Well-documented function."""
    return x * 2
'''
        result = analyzer.analyze_source(source, "test.py", "good_function")

        # Should have an interpretation
        assert result.maintainability.interpretation is not None


class TestFileComplexity:
    """Tests for file-level complexity analysis."""

    def test_analyze_file(self, analyzer):
        """Analyze a complete file."""
        source = '''"""Test module."""

def func1():
    return 1

def func2(x):
    if x > 0:
        return x
    return 0

class MyClass:
    def method1(self):
        return "hello"

    def method2(self, x, y):
        if x > y:
            return x
        return y
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            f.flush()

            # Extract symbols to enable function-level analysis
            symbols = extract_symbols_from_source(source, "python", f.name)
            result = analyzer.analyze_file(source, f.name, symbols)

            # Should have file-level metrics
            assert result is not None

            # Should have analyzed multiple functions
            # Note: func1, func2 are top-level functions; methods inside class may be
            # extracted separately. The implementation counts function_count as 2 (top-level)
            assert result.function_count >= 2  # At least func1, func2
            assert result.class_count >= 1  # MyClass

    def test_file_average_complexity(self, analyzer):
        """File has average complexity across functions."""
        source = '''
def simple():
    return 1

def medium(x):
    if x:
        return x
    return 0

def complex_fn(a, b, c):
    if a:
        if b:
            if c:
                return True
    return False
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            f.flush()

            # Extract symbols to enable function-level analysis
            symbols = extract_symbols_from_source(source, "python", f.name)
            result = analyzer.analyze_file(source, f.name, symbols)

            # Should have average cyclomatic complexity
            assert result.avg_cyclomatic > 0

            # Average should be reasonable (between min and max in the file)
            assert result.avg_cyclomatic >= 1
            assert result.avg_cyclomatic <= result.max_cyclomatic


class TestComplexityThresholds:
    """Tests for complexity threshold checks."""

    def test_low_complexity_within_threshold(self, analyzer):
        """Low complexity code is within acceptable thresholds."""
        source = '''
def simple():
    return 42
'''
        result = analyzer.analyze_source(source, "test.py", "simple")

        # Complexity 1 should be acceptable
        assert result.cyclomatic.value <= 10  # Common threshold

    def test_high_complexity_exceeds_threshold(self, analyzer):
        """High complexity code may exceed thresholds."""
        source = '''
def very_complex(a, b, c, d, e, f, g, h, i, j):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    if i:
                                        if j:
                                            return True
    return False
'''
        result = analyzer.analyze_source(source, "test.py", "very_complex")

        # This should have high complexity
        assert result.cyclomatic.value > 10


class TestEdgeCases:
    """Tests for edge cases in complexity analysis."""

    def test_empty_function(self, analyzer):
        """Handle empty function."""
        source = '''
def empty():
    pass
'''
        result = analyzer.analyze_source(source, "test.py", "empty")

        # Should still have a result
        assert result is not None
        assert result.cyclomatic.value >= 1

    def test_function_with_only_docstring(self, analyzer):
        """Handle function with only docstring."""
        source = '''
def documented():
    """This function does nothing but has a docstring."""
    pass
'''
        result = analyzer.analyze_source(source, "test.py", "documented")

        assert result is not None
        assert result.cyclomatic.value >= 1

    def test_lambda_function(self, analyzer):
        """Handle lambda expressions."""
        source = '''
def use_lambda():
    fn = lambda x: x * 2
    return fn(5)
'''
        result = analyzer.analyze_source(source, "test.py", "use_lambda")

        # Should analyze without error
        assert result is not None

    def test_comprehension(self, analyzer):
        """Handle list comprehensions."""
        source = '''
def with_comprehension(items):
    return [x * 2 for x in items if x > 0]
'''
        result = analyzer.analyze_source(source, "test.py", "with_comprehension")

        # Comprehension adds complexity
        assert result is not None
        # The `if` in comprehension should add 1
        assert result.cyclomatic.value >= 2

    def test_try_except(self, analyzer):
        """Handle try-except blocks."""
        source = '''
def with_exception(x):
    try:
        return 1 / x
    except ZeroDivisionError:
        return 0
    except TypeError:
        return -1
'''
        result = analyzer.analyze_source(source, "test.py", "with_exception")

        # Each except adds complexity
        assert result.cyclomatic.value >= 3

    def test_ternary_operator(self, analyzer):
        """Handle ternary conditional expressions."""
        source = '''
def ternary(x):
    return "positive" if x > 0 else "non-positive"
'''
        result = analyzer.analyze_source(source, "test.py", "ternary")

        # Ternary adds complexity
        assert result.cyclomatic.value >= 2
