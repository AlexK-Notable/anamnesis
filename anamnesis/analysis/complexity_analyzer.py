"""Code complexity analyzer.

This module provides complexity analysis including:
- Cyclomatic complexity (control flow paths)
- Cognitive complexity (mental effort to understand)
- Maintainability index
- Lines of code metrics
- Halstead complexity metrics
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anamnesis.extractors.symbol_extractor import ExtractedSymbol


# Pre-compiled regex patterns for hot-path methods
_RE_IDENTIFIER = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_RE_NUMBER = re.compile(r"\b\d+\.?\d*\b")
_RE_MULTILINE_DQUOTE = re.compile(r'""".*?"""', re.DOTALL)
_RE_MULTILINE_SQUOTE = re.compile(r"'''.*?'''", re.DOTALL)
_RE_SINGLE_DQUOTE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')
_RE_SINGLE_SQUOTE = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'")
_RE_HASH_COMMENT = re.compile(r"#.*$", re.MULTILINE)
_RE_SLASH_COMMENT = re.compile(r"//.*$", re.MULTILINE)
_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RE_WORD = re.compile(r"\b\w+\b")
_RE_PYTHON_TERNARY = re.compile(r"\bif\b.*\belse\b")


class ComplexityLevel(str, Enum):
    """Complexity level classifications."""

    LOW = "low"  # 1-10
    MODERATE = "moderate"  # 11-20
    HIGH = "high"  # 21-50
    VERY_HIGH = "very_high"  # 50+


@dataclass
class CyclomaticComplexity:
    """Cyclomatic complexity result for a code unit."""

    value: int
    decision_points: int = 0
    level: ComplexityLevel = ComplexityLevel.LOW

    def __post_init__(self) -> None:
        """Set complexity level based on value."""
        if self.value <= 10:
            self.level = ComplexityLevel.LOW
        elif self.value <= 20:
            self.level = ComplexityLevel.MODERATE
        elif self.value <= 50:
            self.level = ComplexityLevel.HIGH
        else:
            self.level = ComplexityLevel.VERY_HIGH

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "value": self.value,
            "decision_points": self.decision_points,
            "level": self.level.value,
        }


@dataclass
class CognitiveComplexity:
    """Cognitive complexity result for a code unit."""

    value: int
    nesting_penalty: int = 0
    structural_complexity: int = 0
    level: ComplexityLevel = ComplexityLevel.LOW

    def __post_init__(self) -> None:
        """Set complexity level based on value."""
        if self.value <= 5:
            self.level = ComplexityLevel.LOW
        elif self.value <= 15:
            self.level = ComplexityLevel.MODERATE
        elif self.value <= 25:
            self.level = ComplexityLevel.HIGH
        else:
            self.level = ComplexityLevel.VERY_HIGH

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "value": self.value,
            "nesting_penalty": self.nesting_penalty,
            "structural_complexity": self.structural_complexity,
            "level": self.level.value,
        }


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics."""

    n1: int = 0  # Distinct operators
    n2: int = 0  # Distinct operands
    N1: int = 0  # Total operators
    N2: int = 0  # Total operands

    @property
    def vocabulary(self) -> int:
        """Program vocabulary (n = n1 + n2)."""
        return self.n1 + self.n2

    @property
    def length(self) -> int:
        """Program length (N = N1 + N2)."""
        return self.N1 + self.N2

    @property
    def calculated_length(self) -> float:
        """Calculated program length."""
        if self.n1 == 0 or self.n2 == 0:
            return 0
        return self.n1 * math.log2(self.n1) + self.n2 * math.log2(self.n2)

    @property
    def volume(self) -> float:
        """Program volume (V = N * log2(n))."""
        if self.vocabulary == 0:
            return 0
        return self.length * math.log2(self.vocabulary)

    @property
    def difficulty(self) -> float:
        """Program difficulty (D = (n1/2) * (N2/n2))."""
        if self.n2 == 0:
            return 0
        return (self.n1 / 2) * (self.N2 / self.n2)

    @property
    def effort(self) -> float:
        """Programming effort (E = D * V)."""
        return self.difficulty * self.volume

    @property
    def time_to_program(self) -> float:
        """Time to program in seconds (T = E / 18)."""
        return self.effort / 18

    @property
    def bugs_delivered(self) -> float:
        """Estimated bugs (B = V / 3000)."""
        return self.volume / 3000

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "distinct_operators": self.n1,
            "distinct_operands": self.n2,
            "total_operators": self.N1,
            "total_operands": self.N2,
            "vocabulary": self.vocabulary,
            "length": self.length,
            "volume": round(self.volume, 2),
            "difficulty": round(self.difficulty, 2),
            "effort": round(self.effort, 2),
            "time_to_program": round(self.time_to_program, 2),
            "bugs_delivered": round(self.bugs_delivered, 4),
        }


@dataclass
class LinesOfCode:
    """Lines of code metrics."""

    total: int = 0
    code: int = 0  # Non-blank, non-comment
    comments: int = 0
    blank: int = 0
    logical: int = 0  # Logical lines (statements)

    @property
    def comment_ratio(self) -> float:
        """Ratio of comments to code."""
        if self.code == 0:
            return 0
        return self.comments / self.code

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total": self.total,
            "code": self.code,
            "comments": self.comments,
            "blank": self.blank,
            "logical": self.logical,
            "comment_ratio": round(self.comment_ratio, 3),
        }


@dataclass
class MaintainabilityIndex:
    """Maintainability index result."""

    value: float
    interpretation: str = ""

    def __post_init__(self) -> None:
        """Set interpretation based on value."""
        if self.value >= 85:
            self.interpretation = "highly maintainable"
        elif self.value >= 65:
            self.interpretation = "moderately maintainable"
        elif self.value >= 20:
            self.interpretation = "difficult to maintain"
        else:
            self.interpretation = "unmaintainable"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "value": round(self.value, 2),
            "interpretation": self.interpretation,
        }


@dataclass
class ComplexityResult:
    """Complete complexity analysis result for a code unit."""

    name: str
    file_path: str
    start_line: int
    end_line: int
    cyclomatic: CyclomaticComplexity = field(default_factory=lambda: CyclomaticComplexity(1))
    cognitive: CognitiveComplexity = field(default_factory=lambda: CognitiveComplexity(0))
    halstead: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    loc: LinesOfCode = field(default_factory=LinesOfCode)
    maintainability: MaintainabilityIndex = field(default_factory=lambda: MaintainabilityIndex(100.0))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cyclomatic": self.cyclomatic.to_dict(),
            "cognitive": self.cognitive.to_dict(),
            "halstead": self.halstead.to_dict(),
            "loc": self.loc.to_dict(),
            "maintainability": self.maintainability.to_dict(),
        }


@dataclass
class FileComplexity:
    """Aggregated complexity for an entire file."""

    file_path: str
    total_cyclomatic: int = 0
    total_cognitive: int = 0
    avg_cyclomatic: float = 0.0
    avg_cognitive: float = 0.0
    max_cyclomatic: int = 0
    max_cognitive: int = 0
    loc: LinesOfCode = field(default_factory=LinesOfCode)
    maintainability: MaintainabilityIndex = field(default_factory=lambda: MaintainabilityIndex(100.0))
    function_count: int = 0
    class_count: int = 0
    hotspots: list[str] = field(default_factory=list)  # High complexity functions

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "total_cyclomatic": self.total_cyclomatic,
            "total_cognitive": self.total_cognitive,
            "avg_cyclomatic": round(self.avg_cyclomatic, 2),
            "avg_cognitive": round(self.avg_cognitive, 2),
            "max_cyclomatic": self.max_cyclomatic,
            "max_cognitive": self.max_cognitive,
            "loc": self.loc.to_dict(),
            "maintainability": self.maintainability.to_dict(),
            "function_count": self.function_count,
            "class_count": self.class_count,
            "hotspots": self.hotspots,
        }


class ComplexityAnalyzer:
    """Analyzes code complexity using various metrics."""

    # Python control flow keywords that add to complexity
    PYTHON_DECISION_KEYWORDS = {
        "if", "elif", "else", "for", "while", "except", "with",
        "and", "or", "assert", "case",  # Python 3.10+ match
    }

    # TypeScript/JavaScript control flow keywords
    TS_DECISION_KEYWORDS = {
        "if", "else", "for", "while", "do", "switch", "case",
        "catch", "finally", "&&", "||", "?", ":",
    }

    # Go control flow keywords
    GO_DECISION_KEYWORDS = {
        "if", "else", "for", "switch", "case", "select",
        "defer", "go", "&&", "||",
    }

    # Rust control flow keywords
    RUST_DECISION_KEYWORDS = {
        "if", "else", "for", "while", "loop", "match",
        "?", "&&", "||",
    }

    # Operators for Halstead metrics
    OPERATORS = {
        "=", "+", "-", "*", "/", "%", "**",
        "==", "!=", "<", ">", "<=", ">=",
        "and", "or", "not", "in", "is",
        "&&", "||", "!",
        ".", ",", ":", ";",
        "(", ")", "[", "]", "{", "}",
        "->", "=>", "::",
    }

    def __init__(self, language: str = "python") -> None:
        """Initialize complexity analyzer.

        Args:
            language: Target language for analysis.
        """
        self.language = language
        self._decision_keywords = self._get_decision_keywords(language)

        # Pre-compile regex patterns for word-boundary keyword matching
        self._keyword_patterns: dict[str, re.Pattern[str]] = {}
        for keyword in self._decision_keywords:
            if keyword not in ("&&", "||", "?", ":"):
                self._keyword_patterns[keyword] = re.compile(rf"\b{keyword}\b")

    def _get_decision_keywords(self, language: str) -> set[str]:
        """Get decision keywords for the language."""
        lang_map = {
            "python": self.PYTHON_DECISION_KEYWORDS,
            "typescript": self.TS_DECISION_KEYWORDS,
            "javascript": self.TS_DECISION_KEYWORDS,
            "go": self.GO_DECISION_KEYWORDS,
            "rust": self.RUST_DECISION_KEYWORDS,
        }
        return lang_map.get(language, self.PYTHON_DECISION_KEYWORDS)

    def analyze_source(
        self,
        source: str,
        file_path: str = "<unknown>",
        name: str = "<module>",
    ) -> ComplexityResult:
        """Analyze complexity of source code.

        Args:
            source: Source code string.
            file_path: Path to the source file.
            name: Name of the code unit.

        Returns:
            ComplexityResult with all metrics.
        """
        lines = source.split("\n")

        # Calculate LOC metrics
        loc = self._calculate_loc(lines)

        # Calculate cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic(source)

        # Calculate cognitive complexity
        cognitive = self._calculate_cognitive(source)

        # Calculate Halstead metrics
        halstead = self._calculate_halstead(source)

        # Calculate maintainability index
        maintainability = self._calculate_maintainability(
            halstead.volume, cyclomatic.value, loc.code
        )

        return ComplexityResult(
            name=name,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            cyclomatic=cyclomatic,
            cognitive=cognitive,
            halstead=halstead,
            loc=loc,
            maintainability=maintainability,
        )

    def analyze_symbol(
        self,
        symbol: ExtractedSymbol,
        source: str,
    ) -> ComplexityResult:
        """Analyze complexity of an extracted symbol.

        Args:
            symbol: Extracted symbol to analyze.
            source: Full source code of the file.

        Returns:
            ComplexityResult for the symbol.
        """
        # Extract symbol's source code
        lines = source.split("\n")
        start = symbol.start_line - 1
        end = symbol.end_line
        symbol_source = "\n".join(lines[start:end])

        result = self.analyze_source(
            symbol_source,
            file_path=symbol.file_path,
            name=symbol.name,
        )
        result.start_line = symbol.start_line
        result.end_line = symbol.end_line

        return result

    def analyze_file(
        self,
        source: str,
        file_path: str,
        symbols: list[ExtractedSymbol] | None = None,
    ) -> FileComplexity:
        """Analyze complexity of an entire file.

        Args:
            source: Source code of the file.
            file_path: Path to the file.
            symbols: Optional list of extracted symbols.

        Returns:
            FileComplexity with aggregated metrics.
        """
        from anamnesis.extraction.types import SymbolKind

        lines = source.split("\n")
        loc = self._calculate_loc(lines)

        # Analyze each symbol if provided
        results: list[ComplexityResult] = []
        if symbols:
            for symbol in symbols:
                if symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    result = self.analyze_symbol(symbol, source)
                    results.append(result)

        # Calculate aggregates
        function_count = len(results)
        class_count = sum(1 for s in (symbols or []) if s.kind == SymbolKind.CLASS)

        cyclomatic_values = [r.cyclomatic.value for r in results]
        cognitive_values = [r.cognitive.value for r in results]

        total_cyclomatic = sum(cyclomatic_values)
        total_cognitive = sum(cognitive_values)
        avg_cyclomatic = total_cyclomatic / function_count if function_count else 0
        avg_cognitive = total_cognitive / function_count if function_count else 0
        max_cyclomatic = max(cyclomatic_values) if cyclomatic_values else 0
        max_cognitive = max(cognitive_values) if cognitive_values else 0

        # Identify hotspots (high complexity functions)
        hotspots = [
            r.name for r in results
            if r.cyclomatic.level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)
        ]

        # Calculate file-level maintainability
        file_result = self.analyze_source(source, file_path, file_path)

        return FileComplexity(
            file_path=file_path,
            total_cyclomatic=total_cyclomatic,
            total_cognitive=total_cognitive,
            avg_cyclomatic=avg_cyclomatic,
            avg_cognitive=avg_cognitive,
            max_cyclomatic=max_cyclomatic,
            max_cognitive=max_cognitive,
            loc=loc,
            maintainability=file_result.maintainability,
            function_count=function_count,
            class_count=class_count,
            hotspots=hotspots,
        )

    def _calculate_loc(self, lines: list[str]) -> LinesOfCode:
        """Calculate lines of code metrics."""
        total = len(lines)
        blank = 0
        comments = 0
        code = 0
        logical = 0

        in_multiline_comment = False
        in_multiline_string = False

        for line in lines:
            stripped = line.strip()

            # Empty line
            if not stripped:
                blank += 1
                continue

            # Python multiline strings (as comments)
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                count = stripped.count(quote)
                if count == 1:
                    in_multiline_string = not in_multiline_string
                    comments += 1
                elif count >= 2:
                    comments += 1
                continue

            if in_multiline_string:
                comments += 1
                continue

            # Single-line comments
            if stripped.startswith("#") or stripped.startswith("//"):
                comments += 1
                continue

            # Block comment start (/* ... */)
            if "/*" in stripped:
                in_multiline_comment = True
                if "*/" in stripped:
                    in_multiline_comment = False
                comments += 1
                continue

            if in_multiline_comment:
                if "*/" in stripped:
                    in_multiline_comment = False
                comments += 1
                continue

            # Code line
            code += 1

            # Count logical lines (statements)
            # This is a simplified heuristic
            if any(
                kw in stripped
                for kw in ["def ", "class ", "if ", "for ", "while ", "return ", "yield "]
            ):
                logical += 1
            elif stripped.endswith(";") or stripped.endswith(":"):
                logical += 1
            elif "=" in stripped and not any(
                op in stripped for op in ["==", "!=", "<=", ">="]
            ):
                logical += 1

        return LinesOfCode(
            total=total,
            code=code,
            comments=comments,
            blank=blank,
            logical=logical,
        )

    def _calculate_cyclomatic(self, source: str) -> CyclomaticComplexity:
        """Calculate cyclomatic complexity."""
        # Start with 1 (base complexity)
        complexity = 1
        decision_points = 0

        # Count decision points
        for keyword in self._decision_keywords:
            # Use word boundaries for keyword matching
            if keyword in ("&&", "||", "?", ":"):
                count = source.count(keyword)
            else:
                # Match as whole word (pre-compiled in __init__)
                count = len(self._keyword_patterns[keyword].findall(source))

            decision_points += count
            complexity += count

        # Also count ternary operators and short-circuit operators
        if self.language == "python":
            # Python ternary: x if condition else y
            ternary_count = len(_RE_PYTHON_TERNARY.findall(source))
            # Don't double count - reduce by 1 for each ternary
            complexity -= ternary_count // 2

        return CyclomaticComplexity(
            value=max(1, complexity),
            decision_points=decision_points,
        )

    def _calculate_cognitive(self, source: str) -> CognitiveComplexity:
        """Calculate cognitive complexity (SonarSource method)."""
        complexity = 0
        nesting_penalty = 0
        structural = 0
        nesting_level = 0

        lines = source.split("\n")

        # Nesting-increasing keywords
        nesting_keywords = {"if", "for", "while", "with", "try", "except", "class", "def"}
        # Structural keywords (no nesting penalty)
        structural_keywords = {"elif", "else", "finally", "case"}

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Check for keywords
            words = _RE_WORD.findall(stripped)
            for word in words:
                if word in nesting_keywords:
                    # Add 1 + nesting penalty
                    complexity += 1 + nesting_level
                    nesting_penalty += nesting_level
                    structural += 1
                elif word in structural_keywords:
                    # Add 1, no nesting penalty
                    complexity += 1
                    structural += 1

                # Boolean operators add complexity
                if word in ("and", "or"):
                    complexity += 1
                    structural += 1

            # Track nesting level (simplified)
            if any(kw + ":" in stripped or kw + " " in stripped for kw in nesting_keywords):
                nesting_level += 1
            elif stripped == "pass" or stripped.startswith("return"):
                nesting_level = max(0, nesting_level - 1)

            # Also check for logical operators
            complexity += stripped.count("&&")
            complexity += stripped.count("||")

        return CognitiveComplexity(
            value=complexity,
            nesting_penalty=nesting_penalty,
            structural_complexity=structural,
        )

    def _calculate_halstead(self, source: str) -> HalsteadMetrics:
        """Calculate Halstead complexity metrics."""
        # Tokenize (simplified)
        operators: dict[str, int] = {}
        operands: dict[str, int] = {}

        # Remove comments and strings for cleaner analysis
        cleaned = self._remove_comments_and_strings(source)

        # Find operators — process longest first to avoid substring overlap
        # (e.g., count '==' before '=' so '==' doesn't inflate '=' count)
        remaining = cleaned
        for op in sorted(self.OPERATORS, key=len, reverse=True):
            count = remaining.count(op)
            if count > 0:
                operators[op] = count
                # Remove matched operators to prevent substring double-counting
                remaining = remaining.replace(op, " " * len(op))

        # Find operands (identifiers and literals)
        # Identifiers
        _non_operand = self._decision_keywords | {"def", "class", "return", "import", "from", "as"}
        for ident in _RE_IDENTIFIER.findall(cleaned):
            if ident not in _non_operand:
                operands[ident] = operands.get(ident, 0) + 1

        # Numbers
        for num in _RE_NUMBER.findall(cleaned):
            operands[num] = operands.get(num, 0) + 1

        return HalsteadMetrics(
            n1=len(operators),
            n2=len(operands),
            N1=sum(operators.values()),
            N2=sum(operands.values()),
        )

    def _calculate_maintainability(
        self, volume: float, cyclomatic: int, loc: int
    ) -> MaintainabilityIndex:
        """Calculate Maintainability Index (Microsoft formula).

        MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        Where:
        - V = Halstead Volume
        - G = Cyclomatic Complexity
        - LOC = Lines of Code
        """
        if volume <= 0 or loc <= 0:
            return MaintainabilityIndex(100.0)

        mi = 171 - 5.2 * math.log(volume) - 0.23 * cyclomatic - 16.2 * math.log(loc)

        # Normalize to 0-100 scale
        mi = max(0, min(100, mi * 100 / 171))

        return MaintainabilityIndex(value=mi)

    def _remove_comments_and_strings(self, source: str) -> str:
        """Remove comments and string literals from source."""
        # Remove multiline strings
        source = _RE_MULTILINE_DQUOTE.sub("", source)
        source = _RE_MULTILINE_SQUOTE.sub("", source)

        # Remove single-line strings
        source = _RE_SINGLE_DQUOTE.sub("", source)
        source = _RE_SINGLE_SQUOTE.sub("", source)

        # Remove single-line comments
        source = _RE_HASH_COMMENT.sub("", source)
        source = _RE_SLASH_COMMENT.sub("", source)

        # Remove block comments
        source = _RE_BLOCK_COMMENT.sub("", source)

        return source


def analyze_complexity(
    source: str,
    language: str = "python",
    file_path: str = "<unknown>",
) -> ComplexityResult:
    """Convenience function to analyze source complexity.

    Args:
        source: Source code to analyze.
        language: Programming language.
        file_path: Optional file path.

    Returns:
        ComplexityResult with all metrics.
    """
    analyzer = ComplexityAnalyzer(language=language)
    return analyzer.analyze_source(source, file_path)
