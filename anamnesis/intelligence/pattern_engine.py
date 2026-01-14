"""Pattern detection and learning engine.

Detects coding patterns in source code and provides recommendations
based on learned patterns from the codebase.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    # Design patterns
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    DEPENDENCY_INJECTION = "dependency_injection"
    REPOSITORY = "repository"
    SERVICE = "service"

    # Naming conventions
    CAMEL_CASE_FUNCTION = "camelCase_function_naming"
    PASCAL_CASE_CLASS = "PascalCase_class_naming"
    SNAKE_CASE_VARIABLE = "snake_case_naming"
    SCREAMING_SNAKE_CASE_CONST = "SCREAMING_SNAKE_CASE_constant"

    # Structural patterns
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    CLEAN_ARCHITECTURE = "clean_architecture"

    # Code organization
    TESTING = "testing"
    API_DESIGN = "api_design"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    CONFIGURATION = "configuration"


@dataclass
class DetectedPattern:
    """A pattern detected in source code."""

    pattern_type: PatternType | str
    description: str
    confidence: float
    file_path: str | None = None
    line_range: tuple[int, int] | None = None
    code_snippet: str | None = None
    frequency: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_type": str(self.pattern_type.value if isinstance(self.pattern_type, PatternType) else self.pattern_type),
            "description": self.description,
            "confidence": self.confidence,
            "file_path": self.file_path,
            "line_range": list(self.line_range) if self.line_range else None,
            "code_snippet": self.code_snippet,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectedPattern:
        """Deserialize from dictionary."""
        pattern_type = data["pattern_type"]
        try:
            pattern_type = PatternType(pattern_type)
        except ValueError:
            pass  # Keep as string if not a known pattern type

        return cls(
            pattern_type=pattern_type,
            description=data["description"],
            confidence=data["confidence"],
            file_path=data.get("file_path"),
            line_range=tuple(data["line_range"]) if data.get("line_range") else None,
            code_snippet=data.get("code_snippet"),
            frequency=data.get("frequency", 1),
        )


@dataclass
class PatternRecommendation:
    """A recommended pattern based on context."""

    pattern_type: PatternType | str
    description: str
    confidence: float
    reasoning: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_type": str(self.pattern_type.value if isinstance(self.pattern_type, PatternType) else self.pattern_type),
            "description": self.description,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "examples": self.examples,
            "related_files": self.related_files,
        }


class PatternEngine:
    """Engine for detecting and learning coding patterns."""

    def __init__(self, storage: Any | None = None) -> None:
        """Initialize the pattern engine.

        Args:
            storage: Optional storage backend for persisting patterns.
        """
        self.storage = storage
        self._learned_patterns: dict[str, list[DetectedPattern]] = {}
        self._pattern_frequency: dict[str, int] = {}

        # Pattern detection rules with regex patterns
        self._pattern_rules: dict[str, list[tuple[re.Pattern[str], float]]] = {
            # Design patterns
            PatternType.SINGLETON.value: [
                (re.compile(r"_instance\s*=\s*None|__instance\s*=\s*None", re.MULTILINE), 0.8),
                (re.compile(r"def\s+get_instance|def\s+instance\s*\(", re.MULTILINE), 0.7),
                (re.compile(r"@staticmethod\s+def\s+get_instance", re.MULTILINE), 0.9),
            ],
            PatternType.FACTORY.value: [
                (re.compile(r"class\s+\w*Factory\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"def\s+create_\w+\s*\(|def\s+make_\w+\s*\(", re.MULTILINE), 0.7),
                (re.compile(r"def\s+build_\w+\s*\(", re.MULTILINE), 0.6),
            ],
            PatternType.BUILDER.value: [
                (re.compile(r"class\s+\w*Builder\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"def\s+with_\w+\s*\(|def\s+set_\w+\s*\(", re.MULTILINE), 0.7),
                (re.compile(r"def\s+build\s*\(\s*self\s*\)", re.MULTILINE), 0.8),
            ],
            PatternType.OBSERVER.value: [
                (re.compile(r"def\s+subscribe\s*\(|def\s+add_observer\s*\(", re.MULTILINE), 0.8),
                (re.compile(r"def\s+notify\s*\(|def\s+emit\s*\(", re.MULTILINE), 0.7),
                (re.compile(r"_observers\s*=|_listeners\s*=|_subscribers\s*=", re.MULTILINE), 0.8),
            ],
            PatternType.STRATEGY.value: [
                (re.compile(r"class\s+\w*Strategy\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"def\s+set_strategy\s*\(|def\s+execute\s*\(", re.MULTILINE), 0.7),
            ],
            PatternType.DEPENDENCY_INJECTION.value: [
                (re.compile(r"def\s+__init__\s*\(\s*self\s*,\s*\w+\s*:\s*\w+", re.MULTILINE), 0.6),
                (re.compile(r"@inject|@Inject", re.MULTILINE), 0.9),
                (re.compile(r"container\.resolve|injector\.get", re.MULTILINE), 0.8),
            ],
            PatternType.REPOSITORY.value: [
                (re.compile(r"class\s+\w*Repository\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"def\s+find_by_\w+\s*\(|def\s+get_all\s*\(", re.MULTILINE), 0.7),
                (re.compile(r"def\s+save\s*\(|def\s+delete\s*\(|def\s+update\s*\(", re.MULTILINE), 0.6),
            ],
            PatternType.SERVICE.value: [
                (re.compile(r"class\s+\w*Service\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"class\s+\w*Manager\w*\s*[:(]", re.MULTILINE), 0.7),
            ],
            # Naming conventions
            PatternType.CAMEL_CASE_FUNCTION.value: [
                (re.compile(r"def\s+[a-z]+[A-Z][a-zA-Z]*\s*\(", re.MULTILINE), 0.8),
            ],
            PatternType.PASCAL_CASE_CLASS.value: [
                (re.compile(r"class\s+[A-Z][a-zA-Z]*\s*[:(]", re.MULTILINE), 0.9),
            ],
            PatternType.SNAKE_CASE_VARIABLE.value: [
                (re.compile(r"[a-z]+_[a-z]+\s*=", re.MULTILINE), 0.7),
                (re.compile(r"def\s+[a-z]+_[a-z]+\s*\(", re.MULTILINE), 0.8),
            ],
            PatternType.SCREAMING_SNAKE_CASE_CONST.value: [
                (re.compile(r"^[A-Z][A-Z_0-9]+\s*=", re.MULTILINE), 0.9),
            ],
            # Code organization
            PatternType.TESTING.value: [
                (re.compile(r"def\s+test_\w+\s*\(|class\s+Test\w+", re.MULTILINE), 0.9),
                (re.compile(r"@pytest\.|import\s+pytest|import\s+unittest", re.MULTILINE), 0.9),
                (re.compile(r"assert\s+\w+|self\.assert\w+", re.MULTILINE), 0.7),
            ],
            PatternType.API_DESIGN.value: [
                (re.compile(r"@app\.route|@router\.|@api\.", re.MULTILINE), 0.9),
                (re.compile(r"def\s+(get|post|put|delete|patch)_\w+", re.MULTILINE), 0.7),
            ],
            PatternType.ERROR_HANDLING.value: [
                (re.compile(r"try:\s*\n|except\s+\w+:|raise\s+\w+", re.MULTILINE), 0.7),
                (re.compile(r"class\s+\w*Error\s*\(|class\s+\w*Exception\s*\(", re.MULTILINE), 0.8),
            ],
            PatternType.LOGGING.value: [
                (re.compile(r"logger\.\w+|logging\.\w+|log\.\w+", re.MULTILINE), 0.8),
                (re.compile(r"import\s+logging|from\s+logging\s+import", re.MULTILINE), 0.9),
            ],
            PatternType.CONFIGURATION.value: [
                (re.compile(r"class\s+\w*Config\w*\s*[:(]|class\s+\w*Settings\w*\s*[:(]", re.MULTILINE), 0.9),
                (re.compile(r"\.env|environ\.get|os\.getenv", re.MULTILINE), 0.7),
            ],
        }

        # Pattern descriptions
        self._pattern_descriptions: dict[str, str] = {
            PatternType.SINGLETON.value: "Singleton pattern ensures a class has only one instance",
            PatternType.FACTORY.value: "Factory pattern for creating objects without specifying exact class",
            PatternType.BUILDER.value: "Builder pattern for constructing complex objects step by step",
            PatternType.OBSERVER.value: "Observer pattern for event-based communication",
            PatternType.STRATEGY.value: "Strategy pattern for interchangeable algorithms",
            PatternType.DEPENDENCY_INJECTION.value: "Dependency injection for loose coupling",
            PatternType.REPOSITORY.value: "Repository pattern for data access abstraction",
            PatternType.SERVICE.value: "Service layer pattern for business logic",
            PatternType.CAMEL_CASE_FUNCTION.value: "Functions use camelCase naming convention",
            PatternType.PASCAL_CASE_CLASS.value: "Classes use PascalCase naming convention",
            PatternType.SNAKE_CASE_VARIABLE.value: "Variables and functions use snake_case naming",
            PatternType.SCREAMING_SNAKE_CASE_CONST.value: "Constants use SCREAMING_SNAKE_CASE naming",
            PatternType.TESTING.value: "Test-driven development with pytest/unittest patterns",
            PatternType.API_DESIGN.value: "RESTful API design patterns",
            PatternType.ERROR_HANDLING.value: "Structured error handling with custom exceptions",
            PatternType.LOGGING.value: "Consistent logging practices",
            PatternType.CONFIGURATION.value: "Configuration management patterns",
        }

    def detect_patterns(
        self,
        content: str,
        file_path: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[DetectedPattern]:
        """Detect patterns in source code.

        Args:
            content: Source code content.
            file_path: Optional file path for context.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of detected patterns.
        """
        detected: list[DetectedPattern] = []

        for pattern_type, rules in self._pattern_rules.items():
            max_confidence = 0.0
            matches: list[re.Match[str]] = []

            for regex, confidence in rules:
                found = list(regex.finditer(content))
                if found:
                    matches.extend(found)
                    max_confidence = max(max_confidence, confidence)

            if matches and max_confidence >= min_confidence:
                # Get line range from first match
                first_match = matches[0]
                start_line = content[:first_match.start()].count("\n") + 1
                end_line = content[:first_match.end()].count("\n") + 1

                detected.append(
                    DetectedPattern(
                        pattern_type=pattern_type,
                        description=self._pattern_descriptions.get(
                            pattern_type, f"{pattern_type} pattern detected"
                        ),
                        confidence=max_confidence,
                        file_path=file_path,
                        line_range=(start_line, end_line),
                        code_snippet=first_match.group(0)[:100],  # Truncate long snippets
                        frequency=len(matches),
                    )
                )

        return detected

    def detect_patterns_in_file(
        self,
        file_path: str | Path,
        min_confidence: float = 0.5,
    ) -> list[DetectedPattern]:
        """Detect patterns in a file.

        Args:
            file_path: Path to the file.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of detected patterns.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        return self.detect_patterns(content, str(path), min_confidence)

    def learn_from_file(self, file_path: str | Path, content: str | None = None) -> None:
        """Learn patterns from a file and update frequency counts.

        Args:
            file_path: Path to the file.
            content: Optional content (will be read if not provided).
        """
        path = Path(file_path)
        str_path = str(path)

        if content is None:
            if not path.exists():
                return
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return

        patterns = self.detect_patterns(content, str_path)

        # Store learned patterns
        self._learned_patterns[str_path] = patterns

        # Update frequency counts
        for pattern in patterns:
            pattern_key = str(pattern.pattern_type.value if isinstance(pattern.pattern_type, PatternType) else pattern.pattern_type)
            self._pattern_frequency[pattern_key] = (
                self._pattern_frequency.get(pattern_key, 0) + pattern.frequency
            )

    def get_recommendations(
        self,
        problem_description: str,
        current_file: str | None = None,
        context: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[PatternRecommendation]:
        """Get pattern recommendations based on problem description.

        Args:
            problem_description: Description of what the user wants to implement.
            current_file: Current file being worked on.
            context: Additional context.
            top_k: Number of recommendations to return.

        Returns:
            List of pattern recommendations.
        """
        recommendations: list[PatternRecommendation] = []
        problem_lower = problem_description.lower()

        # Keywords to pattern mappings
        keyword_patterns: list[tuple[list[str], str, str]] = [
            (
                ["service", "business logic", "handler"],
                PatternType.SERVICE.value,
                "For business logic and coordination between components",
            ),
            (
                ["repository", "data access", "database", "crud", "storage"],
                PatternType.REPOSITORY.value,
                "For abstracting data persistence and retrieval",
            ),
            (
                ["factory", "create", "instantiate", "construct"],
                PatternType.FACTORY.value,
                "For creating objects without specifying exact classes",
            ),
            (
                ["builder", "complex object", "step by step", "fluent"],
                PatternType.BUILDER.value,
                "For constructing complex objects incrementally",
            ),
            (
                ["singleton", "single instance", "global", "shared"],
                PatternType.SINGLETON.value,
                "For ensuring only one instance exists",
            ),
            (
                ["observer", "event", "subscribe", "notify", "listener", "callback"],
                PatternType.OBSERVER.value,
                "For event-driven communication between components",
            ),
            (
                ["strategy", "algorithm", "interchangeable", "policy"],
                PatternType.STRATEGY.value,
                "For making algorithms interchangeable at runtime",
            ),
            (
                ["dependency injection", "di", "inject", "container", "ioc"],
                PatternType.DEPENDENCY_INJECTION.value,
                "For loose coupling and testability",
            ),
            (
                ["test", "testing", "unittest", "pytest", "spec"],
                PatternType.TESTING.value,
                "For test-driven development practices",
            ),
            (
                ["api", "endpoint", "route", "rest", "http"],
                PatternType.API_DESIGN.value,
                "For RESTful API design",
            ),
            (
                ["error", "exception", "handle", "catch"],
                PatternType.ERROR_HANDLING.value,
                "For structured error handling",
            ),
            (
                ["log", "logging", "debug", "trace"],
                PatternType.LOGGING.value,
                "For consistent logging practices",
            ),
            (
                ["config", "configuration", "settings", "environment"],
                PatternType.CONFIGURATION.value,
                "For configuration management",
            ),
        ]

        # Score patterns based on keyword matches
        pattern_scores: dict[str, tuple[float, str]] = {}
        for keywords, pattern_type, reasoning in keyword_patterns:
            score = sum(1 for kw in keywords if kw in problem_lower)
            if score > 0:
                # Boost score by learned frequency
                frequency_boost = min(
                    self._pattern_frequency.get(pattern_type, 0) / 100, 0.3
                )
                total_score = score / len(keywords) + frequency_boost
                pattern_scores[pattern_type] = (total_score, reasoning)

        # Sort by score and create recommendations
        sorted_patterns = sorted(
            pattern_scores.items(), key=lambda x: x[1][0], reverse=True
        )[:top_k]

        for pattern_type, (score, reasoning) in sorted_patterns:
            # Find related files using this pattern
            related_files = [
                path
                for path, patterns in self._learned_patterns.items()
                for p in patterns
                if str(p.pattern_type.value if isinstance(p.pattern_type, PatternType) else p.pattern_type) == pattern_type
            ]

            # Get examples from learned patterns
            examples = []
            for path, patterns in self._learned_patterns.items():
                for p in patterns:
                    if str(p.pattern_type.value if isinstance(p.pattern_type, PatternType) else p.pattern_type) == pattern_type and p.code_snippet:
                        examples.append({
                            "file": path,
                            "code": p.code_snippet,
                            "confidence": p.confidence,
                        })
                        if len(examples) >= 3:
                            break
                if len(examples) >= 3:
                    break

            recommendations.append(
                PatternRecommendation(
                    pattern_type=pattern_type,
                    description=self._pattern_descriptions.get(
                        pattern_type, f"{pattern_type} pattern"
                    ),
                    confidence=min(score, 1.0),
                    reasoning=reasoning,
                    examples=examples,
                    related_files=related_files[:5],
                )
            )

        return recommendations

    def get_learned_patterns(self) -> dict[str, int]:
        """Get all learned patterns with their frequencies.

        Returns:
            Dictionary mapping pattern types to frequencies.
        """
        return dict(self._pattern_frequency)

    def get_file_patterns(self, file_path: str) -> list[DetectedPattern]:
        """Get patterns learned from a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            List of patterns from that file.
        """
        return self._learned_patterns.get(file_path, [])

    def clear_learned_patterns(self) -> None:
        """Clear all learned patterns."""
        self._learned_patterns.clear()
        self._pattern_frequency.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine state to dictionary."""
        return {
            "learned_patterns": {
                path: [p.to_dict() for p in patterns]
                for path, patterns in self._learned_patterns.items()
            },
            "pattern_frequency": self._pattern_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage: Any | None = None) -> PatternEngine:
        """Deserialize engine state from dictionary.

        Args:
            data: Dictionary with serialized state.
            storage: Optional storage backend.

        Returns:
            Restored PatternEngine instance.
        """
        engine = cls(storage=storage)

        # Restore learned patterns
        for path, patterns_data in data.get("learned_patterns", {}).items():
            engine._learned_patterns[path] = [
                DetectedPattern.from_dict(p) for p in patterns_data
            ]

        # Restore frequency counts
        engine._pattern_frequency = data.get("pattern_frequency", {})

        return engine
