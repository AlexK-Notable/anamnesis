"""Pattern Detection and Matching Engine.

This package provides pattern-based code search and analysis capabilities,
including regex patterns and AST patterns using tree-sitter.

Components:
- PatternMatch: Result type for pattern matches
- PatternMatcher: Abstract base class for matchers
- RegexPatternMatcher: Regex-based pattern matching with builtins
- ASTPatternMatcher: Tree-sitter based AST pattern matching

Usage:
    from anamnesis.patterns import RegexPatternMatcher, ASTPatternMatcher

    # Regex patterns
    regex = RegexPatternMatcher.with_builtins()
    for match in regex.match(content, "src/auth.py"):
        print(f"{match.pattern_name}: {match.matched_text}")

    # AST patterns (structural)
    ast = ASTPatternMatcher()
    for match in ast.match(content, "src/auth.py"):
        print(f"{match.pattern_name}: {match.matched_text}")
"""

from .matcher import PatternMatch, PatternMatcher
from .regex_matcher import RegexPattern, RegexPatternMatcher
from .ast_matcher import ASTPatternMatcher, ASTQuery

__all__ = [
    "PatternMatch",
    "PatternMatcher",
    "RegexPattern",
    "RegexPatternMatcher",
    "ASTPatternMatcher",
    "ASTQuery",
]
