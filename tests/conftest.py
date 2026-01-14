"""
Pytest configuration and shared fixtures for In-Memoria Python core tests.

This file provides:
- Common fixtures used across all test phases
- Golden test comparison utilities
- Test data generators matching Rust test patterns
"""

import json
from pathlib import Path
from typing import Any

import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "golden" / "fixtures"
RUST_OUTPUT_DIR = FIXTURES_DIR / "rust_output"
TEST_CODE_DIR = FIXTURES_DIR / "test_code"


@pytest.fixture
def rust_output_dir() -> Path:
    """Path to Rust golden output fixtures."""
    return RUST_OUTPUT_DIR


@pytest.fixture
def test_code_dir() -> Path:
    """Path to test code samples."""
    return TEST_CODE_DIR


# ============================================================================
# Test Code Samples (matching rust-core tests)
# ============================================================================

@pytest.fixture
def typescript_sample() -> tuple[str, str]:
    """TypeScript test code and expected filename."""
    return (
        "test.ts",
        'export class UserService { getName(): string { return "test"; } }',
    )


@pytest.fixture
def javascript_sample() -> tuple[str, str]:
    """JavaScript test code and expected filename."""
    return (
        "test.js",
        'function hello() { return "world"; }',
    )


@pytest.fixture
def python_sample() -> tuple[str, str]:
    """Python test code and expected filename."""
    return (
        "test.py",
        'class User:\n    def __init__(self):\n        self.name = "test"',
    )


@pytest.fixture
def rust_sample() -> tuple[str, str]:
    """Rust test code and expected filename."""
    return (
        "test.rs",
        "pub struct User { name: String }",
    )


@pytest.fixture
def go_sample() -> tuple[str, str]:
    """Go test code and expected filename."""
    return (
        "test.go",
        'package main\nfunc main() {\n    println("Hello World")\n}',
    )


@pytest.fixture
def java_sample() -> tuple[str, str]:
    """Java test code and expected filename."""
    return (
        "test.java",
        'public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}',
    )


@pytest.fixture
def c_sample() -> tuple[str, str]:
    """C test code and expected filename."""
    return (
        "test.c",
        '#include <stdio.h>\nint main() {\n    printf("Hello World");\n    return 0;\n}',
    )


@pytest.fixture
def cpp_sample() -> tuple[str, str]:
    """C++ test code and expected filename."""
    return (
        "test.cpp",
        '#include <iostream>\nclass HelloWorld {\npublic:\n    void sayHello() {\n        std::cout << "Hello";\n    }\n};',
    )


@pytest.fixture
def csharp_sample() -> tuple[str, str]:
    """C# test code and expected filename."""
    return (
        "test.cs",
        'using System;\npublic class Program {\n    public static void Main() {\n        Console.WriteLine("Hello World");\n    }\n}',
    )


@pytest.fixture
def sql_sample() -> tuple[str, str]:
    """SQL test code and expected filename."""
    return (
        "test.sql",
        "CREATE TABLE users (\n  id INTEGER PRIMARY KEY,\n  name VARCHAR(255)\n);\nSELECT * FROM users;",
    )


@pytest.fixture
def svelte_sample() -> tuple[str, str]:
    """Svelte test code and expected filename."""
    return (
        "test.svelte",
        '<script>\n  let name = "world";\n  function greet() {\n    alert(`Hello ${name}!`);\n  }\n</script>\n<button on:click={greet}>Greet</button>',
    )


@pytest.fixture
def all_language_samples(
    typescript_sample,
    javascript_sample,
    python_sample,
    rust_sample,
    go_sample,
    java_sample,
    c_sample,
    cpp_sample,
    csharp_sample,
    sql_sample,
    svelte_sample,
) -> dict[str, tuple[str, str]]:
    """All language samples as a dict keyed by language name."""
    return {
        "typescript": typescript_sample,
        "javascript": javascript_sample,
        "python": python_sample,
        "rust": rust_sample,
        "go": go_sample,
        "java": java_sample,
        "c": c_sample,
        "cpp": cpp_sample,
        "csharp": csharp_sample,
        "sql": sql_sample,
        "svelte": svelte_sample,
    }


# ============================================================================
# Golden Test Utilities
# ============================================================================

def load_rust_golden(language: str) -> dict[str, Any] | None:
    """Load Rust golden output for a language, if it exists."""
    golden_file = RUST_OUTPUT_DIR / f"{language}_extraction.json"
    if golden_file.exists():
        return json.loads(golden_file.read_text())
    return None


def normalize_concepts(concepts: list[dict]) -> list[dict]:
    """
    Normalize concept ordering for deterministic comparison.

    Sorts by (name, concept_type, line_range.start) to ensure consistent
    ordering regardless of extraction traversal differences between
    Rust and Python implementations.
    """
    return sorted(
        concepts,
        key=lambda c: (
            c.get("name", ""),
            c.get("concept_type", ""),
            c.get("line_range", {}).get("start", 0),
        ),
    )


def compare_concepts(rust_concepts: list[dict], python_concepts: list[dict]) -> list[str]:
    """
    Compare Rust and Python concept extraction results.
    Returns a list of differences (empty if identical).

    Normalizes ordering before comparison to handle traversal differences.
    """
    differences = []

    # Normalize ordering for fair comparison
    rust_sorted = normalize_concepts(rust_concepts)
    python_sorted = normalize_concepts(python_concepts)

    if len(rust_sorted) != len(python_sorted):
        differences.append(
            f"Concept count mismatch: Rust={len(rust_sorted)}, Python={len(python_sorted)}"
        )
        # Continue comparing what we can
        min_len = min(len(rust_sorted), len(python_sorted))
        rust_sorted = rust_sorted[:min_len]
        python_sorted = python_sorted[:min_len]

    for i, (r, p) in enumerate(zip(rust_sorted, python_sorted)):
        if r.get("name") != p.get("name"):
            differences.append(f"[{i}] Name mismatch: {r.get('name')} vs {p.get('name')}")

        if r.get("concept_type") != p.get("concept_type"):
            differences.append(
                f"[{i}] Type mismatch for {r.get('name')}: "
                f"{r.get('concept_type')} vs {p.get('concept_type')}"
            )

        r_conf = r.get("confidence", 0)
        p_conf = p.get("confidence", 0)
        if abs(r_conf - p_conf) > 0.001:
            differences.append(
                f"[{i}] Confidence mismatch for {r.get('name')}: {r_conf} vs {p_conf}"
            )

        # Compare line ranges
        r_range = r.get("line_range", {})
        p_range = p.get("line_range", {})
        if r_range.get("start") != p_range.get("start") or r_range.get("end") != p_range.get("end"):
            differences.append(
                f"[{i}] Line range mismatch for {r.get('name')}: "
                f"{r_range} vs {p_range}"
            )

    return differences


@pytest.fixture
def golden_comparator():
    """Fixture providing golden test comparison function."""
    return compare_concepts
