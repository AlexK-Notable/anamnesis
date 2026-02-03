#!/usr/bin/env python3
"""
Golden Test Comparison Tool

Compares Python extraction output against Rust baseline to verify parity.

Usage:
    python tests/golden/compare_output.py --phase 1
    python tests/golden/compare_output.py --language typescript
    python tests/golden/compare_output.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.conftest import RUST_OUTPUT_DIR


def compare_phase(phase: int) -> tuple[int, list[str]]:
    """
    Compare all golden tests for a given phase.
    Returns (error_count, list_of_differences).
    """
    errors = 0
    all_differences = []

    phase_map = {
        1: ["types"],
        2: ["parsing"],
        3: [
            "typescript", "javascript", "python", "rust",
            "go", "java", "c", "cpp", "csharp", "sql", "svelte"
        ],
        4: ["complexity", "blueprint", "semantic"],
        5: ["patterns", "prediction"],
    }

    if phase not in phase_map:
        return 1, [f"Unknown phase: {phase}"]

    for test_name in phase_map[phase]:
        rust_file = RUST_OUTPUT_DIR / f"{test_name}_extraction.json"

        if not rust_file.exists():
            all_differences.append(f"Missing Rust baseline: {rust_file}")
            continue

        # TODO: Run Python extraction and compare
        # For now, just check if baseline exists
        print(f"  [PENDING] {test_name}: Rust baseline exists, Python not implemented")

    return errors, all_differences


def compare_language(language: str) -> tuple[int, list[str]]:
    """
    Compare extraction output for a specific language.
    Returns (error_count, list_of_differences).
    """
    rust_file = RUST_OUTPUT_DIR / f"{language}_extraction.json"

    if not rust_file.exists():
        return 1, [f"Missing Rust baseline: {rust_file}"]

    rust_output = json.loads(rust_file.read_text())

    # TODO: Run Python extraction
    # from anamnesis.extractors import get_extractor
    # from anamnesis.parsing import ParserManager
    #
    # manager = ParserManager()
    # extractor = get_extractor(language)
    # python_output = extractor.extract(...)

    # For now, return pending
    return 0, [f"[PENDING] {language}: Python extraction not implemented"]


def compare_all() -> tuple[int, list[str]]:
    """Compare all golden tests."""
    total_errors = 0
    all_differences = []

    for phase in range(1, 6):
        print(f"\nPhase {phase}:")
        errors, diffs = compare_phase(phase)
        total_errors += errors
        all_differences.extend(diffs)

    return total_errors, all_differences


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs Rust extraction output")
    parser.add_argument("--phase", type=int, help="Compare all tests for a phase (1-5)")
    parser.add_argument("--language", type=str, help="Compare extraction for a language")
    parser.add_argument("--all", action="store_true", help="Compare all golden tests")

    args = parser.parse_args()

    print("=" * 60)
    print("Golden Test Comparison")
    print("=" * 60)

    if args.all:
        errors, diffs = compare_all()
    elif args.phase:
        errors, diffs = compare_phase(args.phase)
    elif args.language:
        errors, diffs = compare_language(args.language)
    else:
        parser.print_help()
        return 1

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    if diffs:
        print("\nDifferences found:")
        for diff in diffs:
            print(f"  - {diff}")

    if errors > 0:
        print(f"\n{errors} error(s) found")
        return 1
    else:
        print("\nAll comparisons passed (or pending implementation)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
