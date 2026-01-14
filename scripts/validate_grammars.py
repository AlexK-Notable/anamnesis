#!/usr/bin/env python3
"""
Validate tree-sitter grammar availability.

Run this BEFORE starting Phase 2 to ensure all required language
grammars are available from tree-sitter-language-pack.

Usage:
    python scripts/validate_grammars.py
"""

import sys

REQUIRED_LANGUAGES = [
    "typescript",
    "javascript",
    "python",
    "rust",
    "go",
    "java",
    "c",
    "cpp",
    "c_sharp",
    "sql",
]

# Language name mappings (if tree-sitter-language-pack uses different names)
LANGUAGE_ALIASES = {
    "c_sharp": ["c_sharp", "csharp", "c-sharp"],
    "cpp": ["cpp", "c++"],
}


def validate_grammars() -> bool:
    """Validate all required grammars are available."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        print("❌ tree-sitter-language-pack not installed")
        print("   Run: pip install tree-sitter-language-pack>=0.23.4")
        return False

    results: dict[str, str] = {}

    for lang in REQUIRED_LANGUAGES:
        # Try the primary name first
        names_to_try = [lang] + LANGUAGE_ALIASES.get(lang, [])
        success = False

        for name in names_to_try:
            try:
                parser = get_parser(name)
                # Test parsing with minimal code
                tree = parser.parse(b"test")
                if tree.root_node is not None:
                    results[lang] = "OK"
                    success = True
                    break
            except Exception:
                continue

        if not success:
            results[lang] = "FAIL: Grammar not available"

    # Print results
    print("=" * 50)
    print("Tree-sitter Grammar Validation")
    print("=" * 50)
    print()

    for lang, status in results.items():
        icon = "✅" if status == "OK" else "❌"
        print(f"  {icon} {lang}: {status}")

    failures = [l for l, s in results.items() if s != "OK"]

    print()
    print("=" * 50)

    if failures:
        print(f"⚠️  {len(failures)} grammar(s) failed validation:")
        for lang in failures:
            print(f"   - {lang}")
        print()
        print("Consider fallback options:")
        print("  1. Use tree-sitter-languages for failed grammars")
        print("  2. Build grammars from source")
        print("  3. Check for alternative package names")
        return False
    else:
        print("✅ All grammars validated successfully!")
        print("   Ready to proceed with Phase 2.")
        return True


if __name__ == "__main__":
    success = validate_grammars()
    sys.exit(0 if success else 1)
