#!/bin/bash
#
# Generate Rust baseline output for golden tests.
# Run this from the In-Memoria project root.
#
# This script runs Rust tests with output capture and saves
# the JSON results for comparison with Python implementation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/fixtures/rust_output"

echo "======================================"
echo "Generating Rust Baseline for Golden Tests"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT/rust-core"

# Build first
echo "Building Rust core..."
cargo build --features all-languages

# Run tests and capture output for each language
LANGUAGES=(
    "typescript"
    "javascript"
    "python"
    "rust"
    "go"
    "java"
    "c"
    "cpp"
    "csharp"
    "sql"
    "svelte"
)

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "Generating baseline for: $lang"

    # Run the test and capture JSON output
    # The actual test implementation would need to output JSON
    # For now, create a placeholder structure

    OUTPUT_FILE="$OUTPUT_DIR/${lang}_extraction.json"

    # Placeholder - in real implementation, this would run:
    # cargo test --features all-languages test_${lang}_extraction -- --nocapture 2>&1 | grep -o '{.*}' > "$OUTPUT_FILE"

    cat > "$OUTPUT_FILE" << EOF
{
  "language": "$lang",
  "concepts": [],
  "note": "Placeholder - run actual Rust tests to populate"
}
EOF

    echo "  -> $OUTPUT_FILE"
done

echo ""
echo "======================================"
echo "Baseline generation complete!"
echo ""
echo "Next steps:"
echo "1. Run actual Rust tests to populate JSON files"
echo "2. Implement Python extractors"
echo "3. Run: python tests/golden/compare_output.py --all"
echo "======================================"
