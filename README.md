# Anamnesis

> **Anamnesis** (Greek: ἀνάμνησις) — Plato's concept of recollection, the idea that learning is really remembering knowledge the soul already possesses.

Semantic code analysis and intelligence library — a Python port of [In-Memoria](https://github.com/In-Memoria/In-Memoria)'s Rust core.

## Overview

Anamnesis provides intelligent code analysis through:

- **Tree-sitter based AST parsing** for 11 languages (TypeScript, JavaScript, Python, Rust, Go, Java, C, C++, C#, SQL, Ruby)
- **Semantic concept extraction** — understands what code *means*, not just its syntax
- **Pattern learning** — learns naming conventions, structural patterns, and implementation idioms from your codebase
- **Approach prediction** — suggests relevant files and patterns for implementing new features
- **Codebase blueprinting** — generates high-level architectural overviews

## Installation

```bash
pip install anamnesis
```

Or with development dependencies:

```bash
pip install anamnesis[dev]
```

## Requirements

- Python 3.11+
- tree-sitter >= 0.23.0
- tree-sitter-language-pack >= 0.23.4
- pydantic >= 2.0

## Quick Start

```python
from anamnesis import SemanticAnalyzer

# Analyze a codebase
analyzer = SemanticAnalyzer()
result = await analyzer.analyze_directory("/path/to/project")

# Get codebase blueprint
blueprint = result.get_blueprint()
print(blueprint.tech_stack)
print(blueprint.entry_points)

# Predict approach for a task
prediction = await analyzer.predict_approach(
    "Add user authentication",
    context=result
)
print(prediction.target_files)
print(prediction.suggested_patterns)
```

## Development Status

This is an active port of the In-Memoria Rust core. Implementation follows a phased approach:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Core Types & Models | 🟡 In Progress |
| 2 | Tree-sitter Parsing | ⚪ Planned |
| 3 | Language Extractors | ⚪ Planned |
| 4 | Semantic Analysis | ⚪ Planned |
| 5 | Pattern Learning | ⚪ Planned |
| 6-11 | Advanced Features | ⚪ Planned |

See [COMPREHENSIVE_RALPH_LOOP.md](../In-Memoria/COMPREHENSIVE_RALPH_LOOP.md) for detailed implementation plan.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/anamnesis
cd anamnesis

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Validate Grammar Support

Before developing parsing features, validate tree-sitter grammar availability:

```bash
python scripts/validate_grammars.py
```

### Run Tests

```bash
pytest
```

### Type Checking

```bash
mypy anamnesis
```

### Linting

```bash
ruff check anamnesis
```

## Architecture

```
anamnesis/
├── types/          # Core data models (Pydantic)
├── parsing/        # Tree-sitter integration
│   ├── parser_manager.py
│   ├── tree_walker.py
│   └── fallback_extractor.py
├── extractors/     # Language-specific extractors
│   ├── typescript.py
│   ├── python.py
│   └── ...
├── analysis/       # Semantic analysis engines
│   ├── semantic_analyzer.py
│   ├── complexity_analyzer.py
│   └── blueprint_analyzer.py
└── patterns/       # Pattern learning & prediction
    ├── pattern_engine.py
    └── approach_predictor.py
```

## Relationship to In-Memoria

Anamnesis is a Python implementation of the core analysis engine from [In-Memoria](https://github.com/pi22by7/In-Memoria), originally created by [Piyush Airani](https://github.com/pi22by7). While In-Memoria is a full MCP server providing codebase intelligence to AI assistants, Anamnesis focuses purely on the analysis capabilities, making it usable as a standalone library.

**In-Memoria** (Rust/TypeScript) → Full MCP server with CLI
**Anamnesis** (Python) → Portable analysis library

## Acknowledgments

- [Piyush Airani](https://github.com/pi22by7) - Creator of [In-Memoria](https://github.com/pi22by7/In-Memoria), the original Rust implementation this project is based on

## License

MIT - see the [LICENSE](LICENSE) file for details.

## Etymology

The name "Anamnesis" comes from Plato's theory of recollection, which holds that all learning is actually a process of remembering what the soul knew before birth. This concept is particularly apt for a code intelligence tool that helps developers "remember" patterns, conventions, and structures across their codebase — turning implicit knowledge into explicit understanding.
