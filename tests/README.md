# Anamnesis Tests

## Parallelism

**Hard limit: `-n 4`** (4 cores maximum). Never use `-n auto`.

```bash
# Full suite
python -m pytest -n 4 -x -q tests/ --ignore=tests/test_lsp_pyright.py

# With coverage
python -m pytest -n 4 --cov=anamnesis --cov-report=term-missing tests/ --ignore=tests/test_lsp_pyright.py
```

## LSP Tests

`test_lsp_pyright.py` requires a running Pyright language server. Always exclude it from normal runs:

```bash
--ignore=tests/test_lsp_pyright.py
```
