# solidlsp — Provenance and Ownership

## Origin
- **Project**: Serena (https://github.com/oraios/serena)
- **Source directory**: `src/solidlsp/`
- **Serena version at time of vendoring**: v0.1.4+ (post-release, upstream HEAD was `b7142cb` on 2026-01-13)
- **Date vendored**: 2026-02-01 (commit `10abd22` in Anamnesis)
- **License**: MIT (OLSP generated components), Apache 2.0 (Serena)

## Ownership Status

**Adopted as first-class Anamnesis code** (2026-02-08). This code is now owned and maintained by the Anamnesis project. There is no automated upstream sync — feel free to modify, document, and evolve it.

## Modifications from Upstream
1. All `from solidlsp.` imports rewritten to `from anamnesis.lsp.solidlsp.`
2. New `compat.py` shim replaces `sensai-utils` and `serena` package imports (Anamnesis-authored)
3. `ls_config.py:get_ls_class()` trimmed to 4 languages (Python, Go, Rust, TypeScript)
4. 40+ language server implementations excluded (only pyright, gopls, rust-analyzer, typescript kept)
5. `util/zip.py` excluded
6. Dead code removed: DotnetVersion, SymbolUtils, quote_arg, quote_windows_path, MatchedConsecutiveLines, Python 2 compat, Serena-specific references (2026-02-08)
7. Module docstrings added to all non-empty files (2026-02-08)

## Generated Files (do not edit)

Three files under `lsp_protocol_handler/` are generated from [OLSP](https://github.com/predragnikolic/OLSP) under the MIT License:
- `lsp_types.py`
- `lsp_requests.py`
- `server.py`

## Supported Languages
- Python (Pyright)
- Go (gopls)
- Rust (rust-analyzer)
- TypeScript (typescript-language-server)
