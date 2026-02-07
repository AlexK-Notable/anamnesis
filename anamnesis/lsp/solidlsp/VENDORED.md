# solidlsp Vendor Record

## Source
- **Project**: Serena (https://github.com/oraios/serena)
- **Source directory**: `src/solidlsp/`
- **Serena version at time of vendoring**: v0.1.4+ (post-release, upstream HEAD was `b7142cb` on 2026-01-13)
- **Date vendored**: 2026-02-01 (commit `10abd22` in Anamnesis)
- **License**: MIT (OLSP components), Apache 2.0 (Serena)

## Modifications from Upstream
1. All `from solidlsp.` imports rewritten to `from anamnesis.lsp.solidlsp.`
2. New `compat.py` shim replaces `sensai-utils` and `serena` package imports (Anamnesis-authored, not vendored)
3. `ls_config.py:get_ls_class()` trimmed to 4 languages (Python, Go, Rust, TypeScript)
4. `util/subprocess_util.py:quote_arg()` uses `shlex.quote()` instead of manual quoting (security improvement)
5. 40+ language server implementations excluded (only pyright, gopls, rust-analyzer, typescript kept)
6. `util/zip.py` excluded

## Supported Languages
Only 4 of the upstream 40+ languages have server implementations:
- Python (Pyright)
- Go (gopls)
- Rust (rust-analyzer)
- TypeScript (typescript-language-server)

## Upstream Sync Status
- Last sync: 2026-02-01
- Known divergence: upstream has ~60 files, vendored has 24
- Do NOT modify vendored files unless porting upstream changes
