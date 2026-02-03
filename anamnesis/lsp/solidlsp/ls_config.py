"""
Configuration objects for language servers
"""

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from anamnesis.lsp.solidlsp import SolidLanguageServer


class FilenameMatcher:
    def __init__(self, *patterns: str) -> None:
        """
        :param patterns: fnmatch-compatible patterns
        """
        self.patterns = patterns

    def is_relevant_filename(self, fn: str) -> bool:
        for pattern in self.patterns:
            if fnmatch.fnmatch(fn, pattern):
                return True
        return False


class Language(str, Enum):
    """
    Enumeration of language servers supported by SolidLSP.
    """

    CSHARP = "csharp"
    PYTHON = "python"
    RUST = "rust"
    JAVA = "java"
    KOTLIN = "kotlin"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUBY = "ruby"
    DART = "dart"
    CPP = "cpp"
    PHP = "php"
    R = "r"
    PERL = "perl"
    CLOJURE = "clojure"
    ELIXIR = "elixir"
    ELM = "elm"
    TERRAFORM = "terraform"
    SWIFT = "swift"
    BASH = "bash"
    ZIG = "zig"
    LUA = "lua"
    NIX = "nix"
    ERLANG = "erlang"
    AL = "al"
    FSHARP = "fsharp"
    REGO = "rego"
    SCALA = "scala"
    JULIA = "julia"
    FORTRAN = "fortran"
    HASKELL = "haskell"
    GROOVY = "groovy"
    VUE = "vue"
    POWERSHELL = "powershell"
    PASCAL = "pascal"
    """Pascal Language Server (pasls) for Free Pascal and Lazarus projects.
    Automatically downloads pasls binary. Requires FPC for full functionality.
    Set PP and FPCDIR environment variables for source navigation.
    """
    MATLAB = "matlab"
    """MATLAB language server using the official MathWorks MATLAB Language Server.
    Requires MATLAB R2021b or later and Node.js.
    Set MATLAB_PATH environment variable or configure matlab_path in ls_specific_settings.
    """
    # Experimental or deprecated Language Servers
    TYPESCRIPT_VTS = "typescript_vts"
    """Use the typescript language server through the natively bundled vscode extension via https://github.com/yioneko/vtsls"""
    PYTHON_JEDI = "python_jedi"
    """Jedi language server for Python (instead of pyright, which is the default)"""
    CSHARP_OMNISHARP = "csharp_omnisharp"
    """OmniSharp language server for C# (instead of the default csharp-ls by microsoft).
    Currently has problems with finding references, and generally seems less stable and performant.
    """
    RUBY_SOLARGRAPH = "ruby_solargraph"
    """Solargraph language server for Ruby (legacy, experimental).
    Use Language.RUBY (ruby-lsp) for better performance and modern LSP features.
    """
    MARKDOWN = "markdown"
    """Marksman language server for Markdown (experimental).
    Must be explicitly specified as the main language, not auto-detected.
    This is an edge case primarily useful when working on documentation-heavy projects.
    """
    YAML = "yaml"
    """YAML language server (experimental).
    Must be explicitly specified as the main language, not auto-detected.
    """
    TOML = "toml"
    """TOML language server using Taplo.
    Supports TOML validation, formatting, and schema support.
    """

    @classmethod
    def iter_all(cls, include_experimental: bool = False) -> Iterable[Self]:
        for lang in cls:
            if include_experimental or not lang.is_experimental():
                yield lang

    def is_experimental(self) -> bool:
        """
        Check if the language server is experimental or deprecated.

        Note for serena users/developers:
        Experimental languages are not autodetected and must be explicitly specified
        in the project.yml configuration.
        """
        return self in {
            self.TYPESCRIPT_VTS,
            self.PYTHON_JEDI,
            self.CSHARP_OMNISHARP,
            self.RUBY_SOLARGRAPH,
            self.MARKDOWN,
            self.YAML,
            self.TOML,
            self.GROOVY,
        }

    def __str__(self) -> str:
        return self.value

    def get_priority(self) -> int:
        """
        :return: priority of the language for breaking ties between languages; higher is more important.
        """
        # experimental languages have the lowest priority
        if self.is_experimental():
            return 0
        # We assign lower priority to languages that are supersets of others, such that
        # the "larger" language is only chosen when it matches more strongly
        match self:
            # languages that are supersets of others (Vue is superset of TypeScript/JavaScript)
            case self.VUE:
                return 1
            # regular languages
            case _:
                return 2

    def get_source_fn_matcher(self) -> FilenameMatcher:
        match self:
            case self.PYTHON | self.PYTHON_JEDI:
                return FilenameMatcher("*.py", "*.pyi")
            case self.JAVA:
                return FilenameMatcher("*.java")
            case self.TYPESCRIPT | self.TYPESCRIPT_VTS:
                # see https://github.com/oraios/serena/issues/204
                path_patterns = []
                for prefix in ["c", "m", ""]:
                    for postfix in ["x", ""]:
                        for base_pattern in ["ts", "js"]:
                            path_patterns.append(f"*.{prefix}{base_pattern}{postfix}")
                return FilenameMatcher(*path_patterns)
            case self.CSHARP | self.CSHARP_OMNISHARP:
                return FilenameMatcher("*.cs")
            case self.RUST:
                return FilenameMatcher("*.rs")
            case self.GO:
                return FilenameMatcher("*.go")
            case self.RUBY:
                return FilenameMatcher("*.rb", "*.erb")
            case self.RUBY_SOLARGRAPH:
                return FilenameMatcher("*.rb")
            case self.CPP:
                return FilenameMatcher("*.cpp", "*.h", "*.hpp", "*.c", "*.hxx", "*.cc", "*.cxx")
            case self.KOTLIN:
                return FilenameMatcher("*.kt", "*.kts")
            case self.DART:
                return FilenameMatcher("*.dart")
            case self.PHP:
                return FilenameMatcher("*.php")
            case self.R:
                return FilenameMatcher("*.R", "*.r", "*.Rmd", "*.Rnw")
            case self.PERL:
                return FilenameMatcher("*.pl", "*.pm", "*.t")
            case self.CLOJURE:
                return FilenameMatcher("*.clj", "*.cljs", "*.cljc", "*.edn")  # codespell:ignore edn
            case self.ELIXIR:
                return FilenameMatcher("*.ex", "*.exs")
            case self.ELM:
                return FilenameMatcher("*.elm")
            case self.TERRAFORM:
                return FilenameMatcher("*.tf", "*.tfvars", "*.tfstate")
            case self.SWIFT:
                return FilenameMatcher("*.swift")
            case self.BASH:
                return FilenameMatcher("*.sh", "*.bash")
            case self.YAML:
                return FilenameMatcher("*.yaml", "*.yml")
            case self.TOML:
                return FilenameMatcher("*.toml")
            case self.ZIG:
                return FilenameMatcher("*.zig", "*.zon")
            case self.LUA:
                return FilenameMatcher("*.lua")
            case self.NIX:
                return FilenameMatcher("*.nix")
            case self.ERLANG:
                return FilenameMatcher("*.erl", "*.hrl", "*.escript", "*.config", "*.app", "*.app.src")
            case self.AL:
                return FilenameMatcher("*.al", "*.dal")
            case self.FSHARP:
                return FilenameMatcher("*.fs", "*.fsx", "*.fsi")
            case self.REGO:
                return FilenameMatcher("*.rego")
            case self.MARKDOWN:
                return FilenameMatcher("*.md", "*.markdown")
            case self.SCALA:
                return FilenameMatcher("*.scala", "*.sbt")
            case self.JULIA:
                return FilenameMatcher("*.jl")
            case self.FORTRAN:
                return FilenameMatcher(
                    "*.f90", "*.F90", "*.f95", "*.F95", "*.f03", "*.F03", "*.f08", "*.F08", "*.f", "*.F", "*.for", "*.FOR", "*.fpp", "*.FPP"
                )
            case self.HASKELL:
                return FilenameMatcher("*.hs", "*.lhs")
            case self.VUE:
                path_patterns = ["*.vue"]
                for prefix in ["c", "m", ""]:
                    for postfix in ["x", ""]:
                        for base_pattern in ["ts", "js"]:
                            path_patterns.append(f"*.{prefix}{base_pattern}{postfix}")
                return FilenameMatcher(*path_patterns)
            case self.POWERSHELL:
                return FilenameMatcher("*.ps1", "*.psm1", "*.psd1")
            case self.PASCAL:
                return FilenameMatcher("*.pas", "*.pp", "*.lpr", "*.dpr", "*.dpk", "*.inc")
            case self.GROOVY:
                return FilenameMatcher("*.groovy", "*.gvy")
            case self.MATLAB:
                return FilenameMatcher("*.m", "*.mlx", "*.mlapp")
            case _:
                raise ValueError(f"Unhandled language: {self}")

    def get_ls_class(self) -> type["SolidLanguageServer"]:
        """Get the language server class for this language.

        Only languages with vendored server implementations are available.
        Others raise ValueError with guidance on adding support.
        """
        match self:
            case self.PYTHON:
                from anamnesis.lsp.solidlsp.language_servers.pyright_server import PyrightServer

                return PyrightServer
            case self.GO:
                from anamnesis.lsp.solidlsp.language_servers.gopls import Gopls

                return Gopls
            case self.RUST:
                from anamnesis.lsp.solidlsp.language_servers.rust_analyzer import RustAnalyzer

                return RustAnalyzer
            case self.TYPESCRIPT:
                from anamnesis.lsp.solidlsp.language_servers.typescript_language_server import TypeScriptLanguageServer

                return TypeScriptLanguageServer
            case _:
                raise ValueError(
                    f"Language server for '{self.value}' is not available in Anamnesis. "
                    f"Supported languages: python, go, rust, typescript"
                )

    @classmethod
    def from_ls_class(cls, ls_class: type["SolidLanguageServer"]) -> Self:
        """
        Get the Language enum value from a SolidLanguageServer class.

        :param ls_class: The SolidLanguageServer class to find the corresponding Language for
        :return: The Language enum value
        :raises ValueError: If the language server class is not supported
        """
        for enum_instance in cls:
            try:
                if enum_instance.get_ls_class() == ls_class:
                    return enum_instance
            except ValueError:
                continue  # Skip unsupported languages
        raise ValueError(f"Unhandled language server class: {ls_class}")


@dataclass
class LanguageServerConfig:
    """
    Configuration parameters
    """

    code_language: Language
    trace_lsp_communication: bool = False
    start_independent_lsp_process: bool = True
    ignored_paths: list[str] = field(default_factory=list)
    """Paths, dirs or glob-like patterns. The matching will follow the same logic as for .gitignore entries"""
    encoding: str = "utf-8"
    """File encoding to use when reading source files"""

    @classmethod
    def from_dict(cls, env: dict) -> Self:
        import inspect

        return cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})
