"""LSP server lifecycle management.

Manages language server processes per project, routing files to the
correct server based on extension and handling lazy initialization.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from anamnesis.lsp.solidlsp.ls_config import Language, LanguageServerConfig
from anamnesis.lsp.solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)

# Languages with vendored server implementations
_SUPPORTED_LANGUAGES: dict[str, Language] = {
    "python": Language.PYTHON,
    "go": Language.GO,
    "rust": Language.RUST,
    "typescript": Language.TYPESCRIPT,
}


class LspManager:
    """Manages LSP server lifecycle for a project.

    Each project gets its own LspManager, which lazily starts language
    servers on first use and shuts them down when the project is deactivated.
    """

    def __init__(self, project_root: str, encoding: str = "utf-8") -> None:
        self._project_root = os.path.abspath(project_root)
        self._encoding = encoding
        self._servers: dict[Language, Any] = {}  # Language -> SolidLanguageServer
        self._settings = SolidLSPSettings()

    @property
    def project_root(self) -> str:
        return self._project_root

    def get_language_for_file(self, relative_path: str) -> Language | None:
        """Determine the language for a file based on its extension.

        Args:
            relative_path: Path relative to the project root.

        Returns:
            The Language enum value, or None if no supported language matches.
        """
        filename = os.path.basename(relative_path)
        best_match: Language | None = None
        best_priority = -1

        for lang in _SUPPORTED_LANGUAGES.values():
            matcher = lang.get_source_fn_matcher()
            if matcher.is_relevant_filename(filename):
                priority = lang.get_priority()
                if priority > best_priority:
                    best_match = lang
                    best_priority = priority

        return best_match

    def get_language_server(self, relative_path: str) -> Any | None:
        """Get a running language server for the given file.

        Lazily starts the server if not already running.

        Args:
            relative_path: Path relative to the project root.

        Returns:
            A SolidLanguageServer instance, or None if the language isn't supported.
        """
        language = self.get_language_for_file(relative_path)
        if language is None:
            return None
        return self.get_server_for_language(language.value)

    def get_server_for_language(self, language: str) -> Any | None:
        """Get a running language server for the given language.

        Args:
            language: Language name (e.g., "python", "go").

        Returns:
            A SolidLanguageServer instance, or None if not available.
        """
        lang_enum = _SUPPORTED_LANGUAGES.get(language)
        if lang_enum is None:
            return None

        if lang_enum in self._servers:
            return self._servers[lang_enum]

        # Lazy start
        if self.start(language):
            return self._servers.get(lang_enum)
        return None

    def is_available(self, language: str) -> bool:
        """Check if a language server is available (supported + binary found)."""
        return language in _SUPPORTED_LANGUAGES

    def is_running(self, language: str) -> bool:
        """Check if a language server is currently running."""
        lang_enum = _SUPPORTED_LANGUAGES.get(language)
        return lang_enum is not None and lang_enum in self._servers

    def start(self, language: str) -> bool:
        """Start a language server for the given language.

        Returns:
            True if the server was started successfully.
        """
        lang_enum = _SUPPORTED_LANGUAGES.get(language)
        if lang_enum is None:
            log.warning("Language '%s' is not supported for LSP", language)
            return False

        if lang_enum in self._servers:
            log.debug("Language server for '%s' already running", language)
            return True

        try:
            config = LanguageServerConfig(
                code_language=lang_enum,
                encoding=self._encoding,
            )
            ls_class = lang_enum.get_ls_class()
            server = ls_class(config, self._project_root, self._settings)
            server._start_server_process()
            self._servers[lang_enum] = server
            log.info("Started %s language server for project %s",
                     language, os.path.basename(self._project_root))
            return True
        except Exception:
            log.error("Failed to start %s language server", language, exc_info=True)
            return False

    def stop(self, language: str) -> None:
        """Stop a running language server."""
        lang_enum = _SUPPORTED_LANGUAGES.get(language)
        if lang_enum is None or lang_enum not in self._servers:
            return

        server = self._servers.pop(lang_enum)
        try:
            server.stop()
            log.info("Stopped %s language server", language)
        except Exception:
            log.warning("Error stopping %s language server", language, exc_info=True)

    def stop_all(self) -> None:
        """Stop all running language servers."""
        for lang_enum in list(self._servers.keys()):
            server = self._servers.pop(lang_enum)
            try:
                server.stop()
                log.info("Stopped %s language server", lang_enum.value)
            except Exception:
                log.warning("Error stopping %s language server",
                            lang_enum.value, exc_info=True)

    def get_status(self) -> dict[str, Any]:
        """Get status of all language servers."""
        return {
            "project_root": self._project_root,
            "supported_languages": list(_SUPPORTED_LANGUAGES.keys()),
            "running_servers": {
                lang.value: {
                    "class": type(server).__name__,
                }
                for lang, server in self._servers.items()
            },
        }
