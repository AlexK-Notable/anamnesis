"""SolidLSP — language server protocol client library.

Originally from the Serena project (https://github.com/semgrep/serena),
adopted into Anamnesis as first-class maintained code. The three files under
``lsp_protocol_handler/`` (lsp_types.py, lsp_requests.py, server.py) are
generated from the OLSP project and licensed under the MIT License — do not
edit those directly.
"""

# ruff: noqa
from .ls import SolidLanguageServer
