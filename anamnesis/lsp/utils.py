"""Shared utilities for the LSP integration layer.

Contains path safety and URI conversion functions used by both
SymbolRetriever and CodeEditor.
"""

from __future__ import annotations

import os
import pathlib


def safe_join(root: str, relative_path: str) -> str:
    """Join root and relative path, ensuring result stays within root.

    Raises:
        ValueError: If the resolved path escapes the project root.
    """
    abs_path = os.path.realpath(os.path.join(root, relative_path))
    root_real = os.path.realpath(root)
    if not (abs_path == root_real or abs_path.startswith(root_real + os.sep)):
        raise ValueError(
            f"Path traversal denied: '{relative_path}' resolves outside '{root_real}'"
        )
    return abs_path


def uri_to_relative(uri: str, project_root: str) -> str:
    """Convert a file:// URI to a project-relative path."""
    if uri.startswith("file://"):
        abs_path = uri[7:]
        try:
            return str(pathlib.Path(abs_path).relative_to(project_root))
        except ValueError:
            return abs_path
    return uri
