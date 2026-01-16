"""Entry point for running MCP server as a module.

Usage:
    python -m anamnesis.mcp_server [path]

Args:
    path: Optional working directory for the server (defaults to cwd)
"""

import os
import sys
from pathlib import Path

from anamnesis.mcp_server.server import _set_current_path, mcp


def main() -> None:
    """Run the MCP server."""
    # Get optional path argument
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve()
        if path.exists() and path.is_dir():
            os.chdir(path)
            _set_current_path(str(path))

    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
