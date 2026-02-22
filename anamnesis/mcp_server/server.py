"""MCP Server coordinator — imports tool modules and exposes create_server()."""

from fastmcp import FastMCP

import anamnesis.mcp_server.tools as _tools  # triggers @mcp.tool registration
from anamnesis.mcp_server._shared import mcp

# Reference to prevent ruff F401 — tools must be imported for registration.
_tool_modules = _tools


def create_server() -> FastMCP:
    """Create and return the configured MCP server instance."""
    return mcp


if __name__ == "__main__":
    mcp.run()
