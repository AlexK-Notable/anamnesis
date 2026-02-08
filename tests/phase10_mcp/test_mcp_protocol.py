"""End-to-end MCP protocol tests.

Tests the actual MCP server over stdio transport, verifying:
- JSON-RPC protocol compliance
- MCP initialization handshake
- Tool listing and schemas
- Tool execution via protocol
- Error handling at protocol level
"""

import json
import select
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class MCPClient:
    """Simple MCP client for testing over stdio."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.request_id = 0

    def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and get response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id,
        }
        if params is not None:
            request["params"] = params

        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()

        # Read response with timeout
        ready, _, _ = select.select([self.process.stdout], [], [], 30.0)
        if not ready:
            raise TimeoutError(f"MCP server did not respond within 30s for '{method}'")

        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")

        return json.loads(response_line)

    def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line)
        self.process.stdin.flush()


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal Python project
        (Path(tmpdir) / "main.py").write_text(
            '''"""Main module."""

def hello():
    """Say hello."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
'''
        )
        (Path(tmpdir) / "utils.py").write_text(
            '''"""Utility functions."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        )
        yield tmpdir


@pytest.fixture
def mcp_server(temp_project):
    """Start MCP server as subprocess."""
    # Start the server pointing to temp project
    process = subprocess.Popen(
        [sys.executable, "-m", "anamnesis.mcp_server", temp_project],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        cwd=temp_project,
    )

    client = MCPClient(process)

    yield client

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


class TestMCPInitialization:
    """Tests for MCP initialization handshake."""

    def test_initialize_request(self, mcp_server):
        """Server responds to initialize request."""
        response = mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0",
                },
            },
        )

        assert "result" in response, f"Expected result, got: {response}"
        result = response["result"]

        # Verify server info
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "anamnesis"

        # Verify protocol version
        assert "protocolVersion" in result

        # Verify capabilities
        assert "capabilities" in result
        assert "tools" in result["capabilities"]

    def test_initialized_notification(self, mcp_server):
        """Server accepts initialized notification."""
        # First initialize
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        # Send initialized notification (no response expected)
        mcp_server.send_notification("notifications/initialized")

        # Verify server still responds to requests
        response = mcp_server.send_request("tools/list", {})
        assert "result" in response


class TestMCPToolListing:
    """Tests for tool listing."""

    def test_list_tools(self, mcp_server):
        """Server lists available tools."""
        # Initialize first
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request("tools/list", {})

        assert "result" in response, f"Expected result, got: {response}"
        result = response["result"]

        assert "tools" in result
        tools = result["tools"]
        assert len(tools) > 0

        # Verify tool structure
        tool_names = [t["name"] for t in tools]
        assert "get_system_status" in tool_names
        assert "auto_learn_if_needed" in tool_names
        assert "analyze_project" in tool_names

    def test_tool_has_schema(self, mcp_server):
        """Each tool has proper input schema."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request("tools/list", {})
        tools = response["result"]["tools"]

        for tool in tools:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"

            schema = tool["inputSchema"]
            assert schema.get("type") == "object", f"Tool {tool['name']} schema not object type"


class TestMCPToolExecution:
    """Tests for tool execution via protocol."""

    def test_system_status_health_tool(self, mcp_server, temp_project):
        """Execute get_system_status tool with health section via MCP."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "get_system_status",
                "arguments": {"sections": "health", "path": temp_project},
            },
        )

        assert "result" in response, f"Expected result, got: {response}"
        result = response["result"]

        # MCP tool results have content array
        assert "content" in result
        assert len(result["content"]) > 0

        # Parse the text content
        content = result["content"][0]
        assert content["type"] == "text"

        # The text should be JSON with health section under data
        status_data = json.loads(content["text"])
        assert "health" in status_data["data"]
        assert status_data["data"]["health"]["healthy"] is True
        assert "checks" in status_data["data"]["health"]

    def test_auto_learn_tool(self, mcp_server, temp_project):
        """Execute auto_learn_if_needed tool via MCP."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "auto_learn_if_needed",
                "arguments": {"path": temp_project, "force": True},
            },
        )

        assert "result" in response, f"Expected result, got: {response}"
        content = response["result"]["content"][0]
        learn_data = json.loads(content["text"])

        assert learn_data["data"]["status"] == "learned"
        assert "concepts_learned" in learn_data["data"]

    def test_analyze_project_tool(self, mcp_server, temp_project):
        """Execute analyze_project tool via MCP."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        # Learn first
        mcp_server.send_request(
            "tools/call",
            {
                "name": "auto_learn_if_needed",
                "arguments": {"path": temp_project, "force": True},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "analyze_project",
                "arguments": {"path": temp_project, "scope": "project"},
            },
        )

        assert "result" in response
        content = response["result"]["content"][0]
        blueprint = json.loads(content["text"])

        assert "blueprint" in blueprint["data"]
        bp = blueprint["data"]["blueprint"]
        assert "tech_stack" in bp
        assert "learning_status" in bp

    def test_auto_learn_if_needed_tool(self, mcp_server, temp_project):
        """Execute auto_learn_if_needed tool via MCP."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "auto_learn_if_needed",
                "arguments": {"path": temp_project, "force": True},
            },
        )

        assert "result" in response
        content = response["result"]["content"][0]
        data = json.loads(content["text"])

        assert data["data"]["status"] in ["learned", "already_learned", "skipped"]

    def test_get_system_status_tool(self, mcp_server, temp_project):
        """Execute get_system_status tool via MCP."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "get_system_status",
                "arguments": {},
            },
        )

        assert "result" in response
        content = response["result"]["content"][0]
        status = json.loads(content["text"])

        # Default sections are summary + metrics
        assert "summary" in status["data"]
        assert status["data"]["summary"]["status"] in ["healthy", "degraded", "unhealthy"]
        assert "metrics" in status["data"]


class TestMCPErrorHandling:
    """Tests for protocol-level error handling."""

    def test_invalid_tool_name(self, mcp_server):
        """Server handles invalid tool name."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        )

        # Should get an error response
        assert "error" in response or (
            "result" in response
            and response["result"].get("isError", False)
        )

    @pytest.mark.xfail(reason="FastMCP may not respond to unknown methods (treated as notifications)")
    def test_invalid_method(self, mcp_server):
        """Server handles invalid method."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request("invalid/method", {})

        # Should get an error
        assert "error" in response

    def test_tool_with_invalid_arguments(self, mcp_server):
        """Server handles invalid tool arguments gracefully."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request(
            "tools/call",
            {
                "name": "get_system_status",
                "arguments": {"sections": "health", "path": "/nonexistent/path/that/does/not/exist"},
            },
        )

        # Should return result with health.healthy=False, not protocol error
        assert "result" in response
        content = response["result"]["content"][0]
        data = json.loads(content["text"])
        assert data["data"]["health"]["healthy"] is False


class TestMCPProtocolCompliance:
    """Tests for JSON-RPC and MCP protocol compliance."""

    def test_response_has_jsonrpc_version(self, mcp_server):
        """All responses include jsonrpc version."""
        response = mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        assert response.get("jsonrpc") == "2.0"

    def test_response_has_matching_id(self, mcp_server):
        """Response ID matches request ID."""
        # Send multiple requests and verify IDs match
        for i in range(3):
            response = mcp_server.send_request(
                "initialize" if i == 0 else "tools/list",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                }
                if i == 0
                else {},
            )
            assert response["id"] == i + 1

    @pytest.mark.xfail(reason="FastMCP may not respond to unknown methods (treated as notifications)")
    def test_error_response_format(self, mcp_server):
        """Error responses follow JSON-RPC format."""
        mcp_server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )

        response = mcp_server.send_request("invalid/method", {})

        if "error" in response:
            error = response["error"]
            assert "code" in error
            assert "message" in error
            assert isinstance(error["code"], int)
            assert isinstance(error["message"], str)


class TestLearnQueryRecommendE2E:
    """End-to-end test: learn codebase -> query insights -> get recommendations.

    Uses _impl functions directly (not subprocess) for a multi-step workflow
    that exercises the intelligence pipeline from learning through querying.
    """

    @pytest.fixture(autouse=True)
    def _setup_project(self, temp_project):
        """Reset server state and activate temp_project."""
        import anamnesis.mcp_server._shared as shared_module

        orig_persist = shared_module._registry._persist_path
        shared_module._registry._persist_path = None
        shared_module._registry.reset()
        shared_module._registry.activate(temp_project)

        self._project_path = temp_project

        yield

        shared_module._registry.reset()
        shared_module._registry._persist_path = orig_persist

    def test_learn_query_recommend_e2e(self):
        """Full pipeline: auto_learn -> manage_concepts(query) -> get_coding_guidance."""
        from anamnesis.mcp_server.tools.intelligence import (
            _get_coding_guidance_impl,
            _manage_concepts_impl,
        )
        from anamnesis.mcp_server.tools.learning import _auto_learn_if_needed_impl
        from anamnesis.utils.toon_encoder import ToonEncoder

        _toon = ToonEncoder()

        def as_dict(result):
            if isinstance(result, str):
                return _toon.decode(result)
            return result

        # Step 1: Learn from the sample project
        learn_result = as_dict(
            _auto_learn_if_needed_impl(path=self._project_path, force=True)
        )
        assert learn_result["success"] is True
        assert learn_result["data"]["status"] == "learned"
        assert "concepts_learned" in learn_result["data"]

        # Step 2: Query semantic insights (should find symbols from learned project)
        insights_result = as_dict(_manage_concepts_impl(action="query"))
        assert insights_result["success"] is True
        assert isinstance(insights_result["data"], list)
        assert isinstance(insights_result["metadata"]["total"], int)

        # Step 3: Get coding guidance (pattern recommendations) for a task
        rec_result = as_dict(
            _get_coding_guidance_impl(
                problem_description="add a utility function",
                include_patterns=True,
                include_file_routing=False,
            )
        )
        assert rec_result["success"] is True
        assert "recommendations" in rec_result["data"]
        assert isinstance(rec_result["data"]["recommendations"], list)
        assert rec_result["metadata"]["problem_description"] == "add a utility function"


class TestMemoryCrudLifecycleE2E:
    """End-to-end test: write -> read -> list -> search -> delete memory.

    Exercises the full memory CRUD pipeline through _impl functions
    to verify the MemoryService integration end-to-end.
    """

    @pytest.fixture(autouse=True)
    def _setup_project(self, temp_project):
        """Reset server state and activate temp_project."""
        import anamnesis.mcp_server._shared as shared_module

        orig_persist = shared_module._registry._persist_path
        shared_module._registry._persist_path = None
        shared_module._registry.reset()
        shared_module._registry.activate(temp_project)

        yield

        shared_module._registry.reset()
        shared_module._registry._persist_path = orig_persist

    def test_memory_crud_lifecycle(self):
        """Full pipeline: write -> read -> list -> search -> delete -> verify gone."""
        from anamnesis.mcp_server.tools.memory import (
            _delete_memory_impl,
            _read_memory_impl,
            _search_memories_impl,
            _write_memory_impl,
        )
        from anamnesis.utils.toon_encoder import ToonEncoder

        _toon = ToonEncoder()

        def as_dict(result):
            if isinstance(result, str):
                return _toon.decode(result)
            return result

        # Step 1: Write a memory
        write_result = as_dict(
            _write_memory_impl("test-decisions", "# Architecture Decisions\n\nUse SQLite for storage.")
        )
        assert write_result["success"] is True
        assert write_result["data"]["name"] == "test-decisions"

        # Step 2: Read it back
        read_result = as_dict(_read_memory_impl("test-decisions"))
        assert read_result["success"] is True
        assert "Architecture Decisions" in read_result["data"]["content"]

        # Step 3: List all memories — should include the new one
        list_result = as_dict(_search_memories_impl(query=None))
        assert list_result["success"] is True
        names = [m["name"] for m in list_result["data"]]
        assert "test-decisions" in names

        # Step 4: Search for the memory
        search_result = as_dict(_search_memories_impl("architecture"))
        assert search_result["success"] is True
        assert len(search_result["data"]) > 0

        # Step 5: Delete the memory
        delete_result = as_dict(_delete_memory_impl("test-decisions"))
        assert delete_result["success"] is True

        # Step 6: Verify it's gone
        list_after = as_dict(_search_memories_impl(query=None))
        names_after = [m["name"] for m in list_after["data"]]
        assert "test-decisions" not in names_after
