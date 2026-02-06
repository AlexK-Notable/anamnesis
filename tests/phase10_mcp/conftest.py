"""Shared test infrastructure for phase10_mcp tests.

Provides unified fixtures for resetting server state, decoding TOON
responses, and creating sample Python projects for _impl testing.
"""

import pytest

from anamnesis.utils.toon_encoder import ToonEncoder

_toon = ToonEncoder()


def _as_dict(result):
    """Decode TOON-encoded responses back to dict for assertions.

    _impl functions go through _with_error_handling which may TOON-encode
    eligible success responses. Tests need dict access for assertions.
    """
    if isinstance(result, str):
        return _toon.decode(result)
    return result


@pytest.fixture()
def reset_server_state(tmp_path):
    """Reset server global state and activate tmp_path as project.

    Unified fixture replacing the three separate versions in:
    - test_session_tools.py
    - test_project_registry.py
    - test_memory_and_metacognition.py

    Usage: Either accept as parameter or set autouse=True in class/module.
    """
    import anamnesis.mcp_server._shared as shared_module

    orig_persist = shared_module._registry._persist_path
    shared_module._registry._persist_path = None
    shared_module._registry.reset()
    shared_module._registry.activate(str(tmp_path))

    yield tmp_path

    shared_module._registry.reset()
    shared_module._registry._persist_path = orig_persist


@pytest.fixture()
def sample_python_project(tmp_path):
    """Create a sample Python project for _impl integration tests.

    Contains classes, functions with varying complexity, and multiple
    files for cross-file operations. Uses real Python code that
    tree-sitter can parse.
    """
    src = tmp_path / "src"
    src.mkdir()

    (src / "service.py").write_text(
        '''\
class UserService:
    """Service for user operations."""

    def get_user(self, user_id: int) -> dict:
        """Fetch a user by ID."""
        return {"id": user_id}

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        return {"name": name}

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID."""
        return True


def complex_handler(request, data, config):
    """Handler with moderate cyclomatic complexity."""
    if request.method == "POST":
        if data.get("validate"):
            for item in data["items"]:
                if item.get("type") == "a":
                    process_a(item)
                elif item.get("type") == "b":
                    process_b(item)
                else:
                    raise ValueError("unknown type")
        else:
            raise ValueError("invalid")
    elif request.method == "GET":
        return fetch(config)
    elif request.method == "DELETE":
        return remove(config)
    return None


def simple_add(a, b):
    """Simple function with low complexity."""
    return a + b


def process_a(item):
    return item


def process_b(item):
    return item


def fetch(config):
    return config


def remove(config):
    return True
'''
    )

    (src / "utils.py").write_text(
        '''\
def helper_func():
    """General helper."""
    return "help"


def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first} {last}"


CONSTANT_VALUE = 42
MAX_RETRIES = 3
'''
    )

    (src / "__init__.py").write_text("")

    return tmp_path
