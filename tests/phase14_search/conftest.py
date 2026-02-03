"""Fixtures for Phase 14 search tests.

All fixtures create REAL files in temp directories. No mocks.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_python_files(tmp_path: Path) -> Path:
    """Create real Python files for search testing.

    Creates a realistic mini-codebase structure with searchable content.
    """
    # src/auth/service.py
    auth_dir = tmp_path / "src" / "auth"
    auth_dir.mkdir(parents=True)
    (auth_dir / "service.py").write_text('''"""Authentication service module."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    email: str
    is_active: bool = True


class AuthenticationService:
    """Service for user authentication and session management."""

    def __init__(self, secret_key: str):
        self._secret_key = secret_key
        self._sessions: dict[str, User] = {}

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password.

        Args:
            email: User email address
            password: User password

        Returns:
            User object if authenticated, None otherwise
        """
        # Authentication logic here
        if self._verify_credentials(email, password):
            user = User(user_id="123", email=email)
            return user
        return None

    def _verify_credentials(self, email: str, password: str) -> bool:
        """Verify user credentials against database."""
        # In real implementation, check against database
        return True

    def create_session(self, user: User) -> str:
        """Create a new session for authenticated user."""
        session_id = f"session_{user.user_id}"
        self._sessions[session_id] = user
        return session_id

    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate a session and return the user."""
        return self._sessions.get(session_id)
''')

    (auth_dir / "__init__.py").write_text('"""Auth module."""\nfrom .service import AuthenticationService, User\n')

    # src/database/connection.py
    db_dir = tmp_path / "src" / "database"
    db_dir.mkdir(parents=True)
    (db_dir / "connection.py").write_text('''"""Database connection management."""

import asyncio
from typing import Optional


class DatabaseConnection:
    """Manages database connections with pooling."""

    def __init__(self, connection_string: str, pool_size: int = 10):
        self._connection_string = connection_string
        self._pool_size = pool_size
        self._pool: list = []
        self._connected = False

    async def connect(self) -> None:
        """Establish database connection."""
        if not self._connected:
            # Initialize connection pool
            self._pool = [None] * self._pool_size
            self._connected = True

    async def disconnect(self) -> None:
        """Close database connection."""
        self._pool.clear()
        self._connected = False

    async def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """Execute a database query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query results as list of rows
        """
        if not self._connected:
            raise RuntimeError("Not connected to database")
        # Execute query
        return []

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected
''')

    (db_dir / "__init__.py").write_text('"""Database module."""\n')

    # src/utils/helpers.py
    utils_dir = tmp_path / "src" / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "helpers.py").write_text('''"""Utility helper functions."""

import re
from typing import Any


def sanitize_input(value: str) -> str:
    """Sanitize user input by removing dangerous characters."""
    return re.sub(r'[<>&"\\'']', '', value)


def format_response(data: Any, status: str = "success") -> dict:
    """Format API response."""
    return {
        "status": status,
        "data": data,
    }


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


class ConfigLoader:
    """Loads configuration from files."""

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._cache: dict = {}

    def load(self) -> dict:
        """Load configuration file."""
        # Load config logic
        return self._cache

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._cache.get(key, default)
''')

    return tmp_path


@pytest.fixture
def sample_javascript_files(tmp_path: Path) -> Path:
    """Create real JavaScript/TypeScript files for search testing."""
    js_dir = tmp_path / "src"
    js_dir.mkdir(parents=True, exist_ok=True)

    (js_dir / "api.js").write_text('''/**
 * API client module
 */

class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  async get(endpoint) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
      headers: this.headers,
    });
    return response.json();
  }

  async post(endpoint, data) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data),
    });
    return response.json();
  }
}

function createApiClient(baseUrl) {
  return new ApiClient(baseUrl);
}

export { ApiClient, createApiClient };
''')

    (js_dir / "utils.ts").write_text('''/**
 * TypeScript utility functions
 */

interface User {
  id: string;
  name: string;
  email: string;
}

interface ApiResponse<T> {
  status: string;
  data: T;
  error?: string;
}

function formatUser(user: User): string {
  return `${user.name} <${user.email}>`;
}

async function fetchUsers(): Promise<User[]> {
  // Fetch users from API
  return [];
}

type ValidationResult = {
  valid: boolean;
  errors: string[];
};

function validateInput(input: string): ValidationResult {
  const errors: string[] = [];
  if (!input) {
    errors.push('Input is required');
  }
  return {
    valid: errors.length === 0,
    errors,
  };
}

export { User, ApiResponse, formatUser, fetchUsers, validateInput };
''')

    return tmp_path


@pytest.fixture
def sample_go_files(tmp_path: Path) -> Path:
    """Create real Go files for search testing."""
    go_dir = tmp_path / "pkg"
    go_dir.mkdir(parents=True)

    (go_dir / "handler.go").write_text('''package handler

import (
	"encoding/json"
	"net/http"
)

// User represents a user in the system
type User struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

// UserHandler handles user-related HTTP requests
type UserHandler struct {
	users map[string]*User
}

// NewUserHandler creates a new UserHandler
func NewUserHandler() *UserHandler {
	return &UserHandler{
		users: make(map[string]*User),
	}
}

// GetUser handles GET /users/:id
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("id")
	user, ok := h.users[userID]
	if !ok {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}
	json.NewEncoder(w).Encode(user)
}

// CreateUser handles POST /users
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
	var user User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	h.users[user.ID] = &user
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}
''')

    return tmp_path


@pytest.fixture
def mixed_language_codebase(
    sample_python_files: Path,
    sample_javascript_files: Path,
    sample_go_files: Path,
) -> Path:
    """Create a mixed-language codebase for comprehensive testing.

    Combines Python, JavaScript/TypeScript, and Go files.
    """
    # Copy JS files to Python codebase
    src = sample_javascript_files / "src"
    dst = sample_python_files / "frontend"
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():  # Only copy files, not directories
            (dst / f.name).write_text(f.read_text())

    # Copy Go files to Python codebase
    src = sample_go_files / "pkg"
    dst = sample_python_files / "services"
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():  # Only copy files, not directories
            (dst / f.name).write_text(f.read_text())

    return sample_python_files
