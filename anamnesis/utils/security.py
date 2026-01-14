"""
Security utilities for Anamnesis.

Provides path validation, traversal attack prevention, sensitive file detection,
and safe escaping utilities for database queries.

Ported from TypeScript path-validator.ts and security patterns in In-Memoria.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .logger import Logger


# ============================================================================
# Path Validation
# ============================================================================


@dataclass
class PathValidationResult:
    """Result of path validation."""

    is_valid: bool
    resolved_path: str | None
    error: str | None = None
    warnings: list[str] | None = None


class PathValidator:
    """
    Utility class for validating and resolving project paths.

    Used across MCP tools to ensure consistent path handling and
    prevent path traversal attacks.
    """

    # Common project root markers for different languages/frameworks
    PROJECT_MARKERS = [
        "package.json",      # Node.js/JavaScript
        "Cargo.toml",        # Rust
        "go.mod",            # Go
        "requirements.txt",  # Python (pip)
        "pyproject.toml",    # Python (modern)
        ".git",              # Git repository
        "pom.xml",           # Java (Maven)
        "build.gradle",      # Java/Kotlin (Gradle)
        "composer.json",     # PHP
        "Gemfile",           # Ruby
        "CMakeLists.txt",    # C/C++
        "Makefile",          # Generic
    ]

    @classmethod
    def validate_project_path(
        cls,
        path: str | None,
        context: str,
    ) -> str:
        """
        Validate and resolve a path for tool usage.

        Args:
            path: The path to validate
            context: Context for error messages (e.g., tool name)

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is invalid or doesn't exist
        """
        # If no path provided, use cwd but warn
        if not path:
            cwd = os.getcwd()
            Logger.warn(
                f"⚠️  Warning: No path provided to {context}.\n"
                f"   Using current directory: {cwd}\n"
                "   This may not be the intended project directory in MCP context.\n"
                "   Please provide an explicit path parameter to avoid issues."
            )
            return cwd

        # Resolve to absolute path
        abs_path = Path(path)
        if not abs_path.is_absolute():
            abs_path = Path(os.getcwd()) / path
        abs_path = abs_path.resolve()
        absolute_path = str(abs_path)

        # Check if path exists
        if not abs_path.exists():
            raise ValueError(
                f"Invalid path for {context}: {absolute_path}\n"
                "Path does not exist. Please provide a valid project directory path."
            )

        return absolute_path

    @classmethod
    def looks_like_project_root(cls, path: str) -> bool:
        """
        Check if a path looks like a project root.

        Checks for presence of package.json, .git, or other project markers.

        Args:
            path: The path to check

        Returns:
            True if path looks like a project root
        """
        path_obj = Path(path)

        for marker in cls.PROJECT_MARKERS:
            try:
                if (path_obj / marker).exists():
                    return True
            except (OSError, PermissionError):
                continue

        return False

    @classmethod
    def validate_and_warn_project_path(
        cls,
        path: str | None,
        context: str,
    ) -> str:
        """
        Validate path and warn if it doesn't look like a project root.

        Args:
            path: The path to validate
            context: Context for error messages

        Returns:
            Validated absolute path
        """
        validated_path = cls.validate_project_path(path, context)

        if not cls.looks_like_project_root(validated_path):
            Logger.warn(
                f"⚠️  Warning: {validated_path} doesn't look like a project root.\n"
                "   No package.json, Cargo.toml, .git, or other project markers found.\n"
                "   Make sure you're pointing to the correct directory.\n"
                f"   Tool: {context}"
            )

        return validated_path

    @classmethod
    def describe_path(cls, path: str) -> str:
        """
        Get a human-readable description of what's at the path.

        Args:
            path: The path to describe

        Returns:
            Description string
        """
        path_obj = Path(path)

        if not path_obj.exists():
            return "Path does not exist"

        markers = []

        if (path_obj / "package.json").exists():
            markers.append("Node.js project")
        if (path_obj / "Cargo.toml").exists():
            markers.append("Rust project")
        if (path_obj / ".git").exists():
            markers.append("Git repository")
        if (path_obj / "pyproject.toml").exists() or (path_obj / "requirements.txt").exists():
            markers.append("Python project")
        if (path_obj / "go.mod").exists():
            markers.append("Go project")

        if not markers:
            return "Directory (no project markers detected)"

        return ", ".join(markers)

    @classmethod
    def validate_safe_path(
        cls,
        path: str,
        base_path: str | None = None,
    ) -> PathValidationResult:
        """
        Validate that a path is safe (no traversal attacks).

        Args:
            path: The path to validate
            base_path: Optional base path that the resolved path must be within

        Returns:
            PathValidationResult with validation outcome
        """
        warnings: list[str] = []

        # Check for path traversal patterns
        if ".." in path:
            # Remove path traversal attempts
            sanitized = re.sub(r"\.\.[\\/]?", "", path)
            if sanitized != path:
                warnings.append(f"Path traversal detected and sanitized: {path}")
                path = sanitized

        try:
            resolved = Path(path).resolve()
            resolved_str = str(resolved)

            # If base_path is provided, ensure resolved path is within it
            if base_path:
                base_resolved = Path(base_path).resolve()
                try:
                    resolved.relative_to(base_resolved)
                except ValueError:
                    return PathValidationResult(
                        is_valid=False,
                        resolved_path=None,
                        error=f"Path escapes base directory: {resolved_str} is not within {base_resolved}",
                    )

            return PathValidationResult(
                is_valid=True,
                resolved_path=resolved_str,
                warnings=warnings if warnings else None,
            )

        except (OSError, ValueError) as e:
            return PathValidationResult(
                is_valid=False,
                resolved_path=None,
                error=f"Invalid path: {e}",
            )


# ============================================================================
# Sensitive File Detection
# ============================================================================


# Patterns for detecting sensitive files
SENSITIVE_FILE_PATTERNS = [
    # Environment and secrets
    r"\.env$",
    r"\.env\.[^/]+$",
    r"\.env\.local$",
    r"\.env\.production$",
    r"\.secret[s]?$",
    # Private keys
    r"\.pem$",
    r"\.key$",
    r"\.p12$",
    r"\.pfx$",
    r"id_rsa$",
    r"id_dsa$",
    r"id_ed25519$",
    r"id_ecdsa$",
    # Credentials
    r"credentials\.json$",
    r"credentials\.yaml$",
    r"credentials\.yml$",
    r"\.htpasswd$",
    r"\.netrc$",
    # AWS
    r"\.aws/credentials$",
    r"\.aws/config$",
    # SSH
    r"\.ssh/",
    r"authorized_keys$",
    r"known_hosts$",
    # Kubernetes
    r"kubeconfig$",
    r"\.kube/config$",
    # Docker
    r"\.docker/config\.json$",
    # Database
    r"\.sqlite$",
    r"\.db$",
    # History files
    r"\.bash_history$",
    r"\.zsh_history$",
    r"\.history$",
]

# Compiled patterns for efficiency
_SENSITIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_FILE_PATTERNS]


def is_sensitive_file(file_path: str) -> bool:
    """
    Check if a file path matches sensitive file patterns.

    Args:
        file_path: Path to check

    Returns:
        True if the file appears to be sensitive
    """
    # Normalize path separators
    normalized = file_path.replace("\\", "/")

    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(normalized):
            return True

    return False


def get_sensitivity_reason(file_path: str) -> str | None:
    """
    Get the reason why a file is considered sensitive.

    Args:
        file_path: Path to check

    Returns:
        Reason string if sensitive, None otherwise
    """
    normalized = file_path.replace("\\", "/").lower()

    if re.search(r"\.env", normalized):
        return "Environment file may contain secrets"
    if re.search(r"\.(pem|key|p12|pfx)$", normalized):
        return "Private key file"
    if re.search(r"credentials\.(json|yaml|yml)$", normalized):
        return "Credentials file"
    if re.search(r"id_(rsa|dsa|ed25519|ecdsa)$", normalized):
        return "SSH private key"
    if re.search(r"\.aws/", normalized):
        return "AWS configuration"
    if re.search(r"\.ssh/", normalized):
        return "SSH configuration"
    if re.search(r"\.kube/|kubeconfig", normalized):
        return "Kubernetes configuration"
    if re.search(r"\.db$|\.sqlite$", normalized):
        return "Database file"

    return None


# ============================================================================
# SQL Escaping
# ============================================================================


def escape_sql_like(value: str) -> str:
    r"""
    Escape special characters for SQL LIKE patterns.

    Escapes %, _, and \ characters to prevent SQL injection
    in LIKE clauses.

    Args:
        value: The value to escape

    Returns:
        Escaped string safe for LIKE patterns
    """
    # Escape backslash first (it's the escape character)
    escaped = value.replace("\\", "\\\\")
    # Escape LIKE wildcards
    escaped = escaped.replace("%", "\\%")
    escaped = escaped.replace("_", "\\_")
    return escaped


def escape_sql_string(value: str) -> str:
    """
    Escape special characters for SQL string literals.

    Note: Prefer parameterized queries over string escaping when possible.

    Args:
        value: The value to escape

    Returns:
        Escaped string safe for SQL literals
    """
    # Escape single quotes by doubling them
    return value.replace("'", "''")


# ============================================================================
# Path Sanitization
# ============================================================================


def sanitize_path(path: str) -> str:
    """
    Sanitize a path by removing path traversal attempts.

    Args:
        path: Path to sanitize

    Returns:
        Sanitized path
    """
    # Remove path traversal patterns
    sanitized = re.sub(r"\.\.[\\/]?", "", path)

    # Remove null bytes
    sanitized = sanitized.replace("\x00", "")

    # Normalize multiple slashes
    sanitized = re.sub(r"[\\/]+", os.sep, sanitized)

    return sanitized


def is_safe_filename(filename: str) -> bool:
    """
    Check if a filename is safe (no path separators or special chars).

    Args:
        filename: Filename to check

    Returns:
        True if filename is safe
    """
    # Check for path separators
    if "/" in filename or "\\" in filename:
        return False

    # Check for null bytes
    if "\x00" in filename:
        return False

    # Check for path traversal
    if ".." in filename:
        return False

    # Check for reserved names on Windows
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    base_name = filename.split(".")[0].upper()
    if base_name in reserved_names:
        return False

    return True


# ============================================================================
# Input Validation
# ============================================================================


def validate_string_length(
    value: str,
    name: str,
    min_length: int = 0,
    max_length: int | None = None,
) -> None:
    """
    Validate string length is within bounds.

    Args:
        value: String to validate
        name: Parameter name for error messages
        min_length: Minimum length (default: 0)
        max_length: Maximum length (None = no limit)

    Raises:
        ValueError: If length is out of bounds
    """
    if len(value) < min_length:
        raise ValueError(f"{name} must be at least {min_length} characters")

    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")


def validate_positive_integer(
    value: int,
    name: str,
    allow_zero: bool = False,
) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether zero is allowed

    Raises:
        ValueError: If value is invalid
    """
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def validate_enum_value(
    value: str,
    name: str,
    allowed: list[str],
) -> None:
    """
    Validate that a value is one of allowed options.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allowed: List of allowed values

    Raises:
        ValueError: If value is not in allowed list
    """
    if value not in allowed:
        allowed_str = ", ".join(f"'{v}'" for v in allowed)
        raise ValueError(f"{name} must be one of: {allowed_str}")
