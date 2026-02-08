"""
Security utilities for Anamnesis.

Provides sensitive file detection, safe escaping for database queries,
path sanitization, and input validation utilities.
"""

import os
import re

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

    for pattern, reason in _SENSITIVE_REASON_PATTERNS:
        if pattern.search(normalized):
            return reason

    return None


_SENSITIVE_REASON_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.env", re.IGNORECASE), "Environment file may contain secrets"),
    (re.compile(r"\.(pem|key|p12|pfx)$", re.IGNORECASE), "Private key file"),
    (re.compile(r"credentials\.(json|yaml|yml)$", re.IGNORECASE), "Credentials file"),
    (re.compile(r"id_(rsa|dsa|ed25519|ecdsa)$", re.IGNORECASE), "SSH private key"),
    (re.compile(r"\.aws/", re.IGNORECASE), "AWS configuration"),
    (re.compile(r"\.ssh/", re.IGNORECASE), "SSH configuration"),
    (re.compile(r"\.kube/|kubeconfig", re.IGNORECASE), "Kubernetes configuration"),
    (re.compile(r"\.db$|\.sqlite$", re.IGNORECASE), "Database file"),
]


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
