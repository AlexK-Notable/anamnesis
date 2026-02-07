"""
Anamnesis type definitions.

This module exports all type definitions for the Anamnesis semantic code analysis system.
"""

# Core types
from .core import LineRange

# Error types
from .errors import (
    AnamnesisError,
    AnamnesisSystemError,
    ConfigurationError,
    DatabaseError,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    ExecutionError,
    JSONRPCError,
    LearningError,
    MCPErrorCode,
    MCPErrorResponse,
    RecoveryAction,
    ResourceError,
    ValidationError,
)

__all__ = [
    # Core types
    "LineRange",
    # Error types
    "MCPErrorCode",
    "ErrorCode",
    "ErrorSeverity",
    "RecoveryAction",
    "ErrorContext",
    "MCPErrorResponse",
    "JSONRPCError",
    "AnamnesisError",
    "ConfigurationError",
    "ValidationError",
    "DatabaseError",
    "ExecutionError",
    "LearningError",
    "ResourceError",
    "AnamnesisSystemError",
]
