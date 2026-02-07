"""
MCP-compliant error handling system for Anamnesis.

Provides structured error types following JSON-RPC 2.0 specification,
compatible with Model Context Protocol requirements.

Ported from TypeScript error-types.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

from anamnesis.constants import utcnow


class MCPErrorCode(IntEnum):
    """MCP error codes following JSON-RPC 2.0 specification."""

    # Standard JSON-RPC 2.0 errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific error codes (reserved range: -32000 to -32099)
    RESOURCE_NOT_FOUND = -32001
    RESOURCE_ACCESS_DENIED = -32002
    TOOL_EXECUTION_FAILED = -32003
    PROTOCOL_VIOLATION = -32004
    INITIALIZATION_FAILED = -32005
    TRANSPORT_ERROR = -32006
    AUTHENTICATION_FAILED = -32007
    SESSION_EXPIRED = -32008
    RATE_LIMITED = -32009
    SERVICE_UNAVAILABLE = -32010


class ErrorCode(IntEnum):
    """Internal error codes for categorization."""

    # System/Infrastructure Errors (1000-1999)
    PLATFORM_UNSUPPORTED = 1001
    NATIVE_BINDING_FAILED = 1002
    DATABASE_INIT_FAILED = 1003
    DATABASE_MIGRATION_FAILED = 1004
    VECTOR_DB_INIT_FAILED = 1005
    CIRCUIT_BREAKER_OPEN = 1006

    # File System Errors (2000-2999)
    FILE_NOT_FOUND = 2001
    FILE_READ_FAILED = 2002
    FILE_WRITE_FAILED = 2003
    DIRECTORY_NOT_FOUND = 2004
    PERMISSION_DENIED = 2005
    INVALID_PATH = 2006

    # Parsing/Analysis Errors (3000-3999)
    LANGUAGE_UNSUPPORTED = 3001
    PARSE_FAILED = 3002
    TREE_SITTER_FAILED = 3003
    CONCEPT_EXTRACTION_FAILED = 3004
    PATTERN_ANALYSIS_FAILED = 3005
    SEMANTIC_ANALYSIS_FAILED = 3006

    # Configuration Errors (4000-4999)
    INVALID_CONFIG = 4001
    MISSING_CONFIG = 4002
    CONFIG_VALIDATION_FAILED = 4003
    SETUP_INCOMPLETE = 4004

    # Network/External Errors (5000-5999)
    NETWORK_TIMEOUT = 5001
    EXTERNAL_SERVICE_FAILED = 5002
    RATE_LIMIT_EXCEEDED = 5003

    # User Input Errors (6000-6999)
    INVALID_ARGS = 6001
    MISSING_ARGS = 6002
    VALIDATION_FAILED = 6003


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryAction:
    """Suggested action to recover from an error."""

    description: str
    command: str | None = None
    automated: bool = False


@dataclass
class ErrorContext:
    """Context information for an error."""

    operation: str | None = None
    file_path: str | None = None
    language: str | None = None
    component: str | None = None
    timestamp: datetime = field(default_factory=utcnow)
    stack: str | None = None
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPErrorResponse:
    """MCP-compliant error response following JSON-RPC 2.0 specification."""

    code: MCPErrorCode
    message: str
    data: dict[str, Any] | None = None


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 error object for MCP protocol compliance."""

    error: MCPErrorResponse
    id: str | int | None = None
    jsonrpc: str = "2.0"


class AnamnesisError(Exception):
    """Base error class for Anamnesis with MCP compliance."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        user_message: str,
        severity: str = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.severity = severity
        self.user_message = user_message
        self.context = context or ErrorContext()
        self.recovery_actions = recovery_actions or []
        self.original_error = original_error

        # Update context with timestamp and stack trace
        self.context.timestamp = utcnow()
        if original_error:
            self.context.stack = str(original_error.__traceback__)

    def get_formatted_message(self) -> str:
        """Get a formatted error message for display to users."""
        parts = [
            f"[Error] {self.user_message}",
            f"   Code: {self.code.value}",
        ]

        if self.context.operation:
            parts.append(f"   Operation: {self.context.operation}")
        if self.context.file_path:
            parts.append(f"   File: {self.context.file_path}")
        if self.context.component:
            parts.append(f"   Component: {self.context.component}")

        if self.recovery_actions:
            parts.append("")
            parts.append("Suggested actions:")
            for i, action in enumerate(self.recovery_actions, 1):
                parts.append(f"   {i}. {action.description}")
                if action.command:
                    parts.append(f"      Run: {action.command}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "name": self.__class__.__name__,
            "code": self.code.value,
            "message": str(self),
            "user_message": self.user_message,
            "severity": self.severity,
            "context": {
                "operation": self.context.operation,
                "file_path": self.context.file_path,
                "language": self.context.language,
                "component": self.context.component,
                "timestamp": self.context.timestamp.isoformat(),
                "additional_info": self.context.additional_info,
            },
            "recovery_actions": [
                {"description": a.description, "command": a.command, "automated": a.automated}
                for a in self.recovery_actions
            ],
            "original_error": str(self.original_error) if self.original_error else None,
        }

    def to_mcp_error(self, request_id: str | int | None = None) -> MCPErrorResponse:
        """Convert to MCP-compliant error response."""
        mcp_code = self._get_mcp_error_code()

        return MCPErrorResponse(
            code=mcp_code,
            message=self.user_message,
            data={
                "type": self.__class__.__name__,
                "context": {
                    "operation": self.context.operation,
                    "file_path": self.context.file_path,
                    "component": self.context.component,
                },
                "recovery_actions": [
                    {"description": a.description, "command": a.command, "automated": a.automated}
                    for a in self.recovery_actions
                ],
                "timestamp": self.context.timestamp.isoformat(),
                "request_id": str(request_id) if request_id else None,
                "internal_code": self.code.value,
                "severity": self.severity,
                "original_message": str(self),
            },
        )

    def to_jsonrpc_error(self, request_id: str | int | None) -> JSONRPCError:
        """Convert to JSON-RPC 2.0 error response."""
        return JSONRPCError(
            error=self.to_mcp_error(request_id),
            id=request_id,
        )

    def _get_mcp_error_code(self) -> MCPErrorCode:
        """Map internal error codes to MCP error codes."""
        mapping = {
            ErrorCode.FILE_NOT_FOUND: MCPErrorCode.RESOURCE_NOT_FOUND,
            ErrorCode.DIRECTORY_NOT_FOUND: MCPErrorCode.RESOURCE_NOT_FOUND,
            ErrorCode.PERMISSION_DENIED: MCPErrorCode.RESOURCE_ACCESS_DENIED,
            ErrorCode.INVALID_ARGS: MCPErrorCode.INVALID_PARAMS,
            ErrorCode.MISSING_ARGS: MCPErrorCode.INVALID_PARAMS,
            ErrorCode.VALIDATION_FAILED: MCPErrorCode.INVALID_PARAMS,
            ErrorCode.LANGUAGE_UNSUPPORTED: MCPErrorCode.TOOL_EXECUTION_FAILED,
            ErrorCode.CONCEPT_EXTRACTION_FAILED: MCPErrorCode.TOOL_EXECUTION_FAILED,
            ErrorCode.PATTERN_ANALYSIS_FAILED: MCPErrorCode.TOOL_EXECUTION_FAILED,
            ErrorCode.SEMANTIC_ANALYSIS_FAILED: MCPErrorCode.TOOL_EXECUTION_FAILED,
            ErrorCode.PARSE_FAILED: MCPErrorCode.PROTOCOL_VIOLATION,
            ErrorCode.TREE_SITTER_FAILED: MCPErrorCode.PROTOCOL_VIOLATION,
            ErrorCode.DATABASE_INIT_FAILED: MCPErrorCode.INITIALIZATION_FAILED,
            ErrorCode.DATABASE_MIGRATION_FAILED: MCPErrorCode.INITIALIZATION_FAILED,
            ErrorCode.VECTOR_DB_INIT_FAILED: MCPErrorCode.INITIALIZATION_FAILED,
            ErrorCode.SETUP_INCOMPLETE: MCPErrorCode.INITIALIZATION_FAILED,
            ErrorCode.PLATFORM_UNSUPPORTED: MCPErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.NATIVE_BINDING_FAILED: MCPErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.CIRCUIT_BREAKER_OPEN: MCPErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.NETWORK_TIMEOUT: MCPErrorCode.TRANSPORT_ERROR,
            ErrorCode.EXTERNAL_SERVICE_FAILED: MCPErrorCode.TRANSPORT_ERROR,
            ErrorCode.RATE_LIMIT_EXCEEDED: MCPErrorCode.RATE_LIMITED,
        }
        return mapping.get(self.code, MCPErrorCode.INTERNAL_ERROR)


# Specialized error classes for domain-specific error handling
class ConfigurationError(AnamnesisError):
    """Error related to configuration issues."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.INVALID_CONFIG,
            message=message,
            user_message=user_message or "Configuration error occurred.",
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class ValidationError(AnamnesisError):
    """Error related to input validation failures."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.VALIDATION_FAILED,
            message=message,
            user_message=user_message or "Validation failed.",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class DatabaseError(AnamnesisError):
    """Error related to database operations."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.DATABASE_INIT_FAILED,
            message=message,
            user_message=user_message or "Database operation failed.",
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class ExecutionError(AnamnesisError):
    """Error related to code execution/analysis."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.SEMANTIC_ANALYSIS_FAILED,
            message=message,
            user_message=user_message or "Execution failed.",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class LearningError(AnamnesisError):
    """Error related to codebase learning operations."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.CONCEPT_EXTRACTION_FAILED,
            message=message,
            user_message=user_message or "Learning operation failed.",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class ResourceError(AnamnesisError):
    """Error related to resource access (files, network, etc.)."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.FILE_NOT_FOUND,
            message=message,
            user_message=user_message or "Resource access failed.",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )


class AnamnesisSystemError(AnamnesisError):
    """Error related to system/infrastructure issues."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        context: ErrorContext | None = None,
        recovery_actions: list[RecoveryAction] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            code=ErrorCode.PLATFORM_UNSUPPORTED,
            message=message,
            user_message=user_message or "System error occurred.",
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_actions=recovery_actions,
            original_error=original_error,
        )
