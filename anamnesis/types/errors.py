"""
MCP-compliant error handling system for Anamnesis.

Provides structured error types following JSON-RPC 2.0 specification,
compatible with Model Context Protocol requirements.

Ported from TypeScript error-types.ts
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


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


class ErrorSeverity(str):
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
    timestamp: datetime = field(default_factory=datetime.now)
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
        self.context.timestamp = datetime.now()
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


class SystemError(AnamnesisError):
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


class ErrorFactory:
    """Factory functions for creating common errors with appropriate context."""

    @staticmethod
    def platform_unsupported(platform: str, arch: str) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.PLATFORM_UNSUPPORTED,
            message=f"Unsupported platform: {platform}-{arch}",
            user_message="Your platform is not currently supported.",
            severity=ErrorSeverity.CRITICAL,
            context=ErrorContext(
                component="platform",
                additional_info={"platform": platform, "arch": arch},
            ),
            recovery_actions=[
                RecoveryAction(description="Check supported platforms in documentation"),
            ],
        )

    @staticmethod
    def file_not_found(file_path: str, operation: str) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"File not found: {file_path}",
            user_message="The requested file could not be found.",
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(file_path=file_path, operation=operation),
            recovery_actions=[
                RecoveryAction(description="Check if the file path is correct"),
                RecoveryAction(description="Ensure the file exists and is accessible"),
            ],
        )

    @staticmethod
    def language_unsupported(language: str, file_path: str | None = None) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.LANGUAGE_UNSUPPORTED,
            message=f"Unsupported language: {language}",
            user_message="This programming language is not currently supported.",
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(language=language, file_path=file_path),
            recovery_actions=[
                RecoveryAction(description="Check supported languages in documentation"),
                RecoveryAction(description="Use fallback analysis for basic extraction"),
            ],
        )

    @staticmethod
    def parse_failed(
        file_path: str, language: str, original_error: Exception | None = None
    ) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.PARSE_FAILED,
            message=f"Failed to parse {language} file: {file_path}",
            user_message="Could not parse the source code file.",
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(file_path=file_path, language=language, operation="parsing"),
            recovery_actions=[
                RecoveryAction(description="Check if the file contains valid syntax"),
                RecoveryAction(description="Ensure the file is properly encoded (UTF-8)"),
            ],
            original_error=original_error,
        )

    @staticmethod
    def database_init_failed(db_path: str, original_error: Exception) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.DATABASE_INIT_FAILED,
            message=f"Database initialization failed: {original_error}",
            user_message="Could not initialize the database.",
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                operation="database-init",
                additional_info={"db_path": db_path},
            ),
            recovery_actions=[
                RecoveryAction(description="Check database file permissions"),
                RecoveryAction(description="Ensure SQLite is available"),
                RecoveryAction(description="Try removing corrupted database file"),
            ],
            original_error=original_error,
        )

    @staticmethod
    def invalid_config(config_path: str, validation_errors: list[str]) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.INVALID_CONFIG,
            message=f"Invalid configuration: {', '.join(validation_errors)}",
            user_message="The configuration file contains invalid settings.",
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                file_path=config_path,
                operation="config-validation",
                additional_info={"validation_errors": validation_errors},
            ),
            recovery_actions=[
                RecoveryAction(description="Fix configuration file manually"),
                RecoveryAction(description="Reset to default configuration"),
            ],
        )

    @staticmethod
    def circuit_breaker_open(component: str, last_failure: datetime | None = None) -> AnamnesisError:
        return AnamnesisError(
            code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            message=f"Circuit breaker is open for {component}",
            user_message="Service temporarily unavailable due to repeated failures.",
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                component=component,
                operation="circuit-breaker",
                additional_info={"last_failure": last_failure.isoformat() if last_failure else None},
            ),
            recovery_actions=[
                RecoveryAction(description="Wait for circuit breaker to reset"),
                RecoveryAction(description="Check underlying service health"),
                RecoveryAction(description="Use fallback functionality if available"),
            ],
        )


class MCPErrorUtils:
    """MCP-specific utility functions for error handling."""

    @staticmethod
    def create_mcp_error(
        code: MCPErrorCode,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> MCPErrorResponse:
        """Create a standard MCP error response."""
        return MCPErrorResponse(
            code=code,
            message=message,
            data={
                **(data or {}),
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    def create_jsonrpc_error(
        code: MCPErrorCode,
        message: str,
        request_id: str | int | None,
        data: dict[str, Any] | None = None,
    ) -> JSONRPCError:
        """Create a JSON-RPC error response."""
        return JSONRPCError(
            error=MCPErrorUtils.create_mcp_error(code, message, data),
            id=request_id,
        )

    @staticmethod
    def wrap_tool_error(
        tool_name: str,
        error: Exception | str,
        request_id: str | int | None = None,
    ) -> JSONRPCError:
        """Wrap tool execution errors for MCP compliance."""
        error_message = str(error) if isinstance(error, Exception) else error
        return MCPErrorUtils.create_jsonrpc_error(
            code=MCPErrorCode.TOOL_EXECUTION_FAILED,
            message=f"Tool '{tool_name}' execution failed: {error_message}",
            request_id=request_id,
            data={"tool_name": tool_name, "original_error": error_message, "type": "ToolExecutionError"},
        )

    @staticmethod
    def resource_not_found(
        resource_type: str,
        resource_id: str,
        request_id: str | int | None = None,
    ) -> JSONRPCError:
        """Handle resource access errors."""
        return MCPErrorUtils.create_jsonrpc_error(
            code=MCPErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} not found: {resource_id}",
            request_id=request_id,
            data={"resource_type": resource_type, "resource_id": resource_id, "type": "ResourceNotFound"},
        )

    @staticmethod
    def invalid_params(
        param_name: str,
        expected: str,
        received: Any,
        request_id: str | int | None = None,
    ) -> JSONRPCError:
        """Handle invalid parameter errors."""
        return MCPErrorUtils.create_jsonrpc_error(
            code=MCPErrorCode.INVALID_PARAMS,
            message=f"Invalid parameter '{param_name}': expected {expected}, received {type(received).__name__}",
            request_id=request_id,
            data={"param_name": param_name, "expected": expected, "received": type(received).__name__, "type": "InvalidParams"},
        )


class ErrorUtils:
    """Utility functions for error handling."""

    @staticmethod
    def get_error_message(error: Exception | str | Any) -> str:
        """Safely extract error message from unknown error."""
        if isinstance(error, Exception):
            return str(error)
        if isinstance(error, str):
            return error
        return str(error)

    @staticmethod
    def is_recoverable(error: AnamnesisError) -> bool:
        """Check if error is recoverable."""
        return len(error.recovery_actions) > 0 and error.severity != ErrorSeverity.CRITICAL

    @staticmethod
    def from_error(error: Exception, context: ErrorContext | None = None) -> AnamnesisError:
        """Convert standard Error to AnamnesisError."""
        message = str(error).lower()
        ctx = context or ErrorContext()

        if "unsupported language" in message:
            return ErrorFactory.language_unsupported("unknown", ctx.file_path)

        if "file not found" in message or "enoent" in message:
            return ErrorFactory.file_not_found(ctx.file_path or "unknown", ctx.operation or "unknown")

        if "failed to parse" in message:
            return ErrorFactory.parse_failed(
                ctx.file_path or "unknown",
                ctx.language or "unknown",
                error,
            )

        # Generic error conversion
        return AnamnesisError(
            code=ErrorCode.SEMANTIC_ANALYSIS_FAILED,
            message=str(error),
            user_message="An unexpected error occurred during analysis.",
            severity=ErrorSeverity.MEDIUM,
            context=ctx,
            recovery_actions=[
                RecoveryAction(description="Check the error details above"),
            ],
            original_error=error,
        )
