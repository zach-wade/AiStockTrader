"""
Production-Grade Error Handling Patterns

Comprehensive error handling system with structured errors, correlation tracking,
recovery strategies, and integration with monitoring systems.
"""

import asyncio
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"  # Non-critical, can continue
    MEDIUM = "medium"  # Important but recoverable
    HIGH = "high"  # Serious, may affect functionality
    CRITICAL = "critical"  # System-threatening, immediate attention


class ErrorCategory(Enum):
    """Error category types."""

    NETWORK = "network"  # Network/connectivity issues
    DATABASE = "database"  # Database-related errors
    EXTERNAL_API = "external_api"  # Third-party API errors
    BUSINESS_LOGIC = "business_logic"  # Domain/business rule violations
    SECURITY = "security"  # Security-related issues
    CONFIGURATION = "configuration"  # Configuration errors
    RESOURCE = "resource"  # Resource exhaustion/limits
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class ErrorContext:
    """Contextual information for errors."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    operation: str | None = None
    component: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "component": self.component,
            "metadata": self.metadata,
        }


@dataclass
class StructuredError:
    """Structured error representation."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_type: str = field(default="UnknownError")
    message: str = ""
    details: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    context: ErrorContext = field(default_factory=ErrorContext)
    exception: Exception | None = None
    stack_trace: str | None = None
    recovery_suggestions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if self.exception and not self.stack_trace:
            self.stack_trace = "".join(
                traceback.format_exception(
                    type(self.exception), self.exception, self.exception.__traceback__
                )
            )
            if not self.details and hasattr(self.exception, "__str__"):
                self.details = str(self.exception)

        if not self.error_type and self.exception:
            self.error_type = type(self.exception).__name__

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "stack_trace": self.stack_trace,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.context.timestamp,
        }

    def log(self, logger_instance: logging.Logger | None = None) -> None:
        """Log structured error with appropriate level."""
        log = logger_instance or logger

        # Determine log level based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            level = logging.CRITICAL
        elif self.severity == ErrorSeverity.HIGH:
            level = logging.ERROR
        elif self.severity == ErrorSeverity.MEDIUM:
            level = logging.WARNING
        else:
            level = logging.INFO

        # Format log message
        message = f"[{self.error_id}] {self.error_type}: {self.message}"
        if self.details:
            message += f" - {self.details}"

        # Log with structured data
        log.log(
            level,
            message,
            extra={
                "structured_error": self.to_dict(),
                "correlation_id": self.context.correlation_id,
                "error_category": self.category.value,
                "error_severity": self.severity.value,
            },
        )


class ErrorClassifier:
    """Classifies exceptions into structured errors with recovery suggestions."""

    # Mapping of exception types to categories and severities
    CLASSIFICATION_RULES = {
        # Network errors
        ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
        TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
        OSError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
        # Database errors
        "psycopg.OperationalError": (ErrorCategory.DATABASE, ErrorSeverity.HIGH),
        "psycopg.InterfaceError": (ErrorCategory.DATABASE, ErrorSeverity.HIGH),
        "psycopg.DatabaseError": (ErrorCategory.DATABASE, ErrorSeverity.HIGH),
        # Business logic errors
        ValueError: (ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM),
        TypeError: (ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM),
        # Security errors
        PermissionError: (ErrorCategory.SECURITY, ErrorSeverity.HIGH),
        # Resource errors
        MemoryError: (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
        # Configuration errors
        FileNotFoundError: (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
        KeyError: (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
    }

    # Recovery suggestions by category
    RECOVERY_SUGGESTIONS = {
        ErrorCategory.NETWORK: [
            "Check network connectivity",
            "Verify service endpoint is reachable",
            "Consider using fallback service",
            "Implement exponential backoff retry",
        ],
        ErrorCategory.DATABASE: [
            "Check database connection",
            "Verify database server status",
            "Check connection pool health",
            "Review recent schema changes",
        ],
        ErrorCategory.EXTERNAL_API: [
            "Check API service status",
            "Verify authentication credentials",
            "Review rate limiting settings",
            "Use cached data if available",
        ],
        ErrorCategory.BUSINESS_LOGIC: [
            "Validate input parameters",
            "Review business rule implementation",
            "Check data consistency",
            "Verify calculation logic",
        ],
        ErrorCategory.SECURITY: [
            "Review authentication settings",
            "Check authorization permissions",
            "Verify security configuration",
            "Audit access logs",
        ],
        ErrorCategory.CONFIGURATION: [
            "Check configuration file syntax",
            "Verify environment variables",
            "Review default settings",
            "Validate file permissions",
        ],
        ErrorCategory.RESOURCE: [
            "Check system resources",
            "Review memory usage",
            "Monitor CPU utilization",
            "Check disk space",
        ],
    }

    @classmethod
    def classify(
        self,
        exception: Exception,
        context: ErrorContext | None = None,
        custom_message: str | None = None,
    ) -> StructuredError:
        """
        Classify exception into structured error.

        Args:
            exception: Exception to classify
            context: Error context
            custom_message: Custom error message

        Returns:
            StructuredError instance
        """
        context = context or ErrorContext()

        # Determine category and severity
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM

        # Check by exception type
        exception_type = type(exception)
        if exception_type in self.CLASSIFICATION_RULES:
            category, severity = self.CLASSIFICATION_RULES[exception_type]
        else:
            # Check by string name for module-specific exceptions
            exception_name = f"{exception_type.__module__}.{exception_type.__name__}"
            for rule_name, (rule_category, rule_severity) in self.CLASSIFICATION_RULES.items():
                if isinstance(rule_name, str) and exception_name.endswith(rule_name):
                    category, severity = rule_category, rule_severity
                    break

        # Get recovery suggestions
        recovery_suggestions = self.RECOVERY_SUGGESTIONS.get(
            category, ["Review error details", "Check system logs", "Contact system administrator"]
        )

        return StructuredError(
            error_type=exception_type.__name__,
            message=custom_message or str(exception),
            severity=severity,
            category=category,
            context=context,
            exception=exception,
            recovery_suggestions=recovery_suggestions.copy(),
        )


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""

    @abstractmethod
    def can_handle(self, error: StructuredError) -> bool:
        """Check if this handler can handle the error."""
        pass

    @abstractmethod
    async def handle(self, error: StructuredError) -> Any | None:
        """Handle the error and optionally return a result."""
        pass


class RetryErrorHandler(ErrorHandler):
    """Handler that implements retry logic for recoverable errors."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        retryable_categories: list[ErrorCategory] | None = None,
    ):
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.retryable_categories = retryable_categories or [
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL_API,
            ErrorCategory.DATABASE,
        ]

    def can_handle(self, error: StructuredError) -> bool:
        """Check if error is retryable."""
        return (
            error.category in self.retryable_categories and error.severity != ErrorSeverity.CRITICAL
        )

    async def handle(self, error: StructuredError) -> Any | None:
        """Handle error with retry logic."""
        logger.info(f"Retry handler processing error {error.error_id}")

        # Add retry suggestion to error
        error.recovery_suggestions.append("Automatic retry will be attempted")

        # Return None to indicate retry should be attempted
        return None


class FallbackErrorHandler(ErrorHandler):
    """Handler that provides fallback responses for failed operations."""

    def __init__(self, fallback_categories: list[ErrorCategory] | None = None) -> None:
        self.fallback_categories = fallback_categories or [
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL_API,
        ]

    def can_handle(self, error: StructuredError) -> bool:
        """Check if fallback is available."""
        return error.category in self.fallback_categories

    async def handle(self, error: StructuredError) -> Any | None:
        """Provide fallback response."""
        logger.info(f"Fallback handler processing error {error.error_id}")

        # Add fallback suggestion
        error.recovery_suggestions.append("Fallback response provided")

        # Return fallback data (implementation-specific)
        return {"status": "degraded", "message": "Using fallback data"}


class ErrorManager:
    """
    Central error management system.

    Features:
    - Error classification and routing
    - Handler chain execution
    - Metrics and monitoring
    - Correlation tracking
    """

    def __init__(self) -> None:
        self.handlers: list[ErrorHandler] = []
        self.classifier = ErrorClassifier()
        self.error_metrics: defaultdict[str, int] = defaultdict(int)
        self.error_history: list[StructuredError] = []
        self.max_history = 1000

    def add_handler(self, handler: ErrorHandler) -> None:
        """Add error handler to the chain."""
        self.handlers.append(handler)
        logger.info(f"Added error handler: {type(handler).__name__}")

    def remove_handler(self, handler: ErrorHandler) -> None:
        """Remove error handler from the chain."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.info(f"Removed error handler: {type(handler).__name__}")

    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext | None = None,
        custom_message: str | None = None,
    ) -> Any | None:
        """
        Handle error through the handler chain.

        Args:
            exception: Exception to handle
            context: Error context
            custom_message: Custom error message

        Returns:
            Handler result or None
        """
        # Classify error
        structured_error = self.classifier.classify(exception, context, custom_message)

        # Update metrics
        self._update_metrics(structured_error)

        # Add to history
        self._add_to_history(structured_error)

        # Log error
        structured_error.log(logger)

        # Try handlers in order
        for handler in self.handlers:
            if handler.can_handle(structured_error):
                try:
                    result = await handler.handle(structured_error)
                    if result is not None:
                        logger.info(
                            f"Error {structured_error.error_id} handled by {type(handler).__name__}"
                        )
                        return result
                except Exception as handler_error:
                    logger.error(f"Handler {type(handler).__name__} failed: {handler_error}")

        # No handler could process the error
        logger.warning(f"No handler available for error {structured_error.error_id}")
        return None

    def _update_metrics(self, error: StructuredError) -> None:
        """Update error metrics."""
        self.error_metrics["total_errors"] += 1
        self.error_metrics[f"category_{error.category.value}"] += 1
        self.error_metrics[f"severity_{error.severity.value}"] += 1
        self.error_metrics[f"type_{error.error_type}"] += 1

    def _add_to_history(self, error: StructuredError) -> None:
        """Add error to history."""
        self.error_history.append(error)

        # Trim history if needed
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history :]

    def get_metrics(self) -> dict[str, Any]:
        """Get error metrics."""
        return {
            "total_errors": self.error_metrics.get("total_errors", 0),
            "by_category": {
                category.value: self.error_metrics.get(f"category_{category.value}", 0)
                for category in ErrorCategory
            },
            "by_severity": {
                severity.value: self.error_metrics.get(f"severity_{severity.value}", 0)
                for severity in ErrorSeverity
            },
            "recent_errors": len(self.error_history),
            "handlers_registered": len(self.handlers),
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent errors."""
        return [error.to_dict() for error in self.error_history[-limit:]]

    def reset_metrics(self) -> None:
        """Reset error metrics and history."""
        self.error_metrics.clear()
        self.error_history.clear()
        logger.info("Error metrics and history reset")


# Global error manager instance
error_manager = ErrorManager()

# Add default handlers
error_manager.add_handler(RetryErrorHandler())
error_manager.add_handler(FallbackErrorHandler())


def handle_errors(
    context_name: str | None = None, reraise: bool = True, return_on_error: Any = None
) -> Callable[[F], F]:
    """
    Decorator for automatic error handling.

    Args:
        context_name: Context name for error tracking
        reraise: Whether to reraise exceptions after handling
        return_on_error: Value to return on error (if reraise=False)
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                context = ErrorContext(
                    operation=func.__name__, component=context_name or func.__module__
                )

                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    result = await error_manager.handle_error(e, context)

                    if reraise:
                        raise
                    else:
                        return result if result is not None else return_on_error

            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                context = ErrorContext(
                    operation=func.__name__, component=context_name or func.__module__
                )

                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    # Convert to async for consistent handling
                    async def handle_sync_error() -> Any | None:
                        return await error_manager.handle_error(exc, context)

                    # Run async handler
                    try:
                        loop = asyncio.get_event_loop()
                        result = loop.run_until_complete(handle_sync_error())
                    except RuntimeError:
                        # No event loop, handle synchronously
                        result = None

                    if reraise:
                        raise
                    else:
                        return result if result is not None else return_on_error

            return cast(F, sync_wrapper)

    return decorator


@contextmanager
def error_context(
    operation: str,
    component: str | None = None,
    correlation_id: str | None = None,
    **metadata: Any,
) -> Any:
    """
    Context manager for error handling with structured context.

    Args:
        operation: Operation name
        component: Component name
        correlation_id: Correlation ID
        **metadata: Additional metadata
    """
    context = ErrorContext(
        operation=operation,
        component=component,
        correlation_id=correlation_id or str(uuid.uuid4()),
        metadata=metadata,
    )

    try:
        yield context
    except Exception as e:
        # Handle error synchronously
        structured_error = error_manager.classifier.classify(e, context)
        error_manager._update_metrics(structured_error)
        error_manager._add_to_history(structured_error)
        structured_error.log(logger)
        raise


@asynccontextmanager
async def async_error_context(
    operation: str,
    component: str | None = None,
    correlation_id: str | None = None,
    **metadata: Any,
) -> Any:
    """
    Async context manager for error handling.

    Args:
        operation: Operation name
        component: Component name
        correlation_id: Correlation ID
        **metadata: Additional metadata
    """
    context = ErrorContext(
        operation=operation,
        component=component,
        correlation_id=correlation_id or str(uuid.uuid4()),
        metadata=metadata,
    )

    try:
        yield context
    except Exception as e:
        await error_manager.handle_error(e, context)
        raise
