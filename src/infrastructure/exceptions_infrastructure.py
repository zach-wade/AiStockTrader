"""
Infrastructure-specific exception hierarchy for the AI Trading System.

This module provides exceptions for infrastructure-level errors including
database, caching, messaging, and external service integration issues.
"""

from typing import Any
from uuid import UUID


class InfrastructureException(Exception):
    """Base exception for all infrastructure-level errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ============================================================================
# Database Exceptions
# ============================================================================


class DatabaseException(InfrastructureException):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionException(DatabaseException):
    """Raised when database connection fails."""

    def __init__(
        self,
        database: str,
        host: str | None = None,
        port: int | None = None,
        reason: str | None = None,
    ) -> None:
        message = f"Failed to connect to database {database}"
        if host:
            message += f" at {host}:{port}"
        if reason:
            message += f": {reason}"

        details = {"database": database, "host": host, "port": port, "reason": reason}
        super().__init__(message, details)
        self.database = database
        self.host = host
        self.port = port
        self.reason = reason


class DatabaseQueryException(DatabaseException):
    """Raised when database query fails."""

    def __init__(self, query: str, error: str, params: dict[str, Any] | None = None) -> None:
        message = f"Database query failed: {error}"
        details = {"query": query[:500], "error": error, "params": params}  # Truncate long queries
        super().__init__(message, details)
        self.query = query
        self.error = error
        self.params = params


class DatabaseTransactionException(DatabaseException):
    """Raised when database transaction fails."""

    def __init__(
        self,
        transaction_id: str | None = None,
        operation: str | None = None,
        error: str | None = None,
    ) -> None:
        message = "Database transaction failed"
        if operation:
            message += f" during {operation}"
        if error:
            message += f": {error}"

        details = {"transaction_id": transaction_id, "operation": operation, "error": error}
        super().__init__(message, details)
        self.transaction_id = transaction_id
        self.operation = operation
        self.error = error


class DatabaseDeadlockException(DatabaseTransactionException):
    """Raised when database deadlock is detected."""

    def __init__(
        self, transaction_id: str | None = None, resources: list[str] | None = None
    ) -> None:
        message = "Database deadlock detected"
        if resources:
            message += f" on resources: {', '.join(resources)}"

        details = {"transaction_id": transaction_id, "resources": resources}
        super().__init__(transaction_id=transaction_id, operation="deadlock", error=message)
        self.resources = resources


class EntityNotFoundException(DatabaseException):
    """Raised when an entity is not found in the database."""

    def __init__(self, entity_type: str, entity_id: UUID | str, **kwargs: Any) -> None:
        message = f"{entity_type} with ID {entity_id} not found"
        details = {"entity_type": entity_type, "entity_id": str(entity_id), **kwargs}
        super().__init__(message, details)
        self.entity_type = entity_type
        self.entity_id = entity_id


# ============================================================================
# Cache Exceptions
# ============================================================================


class CacheException(InfrastructureException):
    """Base exception for cache-related errors."""

    pass


class CacheConnectionException(CacheException):
    """Raised when cache connection fails."""

    def __init__(self, cache_type: str, host: str | None = None, reason: str | None = None) -> None:
        message = f"Failed to connect to {cache_type} cache"
        if host:
            message += f" at {host}"
        if reason:
            message += f": {reason}"

        details = {"cache_type": cache_type, "host": host, "reason": reason}
        super().__init__(message, details)
        self.cache_type = cache_type
        self.host = host
        self.reason = reason


class CacheKeyNotFoundException(CacheException):
    """Raised when a cache key is not found."""

    def __init__(self, key: str, cache_name: str | None = None) -> None:
        message = f"Cache key '{key}' not found"
        if cache_name:
            message += f" in {cache_name}"

        details = {"key": key, "cache_name": cache_name}
        super().__init__(message, details)
        self.key = key
        self.cache_name = cache_name


class CacheInvalidationException(CacheException):
    """Raised when cache invalidation fails."""

    def __init__(self, pattern: str | None = None, reason: str | None = None) -> None:
        message = "Cache invalidation failed"
        if pattern:
            message += f" for pattern '{pattern}'"
        if reason:
            message += f": {reason}"

        details = {"pattern": pattern, "reason": reason}
        super().__init__(message, details)
        self.pattern = pattern
        self.reason = reason


# ============================================================================
# Message Queue Exceptions
# ============================================================================


class MessageQueueException(InfrastructureException):
    """Base exception for message queue errors."""

    pass


class MessagePublishException(MessageQueueException):
    """Raised when message publishing fails."""

    def __init__(
        self, topic: str, message_type: str | None = None, reason: str | None = None
    ) -> None:
        message = f"Failed to publish message to topic '{topic}'"
        if message_type:
            message += f" (type: {message_type})"
        if reason:
            message += f": {reason}"

        details = {"topic": topic, "message_type": message_type, "reason": reason}
        super().__init__(message, details)
        self.topic = topic
        self.message_type = message_type
        self.reason = reason


class MessageConsumptionException(MessageQueueException):
    """Raised when message consumption fails."""

    def __init__(
        self, topic: str, consumer_group: str | None = None, reason: str | None = None
    ) -> None:
        message = f"Failed to consume message from topic '{topic}'"
        if consumer_group:
            message += f" (group: {consumer_group})"
        if reason:
            message += f": {reason}"

        details = {"topic": topic, "consumer_group": consumer_group, "reason": reason}
        super().__init__(message, details)
        self.topic = topic
        self.consumer_group = consumer_group
        self.reason = reason


# ============================================================================
# External Service Exceptions
# ============================================================================


class ExternalServiceException(InfrastructureException):
    """Base exception for external service errors."""

    def __init__(self, service_name: str, message: str, **kwargs: Any) -> None:
        details = {"service_name": service_name, **kwargs}
        super().__init__(message, details)
        self.service_name = service_name


class ServiceTimeoutException(ExternalServiceException):
    """Raised when external service call times out."""

    def __init__(
        self, service_name: str, timeout_seconds: int, operation: str | None = None
    ) -> None:
        message = f"Service {service_name} timed out after {timeout_seconds}s"
        if operation:
            message += f" during {operation}"

        super().__init__(
            service_name=service_name,
            message=message,
            timeout_seconds=timeout_seconds,
            operation=operation,
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ServiceUnavailableException(ExternalServiceException):
    """Raised when external service is unavailable."""

    def __init__(
        self, service_name: str, reason: str | None = None, retry_after: int | None = None
    ) -> None:
        message = f"Service {service_name} is unavailable"
        if reason:
            message += f": {reason}"

        super().__init__(
            service_name=service_name, message=message, reason=reason, retry_after=retry_after
        )
        self.reason = reason
        self.retry_after = retry_after


# ============================================================================
# Repository Exceptions
# ============================================================================


class RepositoryException(InfrastructureException):
    """Base exception for repository-layer errors."""

    pass


class OptimisticLockingException(RepositoryException):
    """Raised when optimistic locking fails."""

    def __init__(
        self, entity_type: str, entity_id: UUID | str, expected_version: int, actual_version: int
    ) -> None:
        message = (
            f"Optimistic lock failed for {entity_type} {entity_id}: "
            f"expected version {expected_version}, actual {actual_version}"
        )
        details = {
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "expected_version": expected_version,
            "actual_version": actual_version,
        }
        super().__init__(message, details)
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class RepositoryOperationException(RepositoryException):
    """Raised when repository operation fails."""

    def __init__(self, operation: str, entity_type: str, reason: str, **kwargs: Any) -> None:
        message = f"Repository operation '{operation}' failed for {entity_type}: {reason}"
        details = {"operation": operation, "entity_type": entity_type, "reason": reason, **kwargs}
        super().__init__(message, details)
        self.operation = operation
        self.entity_type = entity_type
        self.reason = reason


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationException(InfrastructureException):
    """Base exception for configuration errors."""

    pass


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, config_section: str | None = None) -> None:
        message = f"Missing required configuration: {config_key}"
        if config_section:
            message = f"Missing required configuration in {config_section}: {config_key}"

        details = {"config_key": config_key, "config_section": config_section}
        super().__init__(message, details)
        self.config_key = config_key
        self.config_section = config_section


class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid."""

    def __init__(
        self, config_key: str, value: Any, reason: str, config_section: str | None = None
    ) -> None:
        message = f"Invalid configuration {config_key}={value}: {reason}"
        if config_section:
            message = f"Invalid configuration in {config_section}: {config_key}={value} - {reason}"

        details = {
            "config_key": config_key,
            "value": str(value),
            "reason": reason,
            "config_section": config_section,
        }
        super().__init__(message, details)
        self.config_key = config_key
        self.value = value
        self.reason = reason
        self.config_section = config_section


# ============================================================================
# Security/Auth Exceptions
# ============================================================================


class SecurityException(InfrastructureException):
    """Base exception for security-related errors."""

    pass


class AuthenticationException(SecurityException):
    """Raised when authentication fails."""

    def __init__(self, user_id: str | None = None, reason: str | None = None) -> None:
        message = "Authentication failed"
        if reason:
            message += f": {reason}"

        details = {"user_id": user_id, "reason": reason}
        super().__init__(message, details)
        self.user_id = user_id
        self.reason = reason


class AuthorizationException(SecurityException):
    """Raised when authorization fails."""

    def __init__(
        self, user_id: str, resource: str, action: str, required_permission: str | None = None
    ) -> None:
        message = f"User {user_id} not authorized to {action} {resource}"
        if required_permission:
            message += f" (requires: {required_permission})"

        details = {
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "required_permission": required_permission,
        }
        super().__init__(message, details)
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.required_permission = required_permission


class TokenExpiredException(SecurityException):
    """Raised when a token has expired."""

    def __init__(self, token_type: str = "access", expired_at: str | None = None) -> None:
        message = f"{token_type.capitalize()} token has expired"
        if expired_at:
            message += f" at {expired_at}"

        details = {"token_type": token_type, "expired_at": expired_at}
        super().__init__(message, details)
        self.token_type = token_type
        self.expired_at = expired_at


class RateLimitExceededException(SecurityException):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int | None = None,
        identifier: str | None = None,
    ) -> None:
        message = f"Rate limit exceeded: {limit} requests per {window_seconds}s"
        if retry_after:
            message += f", retry after {retry_after}s"

        details = {
            "limit": limit,
            "window_seconds": window_seconds,
            "retry_after": retry_after,
            "identifier": identifier,
        }
        super().__init__(message, details)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        self.identifier = identifier
