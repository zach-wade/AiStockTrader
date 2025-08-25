"""
Domain-level exceptions for the trading system.

This module defines exceptions that are specific to domain logic and business rules.
These exceptions are raised within domain entities and services.
"""

from typing import Any
from uuid import UUID


class DomainException(Exception):
    """Base exception for all domain-level errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class StaleDataException(DomainException):
    """
    Raised when attempting to update an entity that has been modified by another process.

    This is the domain's optimistic locking exception indicating version conflict.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: UUID | str,
        expected_version: int,
        actual_version: int | None = None,
    ) -> None:
        message = (
            f"{entity_type} {entity_id} has been modified by another process. "
            f"Expected version {expected_version}"
        )
        if actual_version is not None:
            message += f", but found version {actual_version}"

        super().__init__(
            message,
            details={
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "expected_version": expected_version,
                "actual_version": actual_version,
            },
        )
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class ConcurrencyException(DomainException):
    """
    General concurrency-related exception for domain operations.

    Used for race conditions, deadlocks, and other concurrency issues.
    """

    def __init__(
        self,
        message: str,
        entity_type: str | None = None,
        entity_id: UUID | str | None = None,
        operation: str | None = None,
    ) -> None:
        details = {}
        if entity_type:
            details["entity_type"] = entity_type
        if entity_id:
            details["entity_id"] = str(entity_id)
        if operation:
            details["operation"] = operation

        super().__init__(message, details)
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.operation = operation


class OptimisticLockException(ConcurrencyException):
    """
    Specific exception for optimistic locking failures.

    Raised when optimistic locking fails after maximum retries.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: UUID | str,
        retries: int,
        message: str | None = None,
    ) -> None:
        if message is None:
            message = (
                f"Failed to update {entity_type} {entity_id} after {retries} retries "
                "due to concurrent modifications"
            )

        super().__init__(
            message=message,
            entity_type=entity_type,
            entity_id=entity_id,
            operation="update",
        )
        self.retries = retries


class PessimisticLockException(ConcurrencyException):
    """
    Exception for pessimistic locking failures.

    Raised when unable to acquire a lock within timeout.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: UUID | str,
        timeout: float,
        message: str | None = None,
    ) -> None:
        if message is None:
            message = (
                f"Failed to acquire lock for {entity_type} {entity_id} " f"within {timeout} seconds"
            )

        super().__init__(
            message=message,
            entity_type=entity_type,
            entity_id=entity_id,
            operation="lock",
        )
        self.timeout = timeout


class DeadlockException(ConcurrencyException):
    """
    Exception indicating a deadlock was detected.

    Raised when database detects a deadlock situation.
    """

    def __init__(
        self,
        message: str = "Deadlock detected during operation",
        entities: list[tuple[str, UUID | str]] | None = None,
    ) -> None:
        details = {}
        if entities:
            details["entities"] = [
                {"type": entity_type, "id": str(entity_id)} for entity_type, entity_id in entities
            ]

        super().__init__(message=message)
        self.entities = entities or []


class EntityValidationException(DomainException):
    """Exception raised when entity validation fails."""

    def __init__(
        self,
        entity_type: str,
        entity_id: UUID | str | None,
        field: str,
        value: Any,
        constraint: str,
    ) -> None:
        message = f"{entity_type} validation failed for field '{field}': {constraint}"
        if entity_id:
            message = (
                f"{entity_type} {entity_id} validation failed for field '{field}': {constraint}"
            )

        super().__init__(
            message,
            details={
                "entity_type": entity_type,
                "entity_id": str(entity_id) if entity_id else None,
                "field": field,
                "value": value,
                "constraint": constraint,
            },
        )
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.field = field
        self.value = value
        self.constraint = constraint


class InsufficientResourcesException(DomainException):
    """Exception raised when there are insufficient resources for an operation."""

    def __init__(
        self,
        resource_type: str,
        required: Any,
        available: Any,
        message: str | None = None,
    ) -> None:
        if message is None:
            message = (
                f"Insufficient {resource_type}: required {required}, "
                f"but only {available} available"
            )

        super().__init__(
            message,
            details={
                "resource_type": resource_type,
                "required": required,
                "available": available,
            },
        )
        self.resource_type = resource_type
        self.required = required
        self.available = available


class ValidationError(DomainException):
    """Exception raised when domain validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if constraint:
            details["constraint"] = constraint

        super().__init__(message, details)
        self.field = field
        self.value = value
        self.constraint = constraint
