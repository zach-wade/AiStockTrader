"""
Repository Exception Definitions

Defines exceptions that repositories may raise.
Following clean architecture principles - these are application-level exceptions.
"""

# Standard library imports
from typing import Any
from uuid import UUID


class RepositoryError(Exception):
    """Base exception for repository operations."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found in the repository."""

    def __init__(self, entity_type: str, identifier: UUID | str) -> None:
        super().__init__(f"{entity_type} with identifier '{identifier}' not found")
        self.entity_type = entity_type
        self.identifier = identifier


class OrderNotFoundError(EntityNotFoundError):
    """Raised when an order is not found."""

    def __init__(self, order_id: UUID) -> None:
        super().__init__("Order", order_id)
        self.order_id = order_id


class PositionNotFoundError(EntityNotFoundError):
    """Raised when a position is not found."""

    def __init__(self, position_id: UUID) -> None:
        super().__init__("Position", position_id)
        self.position_id = position_id


class PortfolioNotFoundError(EntityNotFoundError):
    """Raised when a portfolio is not found."""

    def __init__(self, portfolio_id: UUID) -> None:
        super().__init__("Portfolio", portfolio_id)
        self.portfolio_id = portfolio_id


class DuplicateEntityError(RepositoryError):
    """Raised when attempting to create an entity that already exists."""

    def __init__(self, entity_type: str, identifier: UUID | str) -> None:
        super().__init__(f"{entity_type} with identifier '{identifier}' already exists")
        self.entity_type = entity_type
        self.identifier = identifier


class ConcurrencyError(RepositoryError):
    """Raised when a concurrency conflict occurs."""

    def __init__(self, entity_type: str, identifier: UUID | str) -> None:
        super().__init__(
            f"{entity_type} with identifier '{identifier}' was modified by another process"
        )
        self.entity_type = entity_type
        self.identifier = identifier


class ValidationError(RepositoryError):
    """Raised when entity validation fails."""

    def __init__(self, entity_type: str, field: str, value: Any, message: str) -> None:
        super().__init__(f"{entity_type}.{field} validation failed: {message}")
        self.entity_type = entity_type
        self.field = field
        self.value = value


class TransactionError(RepositoryError):
    """Base exception for transaction operations."""

    pass


class TransactionNotActiveError(TransactionError):
    """Raised when operation requires active transaction but none exists."""

    def __init__(self) -> None:
        super().__init__("No active transaction")


class TransactionAlreadyActiveError(TransactionError):
    """Raised when attempting to start transaction when one is already active."""

    def __init__(self) -> None:
        super().__init__("Transaction is already active")


class TransactionCommitError(TransactionError):
    """Raised when transaction commit fails."""

    def __init__(self, cause: Exception | None = None) -> None:
        super().__init__("Transaction commit failed", cause)


class TransactionRollbackError(TransactionError):
    """Raised when transaction rollback fails."""

    def __init__(self, cause: Exception | None = None) -> None:
        super().__init__("Transaction rollback failed", cause)


class ConnectionError(RepositoryError):
    """Raised when database connection fails."""

    def __init__(self, message: str = "Database connection failed") -> None:
        super().__init__(message)


class TimeoutError(RepositoryError):
    """Raised when repository operation times out."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds} seconds")
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class IntegrityError(RepositoryError):
    """Raised when database integrity constraint is violated."""

    def __init__(self, constraint: str, message: str | None = None) -> None:
        msg = f"Integrity constraint '{constraint}' violated"
        if message:
            msg += f": {message}"
        super().__init__(msg)
        self.constraint = constraint


class FactoryError(Exception):
    """Raised when factory cannot create an instance."""

    def __init__(self, factory_type: str, message: str) -> None:
        super().__init__(f"{factory_type} factory error: {message}")
        self.factory_type = factory_type


class ConfigurationError(Exception):
    """Raised when configuration-related errors occur."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause
