"""
Unit of Work Interface

Defines the contract for managing transactions across multiple repositories.
Implements the Unit of Work pattern for atomic operations.
"""

# Standard library imports
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

from .repositories import IOrderRepository, IPortfolioRepository, IPositionRepository


class IUnitOfWork(Protocol):
    """
    Unit of Work interface for transaction management.

    Provides atomic operations across multiple repositories.
    Ensures data consistency by managing database transactions.
    """

    # Repository access
    orders: IOrderRepository
    positions: IPositionRepository
    portfolios: IPortfolioRepository

    @abstractmethod
    async def begin_transaction(self) -> None:
        """
        Begin a new database transaction.

        Should be called before performing multiple related operations
        that need to be atomic.

        Raises:
            TransactionError: If transaction cannot be started
        """
        ...

    @abstractmethod
    async def commit(self) -> None:
        """
        Commit the current transaction.

        Persists all changes made within the transaction to the database.

        Raises:
            TransactionError: If commit fails
        """
        ...

    @abstractmethod
    async def rollback(self) -> None:
        """
        Rollback the current transaction.

        Reverts all changes made within the transaction.

        Raises:
            TransactionError: If rollback fails
        """
        ...

    @abstractmethod
    async def is_active(self) -> bool:
        """
        Check if a transaction is currently active.

        Returns:
            True if transaction is active, False otherwise
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """
        Flush pending changes to the database without committing.

        Useful for getting generated IDs or checking constraints
        before final commit.

        Raises:
            TransactionError: If flush fails
        """
        ...

    @abstractmethod
    async def refresh(self, entity) -> None:
        """
        Refresh an entity with the latest data from the database.

        Args:
            entity: The entity to refresh

        Raises:
            TransactionError: If refresh fails
        """
        ...

    @abstractmethod
    async def __aenter__(self):
        """
        Async context manager entry.

        Automatically begins a transaction.

        Returns:
            Self for use in async with statement
        """
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.

        Automatically commits on success or rolls back on exception.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        ...


class IUnitOfWorkFactory(Protocol):
    """
    Factory interface for creating Unit of Work instances.

    Allows for different implementations (e.g., for testing vs production).
    """

    @abstractmethod
    def create_unit_of_work(self) -> IUnitOfWork:
        """
        Create a new Unit of Work instance.

        Returns:
            A new Unit of Work instance

        Raises:
            FactoryError: If Unit of Work cannot be created
        """
        ...


class ITransactionManager(Protocol):
    """
    High-level transaction management interface.

    Provides convenience methods for common transaction patterns.
    """

    @abstractmethod
    async def execute_in_transaction(self, operation: Callable[[IUnitOfWork], Any]) -> Any:
        """
        Execute an operation within a transaction.

        Args:
            operation: Async callable that takes a UnitOfWork parameter

        Returns:
            Result of the operation

        Raises:
            TransactionError: If transaction fails
        """
        ...

    @abstractmethod
    async def execute_with_retry(
        self,
        operation: Callable[[IUnitOfWork], Any],
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> Any:
        """
        Execute an operation with automatic retry on transient failures.

        Args:
            operation: Async callable that takes a UnitOfWork parameter
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor for exponential backoff between retries

        Returns:
            Result of the operation

        Raises:
            TransactionError: If all retries fail
        """
        ...

    @abstractmethod
    async def execute_batch(self, operations: list[Callable[[IUnitOfWork], Any]]) -> list[Any]:
        """
        Execute multiple operations in a single transaction.

        Args:
            operations: List of async callables that take a UnitOfWork parameter

        Returns:
            List of results from each operation

        Raises:
            TransactionError: If any operation fails
        """
        ...
