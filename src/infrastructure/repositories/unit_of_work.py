"""
PostgreSQL Unit of Work Implementation

Concrete implementation of IUnitOfWork using PostgreSQL database.
Manages transactions across multiple repositories ensuring data consistency.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
import logging
from typing import Any

# Local imports
from src.application.interfaces.exceptions import (
    TransactionAlreadyActiveError,
    TransactionCommitError,
    TransactionError,
    TransactionNotActiveError,
    TransactionRollbackError,
)
from src.application.interfaces.repositories import (
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.application.interfaces.unit_of_work import (
    ITransactionManager,
    IUnitOfWork,
    IUnitOfWorkFactory,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import ConnectionFactory

from .order_repository import PostgreSQLOrderRepository
from .portfolio_repository import PostgreSQLPortfolioRepository
from .position_repository import PostgreSQLPositionRepository

logger = logging.getLogger(__name__)


class PostgreSQLUnitOfWork(IUnitOfWork):
    """
    PostgreSQL implementation of IUnitOfWork.

    Manages database transactions across multiple repositories.
    Ensures all operations within a unit of work are atomic.
    """

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize Unit of Work with database adapter.

        Args:
            adapter: PostgreSQL database adapter
        """
        self.adapter = adapter
        self._orders = PostgreSQLOrderRepository(adapter)
        self._positions = PostgreSQLPositionRepository(adapter)
        self._portfolios = PostgreSQLPortfolioRepository(adapter)

    @property
    def orders(self) -> IOrderRepository:
        """Get the orders repository."""
        return self._orders

    @property
    def positions(self) -> IPositionRepository:
        """Get the positions repository."""
        return self._positions

    @property
    def portfolios(self) -> IPortfolioRepository:
        """Get the portfolios repository."""
        return self._portfolios

    async def begin_transaction(self) -> None:
        """
        Begin a new database transaction.

        Should be called before performing multiple related operations
        that need to be atomic.

        Raises:
            TransactionError: If transaction cannot be started
        """
        try:
            if self.adapter.has_active_transaction:
                raise TransactionAlreadyActiveError()

            await self.adapter.begin_transaction()
            logger.debug("Unit of Work transaction started")

        except TransactionAlreadyActiveError:
            raise
        except Exception as e:
            logger.error(f"Failed to begin transaction: {e}")
            raise TransactionError(f"Failed to begin transaction: {e}") from e

    async def commit(self) -> None:
        """
        Commit the current transaction.

        Persists all changes made within the transaction to the database.

        Raises:
            TransactionError: If commit fails
        """
        try:
            if not self.adapter.has_active_transaction:
                raise TransactionNotActiveError()

            await self.adapter.commit_transaction()
            logger.debug("Unit of Work transaction committed")

        except TransactionNotActiveError:
            raise
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            raise TransactionCommitError(e) from e

    async def rollback(self) -> None:
        """
        Rollback the current transaction.

        Reverts all changes made within the transaction.

        Raises:
            TransactionError: If rollback fails
        """
        try:
            if not self.adapter.has_active_transaction:
                logger.warning("No active transaction to rollback")
                return

            await self.adapter.rollback_transaction()
            logger.debug("Unit of Work transaction rolled back")

        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
            raise TransactionRollbackError(e) from e

    async def is_active(self) -> bool:
        """
        Check if a transaction is currently active.

        Returns:
            True if transaction is active, False otherwise
        """
        return self.adapter.has_active_transaction

    async def flush(self) -> None:
        """
        Flush pending changes to the database without committing.

        Useful for getting generated IDs or checking constraints
        before final commit.

        Raises:
            TransactionError: If flush fails
        """
        try:
            if not self.adapter.has_active_transaction:
                raise TransactionNotActiveError()

            # PostgreSQL automatically flushes within transaction
            # This is a no-op for PostgreSQL, but we keep the interface
            logger.debug("Unit of Work flush (no-op for PostgreSQL)")

        except TransactionNotActiveError:
            raise
        except Exception as e:
            logger.error(f"Failed to flush transaction: {e}")
            raise TransactionError(f"Failed to flush transaction: {e}") from e

    async def refresh(self, entity) -> None:
        """
        Refresh an entity with the latest data from the database.

        Args:
            entity: The entity to refresh

        Raises:
            TransactionError: If refresh fails
        """
        try:
            # For PostgreSQL, we would need to re-fetch the entity
            # This is a simplified implementation
            logger.debug(f"Refresh entity {type(entity).__name__} (simplified implementation)")

            # In a full implementation, you would:
            # 1. Determine entity type
            # 2. Use appropriate repository to re-fetch
            # 3. Update entity fields with fresh data

        except Exception as e:
            logger.error(f"Failed to refresh entity: {e}")
            raise TransactionError(f"Failed to refresh entity: {e}") from e

    async def __aenter__(self):
        """
        Async context manager entry.

        Automatically begins a transaction.

        Returns:
            Self for use in async with statement
        """
        await self.begin_transaction()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.

        Automatically commits on success or rolls back on exception.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if exc_type is None:
            # No exception occurred, commit the transaction
            try:
                await self.commit()
            except Exception as commit_error:
                logger.error(f"Failed to commit in context manager: {commit_error}")
                try:
                    await self.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback after commit error: {rollback_error}")
                raise commit_error
        else:
            # Exception occurred, rollback the transaction
            try:
                await self.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback in context manager: {rollback_error}")
                # Don't suppress the original exception

        return False  # Don't suppress exceptions


class PostgreSQLUnitOfWorkFactory(IUnitOfWorkFactory):
    """
    Factory for creating PostgreSQL Unit of Work instances.

    Manages database connections and creates Unit of Work instances
    with properly configured database adapters.
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._connection_factory = ConnectionFactory()

    def create_unit_of_work(self) -> IUnitOfWork:
        """
        Create a new Unit of Work instance.

        Returns:
            A new Unit of Work instance

        Raises:
            FactoryError: If Unit of Work cannot be created
        """
        try:
            # Get database connection
            connection = asyncio.run(self._connection_factory.get_connection())

            # Create adapter with the connection pool
            adapter = PostgreSQLAdapter(connection._pool)

            # Create and return Unit of Work
            uow = PostgreSQLUnitOfWork(adapter)

            logger.debug("Created new PostgreSQL Unit of Work")
            return uow

        except Exception as e:
            logger.error(f"Failed to create Unit of Work: {e}")
            # Local imports
            from src.application.interfaces.exceptions import FactoryError

            raise FactoryError(
                "PostgreSQLUnitOfWorkFactory", f"Failed to create Unit of Work: {e}"
            ) from e

    async def create_unit_of_work_async(self) -> IUnitOfWork:
        """
        Create a new Unit of Work instance asynchronously.

        Returns:
            A new Unit of Work instance

        Raises:
            FactoryError: If Unit of Work cannot be created
        """
        try:
            # Get database connection
            connection = await self._connection_factory.get_connection()

            # Create adapter with the connection pool
            adapter = PostgreSQLAdapter(connection._pool)

            # Create and return Unit of Work
            uow = PostgreSQLUnitOfWork(adapter)

            logger.debug("Created new PostgreSQL Unit of Work (async)")
            return uow

        except Exception as e:
            logger.error(f"Failed to create Unit of Work async: {e}")
            # Local imports
            from src.application.interfaces.exceptions import FactoryError

            raise FactoryError(
                "PostgreSQLUnitOfWorkFactory", f"Failed to create Unit of Work: {e}"
            ) from e


class PostgreSQLTransactionManager(ITransactionManager):
    """
    High-level transaction management implementation.

    Provides convenience methods for common transaction patterns
    using PostgreSQL Unit of Work.
    """

    def __init__(self, factory: IUnitOfWorkFactory) -> None:
        """
        Initialize transaction manager.

        Args:
            factory: Unit of Work factory
        """
        self.factory = factory

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
        if isinstance(self.factory, PostgreSQLUnitOfWorkFactory):
            uow = await self.factory.create_unit_of_work_async()
        else:
            uow = self.factory.create_unit_of_work()

        try:
            async with uow:
                result = await operation(uow)
                logger.debug("Transaction operation completed successfully")
                return result

        except Exception as e:
            logger.error(f"Transaction operation failed: {e}")
            raise TransactionError(f"Transaction operation failed: {e}") from e

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
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.execute_in_transaction(operation)

            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    # Exponential backoff
                    delay = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Transaction attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} transaction attempts failed")
                    break

        raise TransactionError(
            f"Transaction failed after {max_retries + 1} attempts: {last_error}"
        ) from last_error

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

        async def batch_operation(uow: IUnitOfWork) -> list[Any]:
            results = []
            for i, operation in enumerate(operations):
                try:
                    result = await operation(uow)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch operation {i} failed: {e}")
                    raise TransactionError(f"Batch operation {i} failed: {e}") from e
            return results

        return await self.execute_in_transaction(batch_operation)
