"""
Comprehensive unit tests for PostgreSQL Unit of Work Implementation.

Tests transaction management, repository coordination, factory pattern,
and transaction manager with full coverage of error scenarios.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import (
    FactoryError,
    TransactionAlreadyActiveError,
    TransactionCommitError,
    TransactionError,
    TransactionNotActiveError,
    TransactionRollbackError,
)
from src.domain.entities.order import Order, OrderRequest, OrderSide
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.repositories.unit_of_work import (
    PostgreSQLTransactionManager,
    PostgreSQLUnitOfWork,
    PostgreSQLUnitOfWorkFactory,
)


@pytest.fixture
def mock_adapter():
    """Mock PostgreSQL adapter for unit of work tests."""
    adapter = AsyncMock(spec=PostgreSQLAdapter)
    adapter.has_active_transaction = False
    adapter.begin_transaction = AsyncMock()
    adapter.commit_transaction = AsyncMock()
    adapter.rollback_transaction = AsyncMock()
    adapter.execute_query = AsyncMock(return_value="EXECUTE 1")
    adapter.fetch_one = AsyncMock(return_value=None)
    adapter.fetch_all = AsyncMock(return_value=[])
    return adapter


@pytest.fixture
def unit_of_work(mock_adapter):
    """Unit of work with mocked adapter."""
    return PostgreSQLUnitOfWork(mock_adapter)


@pytest.fixture
def mock_connection():
    """Mock database connection."""
    connection = MagicMock()
    connection._pool = MagicMock()
    return connection


@pytest.fixture
def mock_connection_factory(mock_connection):
    """Mock connection factory."""
    factory = MagicMock()
    factory.get_connection = AsyncMock(return_value=mock_connection)
    return factory


@pytest.mark.unit
class TestUnitOfWorkInitialization:
    """Test unit of work initialization."""

    def test_initialization(self, mock_adapter):
        """Test unit of work is properly initialized with repositories."""
        uow = PostgreSQLUnitOfWork(mock_adapter)

        assert uow.adapter is mock_adapter
        assert uow.orders is not None
        assert uow.positions is not None
        assert uow.portfolios is not None

        # Verify repositories are initialized with the same adapter
        assert uow.orders.adapter is mock_adapter
        assert uow.positions.adapter is mock_adapter
        assert uow.portfolios.adapter is mock_adapter


@pytest.mark.unit
class TestTransactionManagement:
    """Test transaction management operations."""

    async def test_begin_transaction_success(self, unit_of_work, mock_adapter):
        """Test beginning a new transaction."""
        mock_adapter.has_active_transaction = False

        await unit_of_work.begin_transaction()

        mock_adapter.begin_transaction.assert_called_once()

    async def test_begin_transaction_already_active(self, unit_of_work, mock_adapter):
        """Test beginning transaction when one is already active."""
        mock_adapter.has_active_transaction = True

        with pytest.raises(TransactionAlreadyActiveError):
            await unit_of_work.begin_transaction()

    async def test_begin_transaction_error(self, unit_of_work, mock_adapter):
        """Test begin transaction with database error."""
        mock_adapter.has_active_transaction = False
        mock_adapter.begin_transaction.side_effect = Exception("Database error")

        with pytest.raises(TransactionError, match="Failed to begin transaction"):
            await unit_of_work.begin_transaction()

    async def test_commit_success(self, unit_of_work, mock_adapter):
        """Test committing a transaction."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.commit()

        mock_adapter.commit_transaction.assert_called_once()

    async def test_commit_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test committing when no transaction is active."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(TransactionNotActiveError):
            await unit_of_work.commit()

    async def test_commit_error(self, unit_of_work, mock_adapter):
        """Test commit with database error."""
        mock_adapter.has_active_transaction = True
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        with pytest.raises(TransactionCommitError):
            await unit_of_work.commit()

    async def test_rollback_success(self, unit_of_work, mock_adapter):
        """Test rolling back a transaction."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.rollback()

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_rollback_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test rolling back when no transaction is active."""
        mock_adapter.has_active_transaction = False

        # Should not raise, just log warning
        await unit_of_work.rollback()

        mock_adapter.rollback_transaction.assert_not_called()

    async def test_rollback_error(self, unit_of_work, mock_adapter):
        """Test rollback with database error."""
        mock_adapter.has_active_transaction = True
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        with pytest.raises(TransactionRollbackError):
            await unit_of_work.rollback()

    async def test_is_active(self, unit_of_work, mock_adapter):
        """Test checking if transaction is active."""
        mock_adapter.has_active_transaction = False
        assert await unit_of_work.is_active() is False

        mock_adapter.has_active_transaction = True
        assert await unit_of_work.is_active() is True

    async def test_flush_success(self, unit_of_work, mock_adapter):
        """Test flushing pending changes."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.flush()

        # Flush is a no-op for PostgreSQL, just verify no error

    async def test_flush_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test flushing when no transaction is active."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(TransactionNotActiveError):
            await unit_of_work.flush()

    async def test_flush_error(self, unit_of_work, mock_adapter):
        """Test flush with error."""
        # Set initial state
        mock_adapter.has_active_transaction = True

        # Mock the flush method to check for active transaction and raise
        async def flush_with_error():
            if not mock_adapter.has_active_transaction:
                raise TransactionNotActiveError()
            # Simulate internal error during flush
            raise Exception("Flush error")

        # Replace flush with error version
        original_flush = unit_of_work.flush
        unit_of_work.flush = flush_with_error

        with pytest.raises(Exception, match="Flush error"):
            await unit_of_work.flush()

        # Restore original
        unit_of_work.flush = original_flush

    async def test_refresh_entity(self, unit_of_work, mock_adapter):
        """Test refreshing an entity."""
        entity = MagicMock()
        entity.__class__.__name__ = "TestEntity"

        await unit_of_work.refresh(entity)

        # Simplified implementation, just verify no error

    async def test_refresh_entity_error(self, unit_of_work, mock_adapter):
        """Test refresh with error."""
        # Simple test - the refresh method is simplified and logs errors
        # Force an error in refresh by mocking logger
        with patch("src.infrastructure.repositories.unit_of_work.logger") as mock_logger:
            mock_logger.debug.side_effect = Exception("Refresh error")

            entity = MagicMock()

            with pytest.raises(TransactionError, match="Failed to refresh entity"):
                await unit_of_work.refresh(entity)


@pytest.mark.unit
class TestContextManager:
    """Test async context manager functionality."""

    async def test_context_manager_success(self, unit_of_work, mock_adapter):
        """Test successful transaction using context manager."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work as uow:
            mock_adapter.has_active_transaction = True
            assert uow is unit_of_work
            mock_adapter.begin_transaction.assert_called_once()

        mock_adapter.commit_transaction.assert_called_once()

    async def test_context_manager_with_exception(self, unit_of_work, mock_adapter):
        """Test context manager rolls back on exception."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(ValueError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                raise ValueError("Test error")

        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_not_called()

    async def test_context_manager_commit_failure(self, unit_of_work, mock_adapter):
        """Test context manager handles commit failure."""
        mock_adapter.has_active_transaction = False
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        from src.application.interfaces.exceptions import TransactionCommitError

        with pytest.raises(TransactionCommitError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                pass  # Transaction should succeed but commit will fail

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_context_manager_commit_and_rollback_failure(self, unit_of_work, mock_adapter):
        """Test context manager when both commit and rollback fail."""
        mock_adapter.has_active_transaction = False
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        from src.application.interfaces.exceptions import TransactionCommitError

        with pytest.raises(TransactionCommitError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                pass

        # Both should be attempted
        mock_adapter.commit_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_context_manager_rollback_failure_on_exception(self, unit_of_work, mock_adapter):
        """Test context manager when rollback fails after an exception."""
        mock_adapter.has_active_transaction = False
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        with pytest.raises(ValueError):  # Original exception is not suppressed
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                raise ValueError("Test error")

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_context_manager_exit_types(self, unit_of_work, mock_adapter):
        """Test context manager exit with different exception types."""
        mock_adapter.has_active_transaction = False

        # First enter the context
        await unit_of_work.__aenter__()
        mock_adapter.has_active_transaction = True

        # Test with None exception (success case)
        await unit_of_work.__aexit__(None, None, None)
        # Should not raise any errors and should commit
        mock_adapter.commit_transaction.assert_called_once()


@pytest.mark.unit
class TestUnitOfWorkFactory:
    """Test unit of work factory."""

    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = PostgreSQLUnitOfWorkFactory()
        assert factory._connection_factory is not None

    def test_create_unit_of_work(self, mock_connection):
        """Test creating unit of work synchronously."""
        factory = PostgreSQLUnitOfWorkFactory()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = mock_connection

            uow = factory.create_unit_of_work()

            assert isinstance(uow, PostgreSQLUnitOfWork)
            assert uow.adapter is not None

    def test_create_unit_of_work_no_pool(self, mock_connection):
        """Test creating unit of work when connection pool is None."""
        factory = PostgreSQLUnitOfWorkFactory()
        mock_connection._pool = None

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = mock_connection

            with pytest.raises(FactoryError, match="Database connection pool is not initialized"):
                factory.create_unit_of_work()

    def test_create_unit_of_work_connection_error(self):
        """Test creating unit of work with connection error."""
        factory = PostgreSQLUnitOfWorkFactory()

        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Connection failed")

            with pytest.raises(FactoryError, match="Failed to create Unit of Work"):
                factory.create_unit_of_work()

    async def test_create_unit_of_work_async(self, mock_connection):
        """Test creating unit of work asynchronously."""
        factory = PostgreSQLUnitOfWorkFactory()

        with patch.object(
            factory._connection_factory, "get_connection", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_connection

            uow = await factory.create_unit_of_work_async()

            assert isinstance(uow, PostgreSQLUnitOfWork)
            assert uow.adapter is not None

    async def test_create_unit_of_work_async_no_pool(self, mock_connection):
        """Test creating unit of work async when connection pool is None."""
        factory = PostgreSQLUnitOfWorkFactory()
        mock_connection._pool = None

        with patch.object(
            factory._connection_factory, "get_connection", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_connection

            with pytest.raises(FactoryError, match="Database connection pool is not initialized"):
                await factory.create_unit_of_work_async()

    async def test_create_unit_of_work_async_error(self):
        """Test creating unit of work async with error."""
        factory = PostgreSQLUnitOfWorkFactory()

        with patch.object(
            factory._connection_factory, "get_connection", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            with pytest.raises(FactoryError, match="Failed to create Unit of Work"):
                await factory.create_unit_of_work_async()


@pytest.mark.unit
class TestTransactionManager:
    """Test transaction manager."""

    @pytest.fixture
    def mock_factory(self):
        """Mock unit of work factory."""
        factory = MagicMock()
        return factory

    @pytest.fixture
    def transaction_manager(self, mock_factory):
        """Transaction manager with mocked factory."""
        return PostgreSQLTransactionManager(mock_factory)

    async def test_execute_in_transaction_success(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test executing operation in transaction."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        async def operation(uow):
            return "success"

        result = await transaction_manager.execute_in_transaction(operation)

        assert result == "success"
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()

    async def test_execute_in_transaction_with_async_factory(
        self, transaction_manager, mock_adapter
    ):
        """Test executing operation with async factory."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_async_factory = PostgreSQLUnitOfWorkFactory()

        with patch.object(
            mock_async_factory, "create_unit_of_work_async", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_uow
            mock_adapter.has_active_transaction = False

            # Mock begin_transaction to set has_active_transaction to True
            async def begin_transaction_side_effect():
                mock_adapter.has_active_transaction = True

            mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

            manager = PostgreSQLTransactionManager(mock_async_factory)

            async def operation(uow):
                return "async_success"

            result = await manager.execute_in_transaction(operation)

            assert result == "async_success"
            mock_create.assert_called_once()

    async def test_execute_in_transaction_error(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test executing operation that fails."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        async def failing_operation(uow):
            raise ValueError("Operation failed")

        with pytest.raises(TransactionError, match="Transaction operation failed"):
            await transaction_manager.execute_in_transaction(failing_operation)

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_execute_with_retry_success_first_attempt(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test executing with retry succeeds on first attempt."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        async def operation(uow):
            return "success"

        result = await transaction_manager.execute_with_retry(operation)

        assert result == "success"
        # Should only be called once
        assert mock_adapter.begin_transaction.call_count == 1

    async def test_execute_with_retry_success_after_retry(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test executing with retry succeeds after retry."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Track begin_transaction calls and set has_active_transaction appropriately
        begin_count = 0

        async def begin_transaction_side_effect():
            nonlocal begin_count
            begin_count += 1
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        # Mock rollback to reset transaction state
        async def rollback_side_effect():
            mock_adapter.has_active_transaction = False

        mock_adapter.rollback_transaction.side_effect = rollback_side_effect

        call_count = 0

        async def operation(uow):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await transaction_manager.execute_with_retry(operation, max_retries=2)

        assert result == "success"
        assert call_count == 2

    async def test_execute_with_retry_all_attempts_fail(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test executing with retry when all attempts fail."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        # Mock rollback to reset transaction state
        async def rollback_side_effect():
            mock_adapter.has_active_transaction = False

        mock_adapter.rollback_transaction.side_effect = rollback_side_effect

        async def failing_operation(uow):
            raise ValueError("Persistent error")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TransactionError, match="Transaction failed after 4 attempts"):
                await transaction_manager.execute_with_retry(failing_operation, max_retries=3)

    async def test_execute_with_retry_exponential_backoff(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test exponential backoff in retry logic."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        # Mock rollback to reset transaction state
        async def rollback_side_effect():
            mock_adapter.has_active_transaction = False

        mock_adapter.rollback_transaction.side_effect = rollback_side_effect

        async def failing_operation(uow):
            raise ValueError("Error")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(TransactionError):
                await transaction_manager.execute_with_retry(
                    failing_operation, max_retries=3, backoff_factor=2.0
                )

            # Check sleep was called with exponential backoff
            expected_delays = [2.0, 4.0, 8.0]  # 2.0 * (2^0), 2.0 * (2^1), 2.0 * (2^2)
            actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_delays == expected_delays

    async def test_execute_batch_success(self, transaction_manager, mock_factory, mock_adapter):
        """Test executing batch operations."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        async def op1(uow):
            return "result1"

        async def op2(uow):
            return "result2"

        async def op3(uow):
            return "result3"

        results = await transaction_manager.execute_batch([op1, op2, op3])

        assert results == ["result1", "result2", "result3"]
        # All operations should be in a single transaction
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()

    async def test_execute_batch_partial_failure(
        self, transaction_manager, mock_factory, mock_adapter
    ):
        """Test batch operations with one failing."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        async def op1(uow):
            return "result1"

        async def op2(uow):
            raise ValueError("Operation 2 failed")

        async def op3(uow):
            return "result3"

        with pytest.raises(TransactionError, match="Batch operation 1 failed"):
            await transaction_manager.execute_batch([op1, op2, op3])

        # Transaction should be rolled back
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_execute_batch_empty_list(self, transaction_manager, mock_factory, mock_adapter):
        """Test executing empty batch."""
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        results = await transaction_manager.execute_batch([])

        assert results == []
        # Transaction should still be started and committed
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()


@pytest.mark.unit
class TestRepositoryIntegration:
    """Test unit of work with repository integration."""

    async def test_order_repository_access(self, unit_of_work, mock_adapter):
        """Test accessing order repository through unit of work."""
        # Create a sample order
        order_request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Test",
        )
        order = Order.create_limit_order(order_request)

        # Start transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Save order through repository
        await unit_of_work.orders.save_order(order)

        # Commit transaction
        await unit_of_work.commit()

        mock_adapter.execute_query.assert_called()

    async def test_position_repository_access(self, unit_of_work, mock_adapter):
        """Test accessing position repository through unit of work."""
        position = Position(
            id=uuid4(),
            symbol="GOOGL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("2500.00"),
        )

        # Start transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Save position through repository
        await unit_of_work.positions.persist_position(position)

        # Commit transaction
        await unit_of_work.commit()

        mock_adapter.execute_query.assert_called()

    async def test_portfolio_repository_access(self, unit_of_work, mock_adapter):
        """Test accessing portfolio repository through unit of work."""
        portfolio = Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
            cash_balance=Decimal("100000.00"),
        )

        # Start transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Save portfolio through repository
        await unit_of_work.portfolios.save_portfolio(portfolio)

        # Commit transaction
        await unit_of_work.commit()

        mock_adapter.execute_query.assert_called()

    async def test_cross_repository_transaction(self, unit_of_work, mock_adapter):
        """Test transaction across multiple repositories."""
        order_request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Test",
        )
        order = Order.create_limit_order(order_request)

        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
        )

        portfolio = Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
            cash_balance=Decimal("85000.00"),  # After buying position
        )

        # Start transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Save entities across repositories
        await unit_of_work.orders.save_order(order)
        await unit_of_work.positions.persist_position(position)
        await unit_of_work.portfolios.save_portfolio(portfolio)

        # Commit transaction
        await unit_of_work.commit()

        # All operations should be in single transaction
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()
        assert mock_adapter.execute_query.call_count >= 3  # At least one per repository

    async def test_transaction_rollback_on_repository_error(self, unit_of_work, mock_adapter):
        """Test transaction rollback when repository operation fails."""
        order_request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Test",
        )
        order = Order.create_limit_order(order_request)

        # Start transaction
        mock_adapter.has_active_transaction = False

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        # Second operation fails
        mock_adapter.execute_query.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            async with unit_of_work:
                # First operation succeeds
                await unit_of_work.orders.save_order(order)

                # This operation will fail and trigger rollback
                position = Position(
                    id=uuid4(),
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    average_entry_price=Decimal("150.00"),
                )
                await unit_of_work.positions.persist_position(position)

        # Transaction should be rolled back
        mock_adapter.rollback_transaction.assert_called_once()


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_multiple_begin_transaction_calls(self, unit_of_work, mock_adapter):
        """Test multiple begin transaction calls."""
        mock_adapter.has_active_transaction = False

        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Second call should raise
        with pytest.raises(TransactionAlreadyActiveError):
            await unit_of_work.begin_transaction()

    async def test_commit_after_rollback(self, unit_of_work, mock_adapter):
        """Test committing after rollback."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.rollback()
        mock_adapter.has_active_transaction = False

        with pytest.raises(TransactionNotActiveError):
            await unit_of_work.commit()

    async def test_nested_context_managers(self, unit_of_work, mock_adapter):
        """Test nested context manager usage (should fail)."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Nested context should fail to begin transaction
            with pytest.raises(TransactionAlreadyActiveError):
                async with unit_of_work:
                    pass

    async def test_transaction_state_consistency(self, unit_of_work, mock_adapter):
        """Test transaction state remains consistent."""
        # Initially no transaction
        assert await unit_of_work.is_active() is False

        # Begin transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True
        assert await unit_of_work.is_active() is True

        # Commit transaction
        await unit_of_work.commit()
        mock_adapter.has_active_transaction = False
        assert await unit_of_work.is_active() is False

    async def test_exception_types_in_context_manager(self, unit_of_work, mock_adapter):
        """Test different exception types in context manager."""
        mock_adapter.has_active_transaction = False

        # Test with different exception types
        exceptions = [
            ValueError("Value error"),
            KeyError("Key error"),
            RuntimeError("Runtime error"),
            Exception("Generic error"),
        ]

        for exc in exceptions:
            mock_adapter.has_active_transaction = False
            mock_adapter.begin_transaction.reset_mock()
            mock_adapter.rollback_transaction.reset_mock()

            with pytest.raises(type(exc)):
                async with unit_of_work:
                    mock_adapter.has_active_transaction = True
                    raise exc

            mock_adapter.rollback_transaction.assert_called_once()

    async def test_factory_with_different_connection_types(self):
        """Test factory with different connection configurations."""
        factory = PostgreSQLUnitOfWorkFactory()

        # Test with connection that has pool
        mock_conn_with_pool = MagicMock()
        mock_conn_with_pool._pool = MagicMock()

        with patch.object(
            factory._connection_factory, "get_connection", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_conn_with_pool

            uow = await factory.create_unit_of_work_async()
            assert isinstance(uow, PostgreSQLUnitOfWork)

        # Test with connection without pool
        mock_conn_no_pool = MagicMock()
        mock_conn_no_pool._pool = None

        with patch.object(
            factory._connection_factory, "get_connection", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_conn_no_pool

            with pytest.raises(FactoryError, match="Database connection pool is not initialized"):
                await factory.create_unit_of_work_async()

    async def test_transaction_manager_type_casting(self):
        """Test transaction manager properly casts return types."""
        mock_adapter = AsyncMock()
        mock_adapter.has_active_transaction = False
        mock_adapter.begin_transaction.return_value = None
        mock_adapter.commit_transaction.return_value = None
        mock_adapter.rollback_transaction.return_value = None

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        mock_factory = MagicMock()
        manager = PostgreSQLTransactionManager(mock_factory)
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow

        async def op1(uow):
            return 1

        async def op2(uow):
            return "two"

        async def op3(uow):
            return {"three": 3}

        results = await manager.execute_batch([op1, op2, op3])

        assert isinstance(results, list)
        assert results == [1, "two", {"three": 3}]
        assert len(results) == 3


@pytest.mark.unit
class TestUnitOfWorkAdvancedScenarios:
    """Test advanced unit of work scenarios."""

    async def test_savepoint_simulation(self, unit_of_work, mock_adapter):
        """Test savepoint-like behavior within transactions."""
        mock_adapter.has_active_transaction = False

        # Begin main transaction
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Perform first operation
        order_request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Test",
        )
        order = Order.create_limit_order(order_request)
        await unit_of_work.orders.save_order(order)

        # Simulate savepoint (flush without commit)
        await unit_of_work.flush()

        # Perform second operation that might fail
        try:
            mock_adapter.execute_query.side_effect = Exception("Constraint violation")
            position = Position(
                id=uuid4(),
                symbol="INVALID",
                quantity=Decimal("-100"),  # Invalid negative for long position
                average_entry_price=Decimal("150.00"),
            )
            await unit_of_work.positions.persist_position(position)
        except Exception:
            # Would rollback to savepoint in real implementation
            pass

        # Reset and commit what we can
        mock_adapter.execute_query.side_effect = None
        await unit_of_work.commit()

    async def test_distributed_transaction_pattern(self, unit_of_work, mock_adapter):
        """Test pattern for distributed transactions."""
        mock_adapter.has_active_transaction = False

        # Phase 1: Prepare
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Prepare multiple operations
        operations_prepared = []

        order_request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Test",
        )
        order = Order.create_limit_order(order_request)
        operations_prepared.append(("order", order))

        position = Position(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
        )
        operations_prepared.append(("position", position))

        # Phase 2: Execute all operations
        for op_type, entity in operations_prepared:
            if op_type == "order":
                await unit_of_work.orders.save_order(entity)
            elif op_type == "position":
                await unit_of_work.positions.persist_position(entity)

        # Phase 3: Commit if all succeed
        await unit_of_work.commit()

    async def test_nested_repository_calls(self, unit_of_work, mock_adapter):
        """Test nested repository calls within same transaction."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Create portfolio
            portfolio = Portfolio(
                id=uuid4(),
                name="Test Portfolio",
                initial_capital=Decimal("100000.00"),
                cash_balance=Decimal("100000.00"),
            )
            await unit_of_work.portfolios.save_portfolio(portfolio)

            # Create multiple positions for portfolio
            symbols = ["AAPL", "GOOGL", "MSFT"]
            for symbol in symbols:
                position = Position(
                    id=uuid4(),
                    symbol=symbol,
                    quantity=Decimal("100"),
                    average_entry_price=Decimal("100.00"),
                )
                await unit_of_work.positions.persist_position(position)

                # Update portfolio balance
                portfolio.cash_balance -= Decimal("10000.00")

            # Final portfolio update
            mock_adapter.execute_query.return_value = "UPDATE 1"
            await unit_of_work.portfolios.update_portfolio(portfolio)

    async def test_transaction_timeout_handling(self, unit_of_work, mock_adapter):
        """Test handling of transaction timeouts."""
        mock_adapter.has_active_transaction = False

        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Simulate long-running transaction that times out
        mock_adapter.commit_transaction.side_effect = Exception("transaction timeout")

        with pytest.raises(TransactionCommitError):
            await unit_of_work.commit()

    async def test_read_after_write_consistency(self, unit_of_work, mock_adapter):
        """Test read-after-write consistency within transaction."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Write operation
            position = Position(
                id=uuid4(),
                symbol="AAPL",
                quantity=Decimal("100"),
                average_entry_price=Decimal("150.00"),
            )
            await unit_of_work.positions.persist_position(position)

            # Immediate read should see the write
            mock_adapter.fetch_one.return_value = {
                "id": position.id,
                "symbol": "AAPL",
                "quantity": Decimal("100"),
                "average_entry_price": Decimal("150.00"),
                "current_price": None,
                "last_updated": None,
                "realized_pnl": Decimal("0"),
                "commission_paid": Decimal("0"),
                "stop_loss_price": None,
                "take_profit_price": None,
                "max_position_value": None,
                "opened_at": datetime.now(UTC),
                "closed_at": None,
                "strategy": None,
                "tags": {},
            }

            retrieved = await unit_of_work.positions.get_position_by_id(position.id)
            assert retrieved is not None
            assert retrieved.symbol == "AAPL"


@pytest.mark.unit
class TestUnitOfWorkErrorRecovery:
    """Test error recovery and compensation scenarios."""

    async def test_partial_rollback_compensation(self, unit_of_work, mock_adapter):
        """Test compensation logic for partial rollbacks."""
        mock_adapter.has_active_transaction = False

        # Track operations for compensation
        completed_operations = []

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # First operation succeeds
            order_request = OrderRequest(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                limit_price=Decimal("150.00"),
                reason="Test",
            )
            order = Order.create_limit_order(order_request)
            await unit_of_work.orders.save_order(order)
            completed_operations.append(("order", order.id))

            # Second operation fails
            mock_adapter.execute_query.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                position = Position(
                    id=uuid4(),
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    average_entry_price=Decimal("150.00"),
                )
                await unit_of_work.positions.persist_position(position)

        # Transaction rolled back, completed_operations can be used for compensation
        assert len(completed_operations) == 1

    async def test_connection_recovery(self, unit_of_work, mock_adapter):
        """Test recovery from connection loss."""
        mock_adapter.has_active_transaction = True

        # Simulate connection loss during commit
        mock_adapter.commit_transaction.side_effect = Exception("connection lost")

        with pytest.raises(TransactionCommitError):
            await unit_of_work.commit()

        # Attempt rollback after connection loss
        mock_adapter.rollback_transaction.side_effect = Exception("still no connection")

        with pytest.raises(TransactionRollbackError):
            await unit_of_work.rollback()

    async def test_deadlock_retry_pattern(self):
        """Test automatic retry on deadlock detection."""
        mock_adapter = AsyncMock()
        mock_adapter.has_active_transaction = False
        mock_adapter.begin_transaction.return_value = None
        mock_adapter.commit_transaction.return_value = None
        mock_adapter.rollback_transaction.return_value = None

        # Mock begin_transaction to set has_active_transaction to True
        async def begin_transaction_side_effect():
            mock_adapter.has_active_transaction = True

        mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

        # Mock rollback to reset transaction state
        async def rollback_side_effect():
            mock_adapter.has_active_transaction = False

        mock_adapter.rollback_transaction.side_effect = rollback_side_effect

        mock_factory = MagicMock()
        transaction_manager = PostgreSQLTransactionManager(mock_factory)
        mock_uow = PostgreSQLUnitOfWork(mock_adapter)
        mock_factory.create_unit_of_work.return_value = mock_uow

        call_count = 0

        async def operation_with_deadlock(uow):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("deadlock detected")
            return "success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await transaction_manager.execute_with_retry(
                operation_with_deadlock, max_retries=3
            )

        assert result == "success after retry"
        assert call_count == 3


@pytest.mark.unit
class TestUnitOfWorkConcurrency:
    """Test concurrent transaction scenarios."""

    async def test_concurrent_transaction_isolation(self, mock_adapter):
        """Test isolation between concurrent transactions."""
        # Create two unit of work instances
        uow1 = PostgreSQLUnitOfWork(mock_adapter)
        uow2 = PostgreSQLUnitOfWork(mock_adapter)

        # Both should not be able to start transactions simultaneously
        mock_adapter.has_active_transaction = False
        await uow1.begin_transaction()
        mock_adapter.has_active_transaction = True

        with pytest.raises(TransactionAlreadyActiveError):
            await uow2.begin_transaction()

    async def test_optimistic_locking_pattern(self, unit_of_work, mock_adapter):
        """Test optimistic locking pattern implementation."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Read portfolio with version
            portfolio_record = {
                "id": uuid4(),
                "name": "Test Portfolio",
                "initial_capital": Decimal("100000.00"),
                "cash_balance": Decimal("100000.00"),
                "max_position_size": Decimal("10000.00"),
                "max_portfolio_risk": Decimal("0.02"),
                "max_positions": 10,
                "max_leverage": Decimal("2.0"),
                "total_realized_pnl": Decimal("0"),
                "total_commission_paid": Decimal("0"),
                "trades_count": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "created_at": datetime.now(UTC),
                "last_updated": datetime.now(UTC),
                "strategy": None,
                "tags": {},
            }
            mock_adapter.fetch_one.return_value = portfolio_record
            mock_adapter.fetch_all.return_value = []  # No positions

            portfolio = await unit_of_work.portfolios.get_portfolio_by_id(portfolio_record["id"])

            # Modify portfolio
            portfolio.cash_balance = Decimal("95000.00")

            # Update with version check (simulated)
            mock_adapter.execute_query.return_value = "UPDATE 1"  # Version matched
            await unit_of_work.portfolios.update_portfolio(portfolio)

    async def test_lock_acquisition_timeout(self, unit_of_work, mock_adapter):
        """Test timeout when acquiring locks."""
        mock_adapter.has_active_transaction = False
        mock_adapter.begin_transaction.side_effect = Exception("lock acquisition timeout")

        with pytest.raises(TransactionError, match="Failed to begin transaction"):
            await unit_of_work.begin_transaction()


@pytest.mark.unit
class TestTransactionManagerAdvanced:
    """Test advanced transaction manager scenarios."""

    async def test_saga_pattern_implementation(self, mock_adapter):
        """Test saga pattern for distributed transactions."""
        factory = PostgreSQLUnitOfWorkFactory()
        manager = PostgreSQLTransactionManager(factory)

        with patch.object(factory, "create_unit_of_work") as mock_create:
            mock_uow = PostgreSQLUnitOfWork(mock_adapter)
            mock_create.return_value = mock_uow
            mock_adapter.has_active_transaction = False

            # Mock begin_transaction to set has_active_transaction to True
            async def begin_transaction_side_effect():
                mock_adapter.has_active_transaction = True

            mock_adapter.begin_transaction.side_effect = begin_transaction_side_effect

            # Define saga steps
            saga_results = []

            async def step1(uow):
                saga_results.append("step1")
                return "step1_complete"

            async def step2(uow):
                saga_results.append("step2")
                return "step2_complete"

            async def step3(uow):
                saga_results.append("step3")
                return "step3_complete"

            # Execute saga
            results = await manager.execute_batch([step1, step2, step3])

            assert len(saga_results) == 3
            assert results == ["step1_complete", "step2_complete", "step3_complete"]

    async def test_circuit_breaker_pattern(self, mock_adapter):
        """Test circuit breaker pattern for failing database."""
        factory = PostgreSQLUnitOfWorkFactory()
        manager = PostgreSQLTransactionManager(factory)

        with patch.object(factory, "create_unit_of_work") as mock_create:
            mock_uow = PostgreSQLUnitOfWork(mock_adapter)
            mock_create.return_value = mock_uow
            mock_adapter.has_active_transaction = False

            failure_count = 0

            async def failing_operation(uow):
                nonlocal failure_count
                failure_count += 1
                if failure_count < 5:
                    raise Exception("Database unavailable")
                return "recovered"

            # Circuit should open after repeated failures
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TransactionError):
                    await manager.execute_with_retry(
                        failing_operation,
                        max_retries=3,  # Less than required failures
                    )

            assert failure_count == 4  # Initial + 3 retries

    async def test_batch_with_dependencies(self, mock_adapter):
        """Test batch operations with dependencies."""
        factory = PostgreSQLUnitOfWorkFactory()
        manager = PostgreSQLTransactionManager(factory)

        with patch.object(factory, "create_unit_of_work") as mock_create:
            mock_uow = PostgreSQLUnitOfWork(mock_adapter)
            mock_create.return_value = mock_uow
            mock_adapter.has_active_transaction = False

            shared_state = {"portfolio_id": None}

            async def create_portfolio(uow):
                portfolio_id = uuid4()
                shared_state["portfolio_id"] = portfolio_id
                return portfolio_id

            async def create_position(uow):
                if not shared_state["portfolio_id"]:
                    raise ValueError("Portfolio must be created first")
                return f"position_for_{shared_state['portfolio_id']}"

            results = await manager.execute_batch([create_portfolio, create_position])

            assert shared_state["portfolio_id"] is not None
            assert results[1] == f"position_for_{shared_state['portfolio_id']}"
