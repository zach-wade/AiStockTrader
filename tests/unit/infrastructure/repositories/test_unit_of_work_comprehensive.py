"""
Comprehensive unit tests for PostgreSQL Unit of Work Implementation.

Tests the concrete implementation of IUnitOfWork including transaction management,
repository coordination, factory pattern, and error handling with full coverage.
"""

# Standard library imports
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
from src.infrastructure.repositories.unit_of_work import (
    PostgreSQLTransactionManager,
    PostgreSQLUnitOfWork,
    PostgreSQLUnitOfWorkFactory,
)


@pytest.fixture
def mock_adapter():
    """Mock PostgreSQL adapter for unit of work tests."""
    adapter = AsyncMock()
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
def mock_connection_factory():
    """Mock connection factory."""
    factory = AsyncMock()
    connection = AsyncMock()
    connection._pool = AsyncMock()
    factory.get_connection = AsyncMock(return_value=connection)
    return factory


@pytest.fixture
def uow_factory(mock_connection_factory):
    """Unit of work factory with mocked connection factory."""
    factory = PostgreSQLUnitOfWorkFactory()
    factory._connection_factory = mock_connection_factory
    return factory


@pytest.mark.unit
class TestUnitOfWorkTransactionManagement:
    """Test unit of work transaction management."""

    async def test_begin_transaction_success(self, unit_of_work, mock_adapter):
        """Test successful transaction start."""
        await unit_of_work.begin_transaction()

        mock_adapter.begin_transaction.assert_called_once()

    async def test_begin_transaction_already_active(self, unit_of_work, mock_adapter):
        """Test starting transaction when one is already active."""
        mock_adapter.has_active_transaction = True

        with pytest.raises(TransactionAlreadyActiveError):
            await unit_of_work.begin_transaction()

    async def test_begin_transaction_error(self, unit_of_work, mock_adapter):
        """Test transaction start with error."""
        mock_adapter.begin_transaction.side_effect = Exception("Database error")

        with pytest.raises(TransactionError, match="Failed to begin transaction"):
            await unit_of_work.begin_transaction()

    async def test_commit_success(self, unit_of_work, mock_adapter):
        """Test successful transaction commit."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.commit()

        mock_adapter.commit_transaction.assert_called_once()

    async def test_commit_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test committing when no transaction is active."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(TransactionNotActiveError):
            await unit_of_work.commit()

    async def test_commit_error(self, unit_of_work, mock_adapter):
        """Test commit with error."""
        mock_adapter.has_active_transaction = True
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        with pytest.raises(TransactionCommitError):
            await unit_of_work.commit()

    async def test_rollback_success(self, unit_of_work, mock_adapter):
        """Test successful transaction rollback."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.rollback()

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_rollback_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test rollback when no transaction is active."""
        mock_adapter.has_active_transaction = False

        # Should not raise error, just log warning
        await unit_of_work.rollback()

        mock_adapter.rollback_transaction.assert_not_called()

    async def test_rollback_error(self, unit_of_work, mock_adapter):
        """Test rollback with error."""
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
        """Test flush operation."""
        mock_adapter.has_active_transaction = True

        await unit_of_work.flush()

        # Flush is a no-op for PostgreSQL, just verify no error

    async def test_flush_no_active_transaction(self, unit_of_work, mock_adapter):
        """Test flush when no transaction is active."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(TransactionNotActiveError):
            await unit_of_work.flush()

    async def test_flush_error(self, unit_of_work, mock_adapter):
        """Test flush with error by simulating exception in logger call."""
        mock_adapter.has_active_transaction = True

        # Mock the logger.debug call to raise an exception
        with patch(
            "src.infrastructure.repositories.unit_of_work.logger.debug",
            side_effect=Exception("Logging error"),
        ):
            with pytest.raises(TransactionError, match="Failed to flush transaction"):
                await unit_of_work.flush()

    async def test_refresh_entity(self, unit_of_work, mock_adapter):
        """Test refreshing an entity."""
        entity = MagicMock()
        entity.__class__.__name__ = "TestEntity"

        await unit_of_work.refresh(entity)

        # Should not raise error (simplified implementation)

    async def test_refresh_entity_error(self, unit_of_work):
        """Test refresh with error during entity refresh."""
        entity = MagicMock()
        entity.__class__.__name__ = "TestEntity"

        # Mock logger to raise an exception during refresh
        with patch(
            "src.infrastructure.repositories.unit_of_work.logger.debug",
            side_effect=Exception("Refresh failed"),
        ):
            with pytest.raises(TransactionError, match="Failed to refresh entity"):
                await unit_of_work.refresh(entity)


@pytest.mark.unit
class TestUnitOfWorkContextManager:
    """Test unit of work as async context manager."""

    async def test_context_manager_success(self, unit_of_work, mock_adapter):
        """Test successful context manager usage."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work as uow:
            mock_adapter.has_active_transaction = True
            assert uow is unit_of_work
            mock_adapter.begin_transaction.assert_called_once()

        mock_adapter.commit_transaction.assert_called_once()

    async def test_context_manager_with_exception(self, unit_of_work, mock_adapter):
        """Test context manager with exception."""
        mock_adapter.has_active_transaction = False

        with pytest.raises(ValueError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                mock_adapter.begin_transaction.assert_called_once()
                raise ValueError("Test error")

        mock_adapter.rollback_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_not_called()

    async def test_context_manager_commit_error(self, unit_of_work, mock_adapter):
        """Test context manager with commit error."""
        mock_adapter.has_active_transaction = False
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        with pytest.raises(TransactionCommitError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                pass

        # Should attempt rollback after commit failure
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_context_manager_rollback_error(self, unit_of_work, mock_adapter):
        """Test context manager with rollback error."""
        mock_adapter.has_active_transaction = False
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        # Original exception should not be suppressed
        with pytest.raises(ValueError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                raise ValueError("Original error")

    async def test_context_manager_commit_and_rollback_error(self, unit_of_work, mock_adapter):
        """Test context manager when both commit and rollback fail."""
        mock_adapter.has_active_transaction = False
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        with pytest.raises(TransactionCommitError):
            async with unit_of_work:
                mock_adapter.has_active_transaction = True
                pass

    async def test_context_manager_begin_transaction_error(self, unit_of_work, mock_adapter):
        """Test context manager with begin transaction error."""
        mock_adapter.begin_transaction.side_effect = Exception("Begin failed")

        with pytest.raises(Exception, match="Begin failed"):
            async with unit_of_work:
                pass


@pytest.mark.unit
class TestUnitOfWorkRepositories:
    """Test unit of work repository access."""

    def test_repository_initialization(self, unit_of_work):
        """Test that repositories are properly initialized."""
        assert unit_of_work.orders is not None
        assert unit_of_work.positions is not None
        assert unit_of_work.portfolios is not None

    async def test_repository_operations_with_transaction(self, unit_of_work, mock_adapter):
        """Test repository operations within a transaction."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Test order repository operation
            mock_adapter.fetch_one.return_value = None
            order_id = uuid4()
            result = await unit_of_work.orders.get_order_by_id(order_id)
            assert result is None

            # Test position repository operation
            position_id = uuid4()
            result = await unit_of_work.positions.get_position_by_id(position_id)
            assert result is None

            # Test portfolio repository operation
            portfolio_id = uuid4()
            result = await unit_of_work.portfolios.get_portfolio_by_id(portfolio_id)
            assert result is None

    async def test_multiple_repository_operations(self, unit_of_work, mock_adapter):
        """Test coordinating operations across multiple repositories."""
        mock_adapter.has_active_transaction = False

        async with unit_of_work:
            mock_adapter.has_active_transaction = True

            # Simulate creating order and updating portfolio
            mock_adapter.execute_query.return_value = "INSERT 1"

            # Create order
            await unit_of_work.orders._insert_order(MagicMock())

            # Update portfolio
            await unit_of_work.portfolios._insert_portfolio(MagicMock())

            # Both operations should use same adapter
            assert mock_adapter.execute_query.call_count >= 2


@pytest.mark.unit
class TestUnitOfWorkFactory:
    """Test unit of work factory."""

    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = PostgreSQLUnitOfWorkFactory()
        assert factory._connection_factory is not None
        from src.infrastructure.database.connection import ConnectionFactory

        assert isinstance(factory._connection_factory, ConnectionFactory)

    def test_factory_direct_initialization(self):
        """Test factory direct initialization to cover the __init__ method."""
        # Import needed to trigger execution
        from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWorkFactory

        factory = PostgreSQLUnitOfWorkFactory()
        # This covers line 242: self._connection_factory = ConnectionFactory()
        assert hasattr(factory, "_connection_factory")

    def test_create_unit_of_work_sync(self, uow_factory, mock_connection_factory):
        """Test synchronous unit of work creation."""
        uow = uow_factory.create_unit_of_work()

        assert isinstance(uow, PostgreSQLUnitOfWork)
        assert uow.adapter is not None
        mock_connection_factory.get_connection.assert_called()

    async def test_create_unit_of_work_async(self, uow_factory, mock_connection_factory):
        """Test asynchronous unit of work creation."""
        uow = await uow_factory.create_unit_of_work_async()

        assert isinstance(uow, PostgreSQLUnitOfWork)
        assert uow.adapter is not None
        mock_connection_factory.get_connection.assert_called()

    def test_create_unit_of_work_no_pool(self, uow_factory, mock_connection_factory):
        """Test unit of work creation when connection pool is not initialized."""
        connection = AsyncMock()
        connection._pool = None
        mock_connection_factory.get_connection.return_value = connection

        with pytest.raises(FactoryError, match="Database connection pool is not initialized"):
            uow_factory.create_unit_of_work()

    async def test_create_unit_of_work_async_no_pool(self, uow_factory, mock_connection_factory):
        """Test async unit of work creation when connection pool is not initialized."""
        connection = AsyncMock()
        connection._pool = None
        mock_connection_factory.get_connection.return_value = connection

        with pytest.raises(FactoryError, match="Database connection pool is not initialized"):
            await uow_factory.create_unit_of_work_async()

    def test_create_unit_of_work_connection_error(self, uow_factory, mock_connection_factory):
        """Test unit of work creation with connection error."""
        mock_connection_factory.get_connection.side_effect = Exception("Connection failed")

        with pytest.raises(FactoryError, match="Failed to create Unit of Work"):
            uow_factory.create_unit_of_work()

    async def test_create_unit_of_work_async_connection_error(
        self, uow_factory, mock_connection_factory
    ):
        """Test async unit of work creation with connection error."""
        mock_connection_factory.get_connection.side_effect = Exception("Connection failed")

        with pytest.raises(FactoryError, match="Failed to create Unit of Work"):
            await uow_factory.create_unit_of_work_async()


@pytest.mark.unit
class TestTransactionManager:
    """Test transaction manager."""

    @pytest.fixture
    def mock_uow_factory(self):
        """Mock unit of work factory."""
        factory = MagicMock()
        factory.create_unit_of_work = MagicMock()
        factory.create_unit_of_work_async = AsyncMock()
        return factory

    @pytest.fixture
    def transaction_manager(self, mock_uow_factory):
        """Transaction manager with mocked factory."""
        return PostgreSQLTransactionManager(mock_uow_factory)

    async def test_execute_in_transaction_success(self):
        """Test successful transaction execution with PostgreSQL factory."""
        mock_factory = MagicMock(spec=PostgreSQLUnitOfWorkFactory)
        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=None)
        mock_factory.create_unit_of_work_async = AsyncMock(return_value=mock_uow)

        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        async def operation(uow):
            return "success"

        result = await transaction_manager.execute_in_transaction(operation)

        assert result == "success"
        mock_factory.create_unit_of_work_async.assert_called_once()
        mock_uow.__aenter__.assert_called_once()
        mock_uow.__aexit__.assert_called_once()

    async def test_execute_in_transaction_error(self):
        """Test transaction execution with error."""
        mock_factory = MagicMock(spec=PostgreSQLUnitOfWorkFactory)
        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=None)
        mock_factory.create_unit_of_work_async = AsyncMock(return_value=mock_uow)

        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        async def operation(uow):
            raise ValueError("Operation failed")

        with pytest.raises(TransactionError, match="Transaction operation failed"):
            await transaction_manager.execute_in_transaction(operation)

    async def test_execute_with_retry_success(self):
        """Test transaction execution with retry on first failure."""
        mock_factory = MagicMock()
        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        call_count = 0

        async def operation(uow):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient error")
            return "success"

        # Mock execute_in_transaction to simulate retry behavior
        with patch.object(
            transaction_manager,
            "execute_in_transaction",
            side_effect=[TransactionError("Transient error"), "success"],
        ):
            result = await transaction_manager.execute_with_retry(
                operation, max_retries=1, backoff_factor=0.01
            )

        assert result == "success"

    async def test_execute_with_retry_all_fail(self):
        """Test transaction execution with all retries failing."""
        mock_factory = MagicMock()
        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        async def operation(uow):
            raise Exception("Persistent error")

        with patch.object(
            transaction_manager,
            "execute_in_transaction",
            side_effect=TransactionError("Persistent error"),
        ):
            with pytest.raises(TransactionError, match="Transaction failed after"):
                await transaction_manager.execute_with_retry(
                    operation, max_retries=2, backoff_factor=0.01
                )

    async def test_execute_batch_success(self):
        """Test batch transaction execution."""
        mock_factory = MagicMock()
        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        async def op1(uow):
            return "result1"

        async def op2(uow):
            return "result2"

        async def op3(uow):
            return "result3"

        operations = [op1, op2, op3]

        # Mock the individual execute_in_transaction call with the batch operation
        with patch.object(transaction_manager, "execute_in_transaction") as mock_execute:

            async def mock_batch_execution(batch_operation):
                # Simulate running the batch operation
                mock_uow = AsyncMock()
                return await batch_operation(mock_uow)

            mock_execute.side_effect = mock_batch_execution

            results = await transaction_manager.execute_batch(operations)

        assert len(results) == 3
        assert results == ["result1", "result2", "result3"]

        # Should have called execute_in_transaction once with the batch operation
        assert mock_execute.call_count == 1

    async def test_execute_batch_partial_failure(self):
        """Test batch execution with one operation failing."""
        mock_factory = MagicMock()
        transaction_manager = PostgreSQLTransactionManager(mock_factory)

        async def op1(uow):
            return "result1"

        async def op2(uow):
            raise ValueError("Operation 2 failed")

        async def op3(uow):
            return "result3"

        operations = [op1, op2, op3]

        # Mock execute_in_transaction to actually run the batch operation
        with patch.object(transaction_manager, "execute_in_transaction") as mock_execute:

            async def mock_batch_execution(batch_operation):
                mock_uow = AsyncMock()
                return await batch_operation(mock_uow)

            mock_execute.side_effect = mock_batch_execution

            with pytest.raises(TransactionError, match="Batch operation 1 failed"):
                await transaction_manager.execute_batch(operations)

    async def test_execute_in_transaction_non_postgresql_factory(self, transaction_manager):
        """Test transaction execution with non-PostgreSQL factory."""
        mock_factory = MagicMock()
        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=None)
        mock_factory.create_unit_of_work.return_value = mock_uow

        transaction_manager.factory = mock_factory

        async def operation(uow):
            return "success"

        result = await transaction_manager.execute_in_transaction(operation)

        assert result == "success"
        mock_factory.create_unit_of_work.assert_called_once()

    async def test_execute_in_transaction_postgresql_factory(self, transaction_manager):
        """Test transaction execution with PostgreSQL factory (async path)."""
        from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWorkFactory

        mock_factory = MagicMock(spec=PostgreSQLUnitOfWorkFactory)
        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=None)
        mock_factory.create_unit_of_work_async = AsyncMock(return_value=mock_uow)

        transaction_manager.factory = mock_factory

        async def operation(uow):
            return "async_success"

        result = await transaction_manager.execute_in_transaction(operation)

        assert result == "async_success"
        mock_factory.create_unit_of_work_async.assert_called_once()

    async def test_transaction_manager_initialization(self):
        """Test transaction manager initialization."""
        mock_factory = MagicMock()
        manager = PostgreSQLTransactionManager(mock_factory)
        assert manager.factory is mock_factory


@pytest.mark.unit
class TestUnitOfWorkIntegration:
    """Test integrated unit of work operations."""

    async def test_complete_transaction_workflow(self, unit_of_work, mock_adapter):
        """Test complete transaction workflow."""
        # Begin transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Perform operations
        assert await unit_of_work.is_active() is True

        # Flush changes
        await unit_of_work.flush()

        # Commit transaction
        await unit_of_work.commit()

        # Verify calls
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()

    async def test_transaction_with_error_and_rollback(self, unit_of_work, mock_adapter):
        """Test transaction with error and rollback."""
        # Begin transaction
        mock_adapter.has_active_transaction = False
        await unit_of_work.begin_transaction()
        mock_adapter.has_active_transaction = True

        # Simulate error and rollback
        await unit_of_work.rollback()

        # Verify calls
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_not_called()
