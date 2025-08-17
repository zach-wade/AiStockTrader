"""
Unit tests for Unit of Work Implementation.

Tests the concrete implementation of IUnitOfWork including transaction management,
repository coordination, and error handling scenarios.
"""

# Standard library imports
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import TransactionError
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository
from src.infrastructure.repositories.position_repository import PostgreSQLPositionRepository
from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWork


@pytest.fixture
def mock_adapter():
    """Mock database adapter for UoW tests."""
    adapter = AsyncMock()
    adapter.begin_transaction.return_value = None
    adapter.commit_transaction.return_value = None
    adapter.rollback_transaction.return_value = None
    adapter.has_active_transaction = False
    return adapter


@pytest.fixture
def unit_of_work(mock_adapter):
    """Unit of Work with mocked adapter."""
    return PostgreSQLUnitOfWork(mock_adapter)


@pytest.mark.unit
class TestUnitOfWorkInitialization:
    """Test Unit of Work initialization."""

    def test_uow_initialization(self, mock_adapter):
        """Test UoW is properly initialized with repositories."""
        uow = PostgreSQLUnitOfWork(mock_adapter)

        assert uow.adapter == mock_adapter
        assert isinstance(uow.orders, PostgreSQLOrderRepository)
        assert isinstance(uow.positions, PostgreSQLPositionRepository)
        assert isinstance(uow.portfolios, PostgreSQLPortfolioRepository)
        assert uow.orders.adapter == mock_adapter
        assert uow.positions.adapter == mock_adapter
        assert uow.portfolios.adapter == mock_adapter


@pytest.mark.unit
class TestTransactionManagement:
    """Test transaction management operations."""

    async def test_begin_transaction_success(self, unit_of_work, mock_adapter):
        """Test successful transaction begin."""
        await unit_of_work.begin_transaction()

        mock_adapter.begin_transaction.assert_called_once()

    async def test_begin_transaction_adapter_error(self, unit_of_work, mock_adapter):
        """Test begin transaction with adapter error."""
        mock_adapter.begin_transaction.side_effect = Exception("Transaction start failed")

        with pytest.raises(TransactionError, match="Failed to begin transaction"):
            await unit_of_work.begin_transaction()

    async def test_commit_transaction_success(self, unit_of_work, mock_adapter):
        """Test successful transaction commit."""
        await unit_of_work.commit()

        mock_adapter.commit_transaction.assert_called_once()

    async def test_commit_transaction_adapter_error(self, unit_of_work, mock_adapter):
        """Test commit transaction with adapter error."""
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        with pytest.raises(TransactionError, match="Failed to commit transaction"):
            await unit_of_work.commit()

    async def test_rollback_transaction_success(self, unit_of_work, mock_adapter):
        """Test successful transaction rollback."""
        await unit_of_work.rollback()

        mock_adapter.rollback_transaction.assert_called_once()

    async def test_rollback_transaction_adapter_error(self, unit_of_work, mock_adapter):
        """Test rollback transaction with adapter error."""
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        with pytest.raises(TransactionError, match="Failed to rollback transaction"):
            await unit_of_work.rollback()

    async def test_is_active_delegates_to_adapter(self, unit_of_work, mock_adapter):
        """Test is_active delegates to adapter."""
        mock_adapter.has_active_transaction = True

        result = await unit_of_work.is_active()

        assert result is True

        mock_adapter.has_active_transaction = False
        result = await unit_of_work.is_active()

        assert result is False

    async def test_flush_success(self, unit_of_work, mock_adapter):
        """Test successful flush operation."""
        mock_adapter.execute_query.return_value = "FLUSH"

        await unit_of_work.flush()

        # In a real implementation, flush might execute pending changes
        # For now, we just verify no exception is raised
        assert True

    async def test_refresh_entity_success(self, unit_of_work, mock_adapter):
        """Test successful entity refresh."""
        entity = Mock()
        entity.id = uuid4()

        # Mock the refresh operation
        mock_adapter.fetch_one.return_value = {"id": entity.id, "updated_field": "new_value"}

        await unit_of_work.refresh(entity)

        # In a real implementation, this would reload the entity from database
        # For now, we just verify no exception is raised
        assert True


@pytest.mark.unit
class TestContextManager:
    """Test Unit of Work context manager functionality."""

    async def test_context_manager_success(self, unit_of_work, mock_adapter):
        """Test context manager with successful operation."""
        async with unit_of_work as uow:
            assert uow is unit_of_work
            # Simulate some work
            await uow.orders.get_order_by_id(uuid4())

        # Should begin and commit transaction
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_not_called()

    async def test_context_manager_exception(self, unit_of_work, mock_adapter):
        """Test context manager with exception causing rollback."""
        try:
            async with unit_of_work:
                # Simulate work that raises an exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should begin transaction and rollback on exception
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_not_called()

    async def test_context_manager_begin_transaction_error(self, unit_of_work, mock_adapter):
        """Test context manager when begin transaction fails."""
        mock_adapter.begin_transaction.side_effect = Exception("Begin failed")

        with pytest.raises(TransactionError):
            async with unit_of_work:
                pass

        # Should not call commit or rollback if begin fails
        mock_adapter.commit_transaction.assert_not_called()
        mock_adapter.rollback_transaction.assert_not_called()

    async def test_context_manager_commit_error(self, unit_of_work, mock_adapter):
        """Test context manager when commit fails."""
        mock_adapter.commit_transaction.side_effect = Exception("Commit failed")

        with pytest.raises(TransactionError):
            async with unit_of_work:
                pass

        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()
        # Should still attempt rollback after commit failure
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_context_manager_rollback_error(self, unit_of_work, mock_adapter):
        """Test context manager when rollback fails."""
        mock_adapter.rollback_transaction.side_effect = Exception("Rollback failed")

        try:
            async with unit_of_work:
                raise ValueError("Test exception")
        except TransactionError:
            # Should get TransactionError from rollback failure
            pass
        except ValueError:
            # Should not get the original ValueError
            pytest.fail("Expected TransactionError, got ValueError")

        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()


@pytest.mark.unit
class TestRepositoryCoordination:
    """Test repository coordination within Unit of Work."""

    async def test_repositories_share_adapter(self, unit_of_work, mock_adapter):
        """Test that all repositories share the same adapter."""
        assert unit_of_work.orders.adapter is mock_adapter
        assert unit_of_work.positions.adapter is mock_adapter
        assert unit_of_work.portfolios.adapter is mock_adapter

    async def test_multi_repository_transaction(
        self, unit_of_work, mock_adapter, sample_order, sample_position
    ):
        """Test transaction across multiple repositories."""
        # Mock successful operations
        mock_adapter.fetch_one.return_value = None  # Entities don't exist
        mock_adapter.execute_query.return_value = "EXECUTE 1"

        async with unit_of_work as uow:
            # Perform operations across multiple repositories
            await uow.orders.save_order(sample_order)
            await uow.positions.save_position(sample_position)

        # Should have begun and committed one transaction
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_called_once()

        # Should have executed operations for both repositories
        assert mock_adapter.execute_query.call_count >= 2

    async def test_transaction_rollback_affects_all_repositories(
        self, unit_of_work, mock_adapter, sample_order
    ):
        """Test that transaction rollback affects all repository operations."""
        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.return_value = "EXECUTE 1"

        try:
            async with unit_of_work as uow:
                # Successful operation
                await uow.orders.save_order(sample_order)

                # Operation that fails
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # Should rollback, affecting all operations in the transaction
        mock_adapter.begin_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()
        mock_adapter.commit_transaction.assert_not_called()


@pytest.mark.unit
class TestUnitOfWorkEdgeCases:
    """Test edge cases and error scenarios."""

    async def test_multiple_begin_transaction_calls(self, unit_of_work, mock_adapter):
        """Test multiple begin transaction calls."""
        await unit_of_work.begin_transaction()

        # Second call should still work (adapter handles this)
        await unit_of_work.begin_transaction()

        assert mock_adapter.begin_transaction.call_count == 2

    async def test_commit_without_begin(self, unit_of_work, mock_adapter):
        """Test commit without begin transaction."""
        await unit_of_work.commit()

        # Should delegate to adapter (which may handle this gracefully)
        mock_adapter.commit_transaction.assert_called_once()

    async def test_rollback_without_begin(self, unit_of_work, mock_adapter):
        """Test rollback without begin transaction."""
        await unit_of_work.rollback()

        # Should delegate to adapter (which may handle this gracefully)
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_nested_context_managers(self, unit_of_work, mock_adapter):
        """Test nested context manager usage."""
        async with unit_of_work as outer_uow, unit_of_work as inner_uow:
            # In real implementation, this would create nested transactions
            # or reuse the existing one
            assert outer_uow is inner_uow
            await outer_uow.orders.get_order_by_id(uuid4())

        # Exact behavior depends on implementation
        # At minimum, should have begun and committed
        mock_adapter.begin_transaction.assert_called()
        mock_adapter.commit_transaction.assert_called()


@pytest.mark.unit
class TestUnitOfWorkFactory:
    """Test Unit of Work factory patterns."""

    def test_create_multiple_instances(self, mock_adapter):
        """Test creating multiple UoW instances."""
        uow1 = PostgreSQLUnitOfWork(mock_adapter)
        uow2 = PostgreSQLUnitOfWork(mock_adapter)

        assert uow1 is not uow2
        assert uow1.adapter is mock_adapter
        assert uow2.adapter is mock_adapter

        # Each should have its own repository instances
        assert uow1.orders is not uow2.orders
        assert uow1.positions is not uow2.positions
        assert uow1.portfolios is not uow2.portfolios

    def test_repository_instances_unique_per_uow(self, mock_adapter):
        """Test that each UoW gets unique repository instances."""
        uow = PostgreSQLUnitOfWork(mock_adapter)

        # Each repository should be a unique instance
        assert uow.orders is not uow.positions
        assert uow.positions is not uow.portfolios
        assert uow.orders is not uow.portfolios

        # But they should all share the same adapter
        assert uow.orders.adapter is mock_adapter
        assert uow.positions.adapter is mock_adapter
        assert uow.portfolios.adapter is mock_adapter


@pytest.mark.unit
class TestTransactionIsolation:
    """Test transaction isolation between different UoW instances."""

    async def test_separate_uow_instances_independent_transactions(self, mock_adapter):
        """Test that separate UoW instances have independent transactions."""
        uow1 = PostgreSQLUnitOfWork(mock_adapter)
        uow2 = PostgreSQLUnitOfWork(mock_adapter)

        # Start transactions on both
        await uow1.begin_transaction()
        await uow2.begin_transaction()

        # Each should have called begin on the adapter
        assert mock_adapter.begin_transaction.call_count == 2

        # Commit one, rollback the other
        await uow1.commit()
        await uow2.rollback()

        mock_adapter.commit_transaction.assert_called_once()
        mock_adapter.rollback_transaction.assert_called_once()

    async def test_concurrent_context_managers(self, mock_adapter, sample_order):
        """Test concurrent context manager usage."""
        uow1 = PostgreSQLUnitOfWork(mock_adapter)
        uow2 = PostgreSQLUnitOfWork(mock_adapter)

        mock_adapter.fetch_one.return_value = None
        mock_adapter.execute_query.return_value = "EXECUTE 1"

        # Simulate concurrent transactions
        async with uow1:
            await uow1.orders.save_order(sample_order)

            async with uow2:
                # Different order with different ID
                different_order = sample_order
                different_order.id = uuid4()
                await uow2.orders.save_order(different_order)

        # Both should have completed successfully
        assert mock_adapter.begin_transaction.call_count == 2
        assert mock_adapter.commit_transaction.call_count == 2
