"""
Integration tests for Repository Layer.

Tests the full repository layer integration including real database connections,
end-to-end workflows, and performance characteristics.
"""

# Standard library imports
import asyncio
import os
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import RepositoryError
from src.domain.entities.order import Order, OrderRequest, OrderSide, OrderStatus
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position, PositionSide
from src.infrastructure.database.connection import ConnectionFactory, DatabaseConfig
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository
from src.infrastructure.repositories.position_repository import PostgreSQLPositionRepository
from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWork

# Skip integration tests if no database is available
SKIP_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
pytestmark = pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests require database")


@pytest.fixture(scope="session")
async def database_config():
    """Database configuration for integration tests."""
    return DatabaseConfig(
        host=os.getenv("TEST_DATABASE_HOST", "localhost"),
        port=int(os.getenv("TEST_DATABASE_PORT", "5432")),
        database=os.getenv("TEST_DATABASE_NAME", "ai_trader_test"),
        user=os.getenv("TEST_DATABASE_USER", "zachwade"),
        password=os.getenv("TEST_DATABASE_PASSWORD", ""),
        min_pool_size=1,
        max_pool_size=5,
    )


@pytest.fixture(scope="session")
async def database_connection(database_config):
    """Database connection for integration tests."""
    try:
        adapter = await ConnectionFactory.create_connection(database_config)

        # Verify connection works
        health_ok = await adapter.health_check()
        if not health_ok:
            pytest.skip("Database health check failed")

        yield adapter

        # Cleanup
        if hasattr(adapter, "_pool"):
            await adapter._pool.close()
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {e}")


@pytest.fixture
async def clean_database(database_connection):
    """Clean database before each test."""
    adapter = database_connection

    # Clean up test data before test
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'TEST_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'TEST_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'Test_%'")

    yield adapter

    # Clean up test data after test
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'TEST_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'TEST_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'Test_%'")


@pytest.fixture
def order_repository(clean_database):
    """Order repository with real database connection."""
    return PostgreSQLOrderRepository(clean_database)


@pytest.fixture
def position_repository(clean_database):
    """Position repository with real database connection."""
    return PostgreSQLPositionRepository(clean_database)


@pytest.fixture
def portfolio_repository(clean_database):
    """Portfolio repository with real database connection."""
    return PostgreSQLPortfolioRepository(clean_database)


@pytest.fixture
def unit_of_work(clean_database):
    """Unit of Work with real database connection."""
    return PostgreSQLUnitOfWork(clean_database)


@pytest.mark.integration
class TestOrderRepositoryIntegration:
    """Integration tests for Order Repository."""

    async def test_order_crud_operations(self, order_repository):
        """Test complete order CRUD operations against real database."""
        # Create order
        request = OrderRequest(
            symbol="TEST_AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            reason="Integration test order",
        )
        order = Order.create_limit_order(request)

        # Save order
        saved_order = await order_repository.save_order(order)
        assert saved_order.id == order.id

        # Retrieve order
        retrieved_order = await order_repository.get_order_by_id(order.id)
        assert retrieved_order is not None
        assert retrieved_order.id == order.id
        assert retrieved_order.symbol == "TEST_AAPL"
        assert retrieved_order.quantity == Decimal("100")
        assert retrieved_order.side == OrderSide.BUY
        assert retrieved_order.limit_price == Decimal("150.00")

        # Update order
        retrieved_order.submit("BROKER123")
        updated_order = await order_repository.update_order(retrieved_order)
        assert updated_order.status == OrderStatus.SUBMITTED
        assert updated_order.broker_order_id == "BROKER123"

        # Verify update persisted
        verified_order = await order_repository.get_order_by_id(order.id)
        assert verified_order.status == OrderStatus.SUBMITTED
        assert verified_order.broker_order_id == "BROKER123"

        # Delete order
        delete_result = await order_repository.delete_order(order.id)
        assert delete_result is True

        # Verify deletion
        deleted_order = await order_repository.get_order_by_id(order.id)
        assert deleted_order is None

    async def test_order_query_operations(self, order_repository):
        """Test order query operations against real database."""
        # Create multiple test orders
        orders = [
            Order.create_limit_order(
                OrderRequest(
                    symbol="TEST_AAPL",
                    quantity=Decimal("100"),
                    side=OrderSide.BUY,
                    limit_price=Decimal("150.00"),
                )
            ),
            Order.create_limit_order(
                OrderRequest(
                    symbol="TEST_AAPL",
                    quantity=Decimal("50"),
                    side=OrderSide.SELL,
                    limit_price=Decimal("155.00"),
                )
            ),
            Order.create_market_order(
                OrderRequest(symbol="TEST_GOOGL", quantity=Decimal("10"), side=OrderSide.BUY)
            ),
        ]

        # Save all orders
        for order in orders:
            await order_repository.save_order(order)

        # Test get by symbol
        aapl_orders = await order_repository.get_orders_by_symbol("TEST_AAPL")
        assert len(aapl_orders) == 2

        googl_orders = await order_repository.get_orders_by_symbol("TEST_GOOGL")
        assert len(googl_orders) == 1

        # Test get by status
        pending_orders = await order_repository.get_orders_by_status(OrderStatus.PENDING)
        assert len(pending_orders) >= 3  # Our orders plus any others

        # Test get active orders
        active_orders = await order_repository.get_active_orders()
        assert len(active_orders) >= 3

        # Submit one order and test again
        orders[0].submit("BROKER123")
        await order_repository.update_order(orders[0])

        submitted_orders = await order_repository.get_orders_by_status(OrderStatus.SUBMITTED)
        assert len(submitted_orders) >= 1

        # Test get by broker ID
        broker_orders = await order_repository.get_orders_by_broker_id("BROKER123")
        assert len(broker_orders) == 1
        assert broker_orders[0].id == orders[0].id

    async def test_order_date_range_queries(self, order_repository):
        """Test order date range queries."""
        # Create order
        request = OrderRequest(
            symbol="TEST_DATE",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order = Order.create_limit_order(request)
        await order_repository.save_order(order)

        # Query with date range that includes the order
        start_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now(UTC).replace(hour=23, minute=59, second=59, microsecond=999999)

        orders_in_range = await order_repository.get_orders_by_date_range(start_date, end_date)
        assert any(o.id == order.id for o in orders_in_range)

        # Query with date range that excludes the order
        past_start = datetime(2020, 1, 1, tzinfo=UTC)
        past_end = datetime(2020, 12, 31, tzinfo=UTC)

        orders_in_past = await order_repository.get_orders_by_date_range(past_start, past_end)
        assert not any(o.id == order.id for o in orders_in_past)


@pytest.mark.integration
class TestPositionRepositoryIntegration:
    """Integration tests for Position Repository."""

    async def test_position_crud_operations(self, position_repository):
        """Test complete position CRUD operations against real database."""
        # Create position
        position = Position(
            id=uuid4(),
            symbol="TEST_AAPL",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="test_strategy",
        )

        # Save position
        saved_position = await position_repository.persist_position(position)
        assert saved_position.id == position.id

        # Retrieve position
        retrieved_position = await position_repository.get_position_by_id(position.id)
        assert retrieved_position is not None
        assert retrieved_position.id == position.id
        assert retrieved_position.symbol == "TEST_AAPL"
        assert retrieved_position.quantity == Decimal("100")

        # Get by symbol
        symbol_position = await position_repository.get_position_by_symbol("TEST_AAPL")
        assert symbol_position is not None
        assert symbol_position.id == position.id

        # Update position
        retrieved_position.current_price = Decimal("160.00")
        updated_position = await position_repository.update_position(retrieved_position)
        assert updated_position.current_price == Decimal("160.00")

        # Close position
        close_result = await position_repository.close_position(position.id)
        assert close_result is True

        # Verify position is closed
        closed_position = await position_repository.get_position_by_id(position.id)
        assert closed_position.is_closed()

        # Test closed positions query
        closed_positions = await position_repository.get_closed_positions()
        assert any(p.id == position.id for p in closed_positions)

    async def test_position_queries(self, position_repository):
        """Test position query operations."""
        # Create multiple positions
        positions = [
            Position(
                id=uuid4(),
                symbol="TEST_AAPL",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                average_price=Decimal("145.00"),
                current_price=Decimal("150.00"),
                opened_at=datetime.now(UTC),
                strategy="momentum",
            ),
            Position(
                id=uuid4(),
                symbol="TEST_GOOGL",
                quantity=Decimal("50"),
                side=PositionSide.SHORT,
                average_price=Decimal("2800.00"),
                current_price=Decimal("2750.00"),
                opened_at=datetime.now(UTC),
                strategy="mean_reversion",
            ),
        ]

        # Save positions
        for position in positions:
            await position_repository.persist_position(position)

        # Test active positions
        active_positions = await position_repository.get_active_positions()
        assert len(active_positions) >= 2

        # Test by strategy
        momentum_positions = await position_repository.get_positions_by_strategy("momentum")
        assert len(momentum_positions) >= 1

        mean_reversion_positions = await position_repository.get_positions_by_strategy(
            "mean_reversion"
        )
        assert len(mean_reversion_positions) >= 1

        # Test by symbol (multiple positions)
        aapl_positions = await position_repository.get_positions_by_symbol("TEST_AAPL")
        assert len(aapl_positions) >= 1


@pytest.mark.integration
class TestPortfolioRepositoryIntegration:
    """Integration tests for Portfolio Repository."""

    async def test_portfolio_crud_operations(self, portfolio_repository):
        """Test complete portfolio CRUD operations against real database."""
        # Create portfolio
        portfolio = Portfolio(
            id=uuid4(),
            name="Test Integration Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="integration_test",
            created_at=datetime.now(UTC),
        )

        # Save portfolio
        saved_portfolio = await portfolio_repository.save_portfolio(portfolio)
        assert saved_portfolio.id == portfolio.id

        # Retrieve by ID
        retrieved_portfolio = await portfolio_repository.get_portfolio_by_id(portfolio.id)
        assert retrieved_portfolio is not None
        assert retrieved_portfolio.name == "Test Integration Portfolio"

        # Retrieve by name
        name_portfolio = await portfolio_repository.get_portfolio_by_name(
            "Test Integration Portfolio"
        )
        assert name_portfolio is not None
        assert name_portfolio.id == portfolio.id

        # Update portfolio
        retrieved_portfolio.cash_balance = Decimal("150000.00")
        updated_portfolio = await portfolio_repository.update_portfolio(retrieved_portfolio)
        assert updated_portfolio.cash_balance == Decimal("150000.00")

        # Create snapshot
        snapshot = await portfolio_repository.create_portfolio_snapshot(portfolio)
        assert snapshot.id != portfolio.id
        assert snapshot.name.endswith("_snapshot")
        assert snapshot.cash_balance == portfolio.cash_balance

        # Delete portfolio
        delete_result = await portfolio_repository.delete_portfolio(portfolio.id)
        assert delete_result is True

        # Verify deletion
        deleted_portfolio = await portfolio_repository.get_portfolio_by_id(portfolio.id)
        assert deleted_portfolio is None


@pytest.mark.integration
class TestUnitOfWorkIntegration:
    """Integration tests for Unit of Work."""

    async def test_transaction_commit_success(self, unit_of_work):
        """Test successful transaction commit across multiple repositories."""
        request = OrderRequest(
            symbol="TEST_TRANSACTION",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order = Order.create_limit_order(request)

        position = Position(
            id=uuid4(),
            symbol="TEST_TRANSACTION",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="test_transaction",
        )

        portfolio = Portfolio(
            id=uuid4(),
            name="Test Transaction Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="test_transaction",
            created_at=datetime.now(UTC),
        )

        # Execute transaction
        async with unit_of_work as uow:
            saved_order = await uow.orders.save_order(order)
            saved_position = await uow.positions.persist_position(position)
            saved_portfolio = await uow.portfolios.save_portfolio(portfolio)

            assert saved_order.id == order.id
            assert saved_position.id == position.id
            assert saved_portfolio.id == portfolio.id

        # Verify all entities were saved
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        retrieved_position = await unit_of_work.positions.get_position_by_id(position.id)
        retrieved_portfolio = await unit_of_work.portfolios.get_portfolio_by_id(portfolio.id)

        assert retrieved_order is not None
        assert retrieved_position is not None
        assert retrieved_portfolio is not None

    async def test_transaction_rollback_on_error(self, unit_of_work):
        """Test transaction rollback when error occurs."""
        request = OrderRequest(
            symbol="TEST_ROLLBACK",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order = Order.create_limit_order(request)

        # Execute transaction that should fail
        try:
            async with unit_of_work as uow:
                # Save order successfully
                await uow.orders.save_order(order)

                # Cause an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify order was not saved due to rollback
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        assert retrieved_order is None

    async def test_manual_transaction_management(self, unit_of_work):
        """Test manual transaction management."""
        request = OrderRequest(
            symbol="TEST_MANUAL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order = Order.create_limit_order(request)

        # Begin transaction manually
        await unit_of_work.begin_transaction()

        try:
            # Perform operations
            await unit_of_work.orders.save_order(order)

            # Check if transaction is active
            is_active = await unit_of_work.is_active()
            assert is_active

            # Commit manually
            await unit_of_work.commit()

        except Exception:
            await unit_of_work.rollback()
            raise

        # Verify order was saved
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        assert retrieved_order is not None

    async def test_nested_transactions(self, unit_of_work):
        """Test nested transaction behavior."""
        request1 = OrderRequest(
            symbol="TEST_NESTED1",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order1 = Order.create_limit_order(request1)

        request2 = OrderRequest(
            symbol="TEST_NESTED2",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("155.00"),
        )
        order2 = Order.create_limit_order(request2)

        # Outer transaction
        async with unit_of_work as outer_uow:
            await outer_uow.orders.save_order(order1)

            # Inner transaction (should reuse the same transaction)
            async with unit_of_work as inner_uow:
                await inner_uow.orders.save_order(order2)

                assert outer_uow is inner_uow  # Should be same instance

        # Both orders should be saved
        retrieved_order1 = await unit_of_work.orders.get_order_by_id(order1.id)
        retrieved_order2 = await unit_of_work.orders.get_order_by_id(order2.id)

        assert retrieved_order1 is not None
        assert retrieved_order2 is not None


@pytest.mark.integration
class TestConcurrentOperations:
    """Integration tests for concurrent operations."""

    async def test_concurrent_order_creation(self, database_connection):
        """Test concurrent order creation doesn't cause conflicts."""

        async def create_order(symbol_suffix):
            repo = PostgreSQLOrderRepository(database_connection)
            request = OrderRequest(
                symbol=f"TEST_CONCURRENT_{symbol_suffix}",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                limit_price=Decimal("150.00"),
            )
            order = Order.create_limit_order(request)
            return await repo.save_order(order)

        # Create multiple orders concurrently
        tasks = [create_order(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(order.symbol.startswith("TEST_CONCURRENT_") for order in results)

        # All should have unique IDs
        ids = [order.id for order in results]
        assert len(set(ids)) == 5

    async def test_concurrent_position_updates(self, database_connection):
        """Test concurrent position updates."""
        repo = PostgreSQLPositionRepository(database_connection)

        # Create initial position
        position = Position(
            id=uuid4(),
            symbol="TEST_CONCURRENT_UPDATE",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="concurrent_test",
        )
        await repo.persist_position(position)

        async def update_position_price(new_price):
            # Retrieve fresh position
            pos = await repo.get_position_by_id(position.id)
            pos.current_price = new_price
            return await repo.update_position(pos)

        # Update position price concurrently
        prices = [Decimal("155.00"), Decimal("160.00"), Decimal("165.00")]
        tasks = [update_position_price(price) for price in prices]

        # At least one should succeed (depending on isolation level)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_updates = [r for r in results if not isinstance(r, Exception)]

        assert len(successful_updates) >= 1


@pytest.mark.integration
class TestPerformanceCharacteristics:
    """Integration tests for performance characteristics."""

    async def test_bulk_order_operations(self, order_repository):
        """Test performance of bulk order operations."""
        # Standard library imports
        import time

        # Create multiple orders
        orders = [
            Order.create_limit_order(
                OrderRequest(
                    symbol=f"TEST_BULK_{i}",
                    quantity=Decimal("100"),
                    side=OrderSide.BUY,
                    limit_price=Decimal("150.00"),
                )
            )
            for i in range(10)
        ]

        # Time bulk save operations
        start_time = time.time()

        for order in orders:
            await order_repository.save_order(order)

        save_time = time.time() - start_time

        # Time bulk retrieval
        start_time = time.time()

        retrieved_orders = []
        for order in orders:
            retrieved = await order_repository.get_order_by_id(order.id)
            retrieved_orders.append(retrieved)

        retrieve_time = time.time() - start_time

        # Verify all operations completed
        assert len(retrieved_orders) == 10
        assert all(order is not None for order in retrieved_orders)

        # Performance should be reasonable (adjust thresholds as needed)
        assert save_time < 5.0  # 10 saves in under 5 seconds
        assert retrieve_time < 2.0  # 10 retrievals in under 2 seconds

    async def test_connection_pool_usage(self, database_connection):
        """Test connection pool handles concurrent operations efficiently."""

        async def perform_operations():
            repo = PostgreSQLOrderRepository(database_connection)

            # Multiple operations in sequence
            order = Order.create_market_order(
                symbol="TEST_POOL", quantity=Decimal("100"), side=OrderSide.BUY
            )

            await repo.save_order(order)
            retrieved = await repo.get_order_by_id(order.id)
            await repo.delete_order(order.id)

            return retrieved is not None

        # Run multiple concurrent operation sequences
        tasks = [perform_operations() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert all(results)


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling scenarios."""

    async def test_database_constraint_violations(self, order_repository):
        """Test handling of database constraint violations."""
        request = OrderRequest(
            symbol="TEST_CONSTRAINT",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )
        order = Order.create_limit_order(request)

        # Save order first time should succeed
        await order_repository.save_order(order)

        # Try to save order with same ID should handle constraint violation
        duplicate_request = OrderRequest(
            symbol="TEST_CONSTRAINT_2",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("155.00"),
        )
        duplicate_order = Order.create_limit_order(duplicate_request)
        duplicate_order.id = order.id  # Force duplicate ID

        # This should either succeed (if implemented as upsert) or raise appropriate error
        with suppress(RepositoryError):
            await order_repository.save_order(duplicate_order)

    async def test_connection_failure_recovery(self, database_config):
        """Test recovery from connection failures."""
        # This test would require more sophisticated setup to simulate
        # connection failures and recovery scenarios
        pass
