"""
Integration tests for Transaction Behavior.

Tests transaction management, ACID properties, and complex transaction scenarios
with real database connections.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
import os
from uuid import uuid4

# Third-party imports
import pytest

# Local imports
from src.domain.entities.order import Order, OrderSide, OrderStatus
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position, PositionSide
from src.infrastructure.database.connection import ConnectionFactory, DatabaseConfig
from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWork

# Skip integration tests if no database is available
SKIP_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
pytestmark = pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests require database")


@pytest.fixture(scope="session")
async def database_config():
    """Database configuration for transaction tests."""
    return DatabaseConfig(
        host=os.getenv("TEST_DATABASE_HOST", "localhost"),
        port=int(os.getenv("TEST_DATABASE_PORT", "5432")),
        database=os.getenv("TEST_DATABASE_NAME", "ai_trader_test"),
        user=os.getenv("TEST_DATABASE_USER", "zachwade"),
        password=os.getenv("TEST_DATABASE_PASSWORD", ""),
        min_pool_size=2,
        max_pool_size=10,
    )


@pytest.fixture(scope="session")
async def database_connection(database_config):
    """Database connection for transaction tests."""
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

    # Clean up test data
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'TX_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'TX_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'TX_%'")

    yield adapter

    # Clean up test data after test
    await adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'TX_%'")
    await adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'TX_%'")
    await adapter.execute_query("DELETE FROM portfolios WHERE name LIKE 'TX_%'")


@pytest.fixture
def unit_of_work(clean_database):
    """Unit of Work with real database connection."""
    return PostgreSQLUnitOfWork(clean_database)


@pytest.mark.integration
class TestTransactionACIDProperties:
    """Test ACID properties of transactions."""

    async def test_atomicity_all_or_nothing(self, unit_of_work):
        """Test atomicity - all operations succeed or all fail."""
        order1 = Order.create_limit_order(
            symbol="TX_ATOMICITY_1",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        order2 = Order.create_limit_order(
            symbol="TX_ATOMICITY_2",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("155.00"),
        )

        # Test successful transaction - all operations should succeed
        async with unit_of_work as uow:
            await uow.orders.save_order(order1)
            await uow.orders.save_order(order2)

        # Both orders should exist
        retrieved_order1 = await unit_of_work.orders.get_order_by_id(order1.id)
        retrieved_order2 = await unit_of_work.orders.get_order_by_id(order2.id)
        assert retrieved_order1 is not None
        assert retrieved_order2 is not None

        # Test failed transaction - no operations should persist
        order3 = Order.create_limit_order(
            symbol="TX_ATOMICITY_3",
            quantity=Decimal("75"),
            side=OrderSide.BUY,
            limit_price=Decimal("160.00"),
        )

        try:
            async with unit_of_work as uow:
                await uow.orders.save_order(order3)
                # Force an error after saving
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Order3 should not exist due to rollback
        retrieved_order3 = await unit_of_work.orders.get_order_by_id(order3.id)
        assert retrieved_order3 is None

    async def test_consistency_data_integrity(self, unit_of_work):
        """Test consistency - data integrity constraints are maintained."""
        portfolio = Portfolio(
            id=uuid4(),
            name="TX_Consistency_Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="consistency_test",
            created_at=datetime.now(UTC),
        )

        position = Position(
            id=uuid4(),
            symbol="TX_CONSISTENCY",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="consistency_test",
        )

        # Save related entities in transaction
        async with unit_of_work as uow:
            await uow.portfolios.save_portfolio(portfolio)
            await uow.positions.save_position(position)

        # Verify both entities exist and are consistent
        retrieved_portfolio = await unit_of_work.portfolios.get_portfolio_by_id(portfolio.id)
        retrieved_position = await unit_of_work.positions.get_position_by_id(position.id)

        assert retrieved_portfolio is not None
        assert retrieved_position is not None
        assert retrieved_portfolio.strategy == retrieved_position.strategy

    async def test_isolation_concurrent_transactions(self, database_connection):
        """Test isolation - concurrent transactions don't interfere."""
        symbol = "TX_ISOLATION"

        # Create two separate UoW instances for concurrent transactions
        uow1 = PostgreSQLUnitOfWork(database_connection)
        uow2 = PostgreSQLUnitOfWork(database_connection)

        # Create initial position
        initial_position = Position(
            id=uuid4(),
            symbol=symbol,
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="isolation_test",
        )

        await uow1.positions.save_position(initial_position)

        async def transaction1():
            """First transaction - update position price."""
            async with uow1 as uow:
                position = await uow.positions.get_position_by_id(initial_position.id)
                position.current_price = Decimal("160.00")
                await uow.positions.update_position(position)

                # Simulate some processing time
                await asyncio.sleep(0.1)
                return position

        async def transaction2():
            """Second transaction - update same position differently."""
            async with uow2 as uow:
                # Small delay to ensure transaction1 starts first
                await asyncio.sleep(0.05)

                position = await uow.positions.get_position_by_id(initial_position.id)
                position.current_price = Decimal("155.00")
                await uow.positions.update_position(position)
                return position

        # Run transactions concurrently
        results = await asyncio.gather(transaction1(), transaction2(), return_exceptions=True)

        # At least one should succeed (depending on isolation level)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 1

        # Final state should be consistent with one of the transactions
        final_position = await uow1.positions.get_position_by_id(initial_position.id)
        assert final_position.current_price in [Decimal("160.00"), Decimal("155.00")]

    async def test_durability_data_persists(self, unit_of_work):
        """Test durability - committed data persists across connections."""
        order = Order.create_limit_order(
            symbol="TX_DURABILITY",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        # Save in transaction and commit
        async with unit_of_work as uow:
            await uow.orders.save_order(order)

        # Create new UoW instance (simulating new connection/session)
        new_adapter = await ConnectionFactory.create_connection(unit_of_work.adapter._pool._config)
        new_uow = PostgreSQLUnitOfWork(new_adapter)

        try:
            # Data should still exist
            retrieved_order = await new_uow.orders.get_order_by_id(order.id)
            assert retrieved_order is not None
            assert retrieved_order.symbol == "TX_DURABILITY"
        finally:
            if hasattr(new_adapter, "_pool"):
                await new_adapter._pool.close()


@pytest.mark.integration
class TestComplexTransactionScenarios:
    """Test complex transaction scenarios."""

    async def test_multi_repository_transaction_success(self, unit_of_work):
        """Test successful transaction across multiple repositories."""
        # Create related entities
        portfolio = Portfolio(
            id=uuid4(),
            name="TX_Multi_Success_Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="multi_success",
            created_at=datetime.now(UTC),
        )

        order = Order.create_limit_order(
            symbol="TX_MULTI_SUCCESS",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        position = Position(
            id=uuid4(),
            symbol="TX_MULTI_SUCCESS",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            average_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
            opened_at=datetime.now(UTC),
            strategy="multi_success",
        )

        # Execute complex transaction
        async with unit_of_work as uow:
            # Save portfolio first
            await uow.portfolios.save_portfolio(portfolio)

            # Save and immediately update order
            await uow.orders.save_order(order)
            order.submit("BROKER123")
            await uow.orders.update_order(order)

            # Save position
            await uow.positions.save_position(position)

            # Update portfolio balance
            portfolio.cash_balance -= Decimal("15000.00")  # Order cost
            await uow.portfolios.update_portfolio(portfolio)

        # Verify all operations persisted
        retrieved_portfolio = await unit_of_work.portfolios.get_portfolio_by_id(portfolio.id)
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        retrieved_position = await unit_of_work.positions.get_position_by_id(position.id)

        assert retrieved_portfolio is not None
        assert retrieved_portfolio.cash_balance == Decimal("85000.00")

        assert retrieved_order is not None
        assert retrieved_order.status == OrderStatus.SUBMITTED
        assert retrieved_order.broker_order_id == "BROKER123"

        assert retrieved_position is not None
        assert retrieved_position.symbol == "TX_MULTI_SUCCESS"

    async def test_multi_repository_transaction_rollback(self, unit_of_work):
        """Test rollback across multiple repositories."""
        portfolio = Portfolio(
            id=uuid4(),
            name="TX_Multi_Rollback_Portfolio",
            cash_balance=Decimal("100000.00"),
            strategy="multi_rollback",
            created_at=datetime.now(UTC),
        )

        order = Order.create_limit_order(
            symbol="TX_MULTI_ROLLBACK",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        try:
            async with unit_of_work as uow:
                # Save portfolio
                await uow.portfolios.save_portfolio(portfolio)

                # Save order
                await uow.orders.save_order(order)

                # Simulate error condition
                raise ValueError("Simulated transaction failure")
        except ValueError:
            pass

        # Verify nothing was saved due to rollback
        retrieved_portfolio = await unit_of_work.portfolios.get_portfolio_by_id(portfolio.id)
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)

        assert retrieved_portfolio is None
        assert retrieved_order is None

    async def test_savepoint_like_behavior(self, unit_of_work):
        """Test savepoint-like behavior with nested operations."""
        # This test demonstrates how nested operations behave
        # Note: PostgreSQL adapter may not implement actual savepoints

        order1 = Order.create_limit_order(
            symbol="TX_SAVEPOINT_1",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        order2 = Order.create_limit_order(
            symbol="TX_SAVEPOINT_2",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("155.00"),
        )

        async with unit_of_work as uow:
            # Save first order
            await uow.orders.save_order(order1)

            try:
                # Attempt to save second order with potential failure
                await uow.orders.save_order(order2)

                # Simulate failure after second save
                raise ValueError("Simulated nested failure")
            except ValueError:
                # In a real savepoint implementation, we could rollback
                # just the nested operation. Here, the entire transaction
                # will be rolled back.
                raise

        # Current implementation: entire transaction rolls back
        retrieved_order1 = await unit_of_work.orders.get_order_by_id(order1.id)
        retrieved_order2 = await unit_of_work.orders.get_order_by_id(order2.id)

        # Both should be None due to complete rollback
        assert retrieved_order1 is None
        assert retrieved_order2 is None


@pytest.mark.integration
class TestTransactionPerformance:
    """Test transaction performance characteristics."""

    async def test_transaction_overhead(self, unit_of_work):
        """Test overhead of transaction management."""
        # Standard library imports
        import time

        orders = [
            Order.create_market_order(
                symbol=f"TX_PERF_{i}", quantity=Decimal("100"), side=OrderSide.BUY
            )
            for i in range(10)
        ]

        # Time transaction with multiple operations
        start_time = time.time()

        async with unit_of_work as uow:
            for order in orders:
                await uow.orders.save_order(order)

        transaction_time = time.time() - start_time

        # Time individual operations without explicit transaction
        start_time = time.time()

        individual_orders = [
            Order.create_market_order(
                symbol=f"TX_INDIVIDUAL_{i}", quantity=Decimal("100"), side=OrderSide.BUY
            )
            for i in range(10)
        ]

        for order in individual_orders:
            await unit_of_work.orders.save_order(order)

        individual_time = time.time() - start_time

        # Transaction should not add excessive overhead
        # (This is a rough heuristic - actual ratio depends on many factors)
        assert transaction_time < individual_time * 2.0

        # Verify all orders were saved
        for order in orders + individual_orders:
            retrieved = await unit_of_work.orders.get_order_by_id(order.id)
            assert retrieved is not None

    async def test_long_running_transaction(self, unit_of_work):
        """Test behavior of long-running transactions."""
        order = Order.create_limit_order(
            symbol="TX_LONG_RUNNING",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        async with unit_of_work as uow:
            # Start transaction
            await uow.orders.save_order(order)

            # Simulate long-running operation
            await asyncio.sleep(0.5)

            # Perform more operations
            order.submit("BROKER123")
            await uow.orders.update_order(order)

            # Another delay
            await asyncio.sleep(0.2)

            # Final operation
            position = Position(
                id=uuid4(),
                symbol="TX_LONG_RUNNING",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                average_price=Decimal("150.00"),
                current_price=Decimal("150.00"),
                opened_at=datetime.now(UTC),
                strategy="long_running_test",
            )
            await uow.positions.save_position(position)

        # Verify transaction completed successfully despite duration
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        retrieved_position = await unit_of_work.positions.get_position_by_id(position.id)

        assert retrieved_order is not None
        assert retrieved_order.status == OrderStatus.SUBMITTED
        assert retrieved_position is not None


@pytest.mark.integration
class TestTransactionErrorScenarios:
    """Test various transaction error scenarios."""

    async def test_connection_failure_during_transaction(self, unit_of_work):
        """Test behavior when connection fails during transaction."""
        # This test is conceptual - actual connection failure simulation
        # would require more sophisticated test setup

        order = Order.create_limit_order(
            symbol="TX_CONNECTION_FAIL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        async with unit_of_work as uow:
            await uow.orders.save_order(order)
            # In a real scenario, connection might fail here
            # The transaction should handle this gracefully

        # Verify order was saved (assuming connection was stable)
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        assert retrieved_order is not None

    async def test_deadlock_handling(self, database_connection):
        """Test deadlock detection and handling."""
        # Create two UoW instances for potential deadlock scenario
        uow1 = PostgreSQLUnitOfWork(database_connection)
        uow2 = PostgreSQLUnitOfWork(database_connection)

        # Create orders that might cause deadlocks if accessed in different orders
        order1 = Order.create_limit_order(
            symbol="TX_DEADLOCK_1",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        order2 = Order.create_limit_order(
            symbol="TX_DEADLOCK_2",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("155.00"),
        )

        # Save initial orders
        await uow1.orders.save_order(order1)
        await uow1.orders.save_order(order2)

        async def transaction_a():
            """Transaction A: Update order1 then order2."""
            async with uow1 as uow:
                # Get and update order1
                ord1 = await uow.orders.get_order_by_id(order1.id)
                ord1.submit("BROKER_A")
                await uow.orders.update_order(ord1)

                # Small delay to increase chance of deadlock
                await asyncio.sleep(0.1)

                # Get and update order2
                ord2 = await uow.orders.get_order_by_id(order2.id)
                ord2.submit("BROKER_A")
                await uow.orders.update_order(ord2)

        async def transaction_b():
            """Transaction B: Update order2 then order1."""
            async with uow2 as uow:
                # Small initial delay
                await asyncio.sleep(0.05)

                # Get and update order2
                ord2 = await uow.orders.get_order_by_id(order2.id)
                ord2.submit("BROKER_B")
                await uow.orders.update_order(ord2)

                # Small delay
                await asyncio.sleep(0.1)

                # Get and update order1
                ord1 = await uow.orders.get_order_by_id(order1.id)
                ord1.submit("BROKER_B")
                await uow.orders.update_order(ord1)

        # Run potentially deadlocking transactions
        results = await asyncio.gather(transaction_a(), transaction_b(), return_exceptions=True)

        # At least one transaction should complete successfully
        # (PostgreSQL will detect and resolve deadlocks)
        successful_count = sum(1 for r in results if not isinstance(r, Exception))
        assert successful_count >= 1

        # Verify final state is consistent
        final_order1 = await uow1.orders.get_order_by_id(order1.id)
        final_order2 = await uow1.orders.get_order_by_id(order2.id)

        assert final_order1.status == OrderStatus.SUBMITTED
        assert final_order2.status == OrderStatus.SUBMITTED
        assert final_order1.broker_order_id in ["BROKER_A", "BROKER_B"]
        assert final_order2.broker_order_id in ["BROKER_A", "BROKER_B"]

    async def test_transaction_timeout_behavior(self, unit_of_work):
        """Test behavior when transaction exceeds reasonable time limits."""
        # This test would require configuration of transaction timeouts
        # For now, we test that very long transactions can complete

        order = Order.create_limit_order(
            symbol="TX_TIMEOUT_TEST",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
        )

        async with unit_of_work as uow:
            await uow.orders.save_order(order)

            # Simulate operations that take time
            for i in range(5):
                await asyncio.sleep(0.1)
                order.tags[f"step_{i}"] = f"completed_at_{datetime.now(UTC).isoformat()}"
                await uow.orders.update_order(order)

        # Verify transaction completed despite duration
        retrieved_order = await unit_of_work.orders.get_order_by_id(order.id)
        assert retrieved_order is not None
        assert len(retrieved_order.tags) == 5
