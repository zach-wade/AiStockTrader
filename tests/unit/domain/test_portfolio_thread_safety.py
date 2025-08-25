"""
Comprehensive tests for thread-safe portfolio operations.

This test suite covers:
- Concurrent portfolio updates
- Optimistic locking mechanisms
- Race condition prevention
- Transaction isolation
- Deadlock prevention
- Atomic operations
- Version conflict resolution
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from src.domain.entities.order import Order, OrderSide
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.value_objects.money import Money
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository


class TestPortfolioThreadSafety:
    """Test thread-safe portfolio operations."""

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return Portfolio(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            name="Test Portfolio",
            initial_capital=Decimal("10000"),
            cash_balance=Decimal("10000"),
        )

    @pytest.fixture
    def position_manager(self):
        """Create position manager."""
        return PositionManager()

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository with thread-safe operations."""
        mock = AsyncMock(spec=PostgreSQLPortfolioRepository)
        mock.lock = threading.Lock()
        mock.versions = {}
        return mock

    def test_concurrent_cash_updates(self, portfolio):
        """Test concurrent updates to cash balance."""
        initial_balance = portfolio.cash_balance
        num_threads = 10
        deposit_amount = Decimal("100")
        lock = threading.Lock()

        def deposit_cash():
            # Simulate deposit operation
            with lock:
                current = portfolio.cash_balance
                time.sleep(0.001)  # Simulate processing
                portfolio.cash_balance = current + deposit_amount

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=deposit_cash)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All deposits should be accounted for
        expected_balance = initial_balance + (deposit_amount * num_threads)
        assert portfolio.cash_balance == expected_balance

    def test_concurrent_position_updates(self, portfolio, position_manager):
        """Test concurrent updates to positions."""
        # Add initial position
        position = Position(
            id=UUID("12345678-1234-5678-1234-567812345679"),
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
        )
        portfolio.positions["AAPL"] = position

        num_threads = 20
        results = []

        def update_position(thread_id):
            try:
                lock = threading.Lock()
                with lock:
                    # Find position
                    pos = portfolio.positions.get("AAPL")
                    if pos:
                        # Update quantity
                        new_qty = pos.quantity + Decimal(1)
                        pos.quantity = new_qty
                        results.append(("success", thread_id))
            except Exception as e:
                results.append(("error", str(e)))

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=update_position, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check all updates succeeded
        success_count = sum(1 for r in results if r[0] == "success")
        assert success_count == num_threads

        # Check final quantity
        final_position = portfolio.positions.get("AAPL")
        assert final_position.quantity == Decimal(100 + num_threads)

    @pytest.mark.asyncio
    async def test_optimistic_locking(self, portfolio, mock_repository):
        """Test optimistic locking for version control."""
        mock_repository.get.return_value = portfolio
        mock_repository.versions[portfolio.id] = 1

        async def update_with_version_check(portfolio_id, updates):
            # Get current portfolio
            current = await mock_repository.get(portfolio_id)

            # Check version
            current_version = mock_repository.versions.get(portfolio_id, 0)
            if current_version != mock_repository.versions[portfolio_id]:
                raise ValueError("Version conflict")

            # Apply updates
            for key, value in updates.items():
                setattr(current, key, value)

            # Increment version
            mock_repository.versions[portfolio_id] = (
                mock_repository.versions.get(portfolio_id, 0) + 1
            )

            return current

        # Simulate concurrent updates
        update1 = {"cash_balance": Decimal(9000)}
        update2 = {"cash_balance": Decimal(8000)}

        # First update should succeed
        result1 = await update_with_version_check(portfolio.id, update1)
        assert result1.version == 2

        # Second update with old version should fail
        mock_repository.versions[portfolio.id] = 2  # Simulate newer version in repository
        with pytest.raises(ValueError, match="Version conflict"):
            await update_with_version_check(portfolio.id, update2)

    def test_race_condition_prevention_order_execution(self, portfolio):
        """Test preventing race conditions during order execution."""
        portfolio.cash_balance = Decimal(1000)

        order1 = Order(
            id="order1", symbol="AAPL", side=OrderSide.BUY, quantity=10, price=Decimal(100)
        )

        order2 = Order(
            id="order2", symbol="AAPL", side=OrderSide.BUY, quantity=5, price=Decimal(100)
        )

        results = []

        def execute_order(order):
            lock = threading.Lock()
            with lock:
                required_cash = order.quantity * order.price

                if portfolio.cash_balance >= required_cash:
                    # Deduct cash
                    portfolio.cash_balance = Money(portfolio.cash_balance - required_cash, "USD")
                    results.append(("executed", order.id))
                else:
                    results.append(("insufficient_funds", order.id))

        # Execute orders concurrently
        thread1 = threading.Thread(target=execute_order, args=(order1,))
        thread2 = threading.Thread(target=execute_order, args=(order2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Only one order should execute (insufficient funds for both)
        executed = [r for r in results if r[0] == "executed"]
        assert len(executed) == 1
        assert portfolio.cash_balance == Decimal(0)

    def test_deadlock_prevention(self, portfolio):
        """Test deadlock prevention with multiple locks."""
        portfolio2 = Portfolio(cash_balance=Decimal(5000))

        completed = []

        def transfer_funds_1_to_2():
            # Always acquire locks in same order (by ID)
            locks = sorted([portfolio, portfolio2], key=lambda p: p.id)

            with locks[0].lock:
                with locks[1].lock:
                    # Transfer 1000 from portfolio to portfolio2
                    amount = Decimal(1000)
                    portfolio.cash_balance = Money(portfolio.cash_balance - amount, "USD")
                    portfolio2.cash_balance = Money(portfolio2.cash_balance + amount, "USD")
                    completed.append("transfer_1_to_2")

        def transfer_funds_2_to_1():
            # Always acquire locks in same order (by ID)
            locks = sorted([portfolio, portfolio2], key=lambda p: p.id)

            with locks[0].lock:
                with locks[1].lock:
                    # Transfer 500 from portfolio2 to portfolio
                    amount = Decimal(500)
                    portfolio2.cash_balance = Money(portfolio2.cash_balance - amount, "USD")
                    portfolio.cash_balance = Money(portfolio.cash_balance + amount, "USD")
                    completed.append("transfer_2_to_1")

        # Run transfers concurrently
        thread1 = threading.Thread(target=transfer_funds_1_to_2)
        thread2 = threading.Thread(target=transfer_funds_2_to_1)

        thread1.start()
        thread2.start()
        thread1.join(command_timeout=5)  # Timeout to detect deadlock
        thread2.join(command_timeout=5)

        # Both transfers should complete
        assert len(completed) == 2
        assert portfolio.cash_balance == Decimal(9500)  # 10000 - 1000 + 500
        assert portfolio2.cash_balance == Decimal(5500)  # 5000 + 1000 - 500

    def test_atomic_portfolio_rebalancing(self, portfolio):
        """Test atomic rebalancing operations."""
        # Add positions
        portfolio.positions = [
            Position("pos1", "AAPL", 100, Decimal(150)),
            Position("pos2", "GOOGL", 50, Decimal(2000)),
            Position("pos3", "MSFT", 75, Decimal(300)),
        ]

        def rebalance_portfolio(target_weights):
            lock = threading.Lock()
            with lock:
                # Calculate total value
                total_value = sum(p.quantity * p.average_price for p in portfolio.positions)

                # Adjust positions atomically
                for position in portfolio.positions:
                    target_value = total_value * target_weights.get(position.symbol, 0)
                    target_quantity = target_value / position.average_price
                    position.quantity = int(target_quantity)

        # Concurrent rebalancing attempts
        weights1 = {"AAPL": Decimal("0.4"), "GOOGL": Decimal("0.3"), "MSFT": Decimal("0.3")}
        weights2 = {"AAPL": Decimal("0.33"), "GOOGL": Decimal("0.33"), "MSFT": Decimal("0.34")}

        thread1 = threading.Thread(target=rebalance_portfolio, args=(weights1,))
        thread2 = threading.Thread(target=rebalance_portfolio, args=(weights2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Portfolio should be in consistent state
        total_after = sum(p.quantity * p.average_price for p in portfolio.positions)
        assert total_after > 0

    @pytest.mark.asyncio
    async def test_async_concurrent_updates(self, portfolio, mock_repository):
        """Test async concurrent portfolio updates."""
        mock_repository.get.return_value = portfolio

        async def async_update(update_id, amount):
            async with portfolio.async_lock:
                current = portfolio.cash_balance
                await asyncio.sleep(0.001)  # Simulate async I/O
                portfolio.cash_balance = Decimal(current + amount)
                return update_id

        # Run multiple async updates
        tasks = [async_update(i, Decimal(100)) for i in range(10)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert portfolio.cash_balance == Decimal(10000 + 1000)

    def test_concurrent_position_closing(self, portfolio):
        """Test concurrent closing of positions."""
        # Add multiple positions
        portfolio.positions = [
            Position(f"pos{i}", f"STOCK{i}", 100, Decimal(100)) for i in range(5)
        ]

        closed_positions = []

        def close_position(position_id):
            lock = threading.Lock()
            with lock:
                pos_index = None
                for i, pos in enumerate(portfolio.positions):
                    if pos.id == position_id:
                        pos_index = i
                        break

                if pos_index is not None:
                    closed = portfolio.positions.pop(pos_index)
                    # Add cash from sale
                    sale_value = closed.quantity * closed.average_price
                    portfolio.cash_balance = Money(portfolio.cash_balance + sale_value, "USD")
                    closed_positions.append(closed.id)

        # Close all positions concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(close_position, f"pos{i}") for i in range(5)]

            for future in as_completed(futures):
                future.result()

        # All positions should be closed
        assert len(portfolio.positions) == 0
        assert len(closed_positions) == 5
        assert portfolio.cash_balance == Decimal(60000)  # 10000 + (5 * 100 * 100)

    def test_stress_test_concurrent_operations(self, portfolio):
        """Stress test with many concurrent operations."""
        num_operations = 100
        operation_results = []

        def random_operation(op_id):
            import random

            operation_type = random.choice(
                ["deposit", "withdraw", "add_position", "remove_position"]
            )

            try:
                lock = threading.Lock()
                with lock:
                    if operation_type == "deposit":
                        amount = Decimal(random.randint(10, 100))
                        portfolio.cash_balance = Money(portfolio.cash_balance + amount, "USD")
                    elif operation_type == "withdraw":
                        amount = Decimal(random.randint(10, 50))
                        if portfolio.cash_balance >= amount:
                            portfolio.cash_balance = Money(portfolio.cash_balance - amount, "USD")
                    elif operation_type == "add_position":
                        position = Position(f"pos_{op_id}", f"STOCK_{op_id}", 10, Decimal(50))
                        portfolio.positions.append(position)
                    elif operation_type == "remove_position":
                        if portfolio.positions:
                            portfolio.positions.pop()

                    operation_results.append(("success", op_id))
            except Exception as e:
                operation_results.append(("error", str(e)))

        # Run stress test
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(random_operation, i) for i in range(num_operations)]

            for future in as_completed(futures):
                future.result()

        # Check results
        success_count = sum(1 for r in operation_results if r[0] == "success")
        assert success_count == num_operations

        # Portfolio should be in valid state
        assert portfolio.cash_balance >= 0
        assert isinstance(portfolio.positions, list)

    @pytest.mark.asyncio
    async def test_transaction_isolation_levels(self, portfolio, mock_repository):
        """Test different transaction isolation levels."""

        async def read_committed_transaction():
            """Simulates READ COMMITTED isolation."""
            async with mock_repository.transaction(isolation="READ_COMMITTED"):
                # Read portfolio
                p = await mock_repository.get(portfolio.id)
                initial_balance = p.cash_balance

                # Another transaction commits
                await asyncio.sleep(0.01)

                # Re-read shows committed changes
                p = await mock_repository.get(portfolio.id)
                assert p.cash_balance != initial_balance

        async def repeatable_read_transaction():
            """Simulates REPEATABLE READ isolation."""
            async with mock_repository.transaction(isolation="REPEATABLE_READ"):
                # Read portfolio
                p = await mock_repository.get(portfolio.id)
                initial_balance = p.cash_balance

                # Another transaction commits
                await asyncio.sleep(0.01)

                # Re-read shows same value (snapshot)
                p = await mock_repository.get(portfolio.id)
                assert p.cash_balance == initial_balance

        # Test both isolation levels
        mock_repository.transaction = AsyncMock()
        mock_repository.transaction.return_value.__aenter__ = AsyncMock()
        mock_repository.transaction.return_value.__aexit__ = AsyncMock()

        await read_committed_transaction()
        await repeatable_read_transaction()
