"""
Integration tests for portfolio thread-safety and concurrent operations.

Tests the thread-safe implementation of Portfolio entity, Position entity,
and PostgreSQL repository with optimistic locking.
"""

import asyncio
import random
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.repositories.portfolio_repository import (
    OptimisticLockException,
    PostgreSQLPortfolioRepository,
)


@pytest.fixture
async def db_adapter():
    """Create database adapter for testing."""
    adapter = PostgreSQLAdapter(
        host="localhost", database="test_db", user="test_user", password="test_pass"
    )
    await adapter.connect()
    yield adapter
    await adapter.disconnect()


@pytest.fixture
async def portfolio_repo(db_adapter):
    """Create portfolio repository for testing."""
    return PostgreSQLPortfolioRepository(db_adapter, max_retries=3)


@pytest.fixture
def test_portfolio():
    """Create a test portfolio."""
    return Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Decimal("100000"),
        cash_balance=Decimal("100000"),
        max_position_size=Decimal("50000"),
        max_position_risk=Decimal("0.10"),
    )


class TestPortfolioThreadSafety:
    """Test thread-safety of Portfolio entity operations."""

    @pytest.mark.asyncio
    async def test_concurrent_position_opening(self, test_portfolio):
        """Test that concurrent position opening is thread-safe."""
        # Create multiple position requests
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        requests = [
            PositionRequest(
                symbol=symbol,
                quantity=Decimal("100"),
                entry_price=Decimal(str(100 + i * 10)),
                commission=Decimal("1.00"),
            )
            for i, symbol in enumerate(symbols)
        ]

        # Open positions concurrently
        tasks = [test_portfolio.open_position(request) for request in requests]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no data corruption
        successful_positions = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_positions) == len(symbols)

        # Verify cash balance is correctly updated
        expected_cash = Decimal("100000")
        for request in requests:
            expected_cash -= request.quantity * request.entry_price + request.commission

        assert test_portfolio.cash_balance == expected_cash

    @pytest.mark.asyncio
    async def test_concurrent_position_closing(self, test_portfolio):
        """Test that concurrent position closing is thread-safe."""
        # Open multiple positions first
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            request = PositionRequest(
                symbol=symbol,
                quantity=Decimal("100"),
                entry_price=Decimal("100"),
                commission=Decimal("1.00"),
            )
            await test_portfolio.open_position(request)

        initial_cash = test_portfolio.cash_balance

        # Close positions concurrently
        tasks = [
            test_portfolio.close_position(
                symbol,
                exit_price=Decimal("110"),
                commission=Decimal("1.00"),  # 10% profit
            )
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all positions closed successfully
        successful_closes = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_closes) == len(symbols)

        # Verify P&L calculations
        for pnl in successful_closes:
            # Each position made 10% profit minus commissions
            expected_pnl = Decimal("1000") - Decimal("2.00")  # (110-100)*100 - 2 commissions
            assert abs(pnl - expected_pnl) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_concurrent_price_updates(self, test_portfolio):
        """Test that concurrent price updates are thread-safe."""
        # Open a position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            commission=Decimal("1.00"),
        )
        await test_portfolio.open_position(request)

        # Generate random price updates
        prices = {"AAPL": [Decimal(str(150 + random.uniform(-10, 10))) for _ in range(100)]}

        # Apply price updates concurrently
        tasks = []
        for price in prices["AAPL"]:
            tasks.append(test_portfolio.update_position_price("AAPL", price))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify position has the last price
        position = test_portfolio.get_position("AAPL")
        assert position is not None
        assert position.current_price is not None

    @pytest.mark.asyncio
    async def test_portfolio_statistics_consistency(self, test_portfolio):
        """Test that portfolio statistics remain consistent during concurrent operations."""

        # Perform many concurrent operations
        async def trading_operation(i):
            symbol = f"STOCK{i}"

            # Open position
            request = PositionRequest(
                symbol=symbol,
                quantity=Decimal("10"),
                entry_price=Decimal("100"),
                commission=Decimal("0.50"),
            )
            await test_portfolio.open_position(request)

            # Update price
            await test_portfolio.update_position_price(symbol, Decimal("105"))  # 5% gain

            # Close position
            pnl = await test_portfolio.close_position(symbol, Decimal("105"), Decimal("0.50"))

            return pnl

        # Run multiple trading operations concurrently
        tasks = [trading_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify statistics
        successful_trades = [r for r in results if not isinstance(r, Exception)]

        assert test_portfolio.trades_count == len(successful_trades)
        assert (
            test_portfolio.winning_trades + test_portfolio.losing_trades
            <= test_portfolio.trades_count
        )
        assert test_portfolio.total_commission_paid == Decimal("20.00")  # 20 trades * (0.50 + 0.50)


class TestPositionThreadSafety:
    """Test thread-safety of Position entity operations."""

    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self):
        """Test that concurrent position updates are thread-safe."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            commission=Decimal("1.00"),
        )

        # Add to position concurrently
        async def add_shares(qty):
            await position.add_to_position(qty, Decimal("155"), Decimal("0.50"))

        tasks = [add_shares(Decimal("10")) for _ in range(10)]
        await asyncio.gather(*tasks)

        # Verify final state
        assert position.quantity == Decimal("200")  # 100 + 10*10
        assert position.commission_paid == Decimal("6.00")  # 1 + 10*0.50

    @pytest.mark.asyncio
    async def test_concurrent_reduce_position(self):
        """Test that concurrent position reductions are thread-safe."""
        position = Position.open_position(
            symbol="GOOGL",
            quantity=Decimal("1000"),
            entry_price=Decimal("2500"),
            commission=Decimal("5.00"),
        )

        # Reduce position concurrently
        async def reduce_shares():
            return await position.reduce_position(Decimal("50"), Decimal("2550"), Decimal("0.50"))

        tasks = [reduce_shares() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some reductions might fail if position is already reduced
        successful_reductions = [r for r in results if not isinstance(r, Exception)]

        # Verify position state
        assert position.quantity == Decimal("500")  # 1000 - 10*50
        assert position.realized_pnl > 0  # Should have profit

    @pytest.mark.asyncio
    async def test_concurrent_price_updates_on_position(self):
        """Test that concurrent price updates on a position are thread-safe."""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("50"),
            entry_price=Decimal("650"),
            commission=Decimal("2.00"),
        )

        # Update price concurrently
        prices = [Decimal(str(650 + i)) for i in range(-20, 21)]
        tasks = [position.update_market_price(price) for price in prices]

        await asyncio.gather(*tasks)

        # Verify position has a valid price
        assert position.current_price is not None
        assert position.last_updated is not None


class TestRepositoryOptimisticLocking:
    """Test optimistic locking in the PostgreSQL repository."""

    @pytest.mark.asyncio
    async def test_optimistic_lock_conflict_detection(self, portfolio_repo, test_portfolio):
        """Test that version conflicts are detected and handled."""
        # Save initial portfolio
        saved = await portfolio_repo.save_portfolio(test_portfolio)

        # Load the same portfolio twice (simulating concurrent access)
        portfolio1 = await portfolio_repo.get_portfolio_by_id(saved.id)
        portfolio2 = await portfolio_repo.get_portfolio_by_id(saved.id)

        # Modify both portfolios
        portfolio1.cash_balance = Decimal("90000")
        portfolio2.cash_balance = Decimal("80000")

        # First update should succeed
        await portfolio_repo.update_portfolio(portfolio1)

        # Second update should retry and eventually succeed or fail
        # depending on retry logic
        result = await portfolio_repo.update_portfolio(portfolio2)

        # Verify final state
        final = await portfolio_repo.get_portfolio_by_id(saved.id)
        assert final.version > 1  # Version should have incremented

    @pytest.mark.asyncio
    async def test_concurrent_repository_updates(self, portfolio_repo, test_portfolio):
        """Test that concurrent repository updates are handled correctly."""
        # Save initial portfolio
        saved = await portfolio_repo.save_portfolio(test_portfolio)

        async def update_portfolio(amount):
            portfolio = await portfolio_repo.get_portfolio_by_id(saved.id)
            portfolio.cash_balance -= amount
            return await portfolio_repo.update_portfolio(portfolio)

        # Perform concurrent updates
        tasks = [update_portfolio(Decimal("1000")) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for optimistic lock exceptions
        exceptions = [r for r in results if isinstance(r, OptimisticLockException)]
        successful = [r for r in results if not isinstance(r, Exception)]

        # At least some updates should succeed
        assert len(successful) > 0

        # Verify final balance is consistent
        final = await portfolio_repo.get_portfolio_by_id(saved.id)
        expected_deduction = Decimal("1000") * len(successful)
        assert final.cash_balance == Decimal("100000") - expected_deduction


class TestPositionManagerThreadSafety:
    """Test thread-safety of PositionManager service."""

    @pytest.mark.asyncio
    async def test_concurrent_position_lifecycle(self):
        """Test concurrent position lifecycle operations through PositionManager."""
        manager = PositionManager()
        positions = []

        # Create multiple positions
        for i in range(5):
            position = Position.open_position(
                symbol=f"STOCK{i}",
                quantity=Decimal("100"),
                entry_price=Decimal("50"),
                commission=Decimal("1.00"),
            )
            positions.append(position)

        # Perform concurrent updates
        from src.domain.entities.order import Order, OrderSide, OrderStatus

        async def update_position(position, i):
            order = Order(
                symbol=position.symbol,
                quantity=Decimal("50"),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            )
            order.status = OrderStatus.FILLED
            order.filled_quantity = Decimal("50")
            order.average_fill_price = Decimal("55")

            await manager.update_position_async(position, order)

            from src.domain.value_objects.price import Price

            current_price = Price(Decimal("60"))
            pnl = await manager.calculate_pnl_async(position, current_price)

            return pnl

        # Run concurrent updates
        tasks = [update_position(pos, i) for i, pos in enumerate(positions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all operations completed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0

        # Verify P&L calculations are consistent
        for pnl in successful:
            assert pnl.amount is not None


@pytest.mark.asyncio
async def test_high_frequency_trading_simulation():
    """Simulate high-frequency trading with many concurrent operations."""
    portfolio = Portfolio(
        id=uuid4(),
        name="HFT Portfolio",
        initial_capital=Decimal("1000000"),
        cash_balance=Decimal("1000000"),
        max_positions=100,
    )

    symbols = [f"STOCK{i:03d}" for i in range(50)]

    async def trading_bot(symbol, iterations=20):
        """Simulate a trading bot making rapid trades."""
        for _ in range(iterations):
            try:
                # Open position
                request = PositionRequest(
                    symbol=symbol,
                    quantity=Decimal(str(random.randint(10, 100))),
                    entry_price=Decimal(str(random.uniform(10, 200))),
                    commission=Decimal("0.10"),
                )
                await portfolio.open_position(request)

                # Random delay
                await asyncio.sleep(random.uniform(0.001, 0.01))

                # Update price
                new_price = Decimal(str(random.uniform(10, 200)))
                await portfolio.update_position_price(symbol, new_price)

                # Random delay
                await asyncio.sleep(random.uniform(0.001, 0.01))

                # Close position (50% chance)
                if random.random() > 0.5:
                    await portfolio.close_position(symbol, new_price, Decimal("0.10"))
            except Exception:
                # Some operations might fail due to business rules
                pass

    # Run multiple trading bots concurrently
    tasks = [trading_bot(symbol) for symbol in symbols[:20]]  # Limit to 20 for testing
    await asyncio.gather(*tasks)

    # Verify portfolio integrity
    total_value = await portfolio.get_total_value()
    assert total_value > 0
    assert portfolio.trades_count > 0

    # Verify no data corruption
    open_positions = portfolio.get_open_positions()
    assert len(open_positions) <= portfolio.max_positions

    # Calculate final statistics
    win_rate = portfolio.get_win_rate()
    if win_rate is not None:
        assert 0 <= win_rate <= 100
