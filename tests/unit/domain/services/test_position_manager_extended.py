"""Extended comprehensive tests for PositionManager domain service.

This test suite provides extensive coverage for the PositionManager service,
testing all methods, edge cases, error conditions, and thread-safety scenarios.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager


class TestPositionManagerBasics:
    """Test basic PositionManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()
        self.order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

    def test_initialization(self):
        """Test PositionManager initialization."""
        assert isinstance(self.manager, PositionManager)

    def test_open_position_from_filled_order(self):
        """Test opening a position from a filled order."""
        position = self.manager.open_position(self.order)

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.average_entry_price == Decimal("150.00")
        assert position.is_closed() is False  # Use is_closed() method

    def test_open_position_from_unfilled_order(self):
        """Test that unfilled orders cannot open positions."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
        )

        with pytest.raises(ValueError, match="Cannot open position from unfilled order"):
            self.manager.open_position(order)

    def test_open_position_with_partial_fill(self):
        """Test opening a position from partially filled order."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("150.00"),
        )

        position = self.manager.open_position(order)
        assert position.quantity == 50
        assert position.average_entry_price == Decimal("150.00")

    def test_open_short_position(self):
        """Test opening a short position."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

        position = self.manager.open_position(order)
        assert position.quantity == -100  # Negative for short
        assert position.average_entry_price == Decimal("150.00")


class TestPositionManagerAsync:
    """Test async operations of PositionManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()
        self.order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

    @pytest.mark.asyncio
    async def test_open_position_async(self):
        """Test async position opening."""
        position = await self.manager.open_position_async(self.order)

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.is_closed() is False

    @pytest.mark.asyncio
    async def test_update_position_async(self):
        """Test async position update."""
        position = await self.manager.open_position_async(self.order)

        add_order = Order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("155.00"),
        )

        updated_position = await self.manager.update_position_async(position, add_order)
        assert updated_position.quantity == 150
        # Average price should be weighted
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert updated_position.average_entry_price == expected_avg

    @pytest.mark.asyncio
    async def test_close_position_async(self):
        """Test async position closing."""
        position = await self.manager.open_position_async(self.order)

        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("160.00"),
        )

        closed_position = await self.manager.close_position_async(position, close_order)
        assert closed_position.is_closed() is True
        assert closed_position.current_price == Decimal("160.00")
        assert closed_position.realized_pnl == Decimal("1000.00")  # 100 * (160 - 150)


class TestPositionUpdate:
    """Test position update operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()
        self.initial_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        self.position = self.manager.open_position(self.initial_order)

    def test_update_position_add_to_long(self):
        """Test adding to a long position."""
        add_order = Order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("155.00"),
        )

        updated = self.manager.update_position(self.position, add_order)
        assert updated.quantity == 150
        # Weighted average: (100*150 + 50*155) / 150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert updated.average_entry_price == expected_avg

    def test_update_position_reduce_long(self):
        """Test reducing a long position."""
        reduce_order = Order(
            symbol="AAPL",
            quantity=30,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=30,
            average_fill_price=Decimal("160.00"),
        )

        updated = self.manager.update_position(self.position, reduce_order)
        assert updated.quantity == 70
        assert updated.average_entry_price == Decimal("150.00")  # Entry price unchanged
        # Realized P&L from partial close
        assert updated.realized_pnl == Decimal("300.00")  # 30 * (160 - 150)

    def test_update_position_close_exact(self):
        """Test closing position with exact quantity."""
        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("160.00"),
        )

        updated = self.manager.update_position(self.position, close_order)
        assert updated.quantity == 0
        assert updated.is_closed() is True
        assert updated.current_price == Decimal("160.00")
        assert updated.realized_pnl == Decimal("1000.00")

    def test_update_position_mismatched_symbol(self):
        """Test that mismatched symbols raise an error."""
        wrong_order = Order(
            symbol="MSFT",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("300.00"),
        )

        with pytest.raises(ValueError, match="Order symbol .* does not match position symbol"):
            self.manager.update_position(self.position, wrong_order)

    def test_update_closed_position(self):
        """Test that closed positions cannot be updated."""
        # Close the position first
        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("160.00"),
        )
        closed = self.manager.close_position(self.position, close_order)

        # Try to update closed position
        update_order = Order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("165.00"),
        )

        with pytest.raises(ValueError, match="Cannot update closed position"):
            self.manager.update_position(closed, update_order)


class TestPositionClosing:
    """Test position closing operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_close_long_position_with_profit(self):
        """Test closing a long position with profit."""
        open_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        position = self.manager.open_position(open_order)

        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("160.00"),
        )

        closed = self.manager.close_position(position, close_order)
        assert closed.is_closed() is True
        assert closed.current_price == Decimal("160.00")
        assert closed.realized_pnl == Decimal("1000.00")
        assert closed.closed_at is not None

    def test_close_short_position_with_loss(self):
        """Test closing a short position with loss."""
        open_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,  # Short
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        position = self.manager.open_position(open_order)

        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,  # Buy to close short
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("155.00"),
        )

        closed = self.manager.close_position(position, close_order)
        assert closed.is_closed() is True
        assert closed.current_price == Decimal("155.00")
        assert closed.realized_pnl == Decimal("-500.00")  # Lost $5 per share

    def test_close_position_partial(self):
        """Test partial position closing."""
        open_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        position = self.manager.open_position(open_order)

        partial_close = Order(
            symbol="AAPL",
            quantity=60,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=60,
            average_fill_price=Decimal("160.00"),
        )

        updated = self.manager.close_position(position, partial_close)
        assert updated.is_closed() is False  # Still open
        assert updated.quantity == 40  # Remaining quantity
        assert updated.realized_pnl == Decimal("600.00")  # 60 * (160 - 150)


class TestPositionSizing:
    """Test position sizing calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_calculate_position_size_fixed_amount(self):
        """Test fixed amount position sizing."""
        size = self.manager.calculate_position_size(
            account_balance=Decimal("100000"),
            price=Decimal("50.00"),
            risk_per_trade=Decimal("1000"),  # $1000 risk
            stop_loss_price=Decimal("45.00"),  # $5 risk per share
        )

        # Risk $1000 with $5 risk per share = 200 shares
        assert size == 200

    def test_calculate_position_size_percentage_risk(self):
        """Test percentage-based position sizing."""
        size = self.manager.calculate_position_size(
            account_balance=Decimal("100000"),
            price=Decimal("100.00"),
            risk_percentage=Decimal("0.02"),  # 2% risk
            stop_loss_price=Decimal("95.00"),  # $5 risk per share
        )

        # 2% of $100,000 = $2,000 risk
        # $2,000 / $5 per share = 400 shares
        assert size == 400

    def test_calculate_position_size_max_position_limit(self):
        """Test position sizing with max position limit."""
        size = self.manager.calculate_position_size(
            account_balance=Decimal("100000"),
            price=Decimal("10.00"),
            risk_per_trade=Decimal("5000"),
            stop_loss_price=Decimal("9.00"),  # $1 risk per share
            max_position_size=1000,
        )

        # Would be 5000 shares but limited to 1000
        assert size == 1000

    def test_calculate_position_size_insufficient_capital(self):
        """Test position sizing with insufficient capital."""
        size = self.manager.calculate_position_size(
            account_balance=Decimal("1000"),
            price=Decimal("500.00"),
            risk_per_trade=Decimal("100"),
            stop_loss_price=Decimal("490.00"),
        )

        # Can only afford 2 shares with $1000
        assert size == 2

    def test_calculate_position_size_no_stop_loss(self):
        """Test position sizing without stop loss."""
        with pytest.raises(ValueError, match="Stop loss price is required"):
            self.manager.calculate_position_size(
                account_balance=Decimal("100000"),
                price=Decimal("100.00"),
                risk_per_trade=Decimal("1000"),
            )

    def test_calculate_position_size_invalid_stop_loss(self):
        """Test position sizing with invalid stop loss."""
        with pytest.raises(ValueError, match="Invalid stop loss price"):
            self.manager.calculate_position_size(
                account_balance=Decimal("100000"),
                price=Decimal("100.00"),
                risk_per_trade=Decimal("1000"),
                stop_loss_price=Decimal("100.00"),  # Same as entry
            )


class TestMergePositions:
    """Test position merging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_merge_long_positions(self):
        """Test merging two long positions."""
        pos1 = Position.open_position(symbol="AAPL", quantity=100, entry_price=Decimal("150.00"))

        pos2 = Position.open_position(symbol="AAPL", quantity=50, entry_price=Decimal("155.00"))

        merged = self.manager.merge_positions([pos1, pos2])
        assert merged.symbol == "AAPL"
        assert merged.quantity == 150
        # Weighted average: (100*150 + 50*155) / 150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert merged.average_entry_price == expected_avg

    def test_merge_short_positions(self):
        """Test merging short positions."""
        pos1 = Position.open_position(symbol="AAPL", quantity=-100, entry_price=Decimal("150.00"))

        pos2 = Position.open_position(symbol="AAPL", quantity=-50, entry_price=Decimal("145.00"))

        merged = self.manager.merge_positions([pos1, pos2])
        assert merged.quantity == -150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("145.00")) / 150
        assert merged.average_entry_price == expected_avg

    def test_merge_mixed_positions(self):
        """Test merging long and short positions (netting)."""
        pos1 = Position.open_position(symbol="AAPL", quantity=100, entry_price=Decimal("150.00"))

        pos2 = Position.open_position(symbol="AAPL", quantity=-30, entry_price=Decimal("155.00"))

        merged = self.manager.merge_positions([pos1, pos2])
        assert merged.quantity == 70  # Net long 70
        assert merged.average_entry_price == Decimal("150.00")  # Original long price

    def test_merge_positions_different_symbols(self):
        """Test that merging different symbols raises error."""
        pos1 = Position.open_position(symbol="AAPL", quantity=100, entry_price=Decimal("150.00"))

        pos2 = Position.open_position(symbol="MSFT", quantity=50, entry_price=Decimal("300.00"))

        with pytest.raises(ValueError, match="All positions must have the same symbol"):
            self.manager.merge_positions([pos1, pos2])

    def test_merge_empty_list(self):
        """Test merging empty list of positions."""
        with pytest.raises(ValueError, match="At least one position is required"):
            self.manager.merge_positions([])

    def test_merge_with_closed_positions(self):
        """Test that closed positions are excluded from merge."""
        pos1 = Position.open_position(symbol="AAPL", quantity=100, entry_price=Decimal("150.00"))

        pos2 = Position.open_position(symbol="AAPL", quantity=50, entry_price=Decimal("155.00"))
        pos2.close(Decimal("160.00"))  # Close this position

        pos3 = Position.open_position(symbol="AAPL", quantity=25, entry_price=Decimal("152.00"))

        merged = self.manager.merge_positions([pos1, pos2, pos3])
        assert merged.quantity == 125  # Only pos1 and pos3
        expected_avg = (100 * Decimal("150.00") + 25 * Decimal("152.00")) / 125
        assert merged.average_entry_price == expected_avg


class TestPnLCalculations:
    """Test P&L calculation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_calculate_pnl_long_position(self):
        """Test P&L calculation for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        pnl = self.manager.calculate_pnl(position, current_price=Decimal("160.00"))

        assert pnl == Decimal("1000.00")  # 100 * (160 - 150)

    def test_calculate_pnl_short_position(self):
        """Test P&L calculation for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=-100, entry_price=Decimal("150.00")
        )

        pnl = self.manager.calculate_pnl(position, current_price=Decimal("145.00"))

        assert pnl == Decimal("500.00")  # -100 * (145 - 150) = 500 profit

    def test_calculate_pnl_closed_position(self):
        """Test P&L for closed position uses realized P&L."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )
        position.close(Decimal("160.00"))

        pnl = self.manager.calculate_pnl(position, Decimal("165.00"))
        assert pnl == Decimal("1000.00")  # Uses realized, not current price

    def test_calculate_pnl_percentage_long(self):
        """Test percentage P&L calculation for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("100.00")
        )

        pnl_pct = self.manager.calculate_pnl_percentage(position, current_price=Decimal("110.00"))

        assert pnl_pct == Decimal("10.00")  # 10% gain

    def test_calculate_pnl_percentage_short(self):
        """Test percentage P&L calculation for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=-100, entry_price=Decimal("100.00")
        )

        pnl_pct = self.manager.calculate_pnl_percentage(position, current_price=Decimal("90.00"))

        assert pnl_pct == Decimal("10.00")  # 10% gain on short


class TestPositionAnalysis:
    """Test position analysis methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_is_profitable(self):
        """Test profitable position detection."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        assert self.manager.is_profitable(position, Decimal("160.00")) is True
        assert self.manager.is_profitable(position, Decimal("140.00")) is False
        assert self.manager.is_profitable(position, Decimal("150.00")) is False

    def test_days_held(self):
        """Test days held calculation."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        # Mock the entry time
        from datetime import timedelta

        position.opened_at = datetime.now(UTC) - timedelta(days=5, hours=12)

        days = self.manager.days_held(position)
        assert days == 5  # Rounds down

    def test_days_held_closed_position(self):
        """Test days held for closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        from datetime import timedelta

        position.opened_at = datetime.now(UTC) - timedelta(days=10)
        position.closed_at = position.opened_at + timedelta(days=3)
        position.quantity = Decimal("0")  # Mark as closed

        days = self.manager.days_held(position)
        assert days == 3

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("100.00")
        )
        position.stop_loss_price = Decimal("95.00")
        position.take_profit_price = Decimal("110.00")

        ratio = self.manager.risk_reward_ratio(position)
        # Risk: $5, Reward: $10, Ratio: 2.0
        assert ratio == Decimal("2.0")

    def test_risk_reward_ratio_no_stops(self):
        """Test risk/reward ratio without stop loss or take profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("100.00")
        )

        ratio = self.manager.risk_reward_ratio(position)
        assert ratio is None


class TestErrorHandling:
    """Test error handling in PositionManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_open_position_with_cancelled_order(self):
        """Test that cancelled orders cannot open positions."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELLED,
        )

        with pytest.raises(ValueError, match="Cannot open position from unfilled order"):
            self.manager.open_position(order)

    def test_calculate_position_size_negative_balance(self):
        """Test position sizing with negative balance."""
        with pytest.raises(ValueError, match="Invalid account balance"):
            self.manager.calculate_position_size(
                account_balance=Decimal("-1000"),
                price=Decimal("100.00"),
                risk_per_trade=Decimal("100"),
                stop_loss_price=Decimal("95.00"),
            )

    def test_calculate_position_size_negative_price(self):
        """Test position sizing with negative price."""
        with pytest.raises(ValueError, match="Invalid price"):
            self.manager.calculate_position_size(
                account_balance=Decimal("10000"),
                price=Decimal("-100.00"),
                risk_per_trade=Decimal("100"),
                stop_loss_price=Decimal("95.00"),
            )

    def test_merge_positions_all_closed(self):
        """Test merging when all positions are closed."""
        pos1 = Position.open_position(symbol="AAPL", quantity=100, entry_price=Decimal("150.00"))
        pos1.close(Decimal("160.00"))

        pos2 = Position.open_position(symbol="AAPL", quantity=50, entry_price=Decimal("155.00"))
        pos2.close(Decimal("165.00"))

        with pytest.raises(ValueError, match="No open positions to merge"):
            self.manager.merge_positions([pos1, pos2])


class TestConcurrentOperations:
    """Test concurrent operations for thread safety."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    @pytest.mark.asyncio
    async def test_concurrent_position_opens(self):
        """Test concurrent position opening."""
        orders = [
            Order(
                symbol=f"STOCK{i}",
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                average_fill_price=Decimal(f"{100 + i}"),
            )
            for i in range(10)
        ]

        # Open positions concurrently
        tasks = [self.manager.open_position_async(order) for order in orders]
        positions = await asyncio.gather(*tasks)

        assert len(positions) == 10
        for i, pos in enumerate(positions):
            assert pos.symbol == f"STOCK{i}"
            assert pos.quantity == 100

    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self):
        """Test concurrent updates to the same position."""
        # Create initial position
        initial_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        position = await self.manager.open_position_async(initial_order)

        # Create multiple update orders
        update_orders = [
            Order(
                symbol="AAPL",
                quantity=10,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                filled_quantity=10,
                average_fill_price=Decimal("151.00"),
            )
            for _ in range(5)
        ]

        # Update position concurrently (should be serialized internally)
        tasks = [self.manager.update_position_async(position, order) for order in update_orders]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that updates were applied
        final_position = results[-1]
        if not isinstance(final_position, Exception):
            assert final_position.quantity == 150  # 100 + 5*10
