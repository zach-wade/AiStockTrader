"""Complete test coverage for PositionManager domain service.

This test suite focuses on testing the actual methods available in PositionManager.
"""

from decimal import Decimal

import pytest

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price


class TestPositionManagerCore:
    """Test core PositionManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_open_position_from_filled_buy_order(self):
        """Test opening a long position from a filled buy order."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

        position = self.manager.open_position(order)

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100  # Positive for long
        assert position.average_entry_price == Decimal("150.00")
        assert not position.is_closed()

    def test_open_position_from_filled_sell_order(self):
        """Test opening a short position from a filled sell order."""
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

        assert position.symbol == "AAPL"
        assert position.quantity == -100  # Negative for short
        assert position.average_entry_price == Decimal("150.00")
        assert position.is_short()

    def test_open_position_with_price_override(self):
        """Test opening position with price override."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

        override_price = Price(Decimal("151.00"))
        position = self.manager.open_position(order, override_price)

        assert position.average_entry_price == Decimal("151.00")

    def test_open_position_unfilled_order_error(self):
        """Test that unfilled orders cannot open positions."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),  # Required for limit orders
            status=OrderStatus.PENDING,
        )

        with pytest.raises(ValueError, match="Cannot open position from PENDING order"):
            self.manager.open_position(order)

    def test_open_position_zero_quantity_error(self):
        """Test that zero filled quantity raises error."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=0,
        )

        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            self.manager.open_position(order)

    def test_open_position_no_price_error(self):
        """Test that missing price raises error."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            # No average_fill_price
        )

        with pytest.raises(ValueError, match="No fill price available"):
            self.manager.open_position(order)


class TestPositionUpdate:
    """Test position update functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

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
        self.position = self.manager.open_position(initial_order)

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

        self.manager.update_position(self.position, add_order)

        assert self.position.quantity == 150
        # Weighted average: (100*150 + 50*155) / 150 = 151.67
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert self.position.average_entry_price == expected_avg

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

        self.manager.update_position(self.position, reduce_order)

        assert self.position.quantity == 70
        assert self.position.average_entry_price == Decimal("150.00")  # Entry price unchanged
        # P&L from partial close: 30 * (160 - 150) = 300
        assert self.position.realized_pnl == Decimal("300.00")

    def test_update_position_close_completely(self):
        """Test closing entire position."""
        close_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("160.00"),
        )

        self.manager.update_position(self.position, close_order)

        assert self.position.quantity == 0
        assert self.position.is_closed()
        assert self.position.realized_pnl == Decimal("1000.00")

    def test_update_position_mismatched_symbol(self):
        """Test updating with mismatched symbol."""
        wrong_order = Order(
            symbol="MSFT",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("300.00"),
        )

        with pytest.raises(ValueError, match="Symbol mismatch"):
            self.manager.update_position(self.position, wrong_order)

    def test_update_position_unfilled_order(self):
        """Test updating with unfilled order."""
        unfilled_order = Order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("155.00"),
            status=OrderStatus.PENDING,
        )

        with pytest.raises(ValueError, match="Cannot update position with PENDING order"):
            self.manager.update_position(self.position, unfilled_order)


class TestPositionAsync:
    """Test async position operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    @pytest.mark.asyncio
    async def test_open_position_async(self):
        """Test async position opening."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )

        position = await self.manager.open_position_async(order)

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert not position.is_closed()

    @pytest.mark.asyncio
    async def test_update_position_async(self):
        """Test async position update."""
        # Open initial position
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

        # Add to position
        add_order = Order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            average_fill_price=Decimal("155.00"),
        )

        await self.manager.update_position_async(position, add_order)

        assert position.quantity == 150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert position.average_entry_price == expected_avg

    @pytest.mark.asyncio
    async def test_close_position_async(self):
        """Test async position closing."""
        # Open position
        open_order = Order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            average_fill_price=Decimal("150.00"),
        )
        position = await self.manager.open_position_async(open_order)

        # Close position
        close_price = Price(Decimal("160.00"))
        pnl = await self.manager.close_position_async(
            position,
            close_price,
            Money(Decimal("5.00"), "USD"),  # Commission
        )

        assert position.is_closed()
        assert pnl.amount == Decimal("995.00")  # 1000 - 5 commission


class TestPnLCalculations:
    """Test P&L calculation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_calculate_pnl_long_profit(self):
        """Test P&L calculation for profitable long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("160.00"))
        pnl = self.manager.calculate_pnl(position, current_price)

        assert isinstance(pnl, Money)
        assert pnl.amount == Decimal("1000.00")  # 100 * (160 - 150)
        assert pnl.currency == "USD"

    def test_calculate_pnl_long_loss(self):
        """Test P&L calculation for losing long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("140.00"))
        pnl = self.manager.calculate_pnl(position, current_price)

        assert pnl.amount == Decimal("-1000.00")  # 100 * (140 - 150)

    def test_calculate_pnl_short_profit(self):
        """Test P&L calculation for profitable short position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=-100,
            entry_price=Decimal("150.00"),  # Short
        )

        current_price = Price(Decimal("140.00"))
        pnl = self.manager.calculate_pnl(position, current_price)

        assert pnl.amount == Decimal("1000.00")  # -100 * (140 - 150) = 1000

    def test_calculate_pnl_short_loss(self):
        """Test P&L calculation for losing short position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=-100,
            entry_price=Decimal("150.00"),  # Short
        )

        current_price = Price(Decimal("160.00"))
        pnl = self.manager.calculate_pnl(position, current_price)

        assert pnl.amount == Decimal("-1000.00")  # -100 * (160 - 150) = -1000

    def test_calculate_pnl_closed_position(self):
        """Test P&L for closed position uses realized P&L."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )
        position.realized_pnl = Decimal("500.00")
        position.quantity = Decimal("0")  # Mark as closed

        current_price = Price(Decimal("165.00"))
        pnl = self.manager.calculate_pnl(position, current_price)

        # Should use realized P&L, not current price
        assert pnl.amount == Decimal("500.00")

    @pytest.mark.asyncio
    async def test_calculate_pnl_async(self):
        """Test async P&L calculation."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("160.00"))
        pnl = await self.manager.calculate_pnl_async(position, current_price)

        assert pnl.amount == Decimal("1000.00")


class TestPositionMerging:
    """Test position merging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_merge_long_positions(self):
        """Test merging multiple long positions."""
        positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("AAPL", 50, Decimal("155.00")),
            Position.open_position("AAPL", 25, Decimal("152.00")),
        ]

        merged = self.manager.merge_positions(positions)

        assert merged is not None
        assert merged.symbol == "AAPL"
        assert merged.quantity == 175

        # Weighted average: (100*150 + 50*155 + 25*152) / 175
        expected_avg = (
            100 * Decimal("150.00") + 50 * Decimal("155.00") + 25 * Decimal("152.00")
        ) / 175
        assert merged.average_entry_price == expected_avg

    def test_merge_short_positions(self):
        """Test merging multiple short positions."""
        positions = [
            Position.open_position("AAPL", -100, Decimal("150.00")),
            Position.open_position("AAPL", -50, Decimal("145.00")),
        ]

        merged = self.manager.merge_positions(positions)

        assert merged.quantity == -150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("145.00")) / 150
        assert merged.average_entry_price == expected_avg

    def test_merge_mixed_positions(self):
        """Test merging long and short positions."""
        positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("AAPL", -30, Decimal("155.00")),
        ]

        merged = self.manager.merge_positions(positions)

        # Net position should be long 70
        assert merged.quantity == 70
        # When netting, use the dominant side's average price
        assert merged.average_entry_price == Decimal("150.00")

    def test_merge_positions_different_symbols(self):
        """Test that merging different symbols returns None."""
        positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("MSFT", 50, Decimal("300.00")),
        ]

        merged = self.manager.merge_positions(positions)
        assert merged is None

    def test_merge_empty_list(self):
        """Test merging empty list returns None."""
        merged = self.manager.merge_positions([])
        assert merged is None

    def test_merge_single_position(self):
        """Test merging single position returns copy."""
        position = Position.open_position("AAPL", 100, Decimal("150.00"))
        merged = self.manager.merge_positions([position])

        assert merged is not None
        assert merged.symbol == position.symbol
        assert merged.quantity == position.quantity
        assert merged.average_entry_price == position.average_entry_price


class TestPositionSizing:
    """Test position sizing calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_calculate_position_size_fixed_risk(self):
        """Test position sizing with fixed risk amount."""
        size = self.manager.calculate_position_size(
            account_value=Money(Decimal("100000"), "USD"),
            entry_price=Price(Decimal("50.00")),
            stop_loss=Price(Decimal("45.00")),
            risk_amount=Money(Decimal("1000"), "USD"),
        )

        # Risk per share: $5 (50 - 45)
        # Risk amount: $1000
        # Position size: 1000 / 5 = 200 shares
        assert size == 200

    def test_calculate_position_size_percentage_risk(self):
        """Test position sizing with percentage risk."""
        size = self.manager.calculate_position_size(
            account_value=Money(Decimal("100000"), "USD"),
            entry_price=Price(Decimal("100.00")),
            stop_loss=Price(Decimal("95.00")),
            risk_percentage=Decimal("0.02"),  # 2% risk
        )

        # Account value: $100,000
        # Risk amount: $2,000 (2% of 100,000)
        # Risk per share: $5 (100 - 95)
        # Position size: 2000 / 5 = 400 shares
        assert size == 400

    def test_calculate_position_size_max_position_limit(self):
        """Test position sizing with max position limit."""
        size = self.manager.calculate_position_size(
            account_value=Money(Decimal("100000"), "USD"),
            entry_price=Price(Decimal("10.00")),
            stop_loss=Price(Decimal("9.00")),
            risk_amount=Money(Decimal("5000"), "USD"),
            max_position_size=1000,
        )

        # Would be 5000 shares but limited to 1000
        assert size == 1000

    def test_calculate_position_size_insufficient_capital(self):
        """Test position sizing with insufficient capital."""
        size = self.manager.calculate_position_size(
            account_value=Money(Decimal("1000"), "USD"),
            entry_price=Price(Decimal("500.00")),
            stop_loss=Price(Decimal("490.00")),
            risk_amount=Money(Decimal("100"), "USD"),
        )

        # Can only afford 2 shares with $1000
        assert size == 2

    def test_calculate_position_size_no_parameters(self):
        """Test position sizing with missing parameters."""
        with pytest.raises(
            ValueError, match="Either risk_amount or risk_percentage must be provided"
        ):
            self.manager.calculate_position_size(
                account_value=Money(Decimal("100000"), "USD"),
                entry_price=Price(Decimal("100.00")),
                stop_loss=Price(Decimal("95.00")),
            )


class TestPositionClosing:
    """Test position closing operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionManager()

    def test_close_long_position_profit(self):
        """Test closing long position with profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        close_price = Price(Decimal("160.00"))
        commission = Money(Decimal("5.00"), "USD")

        pnl = self.manager.close_position(position, close_price, commission)

        assert position.is_closed()
        assert pnl.amount == Decimal("995.00")  # 1000 - 5 commission

    def test_close_short_position_loss(self):
        """Test closing short position with loss."""
        position = Position.open_position(
            symbol="AAPL", quantity=-100, entry_price=Decimal("150.00")
        )

        close_price = Price(Decimal("155.00"))
        commission = Money(Decimal("5.00"), "USD")

        pnl = self.manager.close_position(position, close_price, commission)

        assert position.is_closed()
        assert pnl.amount == Decimal("-505.00")  # -500 - 5 commission

    @pytest.mark.asyncio
    async def test_close_position_async(self):
        """Test async position closing."""
        position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )

        close_price = Price(Decimal("160.00"))
        pnl = await self.manager.close_position_async(position, close_price)

        assert position.is_closed()
        assert pnl.amount == Decimal("1000.00")
