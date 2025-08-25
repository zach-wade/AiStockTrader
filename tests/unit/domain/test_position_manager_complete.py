"""Comprehensive unit tests for PositionManager service to achieve 80%+ coverage.

This module provides extensive test coverage for the PositionManager service,
testing position lifecycle management, P&L calculations, and position sizing.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.entities import Order, OrderSide, OrderStatus, Position
from src.domain.services.position_manager import PositionManager
from src.domain.value_objects import Money, Price, Quantity


class TestPositionManagerInitialization:
    """Test PositionManager initialization."""

    def test_position_manager_creation(self):
        """Test creating a PositionManager instance."""
        manager = PositionManager()
        assert manager is not None
        assert isinstance(manager, PositionManager)


class TestOpenPosition:
    """Test opening new positions."""

    def test_open_position_from_filled_buy_order(self):
        """Test opening long position from filled buy order."""
        manager = PositionManager()

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.50"),
        )
        order.tags["strategy"] = "momentum"

        position = manager.open_position(order)

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")  # Long position
        assert position.average_entry_price == Decimal("150.50")
        assert position.strategy == "momentum"
        assert position.is_long() is True

    def test_open_position_from_filled_sell_order(self):
        """Test opening short position from filled sell order."""
        manager = PositionManager()

        order = Order(
            symbol="TSLA",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("650.00"),
        )

        position = manager.open_position(order)

        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")  # Short position
        assert position.average_entry_price == Decimal("650.00")
        assert position.is_short() is True

    def test_open_position_with_override_price(self):
        """Test opening position with override fill price."""
        manager = PositionManager()

        order = Order(
            symbol="MSFT",
            quantity=Decimal("75"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("75"),
            average_fill_price=Decimal("350.00"),
        )

        override_price = Price(Decimal("349.50"))
        position = manager.open_position(order, override_price)

        assert position.average_entry_price == Decimal("349.50")  # Uses override

    def test_open_position_non_filled_order(self):
        """Test opening position from non-filled order raises error."""
        manager = PositionManager()

        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.SUBMITTED
        )

        with pytest.raises(ValueError, match="Cannot open position from SUBMITTED order"):
            manager.open_position(order)

    def test_open_position_zero_filled_quantity(self):
        """Test opening position with zero filled quantity raises error."""
        manager = PositionManager()

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0"),
        )

        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            manager.open_position(order)

    def test_open_position_no_fill_price(self):
        """Test opening position without fill price raises error."""
        manager = PositionManager()

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            # No average_fill_price set
        )

        with pytest.raises(ValueError, match="No fill price available"):
            manager.open_position(order)

    @pytest.mark.asyncio
    async def test_open_position_async(self):
        """Test async version of open_position."""
        manager = PositionManager()

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.00"),
        )

        position = await manager.open_position_async(order)

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")


class TestUpdatePosition:
    """Test updating existing positions."""

    def test_update_position_add_to_long(self):
        """Test adding to a long position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("155.00"),
        )

        manager.update_position(position, order)

        assert position.quantity == Decimal("150")
        # Average price: (100*150 + 50*155) / 150
        expected_avg = (Decimal("100") * Decimal("150") + Decimal("50") * Decimal("155")) / Decimal(
            "150"
        )
        assert position.average_entry_price == expected_avg

    def test_update_position_add_to_short(self):
        """Test adding to a short position."""
        manager = PositionManager()

        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        order = Order(
            symbol="TSLA",
            quantity=Decimal("30"),
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("30"),
            average_fill_price=Decimal("645.00"),
        )

        manager.update_position(position, order)

        assert position.quantity == Decimal("-80")

    def test_update_position_reduce_long(self):
        """Test reducing a long position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("40"),
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("40"),
            average_fill_price=Decimal("160.00"),
        )

        manager.update_position(position, order)

        assert position.quantity == Decimal("60")
        assert position.realized_pnl > Decimal("0")  # Profit from selling at higher price

    def test_update_position_reduce_short(self):
        """Test reducing a short position."""
        manager = PositionManager()

        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        order = Order(
            symbol="TSLA",
            quantity=Decimal("20"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("20"),
            average_fill_price=Decimal("640.00"),
        )

        manager.update_position(position, order)

        assert position.quantity == Decimal("-30")
        assert position.realized_pnl > Decimal("0")  # Profit from covering at lower price

    def test_update_position_partially_filled_order(self):
        """Test updating with partially filled order."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("30"),
            average_fill_price=Decimal("152.00"),
        )

        manager.update_position(position, order)

        assert position.quantity == Decimal("130")  # Only added filled quantity

    def test_update_position_with_override_price(self):
        """Test updating position with override price."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("155.00"),
        )

        override_price = Price(Decimal("154.50"))
        manager.update_position(position, order, override_price)

        # Should use override price for calculation
        expected_avg = (
            Decimal("100") * Decimal("150") + Decimal("50") * Decimal("154.50")
        ) / Decimal("150")
        assert position.average_entry_price == expected_avg

    def test_update_position_invalid_status(self):
        """Test updating with invalid order status raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL", quantity=Decimal("50"), side=OrderSide.BUY, status=OrderStatus.PENDING
        )

        with pytest.raises(ValueError, match="Cannot update position with PENDING order"):
            manager.update_position(position, order)

    def test_update_position_symbol_mismatch(self):
        """Test updating with mismatched symbol raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("350.00"),
        )

        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.update_position(position, order)

    def test_update_position_no_fill_price(self):
        """Test updating without fill price raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            # No average_fill_price
        )

        with pytest.raises(ValueError, match="No fill price available"):
            manager.update_position(position, order)

    @pytest.mark.asyncio
    async def test_update_position_async(self):
        """Test async version of update_position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("155.00"),
        )

        await manager.update_position_async(position, order)

        assert position.quantity == Decimal("150")


class TestClosePosition:
    """Test closing positions."""

    def test_close_long_position_profit(self):
        """Test closing long position with profit."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("160.00"),
        )

        pnl = manager.close_position(position, order)

        assert pnl == Decimal("1000")  # 100 * (160 - 150)
        assert position.is_closed() is True

    def test_close_short_position_loss(self):
        """Test closing short position with loss."""
        manager = PositionManager()

        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        order = Order(
            symbol="TSLA",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("660.00"),
        )

        pnl = manager.close_position(position, order)

        assert pnl == Decimal("-500")  # 50 * (650 - 660)
        assert position.is_closed() is True

    def test_close_position_with_override_price(self):
        """Test closing position with override exit price."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            average_fill_price=Decimal("160.00"),
        )

        override_price = Price(Decimal("159.50"))
        pnl = manager.close_position(position, order, override_price)

        assert pnl == Decimal("950")  # 100 * (159.50 - 150)

    def test_close_already_closed_position(self):
        """Test closing already closed position raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )

        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.SELL)

        with pytest.raises(ValueError, match="Position is already closed"):
            manager.close_position(position, order)

    def test_close_position_symbol_mismatch(self):
        """Test closing with mismatched symbol raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(symbol="MSFT", quantity=Decimal("100"), side=OrderSide.SELL)

        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.close_position(position, order)

    def test_close_position_no_exit_price(self):
        """Test closing without exit price raises error."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            # No average_fill_price
        )

        with pytest.raises(ValueError, match="No exit price available"):
            manager.close_position(position, order)

    @pytest.mark.asyncio
    async def test_close_position_async(self):
        """Test async version of close_position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            average_fill_price=Decimal("160.00"),
        )

        pnl = await manager.close_position_async(position, order)

        assert pnl == Decimal("1000")
        assert position.is_closed() is True


class TestCalculatePnL:
    """Test P&L calculation methods."""

    def test_calculate_pnl_open_long_position(self):
        """Test P&L calculation for open long position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("200"),
        )

        current_price = Price(Decimal("160.00"))
        pnl = manager.calculate_pnl(position, current_price)

        assert isinstance(pnl, Money)
        assert pnl.currency == "USD"
        # Total P&L: realized (200) + unrealized (1000)
        assert pnl.amount == Decimal("1200")

    def test_calculate_pnl_open_short_position(self):
        """Test P&L calculation for open short position."""
        manager = PositionManager()

        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        current_price = Price(Decimal("640.00"))
        pnl = manager.calculate_pnl(position, current_price)

        # Profit from short: 50 * (650 - 640) = 500
        assert pnl.amount == Decimal("500")

    def test_calculate_pnl_closed_position(self):
        """Test P&L calculation for closed position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("1000"),
            closed_at=datetime.now(UTC),
        )

        current_price = Price(Decimal("170.00"))
        pnl = manager.calculate_pnl(position, current_price)

        # Only realized P&L for closed position
        assert pnl.amount == Decimal("1000")

    @pytest.mark.asyncio
    async def test_calculate_pnl_async(self):
        """Test async version of calculate_pnl."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("160.00"))
        pnl = await manager.calculate_pnl_async(position, current_price)

        assert pnl.amount == Decimal("1000")


class TestMergePositions:
    """Test merging multiple positions."""

    def test_merge_positions_same_symbol(self):
        """Test merging positions of the same symbol."""
        manager = PositionManager()

        positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                average_entry_price=Decimal("150.00"),
                realized_pnl=Decimal("200"),
                commission_paid=Decimal("5"),
            ),
            Position(
                symbol="AAPL",
                quantity=Decimal("50"),
                average_entry_price=Decimal("155.00"),
                realized_pnl=Decimal("100"),
                commission_paid=Decimal("3"),
            ),
            Position(
                symbol="AAPL",
                quantity=Decimal("-30"),
                average_entry_price=Decimal("160.00"),
                realized_pnl=Decimal("-50"),
                commission_paid=Decimal("2"),
            ),
        ]

        merged = manager.merge_positions(positions)

        assert merged is not None
        assert merged.symbol == "AAPL"
        assert merged.quantity == Decimal("120")  # 100 + 50 - 30
        assert merged.realized_pnl == Decimal("250")  # 200 + 100 - 50
        assert merged.commission_paid == Decimal("10")  # 5 + 3 + 2

        # Check weighted average entry price
        # Total cost: 100*150 + 50*155 + 30*160 = 27550
        # Total absolute quantity: 100 + 50 + 30 = 180
        # But net quantity is 120, so we need abs quantities for average
        expected_avg = Decimal("27550") / Decimal("180")
        assert abs(merged.average_entry_price - expected_avg) < Decimal("0.01")

    def test_merge_positions_net_zero(self):
        """Test merging positions that net to zero."""
        manager = PositionManager()

        positions = [
            Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                average_entry_price=Decimal("150.00"),
                realized_pnl=Decimal("500"),
            ),
            Position(
                symbol="AAPL",
                quantity=Decimal("-100"),
                average_entry_price=Decimal("155.00"),
                realized_pnl=Decimal("300"),
            ),
        ]

        merged = manager.merge_positions(positions)

        assert merged.quantity == Decimal("0")
        assert merged.realized_pnl == Decimal("800")
        assert merged.closed_at is not None

    def test_merge_positions_single(self):
        """Test merging single position returns same position."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        merged = manager.merge_positions([position])

        assert merged == position

    def test_merge_positions_empty(self):
        """Test merging empty list returns None."""
        manager = PositionManager()

        merged = manager.merge_positions([])

        assert merged is None

    def test_merge_positions_different_symbols(self):
        """Test merging positions with different symbols raises error."""
        manager = PositionManager()

        positions = [
            Position(symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150")),
            Position(symbol="MSFT", quantity=Decimal("50"), average_entry_price=Decimal("350")),
        ]

        with pytest.raises(ValueError, match="Cannot merge positions with different symbols"):
            manager.merge_positions(positions)


class TestShouldClosePosition:
    """Test position closure evaluation."""

    def test_should_close_stop_loss_triggered(self):
        """Test stop loss trigger detection."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            stop_loss_price=Decimal("145.00"),
        )

        current_price = Price(Decimal("144.00"))
        should_close, reason = manager.should_close_position(position, current_price)

        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_take_profit_triggered(self):
        """Test take profit trigger detection."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            take_profit_price=Decimal("160.00"),
        )

        current_price = Price(Decimal("161.00"))
        should_close, reason = manager.should_close_position(position, current_price)

        assert should_close is True
        assert reason == "Take profit triggered"

    def test_should_close_max_loss_exceeded(self):
        """Test max loss threshold detection."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("140.00"))
        max_loss = Money(Decimal("500"), "USD")

        should_close, reason = manager.should_close_position(
            position, current_price, max_loss=max_loss
        )

        assert should_close is True
        assert "Max loss exceeded" in reason

    def test_should_close_target_profit_reached(self):
        """Test target profit detection."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("160.00"))
        target_profit = Money(Decimal("900"), "USD")

        should_close, reason = manager.should_close_position(
            position, current_price, target_profit=target_profit
        )

        assert should_close is True
        assert "Target profit reached" in reason

    def test_should_close_no_triggers(self):
        """Test no closure triggers."""
        manager = PositionManager()

        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        current_price = Price(Decimal("152.00"))

        should_close, reason = manager.should_close_position(position, current_price)

        assert should_close is False
        assert reason == ""


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_calculate_position_size_normal(self):
        """Test normal position size calculation."""
        manager = PositionManager()

        account_balance = Money(Decimal("10000"), "USD")
        risk_per_trade = Decimal("0.02")  # 2% risk
        entry_price = Price(Decimal("50.00"))
        stop_loss = Price(Decimal("48.00"))

        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss
        )

        assert isinstance(size, Quantity)
        # Risk amount: $10,000 * 0.02 = $200
        # Risk per share: $50 - $48 = $2
        # Position size: $200 / $2 = 100 shares
        assert size.value == Decimal("100")

    def test_calculate_position_size_fractional(self):
        """Test position size with fractional result (rounds down)."""
        manager = PositionManager()

        account_balance = Money(Decimal("10000"), "USD")
        risk_per_trade = Decimal("0.015")  # 1.5% risk
        entry_price = Price(Decimal("47.00"))
        stop_loss = Price(Decimal("45.00"))

        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss
        )

        # Risk amount: $10,000 * 0.015 = $150
        # Risk per share: $47 - $45 = $2
        # Position size: $150 / $2 = 75 shares
        assert size.value == Decimal("75")

    def test_calculate_position_size_tight_stop(self):
        """Test position size with very tight stop loss."""
        manager = PositionManager()

        account_balance = Money(Decimal("50000"), "USD")
        risk_per_trade = Decimal("0.01")  # 1% risk
        entry_price = Price(Decimal("100.00"))
        stop_loss = Price(Decimal("99.50"))  # Tight stop

        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss
        )

        # Risk amount: $50,000 * 0.01 = $500
        # Risk per share: $100 - $99.50 = $0.50
        # Position size: $500 / $0.50 = 1000 shares
        assert size.value == Decimal("1000")

    def test_calculate_position_size_invalid_risk(self):
        """Test position size with invalid risk percentage."""
        manager = PositionManager()

        account_balance = Money(Decimal("10000"), "USD")
        entry_price = Price(Decimal("50.00"))
        stop_loss = Price(Decimal("48.00"))

        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(account_balance, Decimal("1.5"), entry_price, stop_loss)

        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, Decimal("-0.02"), entry_price, stop_loss
            )

    def test_calculate_position_size_invalid_prices(self):
        """Test position size with invalid prices."""
        manager = PositionManager()

        account_balance = Money(Decimal("10000"), "USD")
        risk_per_trade = Decimal("0.02")

        # Negative price
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, Price(Decimal("-50.00")), Price(Decimal("48.00"))
            )

        # Zero stop loss
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, Price(Decimal("50.00")), Price(Decimal("0"))
            )

    def test_calculate_position_size_same_prices(self):
        """Test position size with same entry and stop prices."""
        manager = PositionManager()

        account_balance = Money(Decimal("10000"), "USD")
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("50.00"))
        stop_loss = Price(Decimal("50.00"))  # Same as entry

        with pytest.raises(ValueError, match="Entry and stop loss prices cannot be the same"):
            manager.calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss)
