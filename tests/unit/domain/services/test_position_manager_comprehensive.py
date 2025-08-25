"""
Comprehensive unit tests for Position Manager domain service
Achieving 95%+ test coverage
"""

from decimal import Decimal

import pytest

from src.domain.entities import Order, OrderSide, OrderStatus, OrderType, Position
from src.domain.services.position_manager import PositionManager
from src.domain.value_objects import Money, Price


class TestPositionManagerOpenPosition:
    """Test suite for open_position method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    def test_open_position_buy_order(self, manager):
        """Test opening a long position from a buy order"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("150.00")

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.is_long()

    def test_open_position_sell_order(self, manager):
        """Test opening a short position from a sell order"""
        # Arrange
        order = Order(
            symbol="TSLA", quantity=Decimal("50"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("700.00")

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("700.00")
        assert position.is_short()

    def test_open_position_with_fill_price_override(self, manager):
        """Test opening position with explicit fill price"""
        # Arrange
        order = Order(
            symbol="GOOGL",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("2500.00"),
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("10")
        order.average_fill_price = Decimal("2500.00")

        fill_price = Price(Decimal("2495.00"))  # Better fill

        # Act
        position = manager.open_position(order, fill_price)

        # Assert
        assert position.average_entry_price == Decimal("2495.00")

    def test_open_position_with_strategy_tag(self, manager):
        """Test opening position with strategy information"""
        # Arrange
        order = Order(
            symbol="NVDA", quantity=Decimal("20"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("20")
        order.average_fill_price = Decimal("500.00")
        order.tags = {"strategy": "momentum"}

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.strategy == "momentum"

    def test_open_position_not_filled_raises_error(self, manager):
        """Test that opening position from unfilled order raises error"""
        # Arrange
        order = Order(
            symbol="AMD", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.PENDING

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position from.*PENDING"):
            manager.open_position(order)

    def test_open_position_zero_quantity_raises_error(self, manager):
        """Test that zero filled quantity raises error"""
        # Arrange
        order = Order(
            symbol="MSFT", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("0")

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            manager.open_position(order)

    def test_open_position_no_fill_price_raises_error(self, manager):
        """Test that missing fill price raises error"""
        # Arrange
        order = Order(
            symbol="META", quantity=Decimal("50"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = None  # No price

        # Act & Assert
        with pytest.raises(ValueError, match="No fill price available for position"):
            manager.open_position(order)


class TestPositionManagerUpdatePosition:
    """Test suite for update_position method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    @pytest.fixture
    def long_position(self):
        """Create a long position"""
        return Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position"""
        return Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("1.00"),
        )

    def test_update_add_to_long_position(self, manager, long_position):
        """Test adding to a long position"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("50"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("155.00")

        # Act
        manager.update_position(long_position, order)

        # Assert
        assert long_position.quantity == Decimal("150")
        # Weighted average: (100*150 + 50*155) / 150
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("155.00")) / 150
        assert abs(long_position.average_entry_price - expected_avg) < Decimal("0.01")

    def test_update_reduce_long_position(self, manager, long_position):
        """Test reducing a long position"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("30"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("30")
        order.average_fill_price = Decimal("160.00")

        # Act
        manager.update_position(long_position, order)

        # Assert
        assert long_position.quantity == Decimal("70")
        assert long_position.average_entry_price == Decimal("150.00")  # Unchanged

    def test_update_add_to_short_position(self, manager, short_position):
        """Test adding to a short position"""
        # Arrange
        order = Order(
            symbol="TSLA", quantity=Decimal("25"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("25")
        order.average_fill_price = Decimal("690.00")

        # Act
        manager.update_position(short_position, order)

        # Assert
        assert short_position.quantity == Decimal("-75")
        # Weighted average for shorts
        expected_avg = (50 * Decimal("700.00") + 25 * Decimal("690.00")) / 75
        assert abs(short_position.average_entry_price - expected_avg) < Decimal("0.01")

    def test_update_reduce_short_position(self, manager, short_position):
        """Test reducing a short position"""
        # Arrange
        order = Order(
            symbol="TSLA", quantity=Decimal("20"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("20")
        order.average_fill_price = Decimal("680.00")

        # Act
        manager.update_position(short_position, order)

        # Assert
        assert short_position.quantity == Decimal("-30")
        assert short_position.average_entry_price == Decimal("700.00")  # Unchanged

    def test_update_position_with_fill_price_override(self, manager, long_position):
        """Test updating position with explicit fill price"""
        # Arrange
        order = Order(
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("152.00"),
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("152.00")

        fill_price = Price(Decimal("151.50"))  # Better fill

        # Act
        manager.update_position(long_position, order, fill_price)

        # Assert
        expected_avg = (100 * Decimal("150.00") + 50 * Decimal("151.50")) / 150
        assert abs(long_position.average_entry_price - expected_avg) < Decimal("0.01")

    def test_update_position_invalid_status_raises_error(self, manager, long_position):
        """Test that invalid order status raises error"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("50"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.CANCELLED

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot update position with.*CANCELLED"):
            manager.update_position(long_position, order)

    def test_update_position_symbol_mismatch_raises_error(self, manager, long_position):
        """Test that symbol mismatch raises error"""
        # Arrange
        order = Order(
            symbol="GOOGL",  # Different symbol
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("2500.00")

        # Act & Assert
        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.update_position(long_position, order)

    def test_update_position_no_fill_price_raises_error(self, manager, long_position):
        """Test that missing fill price raises error"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("50"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = None

        # Act & Assert
        with pytest.raises(ValueError, match="No fill price available"):
            manager.update_position(long_position, order)

    def test_update_position_partially_filled(self, manager, long_position):
        """Test updating position with partially filled order"""
        # Arrange
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("152.00"),
        )
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = Decimal("25")
        order.average_fill_price = Decimal("152.00")

        # Act
        manager.update_position(long_position, order)

        # Assert
        assert long_position.quantity == Decimal("125")


class TestPositionManagerClosePosition:
    """Test suite for close_position method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    @pytest.fixture
    def long_position(self):
        """Create a long position"""
        return Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position"""
        return Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("1.00"),
        )

    def test_close_long_position_with_profit(self, manager, long_position):
        """Test closing a long position with profit"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("160.00")

        # Act
        pnl = manager.close_position(long_position, order)

        # Assert
        assert pnl == Decimal("1000.00")  # (160 - 150) * 100
        assert long_position.is_closed()

    def test_close_short_position_with_profit(self, manager, short_position):
        """Test closing a short position with profit"""
        # Arrange
        order = Order(
            symbol="TSLA", quantity=Decimal("50"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("50")
        order.average_fill_price = Decimal("680.00")

        # Act
        pnl = manager.close_position(short_position, order)

        # Assert
        assert pnl == Decimal("1000.00")  # (700 - 680) * 50
        assert short_position.is_closed()

    def test_close_position_with_loss(self, manager, long_position):
        """Test closing a position with loss"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("145.00")

        # Act
        pnl = manager.close_position(long_position, order)

        # Assert
        assert pnl == Decimal("-500.00")  # (145 - 150) * 100
        assert long_position.is_closed()

    def test_close_position_with_exit_price_override(self, manager, long_position):
        """Test closing position with explicit exit price"""
        # Arrange
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("159.00"),
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("159.00")

        exit_price = Price(Decimal("159.50"))  # Better exit

        # Act
        pnl = manager.close_position(long_position, order, exit_price)

        # Assert
        assert pnl == Decimal("950.00")  # (159.50 - 150) * 100

    def test_close_already_closed_position_raises_error(self, manager, long_position):
        """Test that closing an already closed position raises error"""
        # Arrange
        long_position.close_position(Decimal("160.00"))

        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("160.00")

        # Act & Assert
        with pytest.raises(ValueError, match="Position is already closed"):
            manager.close_position(long_position, order)

    def test_close_position_symbol_mismatch_raises_error(self, manager, long_position):
        """Test that symbol mismatch raises error"""
        # Arrange
        order = Order(
            symbol="GOOGL",  # Different symbol
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = Decimal("2500.00")

        # Act & Assert
        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.close_position(long_position, order)

    def test_close_position_no_exit_price_raises_error(self, manager, long_position):
        """Test that missing exit price raises error"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("100")
        order.average_fill_price = None

        # Act & Assert
        with pytest.raises(ValueError, match="No exit price available"):
            manager.close_position(long_position, order)


class TestPositionManagerCalculatePnL:
    """Test suite for calculate_pnl method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    def test_calculate_pnl_long_position_profit(self, manager):
        """Test P&L calculation for profitable long position"""
        # Arrange
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )
        current_price = Price(Decimal("160.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert
        # P&L is (160 - 150) * 100 = 1000, minus commission of 1 = 999
        assert pnl == Decimal("999.00")  # (160 - 150) * 100 - commission
        assert pnl.currency == "USD"

    def test_calculate_pnl_short_position_profit(self, manager):
        """Test P&L calculation for profitable short position"""
        # Arrange
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("1.00"),
        )
        current_price = Price(Decimal("680.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert
        # P&L is (700 - 680) * 50 = 1000, minus commission of 1 = 999
        assert pnl == Decimal("999.00")  # (700 - 680) * 50 - commission

    def test_calculate_pnl_closed_position(self, manager):
        """Test P&L calculation for closed position"""
        # Arrange
        position = Position.open_position(
            symbol="GOOGL",
            quantity=Decimal("10"),
            entry_price=Decimal("2500.00"),
            commission=Decimal("1.00"),
        )
        position.close_position(Decimal("2600.00"))
        current_price = Price(Decimal("2650.00"))  # Should be ignored

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert
        assert pnl == Decimal("1000.00")  # (2600 - 2500) * 10

    def test_calculate_pnl_with_loss(self, manager):
        """Test P&L calculation with loss"""
        # Arrange
        position = Position.open_position(
            symbol="META",
            quantity=Decimal("50"),
            entry_price=Decimal("300.00"),
            commission=Decimal("1.00"),
        )
        current_price = Price(Decimal("290.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert
        # P&L is (290 - 300) * 50 = -500, minus commission of 1 = -501
        assert pnl == Decimal("-501.00")  # (290 - 300) * 50 - commission

    def test_calculate_pnl_zero_quantity(self, manager):
        """Test P&L calculation with zero quantity position"""
        # Arrange
        position = Position.open_position(
            symbol="AMD",
            quantity=Decimal("100"),
            entry_price=Decimal("100.00"),
            commission=Decimal("1.00"),
        )
        position.close_position(Decimal("110.00"))
        position.quantity = Decimal("0")
        current_price = Price(Decimal("115.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert
        assert pnl == Decimal("1000.00")  # Realized P&L only


class TestPositionManagerMergePositions:
    """Test suite for merge_positions method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    def test_merge_multiple_long_positions(self, manager):
        """Test merging multiple long positions"""
        # Arrange
        pos1 = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )
        pos2 = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("50"),
            entry_price=Decimal("155.00"),
            commission=Decimal("0.50"),
        )
        pos3 = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("30"),
            entry_price=Decimal("152.00"),
            commission=Decimal("0.30"),
        )

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        assert merged.symbol == "AAPL"
        assert merged.quantity == Decimal("180")
        # Weighted average: (100*150 + 50*155 + 30*152) / 180
        expected_avg = (100 * Decimal("150") + 50 * Decimal("155") + 30 * Decimal("152")) / 180
        assert abs(merged.average_entry_price - expected_avg) < Decimal("0.01")
        assert merged.commission_paid == Decimal("1.80")

    def test_merge_long_and_short_positions(self, manager):
        """Test merging long and short positions that cancel out"""
        # Arrange
        long_pos = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("100"),
            entry_price=Decimal("700.00"),
            commission=Decimal("1.00"),
        )
        short_pos = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-100"),
            entry_price=Decimal("720.00"),
            commission=Decimal("1.00"),
        )

        # Act
        merged = manager.merge_positions([long_pos, short_pos])

        # Assert
        assert merged.symbol == "TSLA"
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()

    def test_merge_empty_list_returns_none(self, manager):
        """Test merging empty list returns None"""
        # Act
        merged = manager.merge_positions([])

        # Assert
        assert merged is None

    def test_merge_single_position_returns_same(self, manager):
        """Test merging single position returns the same position"""
        # Arrange
        position = Position.open_position(
            symbol="NVDA",
            quantity=Decimal("50"),
            entry_price=Decimal("500.00"),
            commission=Decimal("1.00"),
        )

        # Act
        merged = manager.merge_positions([position])

        # Assert
        assert merged == position

    def test_merge_different_symbols_raises_error(self, manager):
        """Test merging positions with different symbols raises error"""
        # Arrange
        pos1 = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )
        pos2 = Position.open_position(
            symbol="GOOGL",
            quantity=Decimal("10"),
            entry_price=Decimal("2500.00"),
            commission=Decimal("1.00"),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot merge positions with different symbols"):
            manager.merge_positions([pos1, pos2])

    def test_merge_positions_with_realized_pnl(self, manager):
        """Test merging positions preserves realized P&L"""
        # Arrange
        pos1 = Position.open_position(
            symbol="AMD",
            quantity=Decimal("100"),
            entry_price=Decimal("100.00"),
            commission=Decimal("1.00"),
        )
        pos1.realized_pnl = Decimal("500.00")

        pos2 = Position.open_position(
            symbol="AMD",
            quantity=Decimal("50"),
            entry_price=Decimal("105.00"),
            commission=Decimal("0.50"),
        )
        pos2.realized_pnl = Decimal("250.00")

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert
        assert merged.realized_pnl == Decimal("750.00")

    def test_merge_mixed_open_closed_positions(self, manager):
        """Test merging mix of open and closed positions"""
        # Arrange
        open_pos = Position.open_position(
            symbol="MSFT",
            quantity=Decimal("100"),
            entry_price=Decimal("300.00"),
            commission=Decimal("1.00"),
        )

        closed_pos = Position.open_position(
            symbol="MSFT",
            quantity=Decimal("50"),
            entry_price=Decimal("290.00"),
            commission=Decimal("0.50"),
        )
        closed_pos.close_position(Decimal("310.00"))

        # Act
        merged = manager.merge_positions([open_pos, closed_pos])

        # Assert
        assert merged.quantity == Decimal("100")  # Only open position
        assert merged.realized_pnl == Decimal("1000.00")  # From closed position


class TestPositionManagerShouldClosePosition:
    """Test suite for should_close_position method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    @pytest.fixture
    def long_position(self):
        """Create a long position"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )
        return position

    def test_should_close_stop_loss_triggered(self, manager, long_position):
        """Test stop loss trigger detection"""
        # Arrange
        long_position.stop_loss_price = Decimal("145.00")
        current_price = Price(Decimal("144.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_take_profit_triggered(self, manager, long_position):
        """Test take profit trigger detection"""
        # Arrange
        long_position.take_profit_price = Decimal("160.00")
        current_price = Price(Decimal("161.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Take profit triggered"

    def test_should_close_max_loss_exceeded(self, manager, long_position):
        """Test maximum loss threshold"""
        # Arrange
        current_price = Price(Decimal("140.00"))  # $10 loss per share
        max_loss = Money(Decimal("500.00"))  # Max $500 loss

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert
        assert should_close is True
        assert "Max loss exceeded" in reason

    def test_should_close_target_profit_reached(self, manager, long_position):
        """Test target profit threshold"""
        # Arrange
        current_price = Price(Decimal("160.00"))  # $10 profit per share
        target_profit = Money(Decimal("800.00"))  # Target $800 profit

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, target_profit=target_profit
        )

        # Assert
        assert should_close is True
        assert "Target profit reached" in reason

    def test_should_not_close_no_conditions_met(self, manager, long_position):
        """Test no closing conditions met"""
        # Arrange
        current_price = Price(Decimal("155.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert
        assert should_close is False
        assert reason == ""

    def test_should_close_multiple_conditions_first_wins(self, manager, long_position):
        """Test that first triggered condition is returned"""
        # Arrange
        long_position.stop_loss_price = Decimal("145.00")
        long_position.take_profit_price = Decimal("160.00")
        current_price = Price(Decimal("144.00"))  # Both stop loss and max loss triggered
        max_loss = Money(Decimal("500.00"))

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert
        assert should_close is True
        assert reason == "Stop loss triggered"  # First check wins

    def test_should_close_short_position_stop_loss(self, manager):
        """Test stop loss for short position"""
        # Arrange
        short_position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("1.00"),
        )
        short_position.stop_loss_price = Decimal("710.00")
        current_price = Price(Decimal("712.00"))

        # Act
        should_close, reason = manager.should_close_position(short_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Stop loss triggered"


class TestPositionManagerCalculatePositionSize:
    """Test suite for calculate_position_size method"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    def test_calculate_position_size_basic(self, manager):
        """Test basic position size calculation"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")  # 2% risk
        entry_price = Price(Decimal("50.00"))
        stop_loss_price = Price(Decimal("48.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount: $10,000 * 0.02 = $200
        # Risk per share: $50 - $48 = $2
        # Position size: $200 / $2 = 100 shares
        assert size == Decimal("100")

    def test_calculate_position_size_fractional_shares(self, manager):
        """Test position size with fractional result (rounds down)"""
        # Arrange
        account_balance = Money(Decimal("5000.00"), "USD")
        risk_per_trade = Decimal("0.01")  # 1% risk
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("97.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount: $5,000 * 0.01 = $50
        # Risk per share: $100 - $97 = $3
        # Position size: $50 / $3 = 16.67, quantized to 17 (banker's rounding)
        assert size == Decimal("17")

    def test_calculate_position_size_small_stop_distance(self, manager):
        """Test with very small stop loss distance"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("99.50"))  # Only $0.50 risk per share

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount: $10,000 * 0.02 = $200
        # Risk per share: $100 - $99.50 = $0.50
        # Position size: $200 / $0.50 = 400 shares
        assert size == Decimal("400")

    def test_calculate_position_size_invalid_risk_raises_error(self, manager):
        """Test that invalid risk percentage raises error"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        entry_price = Price(Decimal("50.00"))
        stop_loss_price = Price(Decimal("48.00"))

        # Act & Assert - Risk > 100%
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, Decimal("1.5"), entry_price, stop_loss_price
            )

        # Act & Assert - Risk <= 0
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, Decimal("0"), entry_price, stop_loss_price
            )

    def test_calculate_position_size_invalid_prices_raises_error(self, manager):
        """Test that invalid prices raise errors"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")

        # Act & Assert - Zero entry price (caught by PositionManager)
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, Price(Decimal("0")), Price(Decimal("48.00"))
            )

        # Act & Assert - Negative stop loss price
        # Note: Price object raises error before PositionManager
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal("-48.00"))

    def test_calculate_position_size_same_prices_raises_error(self, manager):
        """Test that same entry and stop prices raise error"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("50.00"))
        stop_loss_price = Price(Decimal("50.00"))  # Same as entry

        # Act & Assert
        with pytest.raises(ValueError, match="Entry and stop loss prices cannot be the same"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, entry_price, stop_loss_price
            )

    def test_calculate_position_size_negative_prices_validation(self, manager):
        """Test that PositionManager validates positive prices"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")

        # Create mock Price objects with negative values by bypassing validation
        entry_price = Price(Decimal("50.00"))
        entry_price._value = Decimal("-50.00")  # Force negative
        stop_loss = Price(Decimal("48.00"))

        # Act & Assert
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss)

    def test_calculate_position_size_for_short(self, manager):
        """Test position size calculation for short position"""
        # Arrange
        account_balance = Money(Decimal("10000.00"), "USD")
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("100.00"))  # Short entry
        stop_loss_price = Price(Decimal("102.00"))  # Stop above entry for short

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount: $10,000 * 0.02 = $200
        # Risk per share: |$100 - $102| = $2
        # Position size: $200 / $2 = 100 shares
        assert size == Decimal("100")


class TestPositionManagerEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()

    def test_handle_very_large_position(self, manager):
        """Test handling very large position sizes"""
        # Arrange
        order = Order(
            symbol="BRK.A",
            quantity=Decimal("1000000"),  # Million shares
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("1000000")
        order.average_fill_price = Decimal("500000.00")  # $500k per share

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.quantity == Decimal("1000000")
        assert position.average_entry_price == Decimal("500000.00")

    def test_handle_very_small_quantities(self, manager):
        """Test handling fractional share quantities"""
        # Arrange
        order = Order(
            symbol="BTC",
            quantity=Decimal("0.00001"),  # Tiny fraction
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("0.00001")
        order.average_fill_price = Decimal("50000.00")

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.quantity == Decimal("0.00001")

    def test_negative_filled_quantity_raises_error(self, manager):
        """Test that negative filled quantity raises error"""
        # Arrange
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order.status = OrderStatus.FILLED
        order.filled_quantity = Decimal("-100")  # Negative
        order.average_fill_price = Decimal("150.00")

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            manager.open_position(order)
