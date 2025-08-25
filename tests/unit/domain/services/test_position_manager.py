"""
Unit tests for PositionManager domain service
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.position import Position
from src.domain.services.position_manager import PositionManager
from src.domain.value_objects import Money, Price


class TestPositionManager:
    """Test suite for PositionManager domain service"""

    @pytest.fixture
    def manager(self):
        """Create a PositionManager instance"""
        return PositionManager()

    @pytest.fixture
    def filled_buy_order(self):
        """Create a filled buy order"""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.00"),
            created_at=datetime.now(UTC),
        )
        order.tags = {"strategy": "momentum"}
        return order

    @pytest.fixture
    def filled_sell_order(self):
        """Create a filled sell order"""
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("155.00"),
            created_at=datetime.now(UTC),
        )
        order.tags = {"strategy": "momentum"}
        return order

    @pytest.fixture
    def long_position(self):
        """Create a long position"""
        return Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            strategy="momentum",
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position"""
        return Position.open_position(
            symbol="AAPL",
            quantity=Decimal("-50"),
            entry_price=Decimal("155.00"),
            commission=Decimal("0.50"),
            strategy="mean_reversion",
        )

    # Test open_position method

    def test_open_position_from_filled_buy_order(self, manager, filled_buy_order):
        """Test opening a long position from a filled buy order"""
        # Act
        position = manager.open_position(filled_buy_order)

        # Assert
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.is_long()
        assert position.strategy == "momentum"

    def test_open_position_from_filled_sell_order(self, manager, filled_sell_order):
        """Test opening a short position from a filled sell order"""
        # Act
        position = manager.open_position(filled_sell_order)

        # Assert
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("155.00")
        assert position.is_short()
        assert position.strategy == "momentum"

    def test_open_position_with_override_price(self, manager, filled_buy_order):
        """Test opening a position with override fill price"""
        # Arrange
        override_price = Price(Decimal("149.50"))

        # Act
        position = manager.open_position(filled_buy_order, override_price)

        # Assert
        assert position.average_entry_price == Decimal("149.50")

    def test_open_position_from_unfilled_order_raises_error(self, manager):
        """Test that opening position from unfilled order raises error"""
        # Arrange
        unfilled_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position from OrderStatus.PENDING order"):
            manager.open_position(unfilled_order)

    def test_open_position_with_zero_filled_quantity_raises_error(self, manager):
        """Test that opening position with zero filled quantity raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            manager.open_position(order)

    def test_open_position_without_fill_price_raises_error(self, manager):
        """Test that opening position without fill price raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="No fill price available for position"):
            manager.open_position(order)

    # Test update_position method

    def test_update_position_add_to_long(self, manager, long_position, filled_buy_order):
        """Test adding to a long position"""
        # Arrange
        filled_buy_order.average_fill_price = Decimal("152.00")

        # Act
        manager.update_position(long_position, filled_buy_order)

        # Assert
        assert long_position.quantity == Decimal("200")  # 100 + 100
        # Weighted average: (100*150 + 100*152) / 200 = 151
        assert long_position.average_entry_price == Decimal("151.00")

    def test_update_position_add_to_short(self, manager, short_position, filled_sell_order):
        """Test adding to a short position"""
        # Arrange
        filled_sell_order.average_fill_price = Decimal("157.00")

        # Act
        manager.update_position(short_position, filled_sell_order)

        # Assert
        assert short_position.quantity == Decimal("-100")  # -50 + (-50)
        # Weighted average: (50*155 + 50*157) / 100 = 156
        assert short_position.average_entry_price == Decimal("156.00")

    def test_update_position_reduce_long(self, manager, long_position, filled_sell_order):
        """Test reducing a long position"""
        # Act
        manager.update_position(long_position, filled_sell_order)

        # Assert
        assert long_position.quantity == Decimal("50")  # 100 - 50
        assert long_position.average_entry_price == Decimal("150.00")  # Unchanged
        assert long_position.realized_pnl > 0  # Profit from selling at 155

    def test_update_position_reduce_short(self, manager, short_position):
        """Test reducing a short position"""
        # Arrange
        buy_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("25"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("25"),
            average_fill_price=Decimal("153.00"),
            created_at=datetime.now(UTC),
        )

        # Act
        manager.update_position(short_position, buy_order)

        # Assert
        assert short_position.quantity == Decimal("-25")  # -50 + 25
        assert short_position.average_entry_price == Decimal("155.00")  # Unchanged
        assert short_position.realized_pnl > 0  # Profit from buying back at 153

    def test_update_position_with_override_price(self, manager, long_position, filled_sell_order):
        """Test updating position with override price"""
        # Arrange
        override_price = Price(Decimal("160.00"))

        # Act
        manager.update_position(long_position, filled_sell_order, override_price)

        # Assert
        assert long_position.quantity == Decimal("50")
        # Should have used override price for P&L calculation
        assert long_position.realized_pnl == Decimal("50") * (Decimal("160.00") - Decimal("150.00"))

    def test_update_position_with_unfilled_order_raises_error(self, manager, long_position):
        """Test that updating position with unfilled order raises error"""
        # Arrange
        unfilled_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot update position with OrderStatus.PENDING order"
        ):
            manager.update_position(long_position, unfilled_order)

    def test_update_position_with_symbol_mismatch_raises_error(self, manager, long_position):
        """Test that updating position with different symbol raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="MSFT",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("300.00"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.update_position(long_position, order)

    def test_update_position_partially_filled_order(self, manager, long_position):
        """Test updating position with partially filled order"""
        # Arrange
        partial_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("152.00"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("152.00"),
            created_at=datetime.now(UTC),
        )

        # Act
        manager.update_position(long_position, partial_order)

        # Assert
        assert long_position.quantity == Decimal("150")  # 100 + 50

    # Test close_position method

    def test_close_position_long(self, manager, long_position, filled_sell_order):
        """Test closing a long position"""
        # Arrange
        filled_sell_order.filled_quantity = Decimal("100")
        filled_sell_order.quantity = Decimal("100")

        # Act
        pnl = manager.close_position(long_position, filled_sell_order)

        # Assert
        assert long_position.is_closed()
        assert long_position.quantity == Decimal("0")
        assert pnl == Decimal("100") * (Decimal("155.00") - Decimal("150.00"))  # 500

    def test_close_position_short(self, manager, short_position):
        """Test closing a short position"""
        # Arrange
        buy_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("153.00"),
            created_at=datetime.now(UTC),
        )

        # Act
        pnl = manager.close_position(short_position, buy_order)

        # Assert
        assert short_position.is_closed()
        assert short_position.quantity == Decimal("0")
        assert pnl == Decimal("50") * (Decimal("155.00") - Decimal("153.00"))  # 100

    def test_close_position_with_override_price(self, manager, long_position, filled_sell_order):
        """Test closing position with override price"""
        # Arrange
        filled_sell_order.filled_quantity = Decimal("100")
        override_price = Price(Decimal("160.00"))

        # Act
        pnl = manager.close_position(long_position, filled_sell_order, override_price)

        # Assert
        assert long_position.is_closed()
        assert pnl == Decimal("100") * (Decimal("160.00") - Decimal("150.00"))  # 1000

    def test_close_already_closed_position_raises_error(
        self, manager, long_position, filled_sell_order
    ):
        """Test that closing already closed position raises error"""
        # Arrange
        filled_sell_order.filled_quantity = Decimal("100")
        manager.close_position(long_position, filled_sell_order)

        # Act & Assert
        with pytest.raises(ValueError, match="Position is already closed"):
            manager.close_position(long_position, filled_sell_order)

    def test_close_position_with_symbol_mismatch_raises_error(self, manager, long_position):
        """Test that closing position with different symbol raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="MSFT",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("300.00"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Symbol mismatch"):
            manager.close_position(long_position, order)

    def test_close_position_without_exit_price_raises_error(self, manager, long_position):
        """Test that closing position without exit price raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="No exit price available"):
            manager.close_position(long_position, order)

    # Test calculate_pnl method

    def test_calculate_pnl_open_long_position(self, manager, long_position):
        """Test calculating P&L for open long position"""
        # Arrange
        current_price = Price(Decimal("155.00"))

        # Act
        pnl = manager.calculate_pnl(long_position, current_price)

        # Assert
        assert pnl == Decimal("100") * (Decimal("155.00") - Decimal("150.00")) - Decimal(
            "1.00"
        )  # 499
        assert pnl.currency == "USD"

    def test_calculate_pnl_open_short_position(self, manager, short_position):
        """Test calculating P&L for open short position"""
        # Arrange
        current_price = Price(Decimal("152.00"))

        # Act
        pnl = manager.calculate_pnl(short_position, current_price)

        # Assert
        assert pnl == Decimal("50") * (Decimal("155.00") - Decimal("152.00")) - Decimal(
            "0.50"
        )  # 149.50
        assert pnl.currency == "USD"

    def test_calculate_pnl_closed_position(self, manager, long_position):
        """Test calculating P&L for closed position"""
        # Arrange
        long_position.close_position(Decimal("160.00"))
        current_price = Price(Decimal("165.00"))  # Should be ignored

        # Act
        pnl = manager.calculate_pnl(long_position, current_price)

        # Assert
        assert pnl == long_position.realized_pnl

    def test_calculate_pnl_updates_market_price(self, manager, long_position):
        """Test that calculate_pnl updates position's market price"""
        # Arrange
        current_price = Price(Decimal("155.00"))

        # Act
        manager.calculate_pnl(long_position, current_price)

        # Assert
        assert long_position.current_price == Decimal("155.00")

    # Test merge_positions method

    def test_merge_positions_single_position(self, manager, long_position):
        """Test merging a single position returns the same position"""
        # Act
        merged = manager.merge_positions([long_position])

        # Assert
        assert merged == long_position

    def test_merge_positions_empty_list(self, manager):
        """Test merging empty list returns None"""
        # Act
        merged = manager.merge_positions([])

        # Assert
        assert merged is None

    def test_merge_multiple_long_positions(self, manager):
        """Test merging multiple long positions"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos2 = Position.open_position("AAPL", Decimal("50"), Decimal("152.00"), Decimal("0.50"))
        pos3 = Position.open_position("AAPL", Decimal("25"), Decimal("154.00"), Decimal("0.25"))

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        assert merged.symbol == "AAPL"
        assert merged.quantity == Decimal("175")  # 100 + 50 + 25
        # Weighted average: (100*150 + 50*152 + 25*154) / 175 = 151.14...
        expected_avg = (100 * Decimal("150") + 50 * Decimal("152") + 25 * Decimal("154")) / 175
        assert abs(merged.average_entry_price - expected_avg) < Decimal("0.01")
        assert merged.commission_paid == Decimal("1.75")  # 1.00 + 0.50 + 0.25

    def test_merge_long_and_short_positions(self, manager):
        """Test merging long and short positions that cancel out"""
        # Arrange
        long_pos = Position.open_position(
            "AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00")
        )
        short_pos = Position.open_position(
            "AAPL", Decimal("-100"), Decimal("155.00"), Decimal("1.00")
        )

        # Act
        merged = manager.merge_positions([long_pos, short_pos])

        # Assert
        assert merged.symbol == "AAPL"
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()
        assert merged.commission_paid == Decimal("2.00")

    def test_merge_positions_with_realized_pnl(self, manager):
        """Test merging positions preserves realized P&L"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.realized_pnl = Decimal("500.00")

        pos2 = Position.open_position("AAPL", Decimal("50"), Decimal("152.00"), Decimal("0.50"))
        pos2.realized_pnl = Decimal("250.00")

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert
        assert merged.realized_pnl == Decimal("750.00")  # 500 + 250

    def test_merge_positions_different_symbols_raises_error(self, manager):
        """Test that merging positions with different symbols raises error"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos2 = Position.open_position("MSFT", Decimal("50"), Decimal("300.00"), Decimal("0.50"))

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot merge positions with different symbols"):
            manager.merge_positions([pos1, pos2])

    def test_merge_positions_net_flat(self, manager):
        """Test merging positions that net to flat"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos2 = Position.open_position("AAPL", Decimal("-75"), Decimal("155.00"), Decimal("0.75"))
        pos3 = Position.open_position("AAPL", Decimal("-25"), Decimal("157.00"), Decimal("0.25"))

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()

    # Test should_close_position method

    def test_should_close_position_stop_loss_triggered_long(self, manager, long_position):
        """Test stop loss trigger for long position"""
        # Arrange
        long_position.stop_loss_price = Decimal("145.00")
        current_price = Price(Decimal("144.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_position_stop_loss_triggered_short(self, manager, short_position):
        """Test stop loss trigger for short position"""
        # Arrange
        short_position.stop_loss_price = Decimal("160.00")
        current_price = Price(Decimal("161.00"))

        # Act
        should_close, reason = manager.should_close_position(short_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_position_take_profit_triggered_long(self, manager, long_position):
        """Test take profit trigger for long position"""
        # Arrange
        long_position.take_profit_price = Decimal("160.00")
        current_price = Price(Decimal("161.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Take profit triggered"

    def test_should_close_position_take_profit_triggered_short(self, manager, short_position):
        """Test take profit trigger for short position"""
        # Arrange
        short_position.take_profit_price = Decimal("150.00")
        current_price = Price(Decimal("149.00"))

        # Act
        should_close, reason = manager.should_close_position(short_position, current_price)

        # Assert
        assert should_close is True
        assert reason == "Take profit triggered"

    def test_should_close_position_max_loss_exceeded(self, manager, long_position):
        """Test max loss threshold exceeded"""
        # Arrange
        current_price = Price(Decimal("140.00"))  # $10 loss per share
        max_loss = Money(Decimal("500.00"))

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert
        assert should_close is True
        assert "Max loss exceeded" in reason

    def test_should_close_position_target_profit_reached(self, manager, long_position):
        """Test target profit reached"""
        # Arrange
        current_price = Price(Decimal("160.00"))  # $10 profit per share
        target_profit = Money(Decimal("900.00"))  # After commission

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, target_profit=target_profit
        )

        # Assert
        assert should_close is True
        assert "Target profit reached" in reason

    def test_should_close_position_no_triggers(self, manager, long_position):
        """Test no close triggers"""
        # Arrange
        long_position.stop_loss_price = Decimal("145.00")
        long_position.take_profit_price = Decimal("160.00")
        current_price = Price(Decimal("155.00"))
        max_loss = Money(Decimal("2000.00"))
        target_profit = Money(Decimal("1500.00"))

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss, target_profit=target_profit
        )

        # Assert
        assert should_close is False
        assert reason == ""

    # Test calculate_position_size method

    def test_calculate_position_size_long(self, manager):
        """Test position size calculation for long position"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")  # 2% risk
        entry_price = Price(Decimal("150.00"))
        stop_loss_price = Price(Decimal("145.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 100000 * 0.02 = 2000
        # Price diff = 150 - 145 = 5
        # Position size = 2000 / 5 = 400
        assert size == Decimal("400")

    def test_calculate_position_size_short(self, manager):
        """Test position size calculation for short position"""
        # Arrange
        account_balance = Money(Decimal("50000.00"))
        risk_per_trade = Decimal("0.01")  # 1% risk
        entry_price = Price(Decimal("200.00"))
        stop_loss_price = Price(Decimal("210.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 50000 * 0.01 = 500
        # Price diff = |200 - 210| = 10
        # Position size = 500 / 10 = 50
        assert size == Decimal("50")

    def test_calculate_position_size_fractional_result(self, manager):
        """Test position size calculation with fractional result"""
        # Arrange
        account_balance = Money(Decimal("75000.00"))
        risk_per_trade = Decimal("0.015")  # 1.5% risk
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("97.50"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 75000 * 0.015 = 1125
        # Price diff = 100 - 97.50 = 2.50
        # Position size = 1125 / 2.50 = 450
        assert size == Decimal("450")

    def test_calculate_position_size_invalid_risk_raises_error(self, manager):
        """Test that invalid risk percentage raises error"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        entry_price = Price(Decimal("150.00"))
        stop_loss_price = Price(Decimal("145.00"))

        # Act & Assert - negative risk
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, Decimal("-0.01"), entry_price, stop_loss_price
            )

        # Act & Assert - risk > 100%
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, Decimal("1.5"), entry_price, stop_loss_price
            )

    def test_calculate_position_size_same_prices_raises_error(self, manager):
        """Test that same entry and stop prices raise error"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("150.00"))
        stop_loss_price = Price(Decimal("150.00"))

        # Act & Assert
        with pytest.raises(ValueError, match="Entry and stop loss prices cannot be the same"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, entry_price, stop_loss_price
            )

    # Edge cases and error conditions

    def test_position_manager_handles_decimal_precision(self, manager):
        """Test that PositionManager handles decimal precision correctly"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("33.333333"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("33.333333"),
            average_fill_price=Decimal("149.999999"),
            created_at=datetime.now(UTC),
        )

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.quantity == Decimal("33.333333")
        assert position.average_entry_price == Decimal("149.999999")

    def test_position_manager_handles_large_quantities(self, manager):
        """Test that PositionManager handles large quantities"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="BRK.A",
            quantity=Decimal("1000000"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1000000"),
            average_fill_price=Decimal("450000.00"),
            created_at=datetime.now(UTC),
        )

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.quantity == Decimal("1000000")
        assert position.average_entry_price == Decimal("450000.00")

    def test_position_manager_handles_small_prices(self, manager):
        """Test that PositionManager handles small prices (penny stocks)"""
        # Arrange
        account_balance = Money(Decimal("10000.00"))
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("0.05"))
        stop_loss_price = Price(Decimal("0.04"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 10000 * 0.02 = 200
        # Price diff = 0.05 - 0.04 = 0.01
        # Position size = 200 / 0.01 = 20000
        assert size == Decimal("20000")

    def test_merge_positions_preserves_metadata(self, manager):
        """Test that merging positions preserves metadata correctly"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.strategy = "momentum"
        pos1.tags = {"entry_signal": "breakout"}

        pos2 = Position.open_position("AAPL", Decimal("50"), Decimal("152.00"), Decimal("0.50"))
        pos2.strategy = "momentum"

        # Note: merge_positions doesn't preserve closed_at for non-zero positions
        # It only sets closed_at when positions net to zero

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert
        assert merged.quantity == Decimal("150")  # Combined quantity
        assert merged.commission_paid == Decimal("1.50")  # Combined commission

    # Additional tests for uncovered branches

    def test_open_position_with_negative_filled_quantity(self, manager):
        """Test that opening position with negative filled quantity raises error"""
        # Since Order validation prevents negative filled_quantity,
        # we'll test by manually setting it after creation
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.00"),
            created_at=datetime.now(UTC),
        )
        # Manually override to negative (bypassing validation)
        order.filled_quantity = Decimal("-100")

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position with zero or negative quantity"):
            manager.open_position(order)

    def test_open_position_without_strategy_tag(self, manager):
        """Test opening position without strategy tag"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.00"),
            created_at=datetime.now(UTC),
        )
        order.tags = {}  # No strategy tag

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.strategy is None

    def test_update_position_without_fill_price(self, manager, long_position):
        """Test that updating position without any fill price raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=None,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="No fill price available"):
            manager.update_position(long_position, order)

    def test_calculate_pnl_position_with_no_unrealized(self, manager):
        """Test calculate_pnl with position that has no unrealized P&L"""
        # Arrange
        position = Position.open_position(
            "AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00")
        )
        position.quantity = Decimal("100")
        position.current_price = None  # No current price, so get_total_pnl returns None
        current_price = Price(Decimal("155.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert - Should calculate P&L even when initial state has no price
        assert pnl == Decimal("100") * (Decimal("155.00") - Decimal("150.00")) - Decimal("1.00")

    def test_merge_positions_with_zero_total_quantity_edge_case(self, manager):
        """Test merge positions where abs_quantity could be zero (defensive check)"""
        # This tests the defensive check in merge_positions line 368-372
        # Though the logic flow prevents this, the code has a defensive check
        # Arrange - Create positions that perfectly cancel
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos2 = Position.open_position("AAPL", Decimal("-100"), Decimal("155.00"), Decimal("1.00"))

        # Set some realized P&L
        pos1.realized_pnl = Decimal("500")
        pos2.realized_pnl = Decimal("300")

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()
        assert merged.realized_pnl == Decimal("800")
        assert merged.average_entry_price == Decimal("0")

    def test_calculate_position_size_with_zero_risk(self, manager):
        """Test that zero risk per trade raises error"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0")  # 0% risk
        entry_price = Price(Decimal("150.00"))
        stop_loss_price = Price(Decimal("145.00"))

        # Act & Assert
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, entry_price, stop_loss_price
            )

    def test_calculate_position_size_with_tiny_prices(self, manager):
        """Test position size calculation with very small prices"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")

        # Test with very small positive prices (edge case)
        entry_price = Price(Decimal("0.01"))
        stop_loss_price = Price(Decimal("0.005"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert - Should handle very small prices correctly
        # Risk amount = 100000 * 0.02 = 2000
        # Price diff = 0.01 - 0.005 = 0.005
        # Position size = 2000 / 0.005 = 400000
        assert size == Decimal("400000")

    def test_calculate_position_size_with_risk_exactly_one(self, manager):
        """Test position size calculation with risk = 1 (100%)"""
        # Arrange
        account_balance = Money(Decimal("10000.00"))
        risk_per_trade = Decimal("1")  # 100% risk (edge case, should be allowed)
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("90.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 10000 * 1 = 10000
        # Price diff = 100 - 90 = 10
        # Position size = 10000 / 10 = 1000
        assert size == Decimal("1000")

    def test_calculate_position_size_rounds_down_fractional_shares(self, manager):
        """Test that position size calculation rounds down fractional shares"""
        # Arrange
        account_balance = Money(Decimal("10000.00"))
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("33.33"))
        stop_loss_price = Price(Decimal("30.00"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 10000 * 0.02 = 200
        # Price diff = 33.33 - 30.00 = 3.33
        # Position size = 200 / 3.33 = 60.06... -> should round down to 60
        assert size == Decimal("60")

    def test_merge_positions_with_closed_at_timestamp(self, manager):
        """Test that merge_positions preserves closed_at for zero quantity positions"""
        # Arrange
        now = datetime.now(UTC)
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos2 = Position.open_position("AAPL", Decimal("-100"), Decimal("155.00"), Decimal("1.00"))
        pos2.closed_at = now  # Set closed_at on one position

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()
        assert merged.closed_at == now  # Should use the last position's closed_at

    def test_should_close_position_target_profit_not_reached(self, manager, long_position):
        """Test that position is not closed when target profit is not reached"""
        # Arrange
        current_price = Price(Decimal("152.00"))  # Small profit
        target_profit = Money(Decimal("500.00"))  # Target not met (only ~$200 profit)

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, target_profit=target_profit
        )

        # Assert
        assert should_close is False
        assert reason == ""

    def test_should_close_position_max_loss_not_exceeded(self, manager, long_position):
        """Test that position is not closed when max loss is not exceeded"""
        # Arrange
        current_price = Price(Decimal("148.00"))  # Small loss
        max_loss = Money(Decimal("500.00"))  # Max loss not exceeded (only ~$200 loss)

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert
        assert should_close is False
        assert reason == ""

    def test_calculate_position_size_with_zero_price_raises_error(self, manager):
        """Test that zero price raises error in calculate_position_size"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("0"))  # Zero price
        stop_loss_price = Price(Decimal("145.00"))

        # Act & Assert
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, entry_price, stop_loss_price
            )

    def test_calculate_position_size_with_negative_price_raises_error(self, manager):
        """Test that negative price raises error in calculate_position_size"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")

        # Create a Price object and then manually set its value to negative
        # (bypassing the Price constructor validation to test the position_manager validation)
        entry_price = Price(Decimal("150.00"))
        stop_loss_price = Price(Decimal("5.00"))
        stop_loss_price._value = Decimal("-5.00")  # Manually set negative value

        # Act & Assert
        with pytest.raises(ValueError, match="Prices must be positive"):
            manager.calculate_position_size(
                account_balance, risk_per_trade, entry_price, stop_loss_price
            )

    # Additional comprehensive tests for complete coverage

    def test_open_position_from_canceled_order_raises_error(self, manager):
        """Test that opening position from canceled order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELLED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot open position from OrderStatus.CANCELLED order"
        ):
            manager.open_position(order)

    def test_open_position_from_rejected_order_raises_error(self, manager):
        """Test that opening position from rejected order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.REJECTED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot open position from OrderStatus.REJECTED order"
        ):
            manager.open_position(order)

    def test_open_position_from_expired_order_raises_error(self, manager):
        """Test that opening position from expired order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.EXPIRED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot open position from OrderStatus.EXPIRED order"):
            manager.open_position(order)

    def test_update_position_with_canceled_order_raises_error(self, manager, long_position):
        """Test that updating position with canceled order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELLED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot update position with OrderStatus.CANCELLED order"
        ):
            manager.update_position(long_position, order)

    def test_update_position_with_rejected_order_raises_error(self, manager, long_position):
        """Test that updating position with rejected order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.REJECTED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot update position with OrderStatus.REJECTED order"
        ):
            manager.update_position(long_position, order)

    def test_update_position_with_expired_order_raises_error(self, manager, long_position):
        """Test that updating position with expired order raises error"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.EXPIRED,
            created_at=datetime.now(UTC),
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot update position with OrderStatus.EXPIRED order"
        ):
            manager.update_position(long_position, order)

    def test_update_position_reversal_long_to_short_raises_error(self, manager, long_position):
        """Test that position reversal from long to short raises error (current behavior)"""
        # Arrange - Long position with 100 shares
        assert long_position.quantity == Decimal("100")

        # Create a sell order for 150 shares (would reverse position)
        sell_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("150"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("150"),
            average_fill_price=Decimal("155.00"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert - Current implementation doesn't support reversals
        with pytest.raises(
            ValueError, match="Cannot reduce position by 150, current quantity is 100"
        ):
            manager.update_position(long_position, sell_order)

    def test_update_position_reversal_short_to_long_raises_error(self, manager, short_position):
        """Test that position reversal from short to long raises error (current behavior)"""
        # Arrange - Short position with -50 shares
        assert short_position.quantity == Decimal("-50")

        # Create a buy order for 100 shares (would reverse position)
        buy_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("152.00"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert - Current implementation doesn't support reversals
        with pytest.raises(
            ValueError, match="Cannot reduce position by 100, current quantity is -50"
        ):
            manager.update_position(short_position, buy_order)

    def test_should_close_position_stop_loss_priority_over_take_profit(
        self, manager, long_position
    ):
        """Test that stop loss has priority over take profit when both are triggered"""
        # Arrange - Set both stop loss and take profit
        long_position.stop_loss_price = Decimal("145.00")
        long_position.take_profit_price = Decimal("160.00")
        # Current price triggers both (below stop loss)
        current_price = Price(Decimal("140.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert - Should return stop loss as the reason (checked first)
        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_position_stop_loss_priority_over_max_loss(self, manager, long_position):
        """Test that stop loss has priority over max loss parameter"""
        # Arrange
        long_position.stop_loss_price = Decimal("145.00")
        current_price = Price(Decimal("140.00"))
        max_loss = Money(Decimal("100.00"))  # Would also trigger

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert - Should return stop loss as the reason (checked first)
        assert should_close is True
        assert reason == "Stop loss triggered"

    def test_should_close_position_take_profit_priority_over_target_profit(
        self, manager, long_position
    ):
        """Test that take profit has priority over target profit parameter"""
        # Arrange
        long_position.take_profit_price = Decimal("160.00")
        current_price = Price(Decimal("161.00"))
        target_profit = Money(Decimal("500.00"))  # Would also trigger

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, target_profit=target_profit
        )

        # Assert - Should return take profit as the reason (checked before target profit)
        assert should_close is True
        assert reason == "Take profit triggered"

    def test_should_close_position_without_stop_or_take_profit(self, manager, long_position):
        """Test should_close_position when position has no stop loss or take profit"""
        # Arrange - No stop loss or take profit set
        long_position.stop_loss_price = None
        long_position.take_profit_price = None
        current_price = Price(Decimal("140.00"))
        max_loss = Money(Decimal("500.00"))

        # Act
        should_close, reason = manager.should_close_position(
            long_position, current_price, max_loss=max_loss
        )

        # Assert - Should check max loss
        assert should_close is True
        assert "Max loss exceeded" in reason

    def test_merge_positions_mixed_with_different_realized_pnl(self, manager):
        """Test merging positions with mixed long/short and different P&L"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.realized_pnl = Decimal("1000.00")

        pos2 = Position.open_position("AAPL", Decimal("-30"), Decimal("155.00"), Decimal("0.30"))
        pos2.realized_pnl = Decimal("-200.00")

        pos3 = Position.open_position("AAPL", Decimal("50"), Decimal("152.00"), Decimal("0.50"))
        pos3.realized_pnl = Decimal("300.00")

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        assert merged.symbol == "AAPL"
        assert merged.quantity == Decimal("120")  # 100 - 30 + 50
        assert merged.realized_pnl == Decimal("1100.00")  # 1000 - 200 + 300
        assert merged.commission_paid == Decimal("1.80")  # 1.00 + 0.30 + 0.50

    def test_calculate_pnl_with_position_having_no_total_pnl(self, manager):
        """Test calculate_pnl when position.get_total_pnl() returns None"""
        # Arrange
        position = Position.open_position(
            "AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00")
        )
        # Clear current price to ensure get_total_pnl might return None initially
        position.current_price = None
        current_price = Price(Decimal("155.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert - Should calculate P&L correctly even if initial state had no total P&L
        expected_pnl = Decimal("100") * (Decimal("155.00") - Decimal("150.00")) - Decimal("1.00")
        assert pnl == expected_pnl

    def test_calculate_position_size_with_extremely_small_difference(self, manager):
        """Test position size calculation with very small price difference"""
        # Arrange
        account_balance = Money(Decimal("100000.00"))
        risk_per_trade = Decimal("0.02")
        entry_price = Price(Decimal("100.0001"))
        stop_loss_price = Price(Decimal("100.0000"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 100000 * 0.02 = 2000
        # Price diff = 0.0001
        # Position size = 2000 / 0.0001 = 20000000
        assert size == Decimal("20000000")

    def test_update_position_adding_to_flat_position_raises_error(self, manager):
        """Test that updating a closed/flat position raises error (current behavior)"""
        # Arrange - Create a position and flatten it
        position = Position.open_position(
            "AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00")
        )
        position.reduce_position(Decimal("100"), Decimal("155.00"))
        assert position.quantity == Decimal("0")
        assert position.is_closed()

        # Now try to add to the flat/closed position
        buy_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("50"),
            average_fill_price=Decimal("160.00"),
            created_at=datetime.now(UTC),
        )

        # Act & Assert - Current implementation doesn't allow updating closed positions
        with pytest.raises(ValueError, match="Cannot reduce position by 50, current quantity is 0"):
            manager.update_position(position, buy_order)

    def test_open_position_with_order_no_strategy_in_tags(self, manager):
        """Test opening position when order tags has no strategy key"""
        # Arrange
        order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("100"),
            average_fill_price=Decimal("150.00"),
            created_at=datetime.now(UTC),
        )
        # Set tags to empty dict (no strategy key)
        order.tags = {"other_key": "value"}

        # Act
        position = manager.open_position(order)

        # Assert
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.strategy is None

    def test_merge_positions_preserves_latest_closed_at(self, manager):
        """Test that merge_positions uses the last position's closed_at for zero quantity"""
        # Arrange
        now1 = datetime.now(UTC)
        now2 = datetime.now(UTC)
        now3 = datetime.now(UTC)

        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.closed_at = now1

        pos2 = Position.open_position("AAPL", Decimal("-50"), Decimal("155.00"), Decimal("0.50"))
        pos2.closed_at = now2

        pos3 = Position.open_position("AAPL", Decimal("-50"), Decimal("157.00"), Decimal("0.25"))
        pos3.closed_at = now3  # Latest timestamp

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()
        assert merged.closed_at == now3  # Should use the last position's closed_at

    def test_calculate_position_size_with_inverted_stop_loss(self, manager):
        """Test position size calculation when stop loss is on the other side of entry"""
        # Arrange - For a short position, stop loss would be above entry
        account_balance = Money(Decimal("50000.00"))
        risk_per_trade = Decimal("0.01")
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("105.00"))  # Stop above entry (short position)

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 50000 * 0.01 = 500
        # Price diff = |100 - 105| = 5
        # Position size = 500 / 5 = 100
        assert size == Decimal("100")

    def test_calculate_pnl_with_closed_position_ignores_current_price(self, manager):
        """Test that calculate_pnl for closed position completely ignores current price"""
        # Arrange
        position = Position.open_position(
            "AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00")
        )
        # Close the position with a specific P&L
        position.close_position(Decimal("160.00"))
        realized_pnl_at_close = position.realized_pnl

        # Provide a different current price that should be ignored
        current_price = Price(Decimal("200.00"))

        # Act
        pnl = manager.calculate_pnl(position, current_price)

        # Assert - P&L should be exactly the realized P&L, ignoring current price
        assert pnl == realized_pnl_at_close
        assert pnl.currency == "USD"

    def test_merge_positions_with_all_closed_positions(self, manager):
        """Test merging multiple closed positions"""
        # Arrange
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.close_position(Decimal("155.00"))

        pos2 = Position.open_position("AAPL", Decimal("50"), Decimal("152.00"), Decimal("0.50"))
        pos2.close_position(Decimal("158.00"))

        # Act
        merged = manager.merge_positions([pos1, pos2])

        # Assert - Should combine the closed positions
        assert merged.quantity == Decimal("0")
        assert merged.is_closed()
        assert merged.realized_pnl == pos1.realized_pnl + pos2.realized_pnl

    def test_should_close_position_with_no_conditions_set(self, manager, long_position):
        """Test should_close_position when no conditions are set"""
        # Arrange - No stop loss, take profit, or parameters
        long_position.stop_loss_price = None
        long_position.take_profit_price = None
        current_price = Price(Decimal("155.00"))

        # Act
        should_close, reason = manager.should_close_position(long_position, current_price)

        # Assert - Should not close
        assert should_close is False
        assert reason == ""

    def test_update_position_partial_fill_reduce_position(self, manager, long_position):
        """Test updating position with a partially filled order that reduces position"""
        # Arrange
        partial_sell = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Decimal("75"),  # Original order for 75
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("155.00"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("30"),  # Only 30 filled
            average_fill_price=Decimal("155.00"),
            created_at=datetime.now(UTC),
        )

        # Act
        manager.update_position(long_position, partial_sell)

        # Assert
        assert long_position.quantity == Decimal("70")  # 100 - 30
        assert long_position.realized_pnl > 0  # Should have some profit

    def test_calculate_position_size_with_exact_risk_boundary(self, manager):
        """Test position size calculation at exact risk boundaries"""
        # Arrange
        account_balance = Money(Decimal("10000.00"))
        risk_per_trade = Decimal("0.000001")  # Very small but valid risk
        entry_price = Price(Decimal("100.00"))
        stop_loss_price = Price(Decimal("99.99"))

        # Act
        size = manager.calculate_position_size(
            account_balance, risk_per_trade, entry_price, stop_loss_price
        )

        # Assert
        # Risk amount = 10000 * 0.000001 = 0.01
        # Price diff = 0.01
        # Position size = 0.01 / 0.01 = 1
        assert size == Decimal("1")

    def test_merge_positions_complex_scenario(self, manager):
        """Test merging positions in a complex trading scenario"""
        # Arrange - Simulate a day of trading with multiple partial positions
        pos1 = Position.open_position("AAPL", Decimal("100"), Decimal("150.00"), Decimal("1.00"))
        pos1.add_to_position(Decimal("50"), Decimal("152.00"))  # Add to position
        pos1.reduce_position(Decimal("75"), Decimal("155.00"))  # Take partial profit
        pos1.realized_pnl = Decimal("225.00")  # (155-150)*75 - some of the avg cost

        pos2 = Position.open_position("AAPL", Decimal("-50"), Decimal("156.00"), Decimal("0.50"))
        pos2.reduce_position(Decimal("25"), Decimal("154.00"))  # Cover half
        pos2.realized_pnl = Decimal("50.00")  # (156-154)*25

        pos3 = Position.open_position("AAPL", Decimal("30"), Decimal("153.00"), Decimal("0.30"))

        # Act
        merged = manager.merge_positions([pos1, pos2, pos3])

        # Assert
        # Final quantity: 75 (from pos1) - 25 (from pos2) + 30 (pos3) = 80
        assert merged.quantity == Decimal("80")
        assert merged.realized_pnl == Decimal("275.00")  # 225 + 50
        assert merged.commission_paid == Decimal("1.80")  # 1.00 + 0.50 + 0.30
