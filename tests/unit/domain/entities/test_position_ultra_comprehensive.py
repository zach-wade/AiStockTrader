"""
Ultra-Comprehensive Tests for Position Entity
==========================================

This test suite provides exhaustive coverage for the Position domain entity,
covering all methods, edge cases, P&L calculations, risk management features,
and position lifecycle management for a financial trading system.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity


class TestPositionCreation:
    """Test position creation and initialization."""

    def test_position_default_creation(self):
        """Test creating position with minimum required values."""
        from datetime import UTC, datetime

        # Create a closed position (zero quantity is only allowed for closed positions)
        position = Position(
            symbol="AAPL", average_entry_price=Price(Decimal("100.00")), closed_at=datetime.now(UTC)
        )

        assert position.id is not None
        assert position.symbol == "AAPL"
        assert position.quantity == Quantity(Decimal("0"))
        assert position.average_entry_price == Price(Decimal("100.00"))
        assert position.current_price is None
        assert position.last_updated is None
        assert position.realized_pnl == Money(Decimal("0"))
        assert position.commission_paid == Money(Decimal("0"))
        assert position.stop_loss_price is None
        assert position.take_profit_price is None
        assert position.max_position_value is None
        assert position.opened_at is not None
        assert position.closed_at is not None
        assert position.strategy is None
        assert position.tags == {}

    def test_position_custom_creation(self):
        """Test creating position with custom values."""
        custom_id = uuid4()
        custom_time = datetime.now(UTC)

        position = Position(
            id=custom_id,
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.50")),
            realized_pnl=Money(Decimal("100")),
            commission_paid=Money(Decimal("5.99")),
            opened_at=custom_time,
            strategy="momentum",
            tags={"sector": "tech"},
        )

        assert position.id == custom_id
        assert position.symbol == "AAPL"
        assert position.quantity.value == Decimal("100")
        assert position.average_entry_price.value == Decimal("150.50")
        assert position.realized_pnl.amount == Decimal("100")
        assert position.commission_paid.amount == Decimal("5.99")
        assert position.opened_at == custom_time
        assert position.strategy == "momentum"
        assert position.tags == {"sector": "tech"}

    def test_position_validation_empty_symbol(self):
        """Test position validation with empty symbol."""
        with pytest.raises(ValueError, match="Position symbol cannot be empty"):
            Position(
                symbol="",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("150.00")),
            )

    def test_position_validation_negative_entry_price(self):
        """Test position validation with negative entry price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Position(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("-10")),
            )

    def test_position_validation_zero_quantity_open(self):
        """Test position validation with zero quantity for open position."""
        with pytest.raises(ValueError, match="Open position cannot have zero quantity"):
            Position(
                symbol="AAPL",
                quantity=Quantity(Decimal("0")),
                average_entry_price=Price(Decimal("150.00")),
                closed_at=None,  # Open position
            )


class TestPositionFactoryMethod:
    """Test position factory method for opening new positions."""

    def test_open_position_success(self):
        """Test successful position opening."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150")),
            commission=Money(Decimal("5")),
            strategy="momentum",
        )

        assert position.symbol == "AAPL"
        assert position.quantity.value == Decimal("100")
        assert position.average_entry_price.value == Decimal("150")
        assert position.commission_paid.amount == Decimal("5")
        assert position.strategy == "momentum"
        assert not position.is_closed()

    def test_open_position_zero_quantity(self):
        """Test opening position with zero quantity."""
        with pytest.raises(ValueError, match="Cannot open position with zero quantity"):
            Position.open_position(
                symbol="AAPL", quantity=Quantity(Decimal("0")), entry_price=Price(Decimal("150"))
            )

    def test_open_position_negative_price(self):
        """Test opening position with negative entry price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Position.open_position(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("-10"))
            )

    def test_open_position_zero_price(self):
        """Test opening position with zero entry price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Position.open_position(
                symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("0"))
            )


class TestPositionDirection:
    """Test position direction detection."""

    def test_is_long(self):
        """Test long position detection."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        assert position.is_long() is True
        assert position.is_short() is False

    def test_is_short(self):
        """Test short position detection."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        assert position.is_long() is False
        assert position.is_short() is True

    def test_zero_quantity_after_close(self):
        """Test position direction after closing to zero quantity."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Close position
        position.close_position(Price(Decimal("160")))

        assert position.is_long() is False
        assert position.is_short() is False
        assert position.is_closed() is True


class TestPositionModification:
    """Test position modification operations."""

    def test_add_to_long_position(self):
        """Test adding to long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Add 50 shares at $160
        position.add_to_position(
            quantity=Quantity(Decimal("50")),
            price=Price(Decimal("160")),
            commission=Money(Decimal("3")),
        )

        # New average price: (100 * 150 + 50 * 160) / 150 = 153.333...
        assert position.quantity.value == Decimal("150")
        # Use quantize to round to 2 decimal places for comparison
        assert position.average_entry_price.value.quantize(Decimal("0.01")) == Decimal("153.33")
        assert position.commission_paid.amount == Decimal("3")

    def test_add_to_short_position(self):
        """Test adding to short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        # Add -50 shares (more short) at $140
        position.add_to_position(
            quantity=Quantity(Decimal("-50")),
            price=Price(Decimal("140")),
            commission=Money(Decimal("3")),
        )

        # New average price: (100 * 150 + 50 * 140) / 150 = 146.666...
        assert position.quantity.value == Decimal("-150")
        # Use quantize to round to 2 decimal places for comparison
        assert position.average_entry_price.value.quantize(Decimal("0.01")) == Decimal("146.67")
        assert position.commission_paid.amount == Decimal("3")

    def test_add_wrong_direction_to_long(self):
        """Test adding wrong direction quantity to long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        with pytest.raises(ValueError, match="Cannot add short quantity to long position"):
            position.add_to_position(quantity=Quantity(Decimal("-50")), price=Price(Decimal("160")))

    def test_add_wrong_direction_to_short(self):
        """Test adding wrong direction quantity to short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        with pytest.raises(ValueError, match="Cannot add long quantity to short position"):
            position.add_to_position(quantity=Quantity(Decimal("50")), price=Price(Decimal("160")))

    def test_add_zero_quantity(self):
        """Test adding zero quantity (should be no-op)."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        original_quantity = position.quantity.value
        original_price = position.average_entry_price.value

        position.add_to_position(quantity=Quantity(Decimal("0")), price=Price(Decimal("160")))

        assert position.quantity.value == original_quantity
        assert position.average_entry_price.value == original_price


class TestPositionReduction:
    """Test position reduction and closing operations."""

    def test_reduce_long_position_partial(self):
        """Test partial reduction of long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Reduce by 30 shares at $160 (profit)
        realized_pnl = position.reduce_position(
            quantity=Quantity(Decimal("30")),
            exit_price=Price(Decimal("160")),
            commission=Money(Decimal("2")),
        )

        # P&L: 30 * (160 - 150) - 2 = 300 - 2 = 298
        assert realized_pnl.amount == Decimal("298")
        assert position.quantity.value == Decimal("70")  # Remaining
        assert position.realized_pnl.amount == Decimal("298")
        assert position.commission_paid.amount == Decimal("2")
        assert not position.is_closed()

    def test_reduce_short_position_partial(self):
        """Test partial reduction of short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        # Cover 30 shares at $140 (profit for short)
        realized_pnl = position.reduce_position(
            quantity=Quantity(Decimal("30")),
            exit_price=Price(Decimal("140")),
            commission=Money(Decimal("2")),
        )

        # P&L: 30 * (150 - 140) - 2 = 300 - 2 = 298
        assert realized_pnl.amount == Decimal("298")
        assert position.quantity.value == Decimal("-70")  # Remaining short
        assert position.realized_pnl.amount == Decimal("298")
        assert not position.is_closed()

    def test_reduce_position_complete(self):
        """Test complete position reduction (full close)."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Reduce entire position
        realized_pnl = position.reduce_position(
            quantity=Quantity(Decimal("100")),
            exit_price=Price(Decimal("155")),
            commission=Money(Decimal("5")),
        )

        # P&L: 100 * (155 - 150) - 5 = 500 - 5 = 495
        assert realized_pnl.amount == Decimal("495")
        assert position.quantity.value == Decimal("0")
        assert position.is_closed()
        assert position.closed_at is not None

    def test_reduce_position_excessive(self):
        """Test reducing position by more than available."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        with pytest.raises(
            ValueError, match="Cannot reduce position by 150, current quantity is 100"
        ):
            position.reduce_position(
                quantity=Quantity(Decimal("150")), exit_price=Price(Decimal("160"))
            )

    def test_reduce_position_zero_quantity(self):
        """Test reducing position by zero (should be no-op)."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        realized_pnl = position.reduce_position(
            quantity=Quantity(Decimal("0")), exit_price=Price(Decimal("160"))
        )

        assert realized_pnl.amount == Decimal("0")
        assert position.quantity.value == Decimal("100")  # Unchanged


class TestPositionClosing:
    """Test position closing operations."""

    def test_close_long_position_profit(self):
        """Test closing long position at profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        realized_pnl = position.close_position(
            exit_price=Price(Decimal("160")), commission=Money(Decimal("5"))
        )

        # P&L: 100 * (160 - 150) - 5 = 1000 - 5 = 995
        assert realized_pnl.amount == Decimal("995")
        assert position.is_closed()
        assert position.quantity.value == Decimal("0")
        assert position.closed_at is not None

    def test_close_short_position_profit(self):
        """Test closing short position at profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        realized_pnl = position.close_position(
            exit_price=Price(Decimal("140")), commission=Money(Decimal("5"))
        )

        # P&L: 100 * (150 - 140) - 5 = 1000 - 5 = 995
        assert realized_pnl.amount == Decimal("995")
        assert position.is_closed()
        assert position.quantity.value == Decimal("0")

    def test_close_position_at_loss(self):
        """Test closing position at a loss."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        realized_pnl = position.close_position(
            exit_price=Price(Decimal("140")), commission=Money(Decimal("5"))
        )

        # P&L: 100 * (140 - 150) - 5 = -1000 - 5 = -1005
        assert realized_pnl.amount == Decimal("-1005")
        assert position.is_closed()

    def test_close_already_closed_position(self):
        """Test closing already closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Close position first time
        position.close_position(Price(Decimal("160")))

        # Try to close again
        with pytest.raises(ValueError, match="Position is already closed"):
            position.close_position(Price(Decimal("165")))


class TestPositionPricingAndPnL:
    """Test position pricing updates and P&L calculations."""

    def test_update_market_price(self):
        """Test updating market price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        new_price = Price(Decimal("160"))
        position.update_market_price(new_price)

        assert position.current_price == new_price
        assert position.last_updated is not None

    def test_update_market_price_negative(self):
        """Test updating with negative price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        with pytest.raises(ValueError, match="Price must be positive"):
            position.update_market_price(Price(Decimal("-10")))

    def test_update_market_price_zero(self):
        """Test updating with zero price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        with pytest.raises(ValueError, match="Price must be positive"):
            position.update_market_price(Price(Decimal("0")))

    def test_get_unrealized_pnl_long_profit(self):
        """Test unrealized P&L for long position in profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))

        unrealized_pnl = position.get_unrealized_pnl()
        # 100 * (160 - 150) = 1000
        assert unrealized_pnl.amount == Decimal("1000")

    def test_get_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for long position in loss."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("140")))

        unrealized_pnl = position.get_unrealized_pnl()
        # 100 * (140 - 150) = -1000
        assert unrealized_pnl.amount == Decimal("-1000")

    def test_get_unrealized_pnl_short_profit(self):
        """Test unrealized P&L for short position in profit."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("140")))

        unrealized_pnl = position.get_unrealized_pnl()
        # 100 * (150 - 140) = 1000
        assert unrealized_pnl.amount == Decimal("1000")

    def test_get_unrealized_pnl_short_loss(self):
        """Test unrealized P&L for short position in loss."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))

        unrealized_pnl = position.get_unrealized_pnl()
        # 100 * (150 - 160) = -1000
        assert unrealized_pnl.amount == Decimal("-1000")

    def test_get_unrealized_pnl_no_price(self):
        """Test unrealized P&L when no current price is set."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        unrealized_pnl = position.get_unrealized_pnl()
        assert unrealized_pnl is None

    def test_get_unrealized_pnl_closed_position(self):
        """Test unrealized P&L for closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.close_position(Price(Decimal("160")))

        unrealized_pnl = position.get_unrealized_pnl()
        assert unrealized_pnl is None


class TestPositionTotalPnL:
    """Test total P&L calculations (realized + unrealized)."""

    def test_get_total_pnl_open_position(self):
        """Test total P&L for open position with realized and unrealized."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("200")), entry_price=Price(Decimal("150"))
        )

        # Partially close for realized P&L
        position.reduce_position(
            quantity=Quantity(Decimal("50")),
            exit_price=Price(Decimal("160")),
            commission=Money(Decimal("5")),
        )

        # Update price for unrealized P&L on remaining
        position.update_market_price(Price(Decimal("165")))

        total_pnl = position.get_total_pnl()
        # Realized: 50 * (160 - 150) - 5 = 495
        # Unrealized: 150 * (165 - 150) = 2250
        # Commission already subtracted from realized
        # Total: 495 + 2250 - 5 = 2740
        expected = Money(Decimal("2740"))
        assert total_pnl == expected

    def test_get_total_pnl_closed_position(self):
        """Test total P&L for closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.close_position(exit_price=Price(Decimal("160")), commission=Money(Decimal("5")))

        total_pnl = position.get_total_pnl()
        # Only realized P&L: 100 * (160 - 150) - 5 = 995
        assert total_pnl.amount == Decimal("995")

    def test_get_total_pnl_no_unrealized(self):
        """Test total P&L when no unrealized P&L available."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Set realized P&L but no current price
        position.realized_pnl = Money(Decimal("100"))
        position.commission_paid = Money(Decimal("5"))

        total_pnl = position.get_total_pnl()
        assert total_pnl.amount == Decimal("100")  # Just realized


class TestPositionValue:
    """Test position value calculations."""

    def test_get_position_value(self):
        """Test position market value calculation."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))

        position_value = position.get_position_value()
        # 100 * 160 = 16000
        assert position_value.amount == Decimal("16000")

    def test_get_position_value_short(self):
        """Test position market value calculation for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))

        position_value = position.get_position_value()
        # abs(-100) * 160 = 16000
        assert position_value.amount == Decimal("16000")

    def test_get_position_value_no_price(self):
        """Test position value when no current price is set."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position_value = position.get_position_value()
        assert position_value is None


class TestPositionReturnPercentage:
    """Test return percentage calculations."""

    def test_get_return_percentage_profit(self):
        """Test return percentage for profitable position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("165")))

        return_pct = position.get_return_percentage()
        # Total P&L: 100 * (165 - 150) = 1500
        # Initial value: 100 * 150 = 15000
        # Return: (1500 / 15000) * 100 = 10%
        assert return_pct == Decimal("10")

    def test_get_return_percentage_loss(self):
        """Test return percentage for losing position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("135")))

        return_pct = position.get_return_percentage()
        # Total P&L: 100 * (135 - 150) = -1500
        # Initial value: 100 * 150 = 15000
        # Return: (-1500 / 15000) * 100 = -10%
        assert return_pct == Decimal("-10")

    def test_get_return_percentage_zero_entry_price(self):
        """Test that zero entry price is not allowed."""
        # Zero prices are invalid in financial systems
        # This test verifies that Price validation prevents zero values
        with pytest.raises(ValueError, match="Price must be positive"):
            Position(
                symbol="TEST",
                quantity=Quantity(Decimal("100")),
                average_entry_price=Price(Decimal("0")),
                closed_at=datetime.now(UTC),  # Mark as closed to pass validation
            )

    def test_get_return_percentage_no_total_pnl(self):
        """Test return percentage when no current price set."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # No current price set, unrealized P&L is None, but total P&L is realized P&L (0)
        # Return percentage is 0 when no gains/losses
        return_pct = position.get_return_percentage()
        assert return_pct == Decimal("0")


class TestPositionRiskManagement:
    """Test risk management features (stop loss, take profit)."""

    def test_set_stop_loss_long(self):
        """Test setting stop loss for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))
        stop_price = Price(Decimal("145"))

        position.set_stop_loss(stop_price)
        assert position.stop_loss_price == stop_price

    def test_set_stop_loss_short(self):
        """Test setting stop loss for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("140")))
        stop_price = Price(Decimal("155"))

        position.set_stop_loss(stop_price)
        assert position.stop_loss_price == stop_price

    def test_set_stop_loss_invalid_long(self):
        """Test setting invalid stop loss for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))

        # Stop loss above current price for long position
        with pytest.raises(
            ValueError, match="Stop loss for long position must be below current price"
        ):
            position.set_stop_loss(Price(Decimal("165")))

    def test_set_stop_loss_invalid_short(self):
        """Test setting invalid stop loss for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("140")))

        # Stop loss below current price for short position
        with pytest.raises(
            ValueError, match="Stop loss for short position must be above current price"
        ):
            position.set_stop_loss(Price(Decimal("135")))

    def test_set_take_profit_long(self):
        """Test setting take profit for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("160")))
        take_profit_price = Price(Decimal("180"))

        position.set_take_profit(take_profit_price)
        assert position.take_profit_price == take_profit_price

    def test_set_take_profit_short(self):
        """Test setting take profit for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.update_market_price(Price(Decimal("140")))
        take_profit_price = Price(Decimal("120"))

        position.set_take_profit(take_profit_price)
        assert position.take_profit_price == take_profit_price

    def test_should_stop_loss_long(self):
        """Test stop loss trigger for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.set_stop_loss(Price(Decimal("145")))

        # Price above stop loss
        position.update_market_price(Price(Decimal("147")))
        assert position.should_stop_loss() is False

        # Price at stop loss
        position.update_market_price(Price(Decimal("145")))
        assert position.should_stop_loss() is True

        # Price below stop loss
        position.update_market_price(Price(Decimal("143")))
        assert position.should_stop_loss() is True

    def test_should_stop_loss_short(self):
        """Test stop loss trigger for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.set_stop_loss(Price(Decimal("155")))

        # Price below stop loss
        position.update_market_price(Price(Decimal("153")))
        assert position.should_stop_loss() is False

        # Price at stop loss
        position.update_market_price(Price(Decimal("155")))
        assert position.should_stop_loss() is True

        # Price above stop loss
        position.update_market_price(Price(Decimal("157")))
        assert position.should_stop_loss() is True

    def test_should_take_profit_long(self):
        """Test take profit trigger for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        position.set_take_profit(Price(Decimal("180")))

        # Price below take profit
        position.update_market_price(Price(Decimal("175")))
        assert position.should_take_profit() is False

        # Price at take profit
        position.update_market_price(Price(Decimal("180")))
        assert position.should_take_profit() is True

        # Price above take profit
        position.update_market_price(Price(Decimal("185")))
        assert position.should_take_profit() is True

    def test_should_take_profit_short(self):
        """Test take profit trigger for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        position.set_take_profit(Price(Decimal("120")))

        # Price above take profit
        position.update_market_price(Price(Decimal("125")))
        assert position.should_take_profit() is False

        # Price at take profit
        position.update_market_price(Price(Decimal("120")))
        assert position.should_take_profit() is True

        # Price below take profit
        position.update_market_price(Price(Decimal("115")))
        assert position.should_take_profit() is True


class TestPositionClosingMethod:
    """Test the close() method for final position closing."""

    def test_close_method_long(self):
        """Test close method for long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        final_price = Price(Decimal("160"))
        close_time = datetime.now(UTC)

        position.close(final_price, close_time)

        # Check final state
        assert position.is_closed()
        assert position.quantity.value == Decimal("0")
        assert position.current_price == final_price
        assert position.closed_at == close_time
        assert position.last_updated == close_time
        # P&L: 100 * (160 - 150) = 1000
        assert position.realized_pnl.amount == Decimal("1000")

    def test_close_method_short(self):
        """Test close method for short position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("-100")), entry_price=Price(Decimal("150"))
        )

        final_price = Price(Decimal("140"))
        close_time = datetime.now(UTC)

        position.close(final_price, close_time)

        # Check final state
        assert position.is_closed()
        assert position.quantity.value == Decimal("0")
        assert position.current_price == final_price
        assert position.closed_at == close_time
        # P&L: 100 * (150 - 140) = 1000
        assert position.realized_pnl.amount == Decimal("1000")

    def test_close_method_already_closed(self):
        """Test close method on already closed position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Close first time
        position.close(Price(Decimal("160")), datetime.now(UTC))

        # Try to close again
        with pytest.raises(ValueError, match="Position is already closed"):
            position.close(Price(Decimal("165")), datetime.now(UTC))


class TestPositionStringRepresentation:
    """Test position string representation."""

    def test_str_representation_open_long(self):
        """Test string representation for open long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150.50"))
        )

        position.update_market_price(Price(Decimal("160")))

        str_repr = str(position)

        assert "AAPL" in str_repr
        assert "LONG" in str_repr
        assert "100" in str_repr
        assert "150.50" in str_repr
        assert "OPEN" in str_repr
        assert "Unrealized P&L" in str_repr

    def test_str_representation_open_short(self):
        """Test string representation for open short position."""
        position = Position.open_position(
            symbol="MSFT", quantity=Quantity(Decimal("-50")), entry_price=Price(Decimal("300"))
        )

        position.update_market_price(Price(Decimal("295")))

        str_repr = str(position)

        assert "MSFT" in str_repr
        assert "SHORT" in str_repr
        assert "50" in str_repr
        assert "300" in str_repr
        assert "OPEN" in str_repr
        assert "Unrealized P&L" in str_repr

    def test_str_representation_closed(self):
        """Test string representation for closed position."""
        position = Position.open_position(
            symbol="GOOGL", quantity=Quantity(Decimal("10")), entry_price=Price(Decimal("2500"))
        )

        position.close_position(Price(Decimal("2600")))

        str_repr = str(position)

        assert "GOOGL" in str_repr
        assert "LONG" in str_repr
        assert "10" in str_repr
        assert "2500" in str_repr
        assert "CLOSED" in str_repr
        assert "Realized P&L" in str_repr


class TestPositionComplexScenarios:
    """Test complex position scenarios."""

    def test_multiple_additions_and_reductions(self):
        """Test multiple position modifications."""
        position = Position.open_position(
            symbol="AAPL", quantity=Quantity(Decimal("100")), entry_price=Price(Decimal("150"))
        )

        # Add more shares
        position.add_to_position(
            quantity=Quantity(Decimal("50")),
            price=Price(Decimal("160")),
            commission=Money(Decimal("2")),
        )

        # Partial close
        realized_pnl_1 = position.reduce_position(
            quantity=Quantity(Decimal("75")),
            exit_price=Price(Decimal("165")),
            commission=Money(Decimal("3")),
        )

        # Add more again
        position.add_to_position(
            quantity=Quantity(Decimal("25")),
            price=Price(Decimal("170")),
            commission=Money(Decimal("1")),
        )

        # Final close
        position.update_market_price(Price(Decimal("175")))
        final_value = position.get_position_value()

        # Verify complex state is maintained correctly
        assert position.quantity.value == Decimal("100")  # 150 - 75 + 25
        assert position.commission_paid.amount == Decimal("6")  # 2 + 3 + 1
        assert not position.is_closed()
        assert final_value.amount == Decimal("17500")  # 100 * 175

    def test_fractional_shares(self):
        """Test position with fractional shares."""
        position = Position.open_position(
            symbol="BRK.A", quantity=Quantity(Decimal("0.1")), entry_price=Price(Decimal("500000"))
        )

        position.update_market_price(Price(Decimal("510000")))

        unrealized_pnl = position.get_unrealized_pnl()
        position_value = position.get_position_value()

        # 0.1 * (510000 - 500000) = 1000
        assert unrealized_pnl.amount == Decimal("1000")
        # 0.1 * 510000 = 51000
        assert position_value.amount == Decimal("51000")

    def test_high_precision_calculations(self):
        """Test position calculations maintain precision."""
        position = Position.open_position(
            symbol="CRYPTO",
            quantity=Quantity(Decimal("123.456789")),
            entry_price=Price(Decimal("98.765432")),
        )

        position.update_market_price(Price(Decimal("99.876543")))

        unrealized_pnl = position.get_unrealized_pnl()
        # Should maintain full decimal precision
        expected = Money(Decimal("123.456789") * (Decimal("99.876543") - Decimal("98.765432")))
        assert unrealized_pnl == expected
