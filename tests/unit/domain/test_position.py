"""
Unit tests for Position entity
"""

# Standard library imports
from decimal import Decimal
from uuid import UUID

# Third-party imports
import pytest

# Local imports
from src.domain.entities import Position


class TestPositionCreation:
    """Test Position creation and validation"""

    def test_open_long_position(self):
        """Test opening a long position"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            strategy="momentum",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.commission_paid == Decimal("1.00")
        assert position.strategy == "momentum"
        assert position.is_long()
        assert not position.is_short()
        assert not position.is_closed()
        assert isinstance(position.id, UUID)

    def test_open_short_position(self):
        """Test opening a short position"""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("200.00"),
            commission=Decimal("2.00"),
        )

        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("200.00")
        assert position.commission_paid == Decimal("2.00")
        assert not position.is_long()
        assert position.is_short()
        assert not position.is_closed()

    def test_cannot_open_position_with_zero_quantity(self):
        """Test that position cannot be opened with zero quantity"""
        with pytest.raises(ValueError, match="Cannot open position with zero quantity"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("0"), entry_price=Decimal("150.00")
            )

    def test_cannot_open_position_with_negative_price(self):
        """Test that position cannot be opened with negative price"""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("-150.00")
            )

    def test_position_validation(self):
        """Test position validation rules"""
        # Empty symbol
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Position(symbol="", quantity=Decimal("100"), average_entry_price=Decimal("150"))

        # Negative average entry price
        with pytest.raises(ValueError, match="Average entry price cannot be negative"):
            Position(symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("-150"))


class TestPositionManagement:
    """Test position management operations"""

    def test_add_to_long_position(self):
        """Test adding to a long position"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

        position.add_to_position(
            quantity=Decimal("50"), price=Decimal("155.00"), commission=Decimal("0.50")
        )

        assert position.quantity == Decimal("150")
        # Average price: (100 * 150 + 50 * 155) / 150 = 151.666...
        expected_avg = (Decimal("100") * Decimal("150") + Decimal("50") * Decimal("155")) / Decimal(
            "150"
        )
        assert position.average_entry_price == expected_avg
        assert position.commission_paid == Decimal("1.50")

    def test_add_to_short_position(self):
        """Test adding to a short position"""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )

        position.add_to_position(quantity=Decimal("-50"), price=Decimal("195.00"))

        assert position.quantity == Decimal("-150")
        # Average price: (100 * 200 + 50 * 195) / 150 = 198.333...
        expected_avg = (Decimal("100") * Decimal("200") + Decimal("50") * Decimal("195")) / Decimal(
            "150"
        )
        assert position.average_entry_price == expected_avg

    def test_cannot_add_opposite_direction(self):
        """Test that cannot add opposite direction to position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Cannot add short quantity to long position"):
            position.add_to_position(Decimal("-50"), Decimal("150.00"))

        short_position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )

        with pytest.raises(ValueError, match="Cannot add long quantity to short position"):
            short_position.add_to_position(Decimal("50"), Decimal("200.00"))

    def test_reduce_long_position(self):
        """Test reducing a long position"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

        pnl = position.reduce_position(
            quantity=Decimal("50"), exit_price=Decimal("160.00"), commission=Decimal("0.50")
        )

        assert position.quantity == Decimal("50")
        # P&L: 50 * (160 - 150) - 0.50 = 500 - 0.50 = 499.50
        assert pnl == Decimal("499.50")
        assert position.realized_pnl == Decimal("499.50")
        assert position.commission_paid == Decimal("1.50")
        assert not position.is_closed()

    def test_reduce_short_position(self):
        """Test reducing a short position"""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )

        pnl = position.reduce_position(
            quantity=Decimal("50"), exit_price=Decimal("190.00"), commission=Decimal("1.00")
        )

        assert position.quantity == Decimal("-50")
        # P&L for short: 50 * (200 - 190) - 1.00 = 500 - 1.00 = 499.00
        assert pnl == Decimal("499.00")
        assert position.realized_pnl == Decimal("499.00")

    def test_cannot_reduce_more_than_position(self):
        """Test cannot reduce position by more than current quantity"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Cannot reduce position by"):
            position.reduce_position(Decimal("150"), Decimal("160.00"))

    def test_close_long_position_with_profit(self):
        """Test closing a long position with profit"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

        total_pnl = position.close_position(
            exit_price=Decimal("160.00"), commission=Decimal("1.00")
        )

        assert position.quantity == Decimal("0")
        assert position.is_closed()
        assert position.closed_at is not None
        # P&L: 100 * (160 - 150) - 1.00 = 1000 - 1.00 = 999.00
        assert total_pnl == Decimal("999.00")
        assert position.realized_pnl == Decimal("999.00")
        assert position.commission_paid == Decimal("2.00")

    def test_close_short_position_with_loss(self):
        """Test closing a short position with loss"""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-100"),
            entry_price=Decimal("200.00"),
            commission=Decimal("2.00"),
        )

        total_pnl = position.close_position(
            exit_price=Decimal("210.00"), commission=Decimal("2.00")
        )

        assert position.quantity == Decimal("0")
        assert position.is_closed()
        # P&L for short: 100 * (200 - 210) - 2.00 = -1000 - 2.00 = -1002.00
        assert total_pnl == Decimal("-1002.00")
        assert position.realized_pnl == Decimal("-1002.00")
        assert position.commission_paid == Decimal("4.00")

    def test_cannot_close_already_closed_position(self):
        """Test cannot close an already closed position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.close_position(Decimal("160.00"))

        with pytest.raises(ValueError, match="Position is already closed"):
            position.close_position(Decimal("165.00"))


class TestPnLCalculations:
    """Test P&L calculation methods"""

    def test_unrealized_pnl_long_position(self):
        """Test unrealized P&L for long position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        # No current price set
        assert position.get_unrealized_pnl() is None

        # Price up - profit
        position.update_market_price(Decimal("160.00"))
        # Unrealized: 100 * (160 - 150) = 1000
        assert position.get_unrealized_pnl() == Decimal("1000.00")

        # Price down - loss
        position.update_market_price(Decimal("145.00"))
        # Unrealized: 100 * (145 - 150) = -500
        assert position.get_unrealized_pnl() == Decimal("-500.00")

    def test_unrealized_pnl_short_position(self):
        """Test unrealized P&L for short position"""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )

        # Price down - profit for short
        position.update_market_price(Decimal("190.00"))
        # Unrealized: 100 * (200 - 190) = 1000
        assert position.get_unrealized_pnl() == Decimal("1000.00")

        # Price up - loss for short
        position.update_market_price(Decimal("210.00"))
        # Unrealized: 100 * (200 - 210) = -1000
        assert position.get_unrealized_pnl() == Decimal("-1000.00")

    def test_total_pnl_calculation(self):
        """Test total P&L calculation (realized + unrealized - commission)"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("2.00"),
        )

        # Partially close with profit
        position.reduce_position(
            quantity=Decimal("50"), exit_price=Decimal("160.00"), commission=Decimal("1.00")
        )

        # Set current price
        position.update_market_price(Decimal("165.00"))

        # Realized P&L: 50 * (160 - 150) - 1 = 499
        # Unrealized P&L: 50 * (165 - 150) = 750
        # Total commission: 3.00
        # Total P&L: 499 + 750 - 3 = 1246
        assert position.get_total_pnl() == Decimal("1246.00")

    def test_position_value_calculation(self):
        """Test position value calculation"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        # No current price
        assert position.get_position_value() is None

        # With current price
        position.update_market_price(Decimal("160.00"))
        assert position.get_position_value() == Decimal("16000.00")

        # Short position
        short_position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-50"), entry_price=Decimal("200.00")
        )
        short_position.update_market_price(Decimal("210.00"))
        # Value is absolute: 50 * 210 = 10500
        assert short_position.get_position_value() == Decimal("10500.00")

    def test_return_percentage_calculation(self):
        """Test return percentage calculation"""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        position.update_market_price(Decimal("165.00"))

        # Total P&L: 100 * (165 - 150) - 10 = 1490
        # Initial value: 100 * 150 = 15000
        # Return: 1490 / 15000 * 100 = 9.933...%
        expected_return = (Decimal("1490") / Decimal("15000")) * Decimal("100")
        assert abs(position.get_return_percentage() - expected_return) < Decimal("0.01")


class TestStopLossAndTakeProfit:
    """Test stop loss and take profit functionality"""

    def test_set_stop_loss_long_position(self):
        """Test setting stop loss for long position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.update_market_price(Decimal("155.00"))

        # Valid stop loss below current price
        position.set_stop_loss(Decimal("145.00"))
        assert position.stop_loss_price == Decimal("145.00")

        # Invalid stop loss above current price
        with pytest.raises(ValueError, match="Stop loss for long position must be below"):
            position.set_stop_loss(Decimal("160.00"))

    def test_set_stop_loss_short_position(self):
        """Test setting stop loss for short position"""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )
        position.update_market_price(Decimal("195.00"))

        # Valid stop loss above current price for short
        position.set_stop_loss(Decimal("205.00"))
        assert position.stop_loss_price == Decimal("205.00")

        # Invalid stop loss below current price for short
        with pytest.raises(ValueError, match="Stop loss for short position must be above"):
            position.set_stop_loss(Decimal("190.00"))

    def test_set_take_profit_long_position(self):
        """Test setting take profit for long position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.update_market_price(Decimal("155.00"))

        # Valid take profit above current price
        position.set_take_profit(Decimal("165.00"))
        assert position.take_profit_price == Decimal("165.00")

        # Invalid take profit below current price
        with pytest.raises(ValueError, match="Take profit for long position must be above"):
            position.set_take_profit(Decimal("150.00"))

    def test_should_stop_loss_trigger(self):
        """Test stop loss trigger detection"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.set_stop_loss(Decimal("145.00"))

        # Price above stop loss - no trigger
        position.update_market_price(Decimal("148.00"))
        assert not position.should_stop_loss()

        # Price at stop loss - trigger
        position.update_market_price(Decimal("145.00"))
        assert position.should_stop_loss()

        # Price below stop loss - trigger
        position.update_market_price(Decimal("143.00"))
        assert position.should_stop_loss()

    def test_should_take_profit_trigger(self):
        """Test take profit trigger detection"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.update_market_price(Decimal("155.00"))
        position.set_take_profit(Decimal("160.00"))

        # Price below take profit - no trigger
        position.update_market_price(Decimal("158.00"))
        assert not position.should_take_profit()

        # Price at take profit - trigger
        position.update_market_price(Decimal("160.00"))
        assert position.should_take_profit()

        # Price above take profit - trigger
        position.update_market_price(Decimal("162.00"))
        assert position.should_take_profit()


class TestPositionStringRepresentation:
    """Test Position string representation"""

    def test_long_position_string(self):
        """Test string representation of long position"""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.update_market_price(Decimal("160.00"))

        str_repr = str(position)
        assert "AAPL" in str_repr
        assert "LONG 100" in str_repr
        assert "@ $150.00" in str_repr
        assert "OPEN" in str_repr
        assert "Unrealized P&L" in str_repr

    def test_closed_position_string(self):
        """Test string representation of closed position"""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )
        position.close_position(Decimal("190.00"))

        str_repr = str(position)
        assert "TSLA" in str_repr
        assert "SHORT" in str_repr
        assert "CLOSED" in str_repr
        assert "Realized P&L" in str_repr
