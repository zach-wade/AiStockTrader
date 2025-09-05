"""
Comprehensive test suite for Position entity - achieving full coverage.
Tests all methods, P&L calculations, risk management, and edge cases.
This file consolidates tests from multiple variant files.
"""

# Standard library imports
from datetime import UTC, datetime
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
        with pytest.raises(ValueError, match="Price must be positive"):
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


class TestPositionInitializationAdvanced:
    """Advanced test Position initialization and validation."""

    def test_position_creation_with_all_attributes(self):
        """Test creating a position with all attributes specified."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),  # Short position
            average_entry_price=Decimal("250.00"),
            current_price=Decimal("245.00"),
            last_updated=datetime.now(UTC),
            realized_pnl=Decimal("100.00"),
            commission_paid=Decimal("10.00"),
            stop_loss_price=Decimal("260.00"),
            take_profit_price=Decimal("240.00"),
            strategy="momentum",
            tags={"risk_level": "high"},
        )

        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("250.00")
        assert position.current_price == Decimal("245.00")
        assert position.realized_pnl == Decimal("100.00")
        assert position.commission_paid == Decimal("10.00")
        assert position.stop_loss_price == Decimal("260.00")
        assert position.take_profit_price == Decimal("240.00")
        assert position.strategy == "momentum"
        assert position.tags["risk_level"] == "high"

    def test_position_validation_comprehensive(self):
        """Test comprehensive position validation."""
        # Empty symbol
        with pytest.raises(ValueError, match="Position symbol cannot be empty"):
            Position(symbol="", quantity=Decimal("100"), average_entry_price=Decimal("150"))

        # Zero quantity
        with pytest.raises(ValueError, match="Position quantity cannot be zero"):
            Position(symbol="AAPL", quantity=Decimal("0"), average_entry_price=Decimal("150"))

        # Negative average entry price
        with pytest.raises(ValueError, match="Average entry price cannot be negative"):
            Position(symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("-150"))

        # Zero average entry price should be allowed for some edge cases
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("0")
        )
        assert position.average_entry_price == Decimal("0")

    def test_position_timestamps_tracking(self):
        """Test that all timestamps are properly tracked."""
        before = datetime.now(UTC)
        position = Position(
            symbol="NVDA", quantity=Decimal("50"), average_entry_price=Decimal("700.00")
        )
        after = datetime.now(UTC)

        # Check initial timestamp
        assert before <= position.opened_at <= after
        assert position.opened_at.tzinfo is not None
        assert position.closed_at is None
        assert position.last_updated is None

        # Update market price and check timestamp
        position.update_market_price(Decimal("720.00"))
        assert position.last_updated is not None
        assert position.last_updated >= position.opened_at

        # Close position and check timestamp
        position.close_position(Decimal("730.00"))
        assert position.closed_at is not None
        assert position.closed_at >= position.opened_at


class TestPositionAdvancedOperations:
    """Advanced position management operations."""

    def test_position_with_quantity_object(self):
        """Test position with Quantity-like object."""

        # Create a mock Quantity object
        class MockQuantity:
            def __init__(self, value):
                self.value = value

            def __gt__(self, other):
                return self.value > (other.value if hasattr(other, "value") else other)

            def __lt__(self, other):
                return self.value < (other.value if hasattr(other, "value") else other)

            def __eq__(self, other):
                return self.value == (other.value if hasattr(other, "value") else other)

            def __add__(self, other):
                other_val = other.value if hasattr(other, "value") else other
                return MockQuantity(self.value + other_val)

            def __sub__(self, other):
                other_val = other.value if hasattr(other, "value") else other
                return MockQuantity(self.value - other_val)

        qty = MockQuantity(Decimal("100"))
        position = Position.open_position(
            symbol="AAPL", quantity=qty, entry_price=Decimal("150.00")
        )

        # Verify the position handles the quantity object properly
        assert position.quantity.value == Decimal("100")

    def test_position_with_price_object(self):
        """Test position with Price-like object."""

        # Create a mock Price object
        class MockPrice:
            def __init__(self, value):
                self.value = value

        price = MockPrice(Decimal("150.00"))
        position = Position.open_position(symbol="AAPL", quantity=Decimal("100"), entry_price=price)

        # Verify the position handles the price object properly
        assert position.average_entry_price.value == Decimal("150.00")

    def test_multiple_additions_to_position(self):
        """Test multiple additions with complex average price calculation."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        # Multiple additions at different prices
        additions = [
            (Decimal("50"), Decimal("155.00")),
            (Decimal("75"), Decimal("148.00")),
            (Decimal("25"), Decimal("152.00")),
        ]

        total_cost = Decimal("100") * Decimal("150.00")
        total_qty = Decimal("100")

        for qty, price in additions:
            position.add_to_position(quantity=qty, price=price)
            total_cost += qty * price
            total_qty += qty

        expected_avg = total_cost / total_qty
        assert position.average_entry_price == expected_avg
        assert position.quantity == Decimal("250")

    def test_partial_reductions_complex(self):
        """Test complex partial reductions with P&L tracking."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("1000"),
            entry_price=Decimal("150.00"),
            commission=Decimal("10.00"),
        )

        # Multiple partial reductions
        reductions = [
            (Decimal("200"), Decimal("160.00"), Decimal("2.00")),
            (Decimal("300"), Decimal("155.00"), Decimal("3.00")),
            (Decimal("250"), Decimal("158.00"), Decimal("2.50")),
        ]

        total_realized_pnl = Decimal("0")

        for qty, price, commission in reductions:
            pnl = position.reduce_position(quantity=qty, exit_price=price, commission=commission)
            total_realized_pnl += pnl

        assert position.quantity == Decimal("250")
        assert position.realized_pnl == total_realized_pnl
        assert position.commission_paid == Decimal("17.50")  # 10 + 2 + 3 + 2.5

        # Check individual P&L calculations
        expected_pnl1 = Decimal("200") * (Decimal("160.00") - Decimal("150.00")) - Decimal(
            "2.00"
        )  # 1998
        expected_pnl2 = Decimal("300") * (Decimal("155.00") - Decimal("150.00")) - Decimal(
            "3.00"
        )  # 1497
        expected_pnl3 = Decimal("250") * (Decimal("158.00") - Decimal("150.00")) - Decimal(
            "2.50"
        )  # 1997.5

        expected_total = expected_pnl1 + expected_pnl2 + expected_pnl3
        assert abs(position.realized_pnl - expected_total) < Decimal("0.01")

    def test_position_lifecycle_comprehensive(self):
        """Test complete position lifecycle with all operations."""
        # Open position
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("500"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="long_term_growth",
        )

        assert not position.is_closed()
        assert position.is_long()

        # Set risk management levels
        position.update_market_price(Decimal("155.00"))
        position.set_stop_loss(Decimal("145.00"))
        position.set_take_profit(Decimal("170.00"))

        # Add to position
        position.add_to_position(
            quantity=Decimal("200"), price=Decimal("158.00"), commission=Decimal("2.00")
        )

        # Market moves up
        position.update_market_price(Decimal("165.00"))
        assert position.get_unrealized_pnl() > Decimal("0")

        # Partial reduction for profit taking
        pnl1 = position.reduce_position(
            quantity=Decimal("300"), exit_price=Decimal("165.00"), commission=Decimal("3.00")
        )
        assert pnl1 > Decimal("0")

        # Market moves down, stop loss triggers
        position.update_market_price(Decimal("144.00"))
        assert position.should_stop_loss()

        # Close remaining position
        final_pnl = position.close_position(
            exit_price=Decimal("144.00"), commission=Decimal("2.00")
        )

        assert position.is_closed()
        assert position.closed_at is not None
        assert position.quantity == Decimal("0")

        # Total commission should be tracked
        expected_total_commission = (
            Decimal("5.00") + Decimal("2.00") + Decimal("3.00") + Decimal("2.00")
        )
        assert position.commission_paid == expected_total_commission

    def test_boundary_values_position(self):
        """Test position with boundary values."""
        # Very small quantity
        position1 = Position.open_position(
            symbol="AAPL", quantity=Decimal("0.0001"), entry_price=Decimal("150.00")
        )
        assert position1.quantity == Decimal("0.0001")

        # Very large quantity
        position2 = Position.open_position(
            symbol="TSLA", quantity=Decimal("1000000000"), entry_price=Decimal("200.00")
        )
        assert position2.quantity == Decimal("1000000000")

        # Very high precision price
        position3 = Position.open_position(
            symbol="BTC", quantity=Decimal("1"), entry_price=Decimal("45678.123456789")
        )
        assert position3.average_entry_price == Decimal("45678.123456789")

        # Test P&L calculation with high precision
        position3.update_market_price(Decimal("46678.987654321"))
        expected_pnl = Decimal("1") * (Decimal("46678.987654321") - Decimal("45678.123456789"))
        assert position3.get_unrealized_pnl() == expected_pnl


class TestPositionRiskManagementAdvanced:
    """Advanced risk management tests."""

    def test_stop_loss_edge_cases(self):
        """Test stop loss edge cases and boundary conditions."""
        # Long position at exactly entry price
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.update_market_price(Decimal("150.00"))

        # Can set stop loss below entry
        position.set_stop_loss(Decimal("145.00"))
        assert position.stop_loss_price == Decimal("145.00")

        # Test stop loss trigger at exactly the price
        position.update_market_price(Decimal("145.00"))
        assert position.should_stop_loss()

        # Test stop loss not triggered above the price
        position.update_market_price(Decimal("145.01"))
        assert not position.should_stop_loss()

    def test_take_profit_edge_cases(self):
        """Test take profit edge cases and boundary conditions."""
        # Short position
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-100"), entry_price=Decimal("200.00")
        )
        position.update_market_price(Decimal("195.00"))

        # Can set take profit below current price for short
        position.set_take_profit(Decimal("190.00"))
        assert position.take_profit_price == Decimal("190.00")

        # Test take profit trigger at exactly the price
        position.update_market_price(Decimal("190.00"))
        assert position.should_take_profit()

        # Test take profit not triggered above the price
        position.update_market_price(Decimal("190.01"))
        assert not position.should_take_profit()

    def test_risk_management_without_current_price(self):
        """Test risk management when current price is not set."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        # No current price set
        assert position.current_price is None

        # Should be able to set stop loss and take profit
        position.set_stop_loss(Decimal("145.00"))
        position.set_take_profit(Decimal("160.00"))

        # But triggers should return False without current price
        assert not position.should_stop_loss()
        assert not position.should_take_profit()

    def test_max_position_value_tracking(self):
        """Test maximum position value tracking."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        # Initially no max value
        assert position.max_position_value is None

        # Update price upward
        position.update_market_price(Decimal("160.00"))
        assert position.max_position_value == Decimal("16000.00")

        # Price goes higher
        position.update_market_price(Decimal("170.00"))
        assert position.max_position_value == Decimal("17000.00")

        # Price comes down, max should remain
        position.update_market_price(Decimal("165.00"))
        assert position.max_position_value == Decimal("17000.00")

        # Price goes even higher
        position.update_market_price(Decimal("180.00"))
        assert position.max_position_value == Decimal("18000.00")


class TestPositionTagsAndMetadata:
    """Test position tags and metadata handling."""

    def test_position_tags_manipulation(self):
        """Test position tags and metadata handling."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            strategy="momentum",
        )

        # Add custom tags
        position.tags["entry_signal"] = "golden_cross"
        position.tags["confidence"] = 0.85
        position.tags["risk_level"] = "medium"

        assert position.tags["entry_signal"] == "golden_cross"
        assert position.tags["confidence"] == 0.85
        assert position.tags["risk_level"] == "medium"
        assert position.strategy == "momentum"

        # Update tags during position lifecycle
        position.tags["current_phase"] = "accumulation"
        position.tags["market_regime"] = "bull"

        assert len(position.tags) == 5

    def test_position_strategy_tracking(self):
        """Test position strategy tracking."""
        # Test with strategy
        position_with_strategy = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("50"),
            entry_price=Decimal("200.00"),
            strategy="breakout_momentum",
        )
        assert position_with_strategy.strategy == "breakout_momentum"

        # Test without strategy
        position_no_strategy = Position.open_position(
            symbol="MSFT", quantity=Decimal("75"), entry_price=Decimal("300.00")
        )
        assert position_no_strategy.strategy is None


class TestPositionStringRepresentationAdvanced:
    """Advanced position string representation tests."""

    def test_position_string_with_all_details(self):
        """Test detailed string representation."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="long_term",
        )

        position.update_market_price(Decimal("165.00"))
        position.set_stop_loss(Decimal("145.00"))
        position.set_take_profit(Decimal("180.00"))
        position.tags["risk_level"] = "medium"

        str_repr = str(position)

        # Check all key components are included
        assert "AAPL" in str_repr
        assert "LONG 100" in str_repr
        assert "@ $150.00" in str_repr
        assert "$165.00" in str_repr  # Current price
        assert "OPEN" in str_repr

    def test_short_position_string_comprehensive(self):
        """Test comprehensive short position string representation."""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-50"), entry_price=Decimal("250.00")
        )

        position.update_market_price(Decimal("240.00"))
        position.reduce_position(Decimal("20"), Decimal("235.00"))

        str_repr = str(position)
        assert "TSLA" in str_repr
        assert "SHORT" in str_repr
        assert "OPEN" in str_repr
