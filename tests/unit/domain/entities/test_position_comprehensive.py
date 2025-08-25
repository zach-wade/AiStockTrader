"""
Comprehensive test suite for Position Entity

Achieves >85% coverage by testing:
- Position initialization and validation
- Position lifecycle (open, add, reduce, close)
- P&L calculations
- Stop loss and take profit logic
- Edge cases and error conditions
- Position state transitions
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.entities.position import Position


class TestPositionInitialization:
    """Test suite for Position initialization."""

    def test_default_initialization(self):
        """Test position with minimal required fields."""
        position = Position(symbol="AAPL", quantity=Decimal("10"))

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("10")
        assert position.average_entry_price == Decimal("0")
        assert position.current_price is None
        assert position.last_updated is None
        assert position.realized_pnl == Decimal("0")
        assert position.commission_paid == Decimal("0")
        assert position.stop_loss_price is None
        assert position.take_profit_price is None
        assert position.max_position_value is None
        assert isinstance(position.opened_at, datetime)
        assert position.closed_at is None
        assert position.strategy is None
        assert len(position.tags) == 0

    def test_custom_initialization(self):
        """Test position with custom values."""
        position = Position(
            symbol="GOOGL",
            quantity=Decimal("50"),
            average_entry_price=Decimal("2500.00"),
            current_price=Decimal("2550.00"),
            realized_pnl=Decimal("100.00"),
            commission_paid=Decimal("5.00"),
            stop_loss_price=Decimal("2400.00"),
            take_profit_price=Decimal("2600.00"),
            strategy="momentum",
            tags={"sector": "tech"},
        )

        assert position.symbol == "GOOGL"
        assert position.quantity == Decimal("50")
        assert position.average_entry_price == Decimal("2500.00")
        assert position.current_price == Decimal("2550.00")
        assert position.realized_pnl == Decimal("100.00")
        assert position.commission_paid == Decimal("5.00")
        assert position.stop_loss_price == Decimal("2400.00")
        assert position.take_profit_price == Decimal("2600.00")
        assert position.strategy == "momentum"
        assert position.tags["sector"] == "tech"

    def test_validation_empty_symbol(self):
        """Test validation rejects empty symbol."""
        with pytest.raises(ValueError, match="Position symbol cannot be empty"):
            Position(symbol="")

    def test_validation_negative_entry_price(self):
        """Test validation rejects negative entry price."""
        with pytest.raises(ValueError, match="Average entry price cannot be negative"):
            Position(symbol="AAPL", average_entry_price=Decimal("-100"))

    def test_validation_zero_quantity_open(self):
        """Test validation rejects zero quantity for open position."""
        with pytest.raises(ValueError, match="Open position cannot have zero quantity"):
            Position(symbol="AAPL", quantity=Decimal("0"), average_entry_price=Decimal("100.00"))

    def test_validation_zero_quantity_closed(self):
        """Test validation allows zero quantity for closed position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("100.00"),
            closed_at=datetime.now(UTC),
        )

        assert position.quantity == Decimal("0")
        assert position.is_closed()


class TestPositionFactory:
    """Test suite for Position factory methods."""

    def test_open_position_long(self):
        """Test opening a long position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            strategy="value",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.commission_paid == Decimal("1.00")
        assert position.strategy == "value"
        assert position.is_long() is True
        assert position.is_short() is False
        assert position.is_closed() is False

    def test_open_position_short(self):
        """Test opening a short position."""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("2.00"),
        )

        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("700.00")
        assert position.commission_paid == Decimal("2.00")
        assert position.is_long() is False
        assert position.is_short() is True
        assert position.is_closed() is False

    def test_open_position_zero_quantity(self):
        """Test opening position with zero quantity fails."""
        with pytest.raises(ValueError, match="Cannot open position with zero quantity"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("0"), entry_price=Decimal("150.00")
            )

    def test_open_position_negative_price(self):
        """Test opening position with negative price fails."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("-150.00")
            )

    def test_open_position_zero_price(self):
        """Test opening position with zero price fails."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position.open_position(symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("0"))


class TestPositionManagement:
    """Test suite for position management operations."""

    @pytest.fixture
    def long_position(self):
        """Create a long position."""
        return Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position."""
        return Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("2.00"),
        )

    def test_add_to_long_position(self, long_position):
        """Test adding to a long position."""
        long_position.add_to_position(
            quantity=Decimal("50"), price=Decimal("160.00"), commission=Decimal("1.00")
        )

        assert long_position.quantity == Decimal("150")
        # New average: (100*150 + 50*160) / 150 = 153.33...
        expected_avg = (Decimal("100") * Decimal("150") + Decimal("50") * Decimal("160")) / Decimal(
            "150"
        )
        assert long_position.average_entry_price == expected_avg
        assert long_position.commission_paid == Decimal("2.00")

    def test_add_to_short_position(self, short_position):
        """Test adding to a short position."""
        short_position.add_to_position(
            quantity=Decimal("-25"), price=Decimal("680.00"), commission=Decimal("1.00")
        )

        assert short_position.quantity == Decimal("-75")
        # New average: (50*700 + 25*680) / 75 = 693.33...
        expected_avg = (Decimal("50") * Decimal("700") + Decimal("25") * Decimal("680")) / Decimal(
            "75"
        )
        assert short_position.average_entry_price == expected_avg
        assert short_position.commission_paid == Decimal("3.00")

    def test_add_zero_quantity(self, long_position):
        """Test adding zero quantity does nothing."""
        original_qty = long_position.quantity
        original_avg = long_position.average_entry_price

        long_position.add_to_position(quantity=Decimal("0"), price=Decimal("160.00"))

        assert long_position.quantity == original_qty
        assert long_position.average_entry_price == original_avg

    def test_add_wrong_direction_to_long(self, long_position):
        """Test adding short quantity to long position fails."""
        with pytest.raises(ValueError, match="Cannot add short quantity to long position"):
            long_position.add_to_position(quantity=Decimal("-50"), price=Decimal("160.00"))

    def test_add_wrong_direction_to_short(self, short_position):
        """Test adding long quantity to short position fails."""
        with pytest.raises(ValueError, match="Cannot add long quantity to short position"):
            short_position.add_to_position(quantity=Decimal("50"), price=Decimal("680.00"))

    def test_reduce_long_position_partial(self, long_position):
        """Test partially reducing a long position."""
        pnl = long_position.reduce_position(
            quantity=Decimal("30"), exit_price=Decimal("160.00"), commission=Decimal("1.00")
        )

        expected_pnl = Decimal("30") * (Decimal("160") - Decimal("150")) - Decimal("1")
        assert pnl == expected_pnl  # 300 - 1 = 299
        assert long_position.quantity == Decimal("70")
        assert long_position.realized_pnl == expected_pnl
        assert long_position.commission_paid == Decimal("2.00")
        assert long_position.is_closed() is False

    def test_reduce_short_position_partial(self, short_position):
        """Test partially reducing a short position."""
        pnl = short_position.reduce_position(
            quantity=Decimal("20"), exit_price=Decimal("680.00"), commission=Decimal("1.00")
        )

        expected_pnl = Decimal("20") * (Decimal("700") - Decimal("680")) - Decimal("1")
        assert pnl == expected_pnl  # 400 - 1 = 399
        assert short_position.quantity == Decimal("-30")
        assert short_position.realized_pnl == expected_pnl
        assert short_position.commission_paid == Decimal("3.00")
        assert short_position.is_closed() is False

    def test_reduce_position_fully(self, long_position):
        """Test fully reducing a position."""
        pnl = long_position.reduce_position(
            quantity=Decimal("100"), exit_price=Decimal("160.00"), commission=Decimal("1.00")
        )

        expected_pnl = Decimal("100") * (Decimal("160") - Decimal("150")) - Decimal("1")
        assert pnl == expected_pnl  # 1000 - 1 = 999
        assert long_position.quantity == Decimal("0")
        assert long_position.is_closed() is True
        assert long_position.closed_at is not None

    def test_reduce_position_zero_quantity(self, long_position):
        """Test reducing by zero quantity returns zero P&L."""
        pnl = long_position.reduce_position(quantity=Decimal("0"), exit_price=Decimal("160.00"))

        assert pnl == Decimal("0")
        assert long_position.quantity == Decimal("100")

    def test_reduce_position_excessive_quantity(self, long_position):
        """Test reducing by more than position quantity fails."""
        with pytest.raises(ValueError, match="Cannot reduce position by 150"):
            long_position.reduce_position(quantity=Decimal("150"), exit_price=Decimal("160.00"))

    def test_close_long_position_profit(self, long_position):
        """Test closing long position with profit."""
        pnl = long_position.close_position(exit_price=Decimal("160.00"), commission=Decimal("1.00"))

        expected_pnl = Decimal("100") * (Decimal("160") - Decimal("150")) - Decimal("1")
        assert pnl == expected_pnl  # 1000 - 1 = 999
        assert long_position.quantity == Decimal("0")
        assert long_position.is_closed() is True
        assert long_position.realized_pnl == expected_pnl

    def test_close_short_position_loss(self, short_position):
        """Test closing short position with loss."""
        pnl = short_position.close_position(
            exit_price=Decimal("720.00"), commission=Decimal("1.00")
        )

        expected_pnl = Decimal("50") * (Decimal("700") - Decimal("720")) - Decimal("1")
        assert pnl == expected_pnl  # -1000 - 1 = -1001
        assert short_position.quantity == Decimal("0")
        assert short_position.is_closed() is True
        assert short_position.realized_pnl == expected_pnl

    def test_close_already_closed_position(self, long_position):
        """Test closing already closed position fails."""
        long_position.close_position(Decimal("160.00"))

        with pytest.raises(ValueError, match="Position is already closed"):
            long_position.close_position(Decimal("170.00"))

    def test_close_method(self, long_position):
        """Test the close method."""
        close_time = datetime.now(UTC)

        long_position.close(final_price=Decimal("160.00"), close_time=close_time)

        assert long_position.is_closed() is True
        assert long_position.quantity == Decimal("0")
        assert long_position.current_price == Decimal("160.00")
        assert long_position.closed_at == close_time
        assert long_position.last_updated == close_time

        # P&L without commission
        expected_pnl = Decimal("100") * (Decimal("160") - Decimal("150"))
        assert long_position.realized_pnl == expected_pnl


class TestPositionPriceUpdates:
    """Test suite for position price updates."""

    @pytest.fixture
    def position(self):
        """Create a position for testing."""
        return Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

    def test_update_market_price(self, position):
        """Test updating market price."""
        position.update_market_price(Decimal("155.00"))

        assert position.current_price == Decimal("155.00")
        assert position.last_updated is not None

    def test_update_market_price_negative(self, position):
        """Test updating with negative price fails."""
        with pytest.raises(ValueError, match="Market price must be positive"):
            position.update_market_price(Decimal("-100.00"))

    def test_update_market_price_zero(self, position):
        """Test updating with zero price fails."""
        with pytest.raises(ValueError, match="Market price must be positive"):
            position.update_market_price(Decimal("0"))


class TestPositionRiskManagement:
    """Test suite for stop loss and take profit."""

    @pytest.fixture
    def long_position(self):
        """Create a long position with current price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.current_price = Decimal("155.00")
        return position

    @pytest.fixture
    def short_position(self):
        """Create a short position with current price."""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-50"), entry_price=Decimal("700.00")
        )
        position.current_price = Decimal("680.00")
        return position

    def test_set_stop_loss_long_valid(self, long_position):
        """Test setting valid stop loss for long position."""
        long_position.set_stop_loss(Decimal("145.00"))

        assert long_position.stop_loss_price == Decimal("145.00")

    def test_set_stop_loss_long_invalid(self, long_position):
        """Test setting invalid stop loss for long position."""
        with pytest.raises(
            ValueError, match="Stop loss for long position must be below current price"
        ):
            long_position.set_stop_loss(Decimal("160.00"))

    def test_set_stop_loss_short_valid(self, short_position):
        """Test setting valid stop loss for short position."""
        short_position.set_stop_loss(Decimal("690.00"))

        assert short_position.stop_loss_price == Decimal("690.00")

    def test_set_stop_loss_short_invalid(self, short_position):
        """Test setting invalid stop loss for short position."""
        with pytest.raises(
            ValueError, match="Stop loss for short position must be above current price"
        ):
            short_position.set_stop_loss(Decimal("670.00"))

    def test_set_stop_loss_negative(self, long_position):
        """Test setting negative stop loss fails."""
        with pytest.raises(ValueError, match="Stop loss price must be positive"):
            long_position.set_stop_loss(Decimal("-100.00"))

    def test_set_take_profit_long_valid(self, long_position):
        """Test setting valid take profit for long position."""
        long_position.set_take_profit(Decimal("165.00"))

        assert long_position.take_profit_price == Decimal("165.00")

    def test_set_take_profit_long_invalid(self, long_position):
        """Test setting invalid take profit for long position."""
        with pytest.raises(
            ValueError, match="Take profit for long position must be above current price"
        ):
            long_position.set_take_profit(Decimal("145.00"))

    def test_set_take_profit_short_valid(self, short_position):
        """Test setting valid take profit for short position."""
        short_position.set_take_profit(Decimal("670.00"))

        assert short_position.take_profit_price == Decimal("670.00")

    def test_set_take_profit_short_invalid(self, short_position):
        """Test setting invalid take profit for short position."""
        with pytest.raises(
            ValueError, match="Take profit for short position must be below current price"
        ):
            short_position.set_take_profit(Decimal("690.00"))

    def test_should_stop_loss_long_triggered(self, long_position):
        """Test stop loss trigger for long position."""
        long_position.stop_loss_price = Decimal("145.00")
        long_position.current_price = Decimal("144.00")

        assert long_position.should_stop_loss() is True

    def test_should_stop_loss_long_not_triggered(self, long_position):
        """Test stop loss not triggered for long position."""
        long_position.stop_loss_price = Decimal("145.00")
        long_position.current_price = Decimal("146.00")

        assert long_position.should_stop_loss() is False

    def test_should_stop_loss_short_triggered(self, short_position):
        """Test stop loss trigger for short position."""
        short_position.stop_loss_price = Decimal("690.00")
        short_position.current_price = Decimal("691.00")

        assert short_position.should_stop_loss() is True

    def test_should_stop_loss_no_price_set(self, long_position):
        """Test stop loss check when no stop loss is set."""
        assert long_position.should_stop_loss() is False

    def test_should_stop_loss_no_current_price(self, long_position):
        """Test stop loss check when no current price."""
        long_position.stop_loss_price = Decimal("145.00")
        long_position.current_price = None

        assert long_position.should_stop_loss() is False

    def test_should_take_profit_long_triggered(self, long_position):
        """Test take profit trigger for long position."""
        long_position.take_profit_price = Decimal("165.00")
        long_position.current_price = Decimal("166.00")

        assert long_position.should_take_profit() is True

    def test_should_take_profit_short_triggered(self, short_position):
        """Test take profit trigger for short position."""
        short_position.take_profit_price = Decimal("670.00")
        short_position.current_price = Decimal("669.00")

        assert short_position.should_take_profit() is True


class TestPositionCalculations:
    """Test suite for P&L and value calculations."""

    @pytest.fixture
    def long_position_with_price(self):
        """Create long position with current price."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("2.00"),
        )
        position.current_price = Decimal("160.00")
        return position

    @pytest.fixture
    def short_position_with_price(self):
        """Create short position with current price."""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("700.00"),
            commission=Decimal("3.00"),
        )
        position.current_price = Decimal("680.00")
        return position

    def test_get_unrealized_pnl_long_profit(self, long_position_with_price):
        """Test unrealized P&L for profitable long position."""
        unrealized_pnl = long_position_with_price.get_unrealized_pnl()

        # 100 * (160 - 150) = 1000
        assert unrealized_pnl == Decimal("1000")

    def test_get_unrealized_pnl_short_profit(self, short_position_with_price):
        """Test unrealized P&L for profitable short position."""
        unrealized_pnl = short_position_with_price.get_unrealized_pnl()

        # 50 * (700 - 680) = 1000
        assert unrealized_pnl == Decimal("1000")

    def test_get_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for losing long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.current_price = Decimal("140.00")

        unrealized_pnl = position.get_unrealized_pnl()

        # 100 * (140 - 150) = -1000
        assert unrealized_pnl == Decimal("-1000")

    def test_get_unrealized_pnl_closed_position(self, long_position_with_price):
        """Test unrealized P&L for closed position returns None."""
        long_position_with_price.closed_at = datetime.now(UTC)
        long_position_with_price.quantity = Decimal("0")

        unrealized_pnl = long_position_with_price.get_unrealized_pnl()

        assert unrealized_pnl is None

    def test_get_unrealized_pnl_no_current_price(self):
        """Test unrealized P&L without current price returns None."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        unrealized_pnl = position.get_unrealized_pnl()

        assert unrealized_pnl is None

    def test_get_total_pnl(self, long_position_with_price):
        """Test total P&L calculation."""
        # Add some realized P&L
        long_position_with_price.realized_pnl = Decimal("500")

        total_pnl = long_position_with_price.get_total_pnl()

        # Realized: 500, Unrealized: 1000, Commission: 2
        # Total: 500 + 1000 - 2 = 1498
        assert total_pnl == Decimal("1498")

    def test_get_total_pnl_no_unrealized(self):
        """Test total P&L with no unrealized component."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("2.00"),
        )
        position.realized_pnl = Decimal("500")

        total_pnl = position.get_total_pnl()

        # Only realized P&L
        assert total_pnl == Decimal("500")

    def test_get_position_value(self, long_position_with_price):
        """Test position value calculation."""
        value = long_position_with_price.get_position_value()

        # 100 * 160 = 16000
        assert value == Decimal("16000")

    def test_get_position_value_short(self, short_position_with_price):
        """Test position value for short position."""
        value = short_position_with_price.get_position_value()

        # abs(-50) * 680 = 34000
        assert value == Decimal("34000")

    def test_get_position_value_no_price(self):
        """Test position value without current price."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )

        value = position.get_position_value()

        assert value is None

    def test_get_return_percentage(self, long_position_with_price):
        """Test return percentage calculation."""
        # Set commission for accurate calculation
        long_position_with_price.commission_paid = Decimal("10")

        return_pct = long_position_with_price.get_return_percentage()

        # Total P&L: 1000 - 10 = 990
        # Initial value: 100 * 150 = 15000
        # Return: 990 / 15000 * 100 = 6.6%
        assert return_pct == Decimal("6.6")

    def test_get_return_percentage_zero_entry(self):
        """Test return percentage with zero entry price."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("0"),
            closed_at=datetime.now(UTC),
        )

        return_pct = position.get_return_percentage()

        assert return_pct is None


class TestPositionStateQueries:
    """Test suite for position state queries."""

    def test_is_long(self):
        """Test identifying long positions."""
        long_pos = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )

        assert long_pos.is_long() is True
        assert long_pos.is_short() is False

    def test_is_short(self):
        """Test identifying short positions."""
        short_pos = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("700.00"),
            closed_at=datetime.now(UTC),
        )

        assert short_pos.is_short() is True
        assert short_pos.is_long() is False

    def test_is_closed_by_quantity(self):
        """Test position closed by zero quantity."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )

        assert position.is_closed() is True

    def test_is_closed_by_timestamp(self):
        """Test position closed by timestamp."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),  # Non-zero quantity
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )

        assert position.is_closed() is True

    def test_is_open(self):
        """Test identifying open positions."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            closed_at=None,
        )
        position._validate = lambda: None  # Skip validation

        assert position.is_closed() is False


class TestPositionRepresentation:
    """Test suite for position string representation."""

    def test_str_long_open(self):
        """Test string representation of open long position."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("150.00")
        )
        position.current_price = Decimal("160.00")

        result = str(position)

        assert "AAPL" in result
        assert "LONG" in result
        assert "100" in result
        assert "150.00" in result
        assert "OPEN" in result
        assert "Unrealized P&L" in result

    def test_str_short_closed(self):
        """Test string representation of closed short position."""
        position = Position.open_position(
            symbol="TSLA", quantity=Decimal("-50"), entry_price=Decimal("700.00")
        )
        position.close_position(Decimal("680.00"))

        result = str(position)

        assert "TSLA" in result
        assert "SHORT" in result
        assert "0" in result  # closed position has 0 quantity
        assert "700.00" in result
        assert "CLOSED" in result
        assert "Realized P&L" in result


class TestPositionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_quantity(self):
        """Test position with very small quantity."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("0.001"), entry_price=Decimal("150.00")
        )

        assert position.quantity == Decimal("0.001")

    def test_very_large_quantity(self):
        """Test position with very large quantity."""
        position = Position.open_position(
            symbol="BRK.A", quantity=Decimal("1000000"), entry_price=Decimal("500000.00")
        )

        assert position.quantity == Decimal("1000000")

    def test_high_precision_prices(self):
        """Test position with high precision prices."""
        position = Position.open_position(
            symbol="CRYPTO", quantity=Decimal("100.123456"), entry_price=Decimal("0.000001234567")
        )

        position.add_to_position(quantity=Decimal("50.654321"), price=Decimal("0.000001345678"))

        assert position.quantity == Decimal("150.777777")

    def test_fractional_shares(self):
        """Test position with fractional shares."""
        position = Position.open_position(
            symbol="AAPL", quantity=Decimal("10.5"), entry_price=Decimal("150.00")
        )

        pnl = position.reduce_position(quantity=Decimal("5.25"), exit_price=Decimal("160.00"))

        assert position.quantity == Decimal("5.25")
        expected_pnl = Decimal("5.25") * Decimal("10")
        assert pnl == expected_pnl
