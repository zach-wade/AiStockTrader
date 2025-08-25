"""Comprehensive unit tests for Position entity to achieve 80%+ coverage.

This module provides extensive test coverage for the Position entity,
testing position lifecycle, P&L calculations, thread safety, and edge cases.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.position import Position


class TestPositionInitialization:
    """Test Position initialization and validation."""

    def test_position_default_initialization(self):
        """Test Position with minimal required fields."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.current_price is None
        assert position.last_updated is None
        assert position.realized_pnl == Decimal("0")
        assert position.commission_paid == Decimal("0")
        assert position.stop_loss_price is None
        assert position.take_profit_price is None
        assert position.max_position_value is None
        assert position.closed_at is None
        assert position.strategy is None
        assert isinstance(position.id, type(uuid4()))
        assert isinstance(position.opened_at, datetime)

    def test_position_full_initialization(self):
        """Test Position with all fields."""
        position_id = uuid4()
        opened_time = datetime.now(UTC)
        closed_time = opened_time + timedelta(hours=2)

        position = Position(
            id=position_id,
            symbol="TSLA",
            quantity=Decimal("-50"),  # Short position
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
            last_updated=opened_time + timedelta(minutes=30),
            realized_pnl=Decimal("500.00"),
            commission_paid=Decimal("10.00"),
            stop_loss_price=Decimal("660.00"),
            take_profit_price=Decimal("630.00"),
            max_position_value=Decimal("35000.00"),
            opened_at=opened_time,
            closed_at=closed_time,
            strategy="mean_reversion",
            tags={"risk_level": "high"},
        )

        assert position.id == position_id
        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.is_short() is True
        assert position.average_entry_price == Decimal("650.00")
        assert position.current_price == Decimal("640.00")
        assert position.realized_pnl == Decimal("500.00")
        assert position.commission_paid == Decimal("10.00")
        assert position.stop_loss_price == Decimal("660.00")
        assert position.take_profit_price == Decimal("630.00")
        assert position.strategy == "mean_reversion"
        assert position.tags["risk_level"] == "high"


class TestPositionValidation:
    """Test Position validation logic."""

    def test_empty_symbol_validation(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Position(symbol="", quantity=Decimal("100"), average_entry_price=Decimal("150.00"))

    def test_negative_entry_price_validation(self):
        """Test that negative entry price raises ValueError."""
        with pytest.raises(ValueError, match="Average entry price cannot be negative"):
            Position(symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("-150.00"))

    def test_zero_quantity_open_position_validation(self):
        """Test that open position with zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="Open position cannot have zero quantity"):
            Position(symbol="AAPL", quantity=Decimal("0"), average_entry_price=Decimal("150.00"))

    def test_zero_quantity_closed_position_valid(self):
        """Test that closed position with zero quantity is valid."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )
        assert position.quantity == Decimal("0")
        assert position.is_closed() is True


class TestPositionFactoryMethod:
    """Test Position factory method."""

    def test_open_position_long(self):
        """Test opening a long position."""
        position = Position.open_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            commission=Decimal("5.00"),
            strategy="momentum",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_entry_price == Decimal("150.00")
        assert position.commission_paid == Decimal("5.00")
        assert position.strategy == "momentum"
        assert position.is_long() is True

    def test_open_position_short(self):
        """Test opening a short position."""
        position = Position.open_position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            entry_price=Decimal("650.00"),
            commission=Decimal("10.00"),
        )

        assert position.symbol == "TSLA"
        assert position.quantity == Decimal("-50")
        assert position.average_entry_price == Decimal("650.00")
        assert position.commission_paid == Decimal("10.00")
        assert position.is_short() is True

    def test_open_position_zero_quantity(self):
        """Test opening position with zero quantity raises error."""
        with pytest.raises(ValueError, match="Cannot open position with zero quantity"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("0"), entry_price=Decimal("150.00")
            )

    def test_open_position_negative_price(self):
        """Test opening position with negative price raises error."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position.open_position(
                symbol="AAPL", quantity=Decimal("100"), entry_price=Decimal("-150.00")
            )


class TestPositionOperations:
    """Test position operations (add, reduce, close)."""

    @pytest.mark.asyncio
    async def test_add_to_long_position(self):
        """Test adding to a long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            commission_paid=Decimal("5.00"),
        )

        await position.add_to_position(Decimal("50"), Decimal("155.00"), Decimal("3.00"))

        assert position.quantity == Decimal("150")
        # New average: (100*150 + 50*155) / 150 = 151.67
        expected_avg = (
            Decimal("100") * Decimal("150.00") + Decimal("50") * Decimal("155.00")
        ) / Decimal("150")
        assert position.average_entry_price == expected_avg
        assert position.commission_paid == Decimal("8.00")

    @pytest.mark.asyncio
    async def test_add_to_short_position(self):
        """Test adding to a short position."""
        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        await position.add_to_position(Decimal("-30"), Decimal("645.00"), Decimal("5.00"))

        assert position.quantity == Decimal("-80")
        # New average: (50*650 + 30*645) / 80 = 648.125
        expected_avg = (
            Decimal("50") * Decimal("650.00") + Decimal("30") * Decimal("645.00")
        ) / Decimal("80")
        assert position.average_entry_price == expected_avg
        assert position.commission_paid == Decimal("5.00")

    @pytest.mark.asyncio
    async def test_add_wrong_direction_to_long(self):
        """Test adding short quantity to long position raises error."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Cannot add short quantity to long position"):
            await position.add_to_position(Decimal("-50"), Decimal("155.00"))

    @pytest.mark.asyncio
    async def test_add_wrong_direction_to_short(self):
        """Test adding long quantity to short position raises error."""
        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        with pytest.raises(ValueError, match="Cannot add long quantity to short position"):
            await position.add_to_position(Decimal("30"), Decimal("645.00"))

    @pytest.mark.asyncio
    async def test_reduce_long_position_profit(self):
        """Test reducing long position with profit."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        pnl = await position.reduce_position(Decimal("50"), Decimal("160.00"), Decimal("3.00"))

        # P&L: 50 * (160 - 150) - 3 = 497
        assert pnl == Decimal("497")
        assert position.quantity == Decimal("50")
        assert position.realized_pnl == Decimal("497")
        assert position.commission_paid == Decimal("3.00")
        assert position.is_closed() is False

    @pytest.mark.asyncio
    async def test_reduce_short_position_profit(self):
        """Test reducing short position with profit."""
        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        pnl = await position.reduce_position(Decimal("30"), Decimal("640.00"), Decimal("5.00"))

        # P&L: 30 * (650 - 640) - 5 = 295
        assert pnl == Decimal("295")
        assert position.quantity == Decimal("-20")
        assert position.realized_pnl == Decimal("295")
        assert position.commission_paid == Decimal("5.00")

    @pytest.mark.asyncio
    async def test_reduce_position_to_zero(self):
        """Test reducing position to zero marks it as closed."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        await position.reduce_position(Decimal("100"), Decimal("155.00"), Decimal("5.00"))

        assert position.quantity == Decimal("0")
        assert position.is_closed() is True
        assert position.closed_at is not None

    @pytest.mark.asyncio
    async def test_reduce_more_than_available(self):
        """Test reducing more than available quantity raises error."""
        position = Position(
            symbol="AAPL", quantity=Decimal("50"), average_entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Cannot reduce position by .*, current quantity is"):
            await position.reduce_position(Decimal("100"), Decimal("155.00"))

    def test_close_position_long_profit(self):
        """Test closing long position with profit."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        pnl = position.close_position(Decimal("160.00"), Decimal("5.00"))

        # P&L: 100 * (160 - 150) - 5 = 995
        assert pnl == Decimal("995")
        assert position.quantity == Decimal("0")
        assert position.is_closed() is True
        assert position.realized_pnl == Decimal("995")

    def test_close_position_short_loss(self):
        """Test closing short position with loss."""
        position = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )

        pnl = position.close_position(Decimal("660.00"), Decimal("5.00"))

        # P&L: 50 * (650 - 660) - 5 = -505
        assert pnl == Decimal("-505")
        assert position.is_closed() is True

    def test_close_already_closed_position(self):
        """Test closing already closed position raises error."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            closed_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="Position is already closed"):
            position.close_position(Decimal("155.00"))

    def test_close_method_with_time(self):
        """Test close method with specific time."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        close_time = datetime.now(UTC)
        position.close(Decimal("160.00"), close_time)

        assert position.quantity == Decimal("0")
        assert position.closed_at == close_time
        assert position.current_price == Decimal("160.00")
        assert position.realized_pnl == Decimal("1000")  # 100 * (160 - 150)


class TestPositionDirectMethods:
    """Test domain methods (now synchronous by default)."""

    def test_add_to_position(self):
        """Test add_to_position method."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        position.add_to_position(Decimal("50"), Decimal("155.00"), Decimal("3.00"))

        assert position.quantity == Decimal("150")
        assert position.commission_paid == Decimal("3.00")

    def test_reduce_position(self):
        """Test reduce_position method."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        pnl = position.reduce_position(Decimal("50"), Decimal("160.00"), Decimal("3.00"))

        assert pnl == Decimal("497")
        assert position.quantity == Decimal("50")

    def test_update_market_price(self):
        """Test update_market_price method."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        position.update_market_price(Decimal("155.00"))

        assert position.current_price == Decimal("155.00")
        assert position.last_updated is not None


class TestPositionPriceUpdates:
    """Test position price update methods."""

    @pytest.mark.asyncio
    async def test_update_market_price(self):
        """Test updating market price."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        await position.update_market_price(Decimal("155.00"))

        assert position.current_price == Decimal("155.00")
        assert position.last_updated is not None
        assert isinstance(position.last_updated, datetime)

    @pytest.mark.asyncio
    async def test_update_market_price_negative(self):
        """Test updating with negative price raises error."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Market price must be positive"):
            await position.update_market_price(Decimal("-155.00"))


class TestPositionRiskManagement:
    """Test position risk management features."""

    def test_set_stop_loss_long(self):
        """Test setting stop loss for long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        position.set_stop_loss(Decimal("145.00"))
        assert position.stop_loss_price == Decimal("145.00")

    def test_set_stop_loss_long_above_price(self):
        """Test setting stop loss above current price for long raises error."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        with pytest.raises(
            ValueError, match="Stop loss for long position must be below current price"
        ):
            position.set_stop_loss(Decimal("160.00"))

    def test_set_stop_loss_short(self):
        """Test setting stop loss for short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        position.set_stop_loss(Decimal("655.00"))
        assert position.stop_loss_price == Decimal("655.00")

    def test_set_stop_loss_short_below_price(self):
        """Test setting stop loss below current price for short raises error."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        with pytest.raises(
            ValueError, match="Stop loss for short position must be above current price"
        ):
            position.set_stop_loss(Decimal("635.00"))

    def test_set_stop_loss_negative(self):
        """Test setting negative stop loss raises error."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        with pytest.raises(ValueError, match="Stop loss price must be positive"):
            position.set_stop_loss(Decimal("-145.00"))

    def test_set_take_profit_long(self):
        """Test setting take profit for long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        position.set_take_profit(Decimal("165.00"))
        assert position.take_profit_price == Decimal("165.00")

    def test_set_take_profit_long_below_price(self):
        """Test setting take profit below current price for long raises error."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        with pytest.raises(
            ValueError, match="Take profit for long position must be above current price"
        ):
            position.set_take_profit(Decimal("150.00"))

    def test_set_take_profit_short(self):
        """Test setting take profit for short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        position.set_take_profit(Decimal("630.00"))
        assert position.take_profit_price == Decimal("630.00")

    def test_set_take_profit_short_above_price(self):
        """Test setting take profit above current price for short raises error."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        with pytest.raises(
            ValueError, match="Take profit for short position must be below current price"
        ):
            position.set_take_profit(Decimal("645.00"))

    def test_should_stop_loss_long(self):
        """Test stop loss trigger for long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            stop_loss_price=Decimal("145.00"),
        )

        # Price above stop loss
        position.current_price = Decimal("150.00")
        assert position.should_stop_loss() is False

        # Price at stop loss
        position.current_price = Decimal("145.00")
        assert position.should_stop_loss() is True

        # Price below stop loss
        position.current_price = Decimal("144.00")
        assert position.should_stop_loss() is True

    def test_should_stop_loss_short(self):
        """Test stop loss trigger for short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            stop_loss_price=Decimal("655.00"),
        )

        # Price below stop loss
        position.current_price = Decimal("650.00")
        assert position.should_stop_loss() is False

        # Price at stop loss
        position.current_price = Decimal("655.00")
        assert position.should_stop_loss() is True

        # Price above stop loss
        position.current_price = Decimal("656.00")
        assert position.should_stop_loss() is True

    def test_should_take_profit_long(self):
        """Test take profit trigger for long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            take_profit_price=Decimal("160.00"),
        )

        # Price below take profit
        position.current_price = Decimal("155.00")
        assert position.should_take_profit() is False

        # Price at take profit
        position.current_price = Decimal("160.00")
        assert position.should_take_profit() is True

        # Price above take profit
        position.current_price = Decimal("161.00")
        assert position.should_take_profit() is True

    def test_should_take_profit_short(self):
        """Test take profit trigger for short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            take_profit_price=Decimal("630.00"),
        )

        # Price above take profit
        position.current_price = Decimal("635.00")
        assert position.should_take_profit() is False

        # Price at take profit
        position.current_price = Decimal("630.00")
        assert position.should_take_profit() is True

        # Price below take profit
        position.current_price = Decimal("629.00")
        assert position.should_take_profit() is True


class TestPositionPnLCalculations:
    """Test P&L calculation methods."""

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized P&L for long position with profit."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        unrealized = position.get_unrealized_pnl()
        assert unrealized == Decimal("1000")  # 100 * (160 - 150)

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for long position with loss."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("145.00"),
        )

        unrealized = position.get_unrealized_pnl()
        assert unrealized == Decimal("-500")  # 100 * (145 - 150)

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized P&L for short position with profit."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        unrealized = position.get_unrealized_pnl()
        assert unrealized == Decimal("500")  # 50 * (650 - 640)

    def test_unrealized_pnl_short_loss(self):
        """Test unrealized P&L for short position with loss."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("660.00"),
        )

        unrealized = position.get_unrealized_pnl()
        assert unrealized == Decimal("-500")  # 50 * (650 - 660)

    def test_unrealized_pnl_closed_position(self):
        """Test unrealized P&L for closed position returns None."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
            closed_at=datetime.now(UTC),
        )

        assert position.get_unrealized_pnl() is None

    def test_unrealized_pnl_no_current_price(self):
        """Test unrealized P&L without current price returns None."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        assert position.get_unrealized_pnl() is None

    def test_total_pnl_open_position(self):
        """Test total P&L for open position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
            realized_pnl=Decimal("200"),
            commission_paid=Decimal("10"),
        )

        total = position.get_total_pnl()
        # Realized: 200, Unrealized: 1000, Commission: 10
        assert total == Decimal("1190")  # 200 + 1000 - 10

    def test_total_pnl_closed_position(self):
        """Test total P&L for closed position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            average_entry_price=Decimal("150.00"),
            realized_pnl=Decimal("500"),
            commission_paid=Decimal("10"),
            closed_at=datetime.now(UTC),
        )

        total = position.get_total_pnl()
        assert total == Decimal("500")  # Only realized P&L

    def test_position_value(self):
        """Test get_position_value method."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        value = position.get_position_value()
        assert value == Decimal("15500")  # 100 * 155

        # Short position
        position_short = Position(
            symbol="TSLA",
            quantity=Decimal("-50"),
            average_entry_price=Decimal("650.00"),
            current_price=Decimal("640.00"),
        )

        value_short = position_short.get_position_value()
        assert value_short == Decimal("32000")  # abs(-50) * 640

    def test_position_value_no_price(self):
        """Test position value without current price returns None."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        assert position.get_position_value() is None

    def test_return_percentage(self):
        """Test get_return_percentage method."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("165.00"),
        )

        return_pct = position.get_return_percentage()
        # P&L: 100 * (165 - 150) = 1500
        # Initial value: 100 * 150 = 15000
        # Return: 1500 / 15000 * 100 = 10%
        assert return_pct == Decimal("10")

        # With commission
        position.commission_paid = Decimal("50")
        return_pct = position.get_return_percentage()
        # P&L: 1500 - 50 = 1450
        # Return: 1450 / 15000 * 100 = 9.67%
        expected = (Decimal("1450") / Decimal("15000")) * Decimal("100")
        assert abs(return_pct - expected) < Decimal("0.01")

    def test_return_percentage_zero_entry(self):
        """Test return percentage with zero entry price returns None."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("0"),
            current_price=Decimal("150.00"),
        )

        assert position.get_return_percentage() is None


class TestPositionQueries:
    """Test position query methods."""

    def test_is_long(self):
        """Test is_long method."""
        position_long = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        assert position_long.is_long() is True
        assert position_long.is_short() is False

        position_short = Position(
            symbol="TSLA", quantity=Decimal("-50"), average_entry_price=Decimal("650.00")
        )
        assert position_short.is_long() is False
        assert position_short.is_short() is True

    def test_is_closed(self):
        """Test is_closed method."""
        # Open position
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        assert position.is_closed() is False

        # Closed with zero quantity
        position.quantity = Decimal("0")
        assert position.is_closed() is True

        # Closed with closed_at set
        position2 = Position(
            symbol="MSFT",
            quantity=Decimal("50"),
            average_entry_price=Decimal("350.00"),
            closed_at=datetime.now(UTC),
        )
        assert position2.is_closed() is True


class TestPositionStringRepresentation:
    """Test Position string representation."""

    def test_str_long_open(self):
        """Test string representation of open long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        str_repr = str(position)
        assert "AAPL" in str_repr
        assert "LONG" in str_repr
        assert "100" in str_repr
        assert "150.00" in str_repr
        assert "OPEN" in str_repr
        assert "Unrealized P&L" in str_repr

    def test_str_short_closed(self):
        """Test string representation of closed short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("0"),
            average_entry_price=Decimal("650.00"),
            realized_pnl=Decimal("500.00"),
            closed_at=datetime.now(UTC),
        )
        # Need to set original quantity for display
        position._original_quantity = Decimal("-50")

        str_repr = str(position)
        assert "TSLA" in str_repr
        assert "CLOSED" in str_repr
        assert "Realized P&L: $500.00" in str_repr


class TestPositionThreadSafety:
    """Test thread safety of position operations."""

    @pytest.mark.asyncio
    async def test_concurrent_adds(self):
        """Test concurrent add operations are thread-safe."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        # Create multiple concurrent add tasks
        tasks = []
        for i in range(10):
            tasks.append(
                position.add_to_position(Decimal("10"), Decimal("151.00"), Decimal("1.00"))
            )

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Verify final state
        assert position.quantity == Decimal("200")  # 100 + 10*10
        assert position.commission_paid == Decimal("10.00")  # 10 * 1

    @pytest.mark.asyncio
    async def test_concurrent_reduces(self):
        """Test concurrent reduce operations are thread-safe."""
        position = Position(
            symbol="AAPL", quantity=Decimal("1000"), average_entry_price=Decimal("150.00")
        )

        # Create multiple concurrent reduce tasks
        tasks = []
        for i in range(10):
            tasks.append(
                position.reduce_position(Decimal("50"), Decimal("155.00"), Decimal("2.00"))
            )

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify final state
        assert position.quantity == Decimal("500")  # 1000 - 10*50
        assert position.commission_paid == Decimal("20.00")  # 10 * 2

        # Each reduce should have returned correct P&L
        for pnl in results:
            # P&L: 50 * (155 - 150) - 2 = 248
            assert pnl == Decimal("248")

    @pytest.mark.asyncio
    async def test_concurrent_price_updates(self):
        """Test concurrent price updates are thread-safe."""
        position = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )

        # Create multiple concurrent price update tasks
        tasks = []
        prices = [Decimal(f"15{i}.00") for i in range(10)]
        for price in prices:
            tasks.append(position.update_market_price(price))

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Final price should be one of the prices we set
        assert position.current_price in prices
        assert position.last_updated is not None
