"""
Comprehensive test suite for Order entity - achieving full coverage.
Tests all methods, state transitions, edge cases, and business rules.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from src.domain.entities.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


class TestOrderEnums:
    """Test order enumerations."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"

    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"


class TestOrderRequest:
    """Test OrderRequest dataclass."""

    def test_order_request_creation(self):
        """Test creating an order request."""
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Decimal("150.00"),
            stop_price=Decimal("145.00"),
            time_in_force=TimeInForce.GTC,
            reason="Test order",
        )

        assert request.symbol == "AAPL"
        assert request.quantity == Decimal("100")
        assert request.side == OrderSide.BUY
        assert request.limit_price == Decimal("150.00")
        assert request.stop_price == Decimal("145.00")
        assert request.time_in_force == TimeInForce.GTC
        assert request.reason == "Test order"

    def test_order_request_defaults(self):
        """Test order request with default values."""
        request = OrderRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
        )

        assert request.symbol == "MSFT"
        assert request.quantity == Decimal("50")
        assert request.side == OrderSide.SELL
        assert request.limit_price is None
        assert request.stop_price is None
        assert request.time_in_force == TimeInForce.DAY
        assert request.reason is None


class TestOrderInitialization:
    """Test Order initialization and validation."""

    def test_order_creation_with_defaults(self):
        """Test creating an order with default values."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        assert isinstance(order.id, UUID)
        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")
        assert order.broker_order_id is None
        assert order.average_fill_price is None
        assert isinstance(order.created_at, datetime)
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.cancelled_at is None

    def test_order_creation_with_all_attributes(self):
        """Test creating an order with all attributes specified."""
        order = Order(
            symbol="TSLA",
            quantity=Decimal("200"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("250.00"),
            stop_price=Decimal("245.00"),
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            reason="Risk management",
            tags={"strategy": "momentum"},
        )

        assert order.symbol == "TSLA"
        assert order.quantity == Decimal("200")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("250.00")
        assert order.stop_price == Decimal("245.00")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Risk management"
        assert order.tags == {"strategy": "momentum"}

    def test_order_validation_empty_symbol(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Order symbol cannot be empty"):
            Order(symbol="", quantity=Decimal("100"))

    def test_order_validation_zero_quantity(self):
        """Test that zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("0"))

    def test_order_validation_negative_quantity(self):
        """Test that negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("-100"))

    def test_order_validation_limit_order_without_price(self):
        """Test that limit order without limit price raises ValueError."""
        with pytest.raises(ValueError, match="Limit order requires limit price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                order_type=OrderType.LIMIT,
            )

    def test_order_validation_stop_order_without_price(self):
        """Test that stop order without stop price raises ValueError."""
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                order_type=OrderType.STOP,
            )

    def test_order_validation_stop_limit_without_prices(self):
        """Test that stop-limit order without both prices raises ValueError."""
        # Missing both prices
        with pytest.raises(
            ValueError, match="Stop-limit order requires both stop and limit prices"
        ):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                order_type=OrderType.STOP_LIMIT,
            )

        # Missing limit price
        with pytest.raises(
            ValueError, match="Stop-limit order requires both stop and limit prices"
        ):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                order_type=OrderType.STOP_LIMIT,
                stop_price=Decimal("145.00"),
            )

        # Missing stop price
        with pytest.raises(
            ValueError, match="Stop-limit order requires both stop and limit prices"
        ):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("150.00"),
            )

    def test_order_validation_negative_filled_quantity(self):
        """Test that negative filled quantity raises ValueError."""
        with pytest.raises(ValueError, match="Filled quantity cannot be negative"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                filled_quantity=Decimal("-10"),
            )

    def test_order_validation_filled_exceeds_quantity(self):
        """Test that filled quantity exceeding order quantity raises ValueError."""
        with pytest.raises(ValueError, match="Filled quantity cannot exceed order quantity"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                filled_quantity=Decimal("150"),
            )


class TestOrderFactoryMethods:
    """Test Order factory methods."""

    def test_create_market_order(self):
        """Test creating a market order using factory method."""
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            reason="Test market order",
        )

        order = Order.create_market_order(request)

        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.reason == "Test market order"
        assert order.limit_price is None
        assert order.stop_price is None

    def test_create_limit_order(self):
        """Test creating a limit order using factory method."""
        request = OrderRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("350.00"),
            time_in_force=TimeInForce.GTC,
            reason="Take profit",
        )

        order = Order.create_limit_order(request)

        assert order.symbol == "MSFT"
        assert order.quantity == Decimal("50")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("350.00")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Take profit"

    def test_create_limit_order_without_price(self):
        """Test that creating limit order without price raises ValueError."""
        request = OrderRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
        )

        with pytest.raises(ValueError, match="Limit price is required for limit orders"):
            Order.create_limit_order(request)

    def test_create_stop_order(self):
        """Test creating a stop order using factory method."""
        request = OrderRequest(
            symbol="TSLA",
            quantity=Decimal("25"),
            side=OrderSide.SELL,
            stop_price=Decimal("200.00"),
            reason="Stop loss",
        )

        order = Order.create_stop_order(request)

        assert order.symbol == "TSLA"
        assert order.quantity == Decimal("25")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP
        assert order.stop_price == Decimal("200.00")
        assert order.reason == "Stop loss"

    def test_create_stop_order_without_price(self):
        """Test that creating stop order without price raises ValueError."""
        request = OrderRequest(
            symbol="TSLA",
            quantity=Decimal("25"),
            side=OrderSide.SELL,
        )

        with pytest.raises(ValueError, match="Stop price is required for stop orders"):
            Order.create_stop_order(request)

    def test_create_stop_limit_order(self):
        """Test creating a stop-limit order using factory method."""
        request = OrderRequest(
            symbol="NVDA",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            stop_price=Decimal("450.00"),
            limit_price=Decimal("455.00"),
            time_in_force=TimeInForce.GTC,
            reason="Entry on breakout",
        )

        order = Order.create_stop_limit_order(request)

        assert order.symbol == "NVDA"
        assert order.quantity == Decimal("10")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == Decimal("450.00")
        assert order.limit_price == Decimal("455.00")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Entry on breakout"

    def test_create_stop_limit_order_without_prices(self):
        """Test that creating stop-limit order without prices raises ValueError."""
        request = OrderRequest(
            symbol="NVDA",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
        )

        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)

        # Test with only stop price
        request.stop_price = Decimal("450.00")
        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)

        # Test with only limit price
        request.stop_price = None
        request.limit_price = Decimal("455.00")
        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)


class TestOrderStateTransitions:
    """Test order state transitions and methods."""

    def test_submit_order(self):
        """Test submitting an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        assert order.status == OrderStatus.PENDING
        assert order.broker_order_id is None
        assert order.submitted_at is None

        order.submit("BROKER123")

        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "BROKER123"
        assert order.submitted_at is not None
        assert isinstance(order.submitted_at, datetime)

    def test_submit_non_pending_order(self):
        """Test that submitting a non-pending order raises ValueError."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        # Try to submit again
        with pytest.raises(ValueError, match="Cannot submit order in .* status"):
            order.submit("BROKER456")

    def test_fill_order_completely(self):
        """Test filling an order completely."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        fill_time = datetime.now(UTC)
        order.fill(Decimal("100"), Decimal("150.00"), fill_time)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.average_fill_price == Decimal("150.00")
        assert order.filled_at == fill_time

    def test_fill_order_partially(self):
        """Test partially filling an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        # First partial fill
        order.fill(Decimal("30"), Decimal("150.00"))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("30")
        assert order.average_fill_price == Decimal("150.00")
        assert order.filled_at is None

        # Second partial fill
        order.fill(Decimal("40"), Decimal("151.00"))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("70")
        # Average price: (30 * 150 + 40 * 151) / 70 = 150.5714...
        expected_avg = (
            Decimal("30") * Decimal("150.00") + Decimal("40") * Decimal("151.00")
        ) / Decimal("70")
        assert order.average_fill_price == expected_avg

        # Final fill
        order.fill(Decimal("30"), Decimal("152.00"))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.filled_at is not None

    def test_fill_order_invalid_status(self):
        """Test that filling order in invalid status raises ValueError."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        # Try to fill pending order
        with pytest.raises(ValueError, match="Cannot fill order in .* status"):
            order.fill(Decimal("50"), Decimal("150.00"))

        # Fill completely and try again
        order.submit("BROKER123")
        order.fill(Decimal("100"), Decimal("150.00"))

        with pytest.raises(ValueError, match="Cannot fill order in .* status"):
            order.fill(Decimal("10"), Decimal("151.00"))

    def test_fill_order_invalid_quantity(self):
        """Test that invalid fill quantities raise ValueError."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        # Zero quantity
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("0"), Decimal("150.00"))

        # Negative quantity
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("-10"), Decimal("150.00"))

        # Exceeding order quantity
        with pytest.raises(
            ValueError, match="Total filled quantity 150 exceeds order quantity 100"
        ):
            order.fill(Decimal("150"), Decimal("150.00"))

    def test_fill_order_invalid_price(self):
        """Test that invalid fill prices raise ValueError."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        # Zero price
        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("50"), Decimal("0"))

        # Negative price
        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("50"), Decimal("-150.00"))

    def test_cancel_order(self):
        """Test cancelling an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        order.cancel("User requested")

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert order.tags["cancel_reason"] == "User requested"

    def test_cancel_order_without_reason(self):
        """Test cancelling an order without providing a reason."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        order.cancel()

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert "cancel_reason" not in order.tags

    def test_cancel_invalid_status(self):
        """Test that cancelling order in invalid status raises ValueError."""
        # Test filled order
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")
        order.fill(Decimal("100"), Decimal("150.00"))

        with pytest.raises(ValueError, match="Cannot cancel order in .* status"):
            order.cancel()

        # Test already cancelled order
        order2 = Order(symbol="MSFT", quantity=Decimal("50"))
        order2.submit("BROKER456")
        order2.cancel()

        with pytest.raises(ValueError, match="Cannot cancel order in .* status"):
            order2.cancel()

        # Test rejected order
        order3 = Order(symbol="TSLA", quantity=Decimal("25"))
        order3.reject("Insufficient funds")

        with pytest.raises(ValueError, match="Cannot cancel order in .* status"):
            order3.cancel()

    def test_reject_order(self):
        """Test rejecting an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.tags["reject_reason"] == "Insufficient funds"

    def test_reject_non_pending_order(self):
        """Test that rejecting non-pending order raises ValueError."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        order.submit("BROKER123")

        with pytest.raises(ValueError, match="Cannot reject order in .* status"):
            order.reject("Invalid symbol")


class TestOrderQueryMethods:
    """Test order query and calculation methods."""

    def test_is_active(self):
        """Test checking if order is active."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        # Pending is active
        assert order.is_active() is True

        # Submitted is active
        order.submit("BROKER123")
        assert order.is_active() is True

        # Partially filled is active
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_active() is True

        # Filled is not active
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_active() is False

        # Test other terminal states
        order2 = Order(symbol="MSFT", quantity=Decimal("50"))
        order2.reject("Invalid")
        assert order2.is_active() is False

        order3 = Order(symbol="TSLA", quantity=Decimal("25"))
        order3.submit("BROKER789")
        order3.cancel()
        assert order3.is_active() is False

    def test_is_complete(self):
        """Test checking if order is complete."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        # Pending is not complete
        assert order.is_complete() is False

        # Submitted is not complete
        order.submit("BROKER123")
        assert order.is_complete() is False

        # Partially filled is not complete
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_complete() is False

        # Filled is complete
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_complete() is True

        # Test other terminal states
        order2 = Order(symbol="MSFT", quantity=Decimal("50"))
        order2.reject("Invalid")
        assert order2.is_complete() is True

        order3 = Order(symbol="TSLA", quantity=Decimal("25"))
        order3.submit("BROKER789")
        order3.cancel()
        assert order3.is_complete() is True

        # Test expired status
        order4 = Order(symbol="NVDA", quantity=Decimal("10"), status=OrderStatus.EXPIRED)
        assert order4.is_complete() is True

    def test_get_remaining_quantity(self):
        """Test getting remaining quantity to be filled."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        # Full quantity remaining initially
        assert order.get_remaining_quantity() == Decimal("100")

        # Partial fill
        order.submit("BROKER123")
        order.fill(Decimal("30"), Decimal("150.00"))
        assert order.get_remaining_quantity() == Decimal("70")

        # More partial fill
        order.fill(Decimal("45"), Decimal("151.00"))
        assert order.get_remaining_quantity() == Decimal("25")

        # Complete fill
        order.fill(Decimal("25"), Decimal("152.00"))
        assert order.get_remaining_quantity() == Decimal("0")

    def test_get_fill_ratio(self):
        """Test getting fill ratio."""
        order = Order(symbol="AAPL", quantity=Decimal("100"))

        # No fills yet
        assert order.get_fill_ratio() == Decimal("0")

        # Partial fill
        order.submit("BROKER123")
        order.fill(Decimal("25"), Decimal("150.00"))
        assert order.get_fill_ratio() == Decimal("0.25")

        # More partial fill
        order.fill(Decimal("50"), Decimal("151.00"))
        assert order.get_fill_ratio() == Decimal("0.75")

        # Complete fill
        order.fill(Decimal("25"), Decimal("152.00"))
        assert order.get_fill_ratio() == Decimal("1")

    def test_get_fill_ratio_zero_quantity(self):
        """Test getting fill ratio with zero quantity edge case."""
        # This shouldn't happen due to validation, but test the calculation
        order = Order.__new__(Order)  # Skip __init__ to bypass validation
        order.quantity = Decimal("0")
        order.filled_quantity = Decimal("0")

        assert order.get_fill_ratio() == Decimal("0")

    def test_get_notional_value_market_order(self):
        """Test getting notional value for market order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), order_type=OrderType.MARKET)

        # No fills yet
        assert order.get_notional_value() is None

        # After fill
        order.submit("BROKER123")
        order.fill(Decimal("100"), Decimal("150.00"))
        assert order.get_notional_value() == Decimal("15000.00")

        # Partial fill
        order2 = Order(symbol="MSFT", quantity=Decimal("50"), order_type=OrderType.MARKET)
        order2.submit("BROKER456")
        order2.fill(Decimal("30"), Decimal("350.00"))
        assert order2.get_notional_value() == Decimal("10500.00")

    def test_get_notional_value_limit_order(self):
        """Test getting notional value for limit order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        # Uses limit price * quantity
        assert order.get_notional_value() == Decimal("15000.00")

    def test_get_notional_value_stop_order(self):
        """Test getting notional value for stop order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("145.00"),
        )

        # Stop orders don't have a definite notional value until executed
        assert order.get_notional_value() is None

    def test_get_notional_value_stop_limit_order(self):
        """Test getting notional value for stop-limit order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("145.00"),
            limit_price=Decimal("150.00"),
        )

        # Stop-limit orders don't have a definite notional value until executed
        assert order.get_notional_value() is None


class TestOrderWithQuantityObject:
    """Test Order with Quantity value object (testing duck typing)."""

    class MockQuantity:
        """Mock Quantity value object for testing."""

        def __init__(self, value):
            self = value

        def __gt__(self, other):
            return self > other

        def __le__(self, other):
            return self <= other

    def test_order_with_quantity_object(self):
        """Test that Order works with Quantity-like objects."""
        qty = self.MockDecimal("100")
        order = Order(symbol="AAPL", quantity=qty)

        assert order.quantity == Decimal("100")

        # Test validation with quantity object
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(symbol="AAPL", quantity=self.MockDecimal("0"))

        with pytest.raises(ValueError, match="Order quantity must be positive"):
            Order(symbol="AAPL", quantity=self.MockDecimal("-10"))

    def test_fill_with_quantity_object(self):
        """Test filling order with Quantity-like object."""
        qty = self.MockDecimal("100")
        order = Order(symbol="AAPL", quantity=qty)
        order.submit("BROKER123")

        # Partial fill
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.filled_quantity == Decimal("50")
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Complete fill
        order.fill(Decimal("50"), Decimal("151.00"))
        assert order.filled_quantity == Decimal("100")
        assert order.status == OrderStatus.FILLED

    def test_get_remaining_with_quantity_object(self):
        """Test get_remaining_quantity with Quantity-like object."""
        qty = self.MockDecimal("100")
        order = Order(symbol="AAPL", quantity=qty)

        assert order.get_remaining_quantity() == Decimal("100")

        order.submit("BROKER123")
        order.fill(Decimal("30"), Decimal("150.00"))
        assert order.get_remaining_quantity() == Decimal("70")

    def test_get_notional_with_price_object(self):
        """Test get_notional_value with Price-like object."""

        class MockPrice:
            def __init__(self, value):
                self = value

        price = MockPrice(Decimal("150.00"))
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=price,
        )

        notional = order.get_notional_value()
        assert notional == Decimal("15000.00")


class TestOrderStringRepresentation:
    """Test Order string representation."""

    def test_market_order_string(self):
        """Test string representation of market order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order_str = str(order)

        assert "BUY" in order_str
        assert "100" in order_str
        assert "AAPL" in order_str
        assert "pending" in order_str

    def test_limit_order_string(self):
        """Test string representation of limit order."""
        order = Order(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("350.00"),
        )
        order_str = str(order)

        assert "SELL" in order_str
        assert "50" in order_str
        assert "MSFT" in order_str
        assert "@ $350.00" in order_str

    def test_stop_order_string(self):
        """Test string representation of stop order."""
        order = Order(
            symbol="TSLA",
            quantity=Decimal("25"),
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("200.00"),
        )
        order_str = str(order)

        assert "SELL" in order_str
        assert "25" in order_str
        assert "TSLA" in order_str
        assert "stop @ $200.00" in order_str

    def test_filled_order_string(self):
        """Test string representation of filled order."""
        order = Order(symbol="NVDA", quantity=Decimal("10"), side=OrderSide.BUY)
        order.submit("BROKER123")
        order.fill(Decimal("10"), Decimal("450.00"))
        order_str = str(order)

        assert "BUY" in order_str
        assert "10" in order_str
        assert "NVDA" in order_str
        assert "filled" in order_str


class TestOrderEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_multiple_partial_fills_average_price(self):
        """Test complex average price calculation with multiple fills."""
        order = Order(symbol="AAPL", quantity=Decimal("1000"))
        order.submit("BROKER123")

        # Fill 1: 200 @ 150.00
        order.fill(Decimal("200"), Decimal("150.00"))
        assert order.average_fill_price == Decimal("150.00")

        # Fill 2: 300 @ 151.50
        order.fill(Decimal("300"), Decimal("151.50"))
        expected = (
            Decimal("200") * Decimal("150.00") + Decimal("300") * Decimal("151.50")
        ) / Decimal("500")
        assert order.average_fill_price == expected

        # Fill 3: 250 @ 149.75
        order.fill(Decimal("250"), Decimal("149.75"))
        expected = (
            Decimal("200") * Decimal("150.00")
            + Decimal("300") * Decimal("151.50")
            + Decimal("250") * Decimal("149.75")
        ) / Decimal("750")
        assert order.average_fill_price == expected

        # Fill 4: 250 @ 150.25
        order.fill(Decimal("250"), Decimal("150.25"))
        expected = (
            Decimal("200") * Decimal("150.00")
            + Decimal("300") * Decimal("151.50")
            + Decimal("250") * Decimal("149.75")
            + Decimal("250") * Decimal("150.25")
        ) / Decimal("1000")
        assert order.average_fill_price == expected
        assert order.status == OrderStatus.FILLED

    def test_order_lifecycle_complex(self):
        """Test complex order lifecycle with all transitions."""
        # Create pending order
        order = Order(
            symbol="AAPL",
            quantity=Decimal("500"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            reason="Long-term investment",
            tags={"strategy": "value", "sector": "tech"},
        )

        assert order.is_active() is True
        assert order.is_complete() is False

        # Submit to broker
        order.submit("ALPACA-12345")
        assert order.broker_order_id == "ALPACA-12345"
        assert order.status == OrderStatus.SUBMITTED

        # Partial fills
        order.fill(Decimal("100"), Decimal("149.95"))
        assert order.status == OrderStatus.PARTIALLY_FILLED

        order.fill(Decimal("150"), Decimal("149.90"))
        assert order.status == OrderStatus.PARTIALLY_FILLED

        order.fill(Decimal("200"), Decimal("150.00"))
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Calculate expected average
        total_cost = (
            Decimal("100") * Decimal("149.95")
            + Decimal("150") * Decimal("149.90")
            + Decimal("200") * Decimal("150.00")
        )
        expected_avg = total_cost / Decimal("450")
        assert order.average_fill_price == expected_avg

        # Final fill
        final_time = datetime.now(UTC)
        order.fill(Decimal("50"), Decimal("149.85"), final_time)
        assert order.status == OrderStatus.FILLED
        assert order.filled_at == final_time
        assert order.is_active() is False
        assert order.is_complete() is True

    def test_order_with_all_time_in_force_values(self):
        """Test orders with different time in force values."""
        # DAY order
        day_order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            time_in_force=TimeInForce.DAY,
        )
        assert day_order.time_in_force == TimeInForce.DAY

        # GTC order
        gtc_order = Order(
            symbol="MSFT",
            quantity=Decimal("50"),
            time_in_force=TimeInForce.GTC,
        )
        assert gtc_order.time_in_force == TimeInForce.GTC

        # IOC order
        ioc_order = Order(
            symbol="TSLA",
            quantity=Decimal("25"),
            time_in_force=TimeInForce.IOC,
        )
        assert ioc_order.time_in_force == TimeInForce.IOC

        # FOK order
        fok_order = Order(
            symbol="NVDA",
            quantity=Decimal("10"),
            time_in_force=TimeInForce.FOK,
        )
        assert fok_order.time_in_force == TimeInForce.FOK

    def test_order_tags_manipulation(self):
        """Test order tags and metadata handling."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            tags={"initial": "value", "priority": 1},
        )

        # Initial tags
        assert order.tags["initial"] == "value"
        assert order.tags["priority"] == 1

        # Add tags during lifecycle
        order.tags["executed_by"] = "algo_trader"
        order.tags["risk_score"] = 0.75

        # Cancel with reason adds to tags
        order.submit("BROKER123")
        order.cancel("Risk limit exceeded")
        assert order.tags["cancel_reason"] == "Risk limit exceeded"
        assert order.tags["initial"] == "value"  # Original tags preserved

    def test_order_timestamps_precision(self):
        """Test that timestamps have proper precision and timezone."""
        before = datetime.now(UTC)
        order = Order(symbol="AAPL", quantity=Decimal("100"))
        after = datetime.now(UTC)

        # Created_at should be between before and after
        assert before <= order.created_at <= after
        assert order.created_at.tzinfo is not None

        # Submit and check submitted_at
        order.submit("BROKER123")
        assert order.submitted_at is not None
        assert order.submitted_at.tzinfo is not None
        assert order.submitted_at >= order.created_at

        # Fill and check filled_at
        order.fill(Decimal("100"), Decimal("150.00"))
        assert order.filled_at is not None
        assert order.filled_at.tzinfo is not None
        assert order.filled_at >= order.submitted_at

    def test_boundary_values(self):
        """Test with boundary values."""
        # Very small quantity
        order1 = Order(symbol="AAPL", quantity=Decimal("0.0001"))
        assert order1.quantity == Decimal("0.0001")

        # Very large quantity
        order2 = Order(symbol="MSFT", quantity=Decimal("1000000000"))
        assert order2.quantity == Decimal("1000000000")

        # Very high precision price
        order3 = Order(
            symbol="BTC",
            quantity=Decimal("1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("45678.123456789"),
        )
        assert order3.limit_price == Decimal("45678.123456789")

        # Test fill with high precision
        order3.submit("BROKER123")
        order3.fill(Decimal("0.5"), Decimal("45678.987654321"))
        order3.fill(Decimal("0.5"), Decimal("45679.111111111"))
        expected_avg = (
            Decimal("0.5") * Decimal("45678.987654321")
            + Decimal("0.5") * Decimal("45679.111111111")
        ) / Decimal("1")
        assert order3.average_fill_price == expected_avg
