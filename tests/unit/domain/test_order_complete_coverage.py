"""Comprehensive unit tests for Order entity to achieve 80%+ coverage.

This module provides extensive test coverage for the Order entity,
testing all states, transitions, calculations, edge cases, and error conditions.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

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
    """Test enumeration values."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        assert len(OrderSide) == 2

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert len(OrderType) == 4

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert len(OrderStatus) == 7

    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"
        assert len(TimeInForce) == 4


class TestOrderRequest:
    """Test OrderRequest dataclass."""

    def test_order_request_creation(self):
        """Test creating an OrderRequest."""
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
        """Test OrderRequest default values."""
        request = OrderRequest(symbol="MSFT", quantity=Decimal("50"), side=OrderSide.SELL)

        assert request.limit_price is None
        assert request.stop_price is None
        assert request.time_in_force == TimeInForce.DAY
        assert request.reason is None


class TestOrderInitialization:
    """Test Order initialization and validation."""

    def test_order_default_initialization(self):
        """Test Order with minimal required fields."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)

        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.time_in_force == TimeInForce.DAY
        assert order.filled_quantity == Decimal("0")
        assert order.average_fill_price is None
        assert order.broker_order_id is None
        assert isinstance(order.id, type(uuid4()))
        assert isinstance(order.created_at, datetime)

    def test_order_full_initialization(self):
        """Test Order with all fields."""
        order_id = uuid4()
        created_time = datetime.now(UTC)

        order = Order(
            id=order_id,
            symbol="TSLA",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("650.00"),
            stop_price=Decimal("640.00"),
            status=OrderStatus.SUBMITTED,
            time_in_force=TimeInForce.GTC,
            broker_order_id="BRK123",
            filled_quantity=Decimal("25"),
            average_fill_price=Decimal("649.50"),
            created_at=created_time,
            submitted_at=created_time + timedelta(seconds=1),
            filled_at=None,
            cancelled_at=None,
            reason="Portfolio rebalancing",
            tags={"strategy": "momentum"},
        )

        assert order.id == order_id
        assert order.symbol == "TSLA"
        assert order.quantity == Decimal("50")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("650.00")
        assert order.stop_price == Decimal("640.00")
        assert order.status == OrderStatus.SUBMITTED
        assert order.time_in_force == TimeInForce.GTC
        assert order.broker_order_id == "BRK123"
        assert order.filled_quantity == Decimal("25")
        assert order.average_fill_price == Decimal("649.50")
        assert order.reason == "Portfolio rebalancing"
        assert order.tags["strategy"] == "momentum"


class TestOrderValidation:
    """Test Order validation logic."""

    def test_empty_symbol_validation(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Order(symbol="", quantity=Decimal("100"), side=OrderSide.BUY)

    def test_zero_quantity_validation(self):
        """Test that zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("0"), side=OrderSide.BUY)

    def test_negative_quantity_validation(self):
        """Test that negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("-100"), side=OrderSide.BUY)

    def test_limit_order_without_price_validation(self):
        """Test that limit order without limit price raises ValueError."""
        with pytest.raises(ValueError, match="Limit order requires limit price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
            )

    def test_stop_order_without_price_validation(self):
        """Test that stop order without stop price raises ValueError."""
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
            )

    def test_stop_limit_without_prices_validation(self):
        """Test that stop-limit order without prices raises ValueError."""
        with pytest.raises(
            ValueError, match="Stop-limit order requires both stop and limit prices"
        ):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("150.00"),
            )

    def test_negative_filled_quantity_validation(self):
        """Test that negative filled quantity raises ValueError."""
        with pytest.raises(ValueError, match="Filled quantity cannot be negative"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                filled_quantity=Decimal("-10"),
            )

    def test_filled_exceeds_quantity_validation(self):
        """Test that filled quantity exceeding order quantity raises ValueError."""
        with pytest.raises(ValueError, match="Filled quantity cannot exceed order quantity"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                filled_quantity=Decimal("150"),
            )


class TestOrderFactoryMethods:
    """Test Order factory methods."""

    def test_create_market_order(self):
        """Test creating market order via factory method."""
        request = OrderRequest(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, reason="Test market order"
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
        """Test creating limit order via factory method."""
        request = OrderRequest(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Decimal("350.00"),
            time_in_force=TimeInForce.GTC,
            reason="Test limit order",
        )

        order = Order.create_limit_order(request)

        assert order.symbol == "MSFT"
        assert order.quantity == Decimal("50")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("350.00")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Test limit order"

    def test_create_limit_order_without_price(self):
        """Test creating limit order without price raises error."""
        request = OrderRequest(symbol="MSFT", quantity=Decimal("50"), side=OrderSide.BUY)

        with pytest.raises(ValueError, match="Limit price is required"):
            Order.create_limit_order(request)

    def test_create_stop_order(self):
        """Test creating stop order via factory method."""
        request = OrderRequest(
            symbol="GOOGL",
            quantity=Decimal("25"),
            side=OrderSide.SELL,
            stop_price=Decimal("2500.00"),
            reason="Stop loss",
        )

        order = Order.create_stop_order(request)

        assert order.symbol == "GOOGL"
        assert order.quantity == Decimal("25")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP
        assert order.stop_price == Decimal("2500.00")
        assert order.reason == "Stop loss"

    def test_create_stop_order_without_price(self):
        """Test creating stop order without price raises error."""
        request = OrderRequest(symbol="GOOGL", quantity=Decimal("25"), side=OrderSide.SELL)

        with pytest.raises(ValueError, match="Stop price is required"):
            Order.create_stop_order(request)

    def test_create_stop_limit_order(self):
        """Test creating stop-limit order via factory method."""
        request = OrderRequest(
            symbol="NVDA",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            stop_price=Decimal("500.00"),
            limit_price=Decimal("505.00"),
            time_in_force=TimeInForce.IOC,
            reason="Entry order",
        )

        order = Order.create_stop_limit_order(request)

        assert order.symbol == "NVDA"
        assert order.quantity == Decimal("10")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == Decimal("500.00")
        assert order.limit_price == Decimal("505.00")
        assert order.time_in_force == TimeInForce.IOC
        assert order.reason == "Entry order"

    def test_create_stop_limit_order_without_prices(self):
        """Test creating stop-limit order without prices raises error."""
        request = OrderRequest(
            symbol="NVDA", quantity=Decimal("10"), side=OrderSide.BUY, stop_price=Decimal("500.00")
        )

        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)


class TestOrderStateTransitions:
    """Test Order state transitions."""

    def test_submit_order(self):
        """Test submitting an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)

        assert order.status == OrderStatus.PENDING
        assert order.broker_order_id is None
        assert order.submitted_at is None

        order.submit("BRK-123-456")

        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "BRK-123-456"
        assert order.submitted_at is not None
        assert isinstance(order.submitted_at, datetime)

    def test_submit_non_pending_order(self):
        """Test submitting non-pending order raises error."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.FILLED
        )

        with pytest.raises(ValueError, match="Cannot submit order in OrderStatus.FILLED status"):
            order.submit("BRK-123")

    def test_fill_order_completely(self):
        """Test filling an order completely."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        fill_time = datetime.now(UTC)
        order.fill(Decimal("100"), Decimal("150.50"), fill_time)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.average_fill_price == Decimal("150.50")
        assert order.filled_at == fill_time

    def test_fill_order_partially(self):
        """Test partially filling an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

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
        # Average price: (30*150 + 40*151) / 70 = 150.571...
        expected_avg = (
            Decimal("30") * Decimal("150.00") + Decimal("40") * Decimal("151.00")
        ) / Decimal("70")
        assert order.average_fill_price == expected_avg

        # Final fill
        order.fill(Decimal("30"), Decimal("152.00"))

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.filled_at is not None

    def test_fill_invalid_status(self):
        """Test filling order with invalid status."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.CANCELLED
        )

        with pytest.raises(ValueError, match="Cannot fill order in OrderStatus.CANCELLED status"):
            order.fill(Decimal("50"), Decimal("150.00"))

    def test_fill_zero_quantity(self):
        """Test filling with zero quantity raises error."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("0"), Decimal("150.00"))

    def test_fill_negative_price(self):
        """Test filling with negative price raises error."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("50"), Decimal("-150.00"))

    def test_fill_exceeds_quantity(self):
        """Test filling more than order quantity raises error."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        with pytest.raises(ValueError, match="Total filled quantity .* exceeds order quantity"):
            order.fill(Decimal("150"), Decimal("150.00"))

    def test_cancel_order(self):
        """Test cancelling an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        order.cancel("Market closed")

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert order.tags.get("cancel_reason") == "Market closed"

    def test_cancel_without_reason(self):
        """Test cancelling without reason."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        order.submit("BRK-123")

        order.cancel()

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert "cancel_reason" not in order.tags

    def test_cancel_filled_order(self):
        """Test cancelling filled order raises error."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.FILLED
        )

        with pytest.raises(ValueError, match="Cannot cancel order in OrderStatus.FILLED status"):
            order.cancel()

    def test_cancel_already_cancelled(self):
        """Test cancelling already cancelled order raises error."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.CANCELLED
        )

        with pytest.raises(ValueError, match="Cannot cancel order in OrderStatus.CANCELLED status"):
            order.cancel()

    def test_reject_order(self):
        """Test rejecting an order."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.tags.get("reject_reason") == "Insufficient funds"

    def test_reject_non_pending_order(self):
        """Test rejecting non-pending order raises error."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, status=OrderStatus.SUBMITTED
        )

        with pytest.raises(ValueError, match="Cannot reject order in OrderStatus.SUBMITTED status"):
            order.reject("Invalid symbol")


class TestOrderQueries:
    """Test Order query methods."""

    def test_is_active(self):
        """Test is_active method."""
        # Pending order is active
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        assert order.is_active() is True

        # Submitted order is active
        order.submit("BRK-123")
        assert order.is_active() is True

        # Partially filled is active
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_active() is True

        # Filled order is not active
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_active() is False

        # Cancelled order is not active
        order2 = Order(symbol="MSFT", quantity=Decimal("50"), side=OrderSide.SELL)
        order2.submit("BRK-456")
        order2.cancel()
        assert order2.is_active() is False

        # Rejected order is not active
        order3 = Order(symbol="GOOGL", quantity=Decimal("25"), side=OrderSide.BUY)
        order3.reject("Invalid")
        assert order3.is_active() is False

    def test_is_complete(self):
        """Test is_complete method."""
        # Pending order is not complete
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        assert order.is_complete() is False

        # Submitted order is not complete
        order.submit("BRK-123")
        assert order.is_complete() is False

        # Partially filled is not complete
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_complete() is False

        # Filled order is complete
        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_complete() is True

        # Cancelled order is complete
        order2 = Order(symbol="MSFT", quantity=Decimal("50"), side=OrderSide.SELL)
        order2.submit("BRK-456")
        order2.cancel()
        assert order2.is_complete() is True

        # Rejected order is complete
        order3 = Order(symbol="GOOGL", quantity=Decimal("25"), side=OrderSide.BUY)
        order3.reject("Invalid")
        assert order3.is_complete() is True

        # Expired order is complete
        order4 = Order(
            symbol="NVDA", quantity=Decimal("10"), side=OrderSide.BUY, status=OrderStatus.EXPIRED
        )
        assert order4.is_complete() is True

    def test_get_remaining_quantity(self):
        """Test get_remaining_quantity method."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)

        # Full quantity remaining initially
        assert order.get_remaining_quantity() == Decimal("100")

        # Partial fill reduces remaining
        order.submit("BRK-123")
        order.fill(Decimal("30"), Decimal("150.00"))
        assert order.get_remaining_quantity() == Decimal("70")

        # Another partial fill
        order.fill(Decimal("40"), Decimal("151.00"))
        assert order.get_remaining_quantity() == Decimal("30")

        # Complete fill
        order.fill(Decimal("30"), Decimal("152.00"))
        assert order.get_remaining_quantity() == Decimal("0")

    def test_get_fill_ratio(self):
        """Test get_fill_ratio method."""
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)

        # No fills yet
        assert order.get_fill_ratio() == Decimal("0")

        # Partial fill
        order.submit("BRK-123")
        order.fill(Decimal("25"), Decimal("150.00"))
        assert order.get_fill_ratio() == Decimal("0.25")

        # More fills
        order.fill(Decimal("25"), Decimal("151.00"))
        assert order.get_fill_ratio() == Decimal("0.50")

        # Complete fill
        order.fill(Decimal("50"), Decimal("152.00"))
        assert order.get_fill_ratio() == Decimal("1")

    def test_get_fill_ratio_zero_quantity(self):
        """Test get_fill_ratio with edge case."""
        # This shouldn't happen due to validation, but test defensive code
        order = Order.__new__(Order)  # Bypass __init__ validation
        order.quantity = Decimal("0")
        order.filled_quantity = Decimal("0")

        assert order.get_fill_ratio() == Decimal("0")

    def test_get_notional_value_market_order(self):
        """Test get_notional_value for market order."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )

        # No fills yet
        assert order.get_notional_value() is None

        # After fill
        order.submit("BRK-123")
        order.fill(Decimal("100"), Decimal("150.00"))
        assert order.get_notional_value() == Decimal("15000.00")

        # Partial fill
        order2 = Order(
            symbol="MSFT", quantity=Decimal("50"), side=OrderSide.SELL, order_type=OrderType.MARKET
        )
        order2.submit("BRK-456")
        order2.fill(Decimal("30"), Decimal("350.00"))
        assert order2.get_notional_value() == Decimal("10500.00")

    def test_get_notional_value_limit_order(self):
        """Test get_notional_value for limit order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )

        # Uses limit price * quantity
        assert order.get_notional_value() == Decimal("15000.00")

    def test_get_notional_value_stop_order(self):
        """Test get_notional_value for stop order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("145.00"),
        )

        # Stop orders don't have notional value until filled
        assert order.get_notional_value() is None


class TestOrderStringRepresentation:
    """Test Order string representation."""

    def test_str_market_order(self):
        """Test string representation of market order."""
        order = Order(
            symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )

        str_repr = str(order)
        assert "BUY" in str_repr
        assert "100" in str_repr
        assert "AAPL" in str_repr
        assert "pending" in str_repr

    def test_str_limit_order(self):
        """Test string representation of limit order."""
        order = Order(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("350.00"),
        )

        str_repr = str(order)
        assert "SELL" in str_repr
        assert "50" in str_repr
        assert "MSFT" in str_repr
        assert "@ $350" in str_repr

    def test_str_stop_order(self):
        """Test string representation of stop order."""
        order = Order(
            symbol="GOOGL",
            quantity=Decimal("25"),
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("2500.00"),
        )

        str_repr = str(order)
        assert "BUY" in str_repr
        assert "25" in str_repr
        assert "GOOGL" in str_repr
        assert "stop @ $2500" in str_repr


class TestOrderEdgeCases:
    """Test edge cases and special scenarios."""

    def test_order_with_quantity_object(self):
        """Test order with Quantity-like object."""

        # Create a mock Quantity object
        class MockQuantity:
            def __init__(self, value):
                self.value = value

        # Test validation with Quantity object
        order = Order(symbol="AAPL", quantity=MockQuantity(Decimal("100")), side=OrderSide.BUY)

        # The validation should handle .value attribute
        assert order.quantity.value == Decimal("100")

    def test_order_with_price_object(self):
        """Test order with Price-like object."""

        # Create a mock Price object
        class MockPrice:
            def __init__(self, value):
                self.value = value

        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=MockPrice(Decimal("150.00")),
        )

        # The get_notional_value should handle .value attribute
        notional = order.get_notional_value()
        assert notional == Decimal("15000.00")

    def test_multiple_partial_fills_average_price(self):
        """Test complex average price calculation with multiple fills."""
        order = Order(symbol="AAPL", quantity=Decimal("1000"), side=OrderSide.BUY)
        order.submit("BRK-123")

        # Multiple fills at different prices
        fills = [
            (Decimal("100"), Decimal("150.00")),
            (Decimal("200"), Decimal("151.00")),
            (Decimal("300"), Decimal("149.50")),
            (Decimal("250"), Decimal("150.50")),
            (Decimal("150"), Decimal("151.50")),
        ]

        total_cost = Decimal("0")
        total_qty = Decimal("0")

        for qty, price in fills:
            order.fill(qty, price)
            total_cost += qty * price
            total_qty += qty

        expected_avg = total_cost / total_qty
        assert order.average_fill_price == expected_avg
        assert order.filled_quantity == Decimal("1000")
        assert order.status == OrderStatus.FILLED

    def test_order_lifecycle(self):
        """Test complete order lifecycle."""
        # Create order
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
            time_in_force=TimeInForce.GTC,
            reason="Test lifecycle",
        )

        # Check initial state
        assert order.status == OrderStatus.PENDING
        assert order.is_active() is True
        assert order.is_complete() is False

        # Submit to broker
        order.submit("BRK-LIFECYCLE-001")
        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "BRK-LIFECYCLE-001"

        # Partial fill
        order.fill(Decimal("30"), Decimal("149.50"))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.get_remaining_quantity() == Decimal("70")

        # Another partial fill
        order.fill(Decimal("40"), Decimal("149.75"))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.get_fill_ratio() == Decimal("0.70")

        # Final fill
        order.fill(Decimal("30"), Decimal("150.00"))
        assert order.status == OrderStatus.FILLED
        assert order.is_active() is False
        assert order.is_complete() is True
        assert order.get_remaining_quantity() == Decimal("0")
        assert order.get_fill_ratio() == Decimal("1")
