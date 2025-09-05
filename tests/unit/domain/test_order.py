"""
Comprehensive test suite for Order entity - achieving full coverage.
Tests all methods, state transitions, edge cases, and business rules.
This file consolidates tests from multiple variant files.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

# Third-party imports
import pytest

# Local imports
from src.domain.entities import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.order import OrderRequest
from src.domain.value_objects import Price, Quantity


class TestOrderCreation:
    """Test Order creation and validation"""

    def test_create_market_order(self):
        """Test creating a market order"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            reason="Test order",
        )
        order = Order.create_market_order(request)

        assert order.symbol == "AAPL"
        assert order.quantity == Decimal("100")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.reason == "Test order"
        assert isinstance(order.id, UUID)
        assert order.created_at is not None

    def test_create_limit_order(self):
        """Test creating a limit order"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        request = OrderRequest(
            symbol="GOOGL",
            quantity=Decimal("50"),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("150.50")),
            time_in_force=TimeInForce.GTC,
            reason="Take profit",
        )
        order = Order.create_limit_order(request)

        assert order.symbol == "GOOGL"
        assert order.quantity == Decimal("50")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price.value == Decimal("150.50")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Take profit"

    def test_create_stop_order(self):
        """Test creating a stop order"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        request = OrderRequest(
            symbol="MSFT",
            quantity=Decimal("75"),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("300.00")),
            reason="Stop loss",
        )
        order = Order.create_stop_order(request)

        assert order.symbol == "MSFT"
        assert order.quantity == Decimal("75")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP
        assert order.stop_price.value == Decimal("300.00")
        assert order.reason == "Stop loss"


class TestOrderValidation:
    """Test Order validation rules"""

    def test_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError"""
        # Local imports

        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Order(symbol="", quantity=Decimal("100"), side=OrderSide.BUY)

    def test_negative_quantity_raises_error(self):
        """Test that negative quantity raises ValueError"""
        # Local imports

        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("-10"), side=OrderSide.BUY)

    def test_zero_quantity_raises_error(self):
        """Test that zero quantity raises ValueError"""
        # Local imports

        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Decimal("0"), side=OrderSide.BUY)

    def test_limit_order_without_price_raises_error(self):
        """Test that limit order without price raises ValueError"""
        # Local imports

        with pytest.raises(ValueError, match="Limit order requires limit price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
            )

    def test_stop_order_without_price_raises_error(self):
        """Test that stop order without stop price raises ValueError"""
        # Local imports

        with pytest.raises(ValueError, match="Stop order requires stop price"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
            )

    def test_stop_limit_without_prices_raises_error(self):
        """Test that stop-limit order without both prices raises ValueError"""
        # Local imports
        from src.domain.value_objects.price import Price

        with pytest.raises(ValueError, match="Stop-limit order requires both"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.STOP_LIMIT,
                stop_price=Price(Decimal("150.00")),
                # Missing limit_price
            )

    def test_filled_quantity_exceeds_order_quantity(self):
        """Test that filled quantity cannot exceed order quantity"""
        # Local imports

        with pytest.raises(ValueError, match="Filled quantity cannot exceed"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                filled_quantity=Decimal("150"),
            )


class TestOrderStateTransitions:
    """Test Order state transitions"""

    def test_submit_order(self):
        """Test submitting an order"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        assert order.status == OrderStatus.PENDING
        assert order.broker_order_id is None
        assert order.submitted_at is None

        order.submit("BROKER_123")

        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "BROKER_123"
        assert order.submitted_at is not None

    def test_cannot_submit_non_pending_order(self):
        """Test that non-pending orders cannot be submitted"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="Cannot submit order in OrderStatus.SUBMITTED status"):
            order.submit("BROKER_456")

    def test_fill_order_completely(self):
        """Test filling an order completely"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        order.fill(filled_quantity=Decimal("100"), fill_price=Decimal("150.50"))

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.average_fill_price == Decimal("150.50")
        assert order.filled_at is not None

    def test_fill_order_partially(self):
        """Test partial fill of an order"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        # First partial fill
        order.fill(filled_quantity=Decimal("30"), fill_price=Decimal("150.00"))

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("30")
        assert order.average_fill_price == Decimal("150.00")

        # Second partial fill
        order.fill(filled_quantity=Decimal("40"), fill_price=Decimal("151.00"))

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("70")
        # Average price: (30 * 150 + 40 * 151) / 70 = 150.571...
        expected_avg = (Decimal("30") * Decimal("150") + Decimal("40") * Decimal("151")) / Decimal(
            "70"
        )
        assert order.average_fill_price == expected_avg

        # Final fill
        order.fill(filled_quantity=Decimal("30"), fill_price=Decimal("152.00"))

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")

    def test_cannot_fill_cancelled_order(self):
        """Test that cancelled orders cannot be filled"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")
        order.cancel("User requested")

        with pytest.raises(ValueError, match="Cannot fill order in OrderStatus.CANCELLED status"):
            order.fill(Decimal("50"), Decimal("150.00"))

    def test_cancel_order(self):
        """Test cancelling an order"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        order.cancel("Market closed")

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert order.tags["cancel_reason"] == "Market closed"

    def test_cannot_cancel_filled_order(self):
        """Test that filled orders cannot be cancelled"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")
        order.fill(Decimal("100"), Decimal("150.00"))

        with pytest.raises(ValueError, match="Cannot cancel order in OrderStatus.FILLED status"):
            order.cancel()

    def test_reject_order(self):
        """Test rejecting an order"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.tags["reject_reason"] == "Insufficient funds"

    def test_cannot_reject_submitted_order(self):
        """Test that submitted orders cannot be rejected"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="Cannot reject order in OrderStatus.SUBMITTED status"):
            order.reject("Some reason")


class TestOrderQueries:
    """Test Order query methods"""

    def test_is_active(self):
        """Test is_active method"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        assert order.is_active()  # PENDING

        order.submit("BROKER_123")
        assert order.is_active()  # SUBMITTED

        order.fill(Decimal("50"), Decimal("150.00"))
        assert order.is_active()  # PARTIALLY_FILLED

        order.fill(Decimal("50"), Decimal("150.00"))
        assert not order.is_active()  # FILLED

    def test_is_complete(self):
        """Test is_complete method"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        assert not order.is_complete()  # PENDING

        order.submit("BROKER_123")
        assert not order.is_complete()  # SUBMITTED

        order.cancel()
        assert order.is_complete()  # CANCELLED

    def test_get_remaining_quantity(self):
        """Test get_remaining_quantity method"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        assert order.get_remaining_quantity() == Decimal("100")

        order.submit("BROKER_123")
        order.fill(Decimal("30"), Decimal("150.00"))

        assert order.get_remaining_quantity() == Decimal("70")

        order.fill(Decimal("70"), Decimal("150.00"))

        assert order.get_remaining_quantity() == Decimal("0")

    def test_get_fill_ratio(self):
        """Test get_fill_ratio method"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        assert order.get_fill_ratio() == Decimal("0")

        order.submit("BROKER_123")
        order.fill(Decimal("25"), Decimal("150.00"))

        assert order.get_fill_ratio() == Decimal("0.25")

        order.fill(Decimal("75"), Decimal("150.00"))

        assert order.get_fill_ratio() == Decimal("1")

    def test_get_notional_value(self):
        """Test get_notional_value method"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        # Market order before fill
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        market_order = Order.create_market_order(request)
        assert market_order.get_notional_value() is None

        # Market order after fill
        market_order.submit("BROKER_123")
        market_order.fill(Decimal("100"), Decimal("150.00"))
        assert market_order.get_notional_value() == Decimal("15000")

        # Limit order
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("149.99")),
        )
        limit_order = Order.create_limit_order(request)
        assert limit_order.get_notional_value() == Decimal("14999")

    def test_string_representation(self):
        """Test string representation"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("150.00")),
        )
        order = Order.create_limit_order(request)

        str_repr = str(order)
        assert "BUY 100 AAPL" in str_repr
        assert "@ $150.00" in str_repr
        assert "pending" in str_repr


class TestOrderEdgeCases:
    """Test Order edge cases and additional coverage"""

    def test_negative_filled_quantity_validation(self):
        """Test that negative filled quantity raises error"""
        # Local imports

        with pytest.raises(ValueError, match="Filled quantity cannot be negative"):
            Order(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                filled_quantity=Decimal("-10"),
            )

    def test_create_stop_limit_order_with_missing_prices(self):
        """Test creating stop-limit order with missing prices"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        # Missing both prices
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
        )
        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)

        # Missing limit price
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("145.00")),
        )
        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)

        # Missing stop price
        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("150.00")),
        )
        with pytest.raises(ValueError, match="Both stop price and limit price are required"):
            Order.create_stop_limit_order(request)

    def test_fill_with_negative_quantity(self):
        """Test that fill with negative quantity raises error"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("-10"), Decimal("150.00"))

    def test_fill_with_negative_price(self):
        """Test that fill with negative price raises error"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("10"), Decimal("-150.00"))

    def test_fill_with_overfill(self):
        """Test that overfilling raises error"""
        # Local imports
        from src.domain.entities.order import OrderRequest

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="exceeds order quantity"):
            order.fill(Decimal("150"), Decimal("150.00"))

    def test_get_fill_ratio_with_zero_quantity(self):
        """Test get_fill_ratio with zero quantity order"""
        # Local imports

        # This is an edge case that shouldn't normally happen but we handle it
        order = Order(
            symbol="TEST",
            quantity=Decimal("1"),  # Start with valid quantity
            side=OrderSide.BUY,
        )
        # Directly set to zero to test edge case
        order.quantity = Decimal("0")

        ratio = order.get_fill_ratio()
        assert ratio == Decimal("0")

    def test_get_notional_value_stop_order(self):
        """Test get_notional_value for stop order returns None"""
        # Local imports
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.price import Price

        request = OrderRequest(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("145.00")),
        )
        order = Order.create_stop_order(request)

        # Stop orders don't have a notional value until filled
        assert order.get_notional_value() is None


class TestOrderEnums:
    """Test order enumerations."""

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
            limit_price=Price(Decimal("150.00")),
            stop_price=Price(Decimal("145.00")),
            time_in_force=TimeInForce.GTC,
            reason="Test order",
        )

        assert request.symbol == "AAPL"
        assert request.quantity == Decimal("100")
        assert request.side == OrderSide.BUY
        assert request.limit_price == Price(Decimal("150.00"))
        assert request.stop_price == Price(Decimal("145.00"))
        assert request.time_in_force == TimeInForce.GTC
        assert request.reason == "Test order"

    def test_order_request_defaults(self):
        """Test OrderRequest default values."""
        request = OrderRequest(symbol="MSFT", quantity=Decimal("50"), side=OrderSide.SELL)

        assert request.limit_price is None
        assert request.stop_price is None
        assert request.time_in_force == TimeInForce.DAY
        assert request.reason is None


class TestOrderAdvanced:
    """Advanced Order tests for comprehensive coverage."""

    def test_order_with_all_time_in_force_values(self):
        """Test orders with different time in force values."""
        # DAY order
        day_order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        assert day_order.time_in_force == TimeInForce.DAY

        # GTC order
        gtc_order = Order(
            symbol="MSFT",
            quantity=Decimal("50"),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        assert gtc_order.time_in_force == TimeInForce.GTC

        # IOC order
        ioc_order = Order(
            symbol="TSLA",
            quantity=Decimal("25"),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC,
        )
        assert ioc_order.time_in_force == TimeInForce.IOC

        # FOK order
        fok_order = Order(
            symbol="NVDA",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.FOK,
        )
        assert fok_order.time_in_force == TimeInForce.FOK

    def test_order_with_ioc_time_in_force(self):
        """Test Immediate-or-Cancel order."""
        request = OrderRequest(
            symbol="AMZN",
            quantity=Decimal("150"),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("180.00")),
            time_in_force=TimeInForce.IOC,
        )
        order = Order.create_limit_order(request)

        assert order.time_in_force == TimeInForce.IOC
        order.submit("BROKER_IOC")

        # Simulate partial fill then cancellation
        order.fill(Decimal("50"), Decimal("179.90"))
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # IOC orders cancel remaining quantity
        order.cancel("IOC - unfilled portion cancelled")
        assert order.status == OrderStatus.CANCELLED
        assert order.tags["cancel_reason"] == "IOC - unfilled portion cancelled"

    def test_order_with_fok_time_in_force(self):
        """Test Fill-or-Kill order."""
        request = OrderRequest(
            symbol="GOOG",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("175.00")),
            time_in_force=TimeInForce.FOK,
        )
        order = Order.create_limit_order(request)

        assert order.time_in_force == TimeInForce.FOK
        order.submit("BROKER_FOK")

        # FOK orders must be filled completely or cancelled
        order.cancel("FOK - unable to fill completely")
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == Decimal("0")

    def test_order_expired_status(self):
        """Test order expiration handling."""
        request = OrderRequest(
            symbol="NFLX",
            quantity=Decimal("75"),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("450.00")),
            time_in_force=TimeInForce.DAY,
        )
        order = Order.create_limit_order(request)

        # Manually set to expired status (simulating end of day)
        order.status = OrderStatus.EXPIRED

        assert order.is_complete()
        assert not order.is_active()

    def test_partial_fills_with_custom_timestamps(self):
        """Test partial fills with custom timestamps."""
        request = OrderRequest(
            symbol="META",
            quantity=Decimal("300"),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_789")

        # First fill with custom timestamp
        fill_time1 = datetime(2024, 6, 15, 13, 0, 0, tzinfo=UTC)
        order.fill(Decimal("100"), Decimal("350.00"), fill_time1)

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.filled_at is None  # Not fully filled yet

        # Second fill with custom timestamp
        fill_time2 = datetime(2024, 6, 15, 13, 5, 0, tzinfo=UTC)
        order.fill(Decimal("200"), Decimal("352.00"), fill_time2)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("300")
        assert order.filled_at == fill_time2

        # Check weighted average price
        expected_avg = (
            Decimal("100") * Decimal("350.00") + Decimal("200") * Decimal("352.00")
        ) / Decimal("300")
        assert order.average_fill_price == expected_avg

    def test_multiple_partial_fills_average_price(self):
        """Test complex average price calculation with multiple fills."""
        order = Order(symbol="AAPL", quantity=Decimal("1000"), side=OrderSide.BUY)
        order.submit("BROKER123")

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

    def test_order_lifecycle_complex(self):
        """Test complete order lifecycle with all transitions."""
        # Create pending order
        order = Order(
            symbol="AAPL",
            quantity=Decimal("500"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("150.00")),
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

    def test_order_tags_manipulation(self):
        """Test order tags and metadata handling."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
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
        order = Order(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
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
        order1 = Order(symbol="AAPL", quantity=Decimal("0.0001"), side=OrderSide.BUY)
        assert order1.quantity == Decimal("0.0001")

        # Very large quantity
        order2 = Order(symbol="MSFT", quantity=Decimal("1000000000"), side=OrderSide.BUY)
        assert order2.quantity == Decimal("1000000000")

        # Very high precision price
        order3 = Order(
            symbol="BTC",
            quantity=Decimal("1"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Price(Decimal("45678.123456789")),
        )
        assert order3.limit_price.value == Decimal("45678.123456789")

        # Test fill with high precision
        order3.submit("BROKER123")
        order3.fill(Decimal("0.5"), Decimal("45678.987654321"))
        order3.fill(Decimal("0.5"), Decimal("45679.111111111"))
        expected_avg = (
            Decimal("0.5") * Decimal("45678.987654321")
            + Decimal("0.5") * Decimal("45679.111111111")
        ) / Decimal("1")
        assert order3.average_fill_price == expected_avg

    def test_cancel_with_various_reasons(self):
        """Test cancellation with different reasons and states."""
        request = OrderRequest(
            symbol="CANCEL",
            quantity=Decimal("100"),
            side=OrderSide.BUY,
        )

        # Test cancel without reason
        order1 = Order.create_market_order(request)
        order1.submit("BROKER_C1")
        order1.cancel()
        assert order1.status == OrderStatus.CANCELLED
        assert "cancel_reason" not in order1.tags

        # Test cancel with reason
        order2 = Order.create_market_order(request)
        order2.submit("BROKER_C2")
        order2.cancel("User requested cancellation")
        assert order2.status == OrderStatus.CANCELLED
        assert order2.tags["cancel_reason"] == "User requested cancellation"

        # Test cancel partially filled order
        order3 = Order.create_market_order(request)
        order3.submit("BROKER_C3")
        order3.fill(Decimal("30"), Decimal("100.00"))
        order3.cancel("Timeout")
        assert order3.status == OrderStatus.CANCELLED
        assert order3.filled_quantity == Decimal("30")
        assert order3.tags["cancel_reason"] == "Timeout"

    def test_reject_with_various_reasons(self):
        """Test rejection with different reasons."""
        request = OrderRequest(
            symbol="REJECT",
            quantity=Decimal("1000"),
            side=OrderSide.BUY,
        )

        # Insufficient buying power
        order1 = Order.create_market_order(request)
        order1.reject("Insufficient buying power")
        assert order1.status == OrderStatus.REJECTED
        assert order1.tags["reject_reason"] == "Insufficient buying power"

        # Symbol not tradeable
        order2 = Order.create_market_order(request)
        order2.reject("Symbol not tradeable")
        assert order2.status == OrderStatus.REJECTED
        assert order2.tags["reject_reason"] == "Symbol not tradeable"

        # Market closed
        order3 = Order.create_market_order(request)
        order3.reject("Market closed")
        assert order3.status == OrderStatus.REJECTED
        assert order3.tags["reject_reason"] == "Market closed"
