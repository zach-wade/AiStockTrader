"""
Unit tests for Order entity
"""

# Standard library imports
from decimal import Decimal
from uuid import UUID

# Third-party imports
import pytest

# Local imports
from src.domain.entities import Order, OrderSide, OrderStatus, OrderType, TimeInForce


class TestOrderCreation:
    """Test Order creation and validation"""

    def test_create_market_order(self):
        """Test creating a market order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            reason="Test order",
        )
        order = Order.create_market_order(request)

        assert order.symbol == "AAPL"
        assert order.quantity.value == Decimal("100")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.reason == "Test order"
        assert isinstance(order.id, UUID)
        assert order.created_at is not None

    def test_create_limit_order(self):
        """Test creating a limit order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        request = OrderRequest(
            symbol="GOOGL",
            quantity=Quantity(Decimal("50")),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("150.50")),
            time_in_force=TimeInForce.GTC,
            reason="Take profit",
        )
        order = Order.create_limit_order(request)

        assert order.symbol == "GOOGL"
        assert order.quantity.value == Decimal("50")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price.value == Decimal("150.50")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Take profit"

    def test_create_stop_order(self):
        """Test creating a stop order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        request = OrderRequest(
            symbol="MSFT",
            quantity=Quantity(Decimal("75")),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("300.00")),
            reason="Stop loss",
        )
        order = Order.create_stop_order(request)

        assert order.symbol == "MSFT"
        assert order.quantity.value == Decimal("75")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP
        assert order.stop_price.value == Decimal("300.00")
        assert order.reason == "Stop loss"


class TestOrderValidation:
    """Test Order validation rules"""

    def test_empty_symbol_raises_error(self):
        """Test that empty symbol raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Order(symbol="", quantity=Quantity(Decimal("100")), side=OrderSide.BUY)

    def test_negative_quantity_raises_error(self):
        """Test that negative quantity raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Quantity(Decimal("-10")), side=OrderSide.BUY)

    def test_zero_quantity_raises_error(self):
        """Test that zero quantity raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(symbol="AAPL", quantity=Quantity(Decimal("0")), side=OrderSide.BUY)

    def test_limit_order_without_price_raises_error(self):
        """Test that limit order without price raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="Limit order requires limit price"):
            Order(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
            )

    def test_stop_order_without_price_raises_error(self):
        """Test that stop order without stop price raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            Order(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
            )

    def test_stop_limit_without_prices_raises_error(self):
        """Test that stop-limit order without both prices raises ValueError"""
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        with pytest.raises(ValueError, match="Stop-limit order requires both"):
            Order(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                side=OrderSide.BUY,
                order_type=OrderType.STOP_LIMIT,
                stop_price=Price(Decimal("150.00")),
                # Missing limit_price
            )

    def test_filled_quantity_exceeds_order_quantity(self):
        """Test that filled quantity cannot exceed order quantity"""
        from src.domain.value_objects.quantity import Quantity
        
        with pytest.raises(ValueError, match="Filled quantity cannot exceed"):
            Order(
                symbol="AAPL",
                quantity=Quantity(Decimal("100")),
                side=OrderSide.BUY,
                filled_quantity=Decimal("150"),
            )


class TestOrderStateTransitions:
    """Test Order state transitions"""

    def test_submit_order(self):
        """Test submitting an order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")

        with pytest.raises(ValueError, match="Cannot submit order in OrderStatus.SUBMITTED status"):
            order.submit("BROKER_456")

    def test_fill_order_completely(self):
        """Test filling an order completely"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")
        order.cancel("User requested")

        with pytest.raises(ValueError, match="Cannot fill order in OrderStatus.CANCELLED status"):
            order.fill(Decimal("50"), Decimal("150.00"))

    def test_cancel_order(self):
        """Test cancelling an order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_123")
        order.fill(Decimal("100"), Decimal("150.00"))

        with pytest.raises(ValueError, match="Cannot cancel order in OrderStatus.FILLED status"):
            order.cancel()

    def test_reject_order(self):
        """Test rejecting an order"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.tags["reject_reason"] == "Insufficient funds"

    def test_cannot_reject_submitted_order(self):
        """Test that submitted orders cannot be rejected"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        # Market order before fill
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
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
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("149.99")),
        )
        limit_order = Order.create_limit_order(request)
        assert limit_order.get_notional_value() == Decimal("14999")

    def test_string_representation(self):
        """Test string representation"""
        from src.domain.entities.order import OrderRequest
        from src.domain.value_objects.quantity import Quantity
        from src.domain.value_objects.price import Price
        
        request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            limit_price=Price(Decimal("150.00")),
        )
        order = Order.create_limit_order(request)

        str_repr = str(order)
        assert "BUY 100 AAPL" in str_repr
        assert "@ $150.00" in str_repr
        assert "pending" in str_repr
