"""
Comprehensive unit tests for Order entity to achieve high coverage.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

# Third-party imports
import pytest
try:
    from freezegun import freeze_time
except ImportError:
    # If freezegun is not available, create a simple mock
    from contextlib import contextmanager
    @contextmanager
    def freeze_time(dt):
        yield

# Local imports
from src.domain.entities import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.order import OrderRequest
from src.domain.value_objects.quantity import Quantity
from src.domain.value_objects.price import Price


class TestOrderComprehensive:
    """Comprehensive test suite for Order entity."""

    def test_order_with_all_attributes(self):
        """Test creating an order with all possible attributes."""
        request = OrderRequest(
            symbol="TSLA",
            quantity=Quantity(Decimal("200")),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("250.00")),
            stop_price=Price(Decimal("245.00")),
            time_in_force=TimeInForce.GTC,
            reason="Take profit with stop loss",
        )
        
        order = Order.create_stop_limit_order(request)
        
        assert order.symbol == "TSLA"
        assert order.quantity.value == Decimal("200")
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.limit_price.value == Decimal("250.00")
        assert order.stop_price.value == Decimal("245.00")
        assert order.time_in_force == TimeInForce.GTC
        assert order.reason == "Take profit with stop loss"
        assert order.status == OrderStatus.PENDING
        assert isinstance(order.id, UUID)

    def test_order_timestamps_tracking(self):
        """Test that all timestamps are properly tracked."""
        request = OrderRequest(
            symbol="NVDA",
            quantity=Quantity(Decimal("50")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        
        # Check initial timestamps
        assert order.created_at is not None
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.cancelled_at is None
        
        # Submit order
        order.submit("BROKER_456")
        assert order.submitted_at is not None
        assert order.submitted_at >= order.created_at
        
        # Fill order
        order.fill(Decimal("50"), Decimal("700.00"))
        assert order.filled_at is not None
        assert order.filled_at >= order.submitted_at

    def test_order_tags_metadata(self):
        """Test order tags and metadata handling."""
        request = OrderRequest(
            symbol="AMD",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            reason="Technical breakout",
        )
        order = Order.create_market_order(request)
        
        # Add custom tags
        order.tags["strategy"] = "momentum"
        order.tags["confidence"] = 0.85
        order.tags["signal_strength"] = "strong"
        
        assert order.tags["strategy"] == "momentum"
        assert order.tags["confidence"] == 0.85
        assert order.tags["signal_strength"] == "strong"
        assert order.reason == "Technical breakout"

    def test_partial_fills_with_custom_timestamps(self):
        """Test partial fills with custom timestamps."""
        request = OrderRequest(
            symbol="META",
            quantity=Quantity(Decimal("300")),
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
        expected_avg = (Decimal("100") * Decimal("350.00") + 
                       Decimal("200") * Decimal("352.00")) / Decimal("300")
        assert order.average_fill_price == expected_avg

    def test_order_expired_status(self):
        """Test order expiration handling."""
        request = OrderRequest(
            symbol="NFLX",
            quantity=Quantity(Decimal("75")),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("450.00")),
            time_in_force=TimeInForce.DAY,
        )
        order = Order.create_limit_order(request)
        
        # Manually set to expired status (simulating end of day)
        order.status = OrderStatus.EXPIRED
        
        assert order.is_complete()
        assert not order.is_active()

    def test_order_with_ioc_time_in_force(self):
        """Test Immediate-or-Cancel order."""
        request = OrderRequest(
            symbol="AMZN",
            quantity=Quantity(Decimal("150")),
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
            quantity=Quantity(Decimal("100")),
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

    def test_get_notional_value_various_order_types(self):
        """Test notional value calculation for different order types."""
        # Market order before fill
        market_request = OrderRequest(
            symbol="SPY",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        market_order = Order.create_market_order(market_request)
        assert market_order.get_notional_value() is None
        
        # Market order after partial fill
        market_order.submit("BROKER_M1")
        market_order.fill(Decimal("50"), Decimal("450.00"))
        assert market_order.get_notional_value() == Decimal("22500.00")  # 50 * 450
        
        # Limit order
        limit_request = OrderRequest(
            symbol="QQQ",
            quantity=Quantity(Decimal("200")),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("400.00")),
        )
        limit_order = Order.create_limit_order(limit_request)
        assert limit_order.get_notional_value() == Decimal("80000.00")  # 200 * 400
        
        # Stop order (no notional value until filled)
        stop_request = OrderRequest(
            symbol="IWM",
            quantity=Quantity(Decimal("150")),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("200.00")),
        )
        stop_order = Order.create_stop_order(stop_request)
        assert stop_order.get_notional_value() is None

    def test_order_string_representations(self):
        """Test string representations for all order types."""
        # Market order
        market_request = OrderRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        market_order = Order.create_market_order(market_request)
        str_repr = str(market_order)
        assert "BUY 100 AAPL" in str_repr
        assert "pending" in str_repr.lower()
        
        # Limit order
        limit_request = OrderRequest(
            symbol="MSFT",
            quantity=Quantity(Decimal("50")),
            side=OrderSide.SELL,
            limit_price=Price(Decimal("400.00")),
        )
        limit_order = Order.create_limit_order(limit_request)
        str_repr = str(limit_order)
        assert "SELL 50 MSFT" in str_repr
        assert "@ $" in str_repr
        assert "400" in str_repr
        
        # Stop order
        stop_request = OrderRequest(
            symbol="GOOGL",
            quantity=Quantity(Decimal("75")),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("150.00")),
        )
        stop_order = Order.create_stop_order(stop_request)
        str_repr = str(stop_order)
        assert "SELL 75 GOOGL" in str_repr
        assert "stop @ $" in str_repr
        assert "150" in str_repr

    def test_edge_case_zero_fill_ratio(self):
        """Test edge case where quantity is zero (should not happen but defensive)."""
        order = Order(
            symbol="TEST",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        
        # Normal case
        assert order.get_fill_ratio() == Decimal("0")
        
        # After partial fill
        order.status = OrderStatus.SUBMITTED
        order.filled_quantity = Decimal("50")
        assert order.get_fill_ratio() == Decimal("0.5")

    def test_validation_combinations(self):
        """Test various validation combinations."""
        # Valid stop-limit order
        order = Order(
            symbol="VALID",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            stop_price=Price(Decimal("100.00")),
            limit_price=Price(Decimal("101.00")),
        )
        assert order.order_type == OrderType.STOP_LIMIT
        
        # Test with filled_quantity at boundary
        order2 = Order(
            symbol="BOUND",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.SELL,
            filled_quantity=Decimal("100"),  # Exactly at boundary
        )
        assert order2.filled_quantity == Decimal("100")

    def test_multiple_fill_error_conditions(self):
        """Test various error conditions during fills."""
        request = OrderRequest(
            symbol="ERR",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        order.submit("BROKER_ERR")
        
        # Test zero fill quantity
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("0"), Decimal("100.00"))
        
        # Test negative fill quantity
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            order.fill(Decimal("-10"), Decimal("100.00"))
        
        # Test zero fill price
        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("10"), Decimal("0"))
        
        # Test negative fill price
        with pytest.raises(ValueError, match="Fill price must be positive"):
            order.fill(Decimal("10"), Decimal("-100.00"))
        
        # Test overfill
        order.fill(Decimal("90"), Decimal("100.00"))
        with pytest.raises(ValueError, match="exceeds order quantity"):
            order.fill(Decimal("20"), Decimal("100.00"))

    def test_cancel_with_various_reasons(self):
        """Test cancellation with different reasons and states."""
        request = OrderRequest(
            symbol="CANCEL",
            quantity=Quantity(Decimal("100")),
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
            quantity=Quantity(Decimal("1000")),
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

    def test_order_state_queries_comprehensive(self):
        """Test all state query methods comprehensively."""
        request = OrderRequest(
            symbol="STATE",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.BUY,
        )
        order = Order.create_market_order(request)
        
        # Pending state
        assert order.is_active()
        assert not order.is_complete()
        assert order.get_remaining_quantity() == Decimal("100")
        assert order.get_fill_ratio() == Decimal("0")
        
        # Submitted state
        order.submit("BROKER_STATE")
        assert order.is_active()
        assert not order.is_complete()
        
        # Partially filled state
        order.fill(Decimal("40"), Decimal("100.00"))
        assert order.is_active()
        assert not order.is_complete()
        assert order.get_remaining_quantity() == Decimal("60")
        assert order.get_fill_ratio() == Decimal("0.4")
        
        # Fully filled state
        order.fill(Decimal("60"), Decimal("101.00"))
        assert not order.is_active()
        assert order.is_complete()
        assert order.get_remaining_quantity() == Decimal("0")
        assert order.get_fill_ratio() == Decimal("1")

    def test_factory_methods_comprehensive(self):
        """Test all factory methods with various parameters."""
        # Test create_stop_limit_order
        sl_request = OrderRequest(
            symbol="STOPLIM",
            quantity=Quantity(Decimal("100")),
            side=OrderSide.SELL,
            stop_price=Price(Decimal("95.00")),
            limit_price=Price(Decimal("94.50")),
            time_in_force=TimeInForce.GTC,
            reason="Stop loss with limit",
        )
        sl_order = Order.create_stop_limit_order(sl_request)
        
        assert sl_order.order_type == OrderType.STOP_LIMIT
        assert sl_order.stop_price.value == Decimal("95.00")
        assert sl_order.limit_price.value == Decimal("94.50")
        assert sl_order.time_in_force == TimeInForce.GTC
        assert sl_order.reason == "Stop loss with limit"