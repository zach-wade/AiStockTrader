"""
Additional edge case tests for AlpacaBroker implementation.

Tests cover:
- API error handling edge cases
- Malformed response handling
- Rate limiting scenarios
- Network timeout handling
- Partial order fills
- Complex order types
- Account restrictions
- Symbol validation
"""

import os
import threading
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading import OrderStatus as AlpacaOrderStatus
from alpaca.trading import TimeInForce as AlpacaTimeInForce

from src.application.interfaces.broker import (
    BrokerConnectionError,
    BrokerError,
    InvalidCredentialsError,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.infrastructure.brokers.alpaca_broker import AlpacaBroker


class TestAlpacaBrokerAPIErrors:
    """Test handling of various API errors."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_submit_order_with_api_rate_limit(self, broker):
        """Test submit order handling rate limit errors."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100"), order_type=OrderType.MARKET
        )

        # Simulate rate limit error
        broker.client.submit_order.side_effect = APIError("Rate limit exceeded")

        # Should be wrapped in BrokerError
        with pytest.raises(BrokerError) as exc_info:
            broker.submit_order(order)

        assert "Rate limit" in str(exc_info.value)

    def test_cancel_order_partial_fill_scenario(self, broker):
        """Test cancelling partially filled order."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-partial"

        # First attempt fails with specific error
        broker.client.cancel_order_by_id.side_effect = APIError(
            "Order partially filled, cannot cancel remaining"
        )

        result = broker.cancel_order(order_id)
        assert result is False

    def test_get_order_status_with_stale_id(self, broker):
        """Test getting status with stale order ID."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-stale"

        # Simulate order not found at Alpaca (expired/purged)
        broker.client.get_order_by_id.side_effect = APIError("Order not found")

        with pytest.raises(APIError):
            broker.get_order_status(order_id)

    def test_positions_with_api_timeout(self, broker):
        """Test getting positions with API timeout."""
        broker.client.get_all_positions.side_effect = APIError("Request timeout")

        with pytest.raises(APIError) as exc_info:
            broker.get_positions()

        assert "timeout" in str(exc_info).lower()

    def test_account_info_insufficient_permissions(self, broker):
        """Test getting account info with insufficient permissions."""
        broker.client.get_account.side_effect = APIError("Insufficient permissions")

        # AlpacaBroker wraps APIError in BrokerConnectionError for get_account_info
        with pytest.raises(BrokerConnectionError) as exc_info:
            broker.get_account_info()

        assert "permissions" in str(exc_info.value).lower()


class TestAlpacaBrokerMalformedResponses:
    """Test handling of malformed API responses."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_submit_order_malformed_response(self, broker):
        """Test submit order with malformed response."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100"), order_type=OrderType.MARKET
        )

        # Return object without id attribute
        mock_response = MagicMock()
        del mock_response.id
        broker.client.submit_order.return_value = mock_response

        # Should handle gracefully and use string representation
        submitted = broker.submit_order(order)
        assert submitted.broker_order_id is not None

    def test_get_order_status_without_status_attr(self, broker):
        """Test get order status when response lacks status attribute."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-no-status"

        # Mock response without status attribute
        mock_order = MagicMock()
        del mock_order.status
        broker.client.get_order_by_id.return_value = mock_order

        # Should use string representation fallback
        status = broker.get_order_status(order_id)
        assert status == OrderStatus.PENDING  # Default fallback

    def test_position_with_missing_attributes(self, broker):
        """Test position handling with missing attributes."""
        # Create position with some missing attributes
        mock_pos = MagicMock()
        mock_pos.symbol = "TSLA"
        mock_pos.qty = 50
        # Missing avg_entry_price
        del mock_pos.avg_entry_price
        mock_pos.current_price = None
        mock_pos.realized_pl = None

        broker.client.get_all_positions.return_value = [mock_pos]

        positions = broker.get_positions()

        # Should handle missing attributes gracefully
        assert len(positions) == 1
        assert positions[0].symbol == "TSLA"
        assert positions[0].quantity == Decimal("50")
        assert positions[0].average_entry_price == Decimal("0")  # Default

    def test_account_info_missing_fields(self, broker):
        """Test account info with missing fields."""
        mock_account = MagicMock()
        # Only provide minimal fields
        mock_account.account_number = "TEST123"
        # Delete optional fields
        del mock_account.equity
        del mock_account.cash
        del mock_account.buying_power
        del mock_account.long_market_value
        del mock_account.pattern_day_trader

        broker.client.get_account.return_value = mock_account

        account_info = broker.get_account_info()

        # Should use defaults for missing fields
        assert account_info.account_id == "TEST123"
        assert account_info.equity == Decimal("0")
        assert account_info.cash == Decimal("0")
        assert account_info.buying_power == Decimal("0")

    def test_market_hours_missing_times(self, broker):
        """Test market hours with missing open/close times."""
        mock_clock = MagicMock()
        mock_clock.is_open = False
        del mock_clock.next_open
        del mock_clock.next_close

        broker.client.get_clock.return_value = mock_clock

        market_hours = broker.get_market_hours()

        assert market_hours.is_open is False
        assert market_hours.next_open is None
        assert market_hours.next_close is None


class TestAlpacaBrokerComplexOrders:
    """Test complex order scenarios."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_submit_stop_limit_order(self, broker):
        """Test submitting stop limit order."""
        order = Order(
            symbol="NVDA",
            side=OrderSide.SELL,
            quantity=Decimal("25"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("700.00"),
            stop_price=Decimal("680.00"),
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-stop-limit"
        broker.client.submit_order.return_value = mock_response

        submitted = broker.submit_order(order)

        # Verify stop price is handled
        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.limit_price == 700.0
        # Note: Current implementation doesn't pass stop_price
        # This test documents current behavior

    def test_submit_order_with_extended_hours(self, broker):
        """Test order submission with extended hours trading."""
        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("450.00"),
            time_in_force=TimeInForce.DAY,
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-extended"
        broker.client.submit_order.return_value = mock_response

        submitted = broker.submit_order(order)

        # The implementation now properly maps order's time_in_force
        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.time_in_force == AlpacaTimeInForce.DAY

    def test_fractional_shares_order(self, broker):
        """Test fractional shares order."""
        order = Order(
            symbol="AMZN",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),  # Fractional share
            order_type=OrderType.MARKET,
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-fractional"
        broker.client.submit_order.return_value = mock_response

        submitted = broker.submit_order(order)

        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.qty == 0.5


class TestAlpacaBrokerConcurrency:
    """Test concurrent operations and thread safety."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_concurrent_mixed_operations(self, broker):
        """Test mixed concurrent operations."""
        results = {"errors": [], "orders": [], "cancellations": []}

        def submit_orders():
            """Submit multiple orders."""
            for i in range(5):
                try:
                    order = Order(
                        symbol=f"SYM{i}",
                        side=OrderSide.BUY,
                        quantity=Decimal("10"),
                        order_type=OrderType.MARKET,
                    )
                    mock_response = MagicMock()
                    mock_response.id = f"alpaca-submit-{i}"
                    broker.client.submit_order.return_value = mock_response

                    submitted = broker.submit_order(order)
                    results["orders"].append(submitted)
                except Exception as e:
                    results["errors"].append(e)

        def cancel_orders():
            """Cancel orders after brief delay."""
            time.sleep(0.01)  # Small delay to let submissions start
            for order in results["orders"][:3]:  # Cancel first 3
                try:
                    if order and order.id in broker._order_map:
                        result = broker.cancel_order(order.id)
                        results["cancellations"].append(result)
                except Exception as e:
                    results["errors"].append(e)

        def get_statuses():
            """Get order statuses."""
            time.sleep(0.02)  # Small delay
            statuses = []
            for order in results["orders"]:
                try:
                    if order and order.id in broker._order_map:
                        mock_order = MagicMock()
                        mock_order.status = AlpacaOrderStatus.FILLED
                        broker.client.get_order_by_id.return_value = mock_order

                        status = broker.get_order_status(order.id)
                        statuses.append(status)
                except Exception as e:
                    results["errors"].append(e)
            return statuses

        # Run operations concurrently
        threads = [
            threading.Thread(target=submit_orders),
            threading.Thread(target=cancel_orders),
            threading.Thread(target=get_statuses),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=1.0)

        # Verify no errors occurred
        assert len(results["errors"]) == 0
        # Verify order map consistency
        assert len(broker._order_map) >= len(results["orders"])

    def test_order_map_cleanup_race_condition(self, broker):
        """Test order map cleanup under race conditions."""
        order_ids = [uuid4() for _ in range(10)]

        # Populate order map
        for i, order_id in enumerate(order_ids):
            broker._order_map[order_id] = f"alpaca-{i}"

        def remove_orders(ids_to_remove):
            """Remove orders from map."""
            for order_id in ids_to_remove:
                with broker._order_map_lock:
                    if order_id in broker._order_map:
                        del broker._order_map[order_id]

        def add_orders(ids_to_add):
            """Add new orders to map."""
            for i, order_id in enumerate(ids_to_add):
                with broker._order_map_lock:
                    broker._order_map[order_id] = f"alpaca-new-{i}"

        # Create new IDs for adding
        new_ids = [uuid4() for _ in range(5)]

        # Run add and remove concurrently
        threads = [
            threading.Thread(target=remove_orders, args=(order_ids[:5],)),
            threading.Thread(target=add_orders, args=(new_ids,)),
            threading.Thread(target=remove_orders, args=(order_ids[5:],)),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify final state is consistent
        assert len(broker._order_map) == 5  # Only new orders remain
        for new_id in new_ids:
            assert new_id in broker._order_map


class TestAlpacaBrokerEdgeCases:
    """Test various edge cases."""

    def test_initialization_with_empty_strings(self):
        """Test initialization with empty credential strings."""
        with pytest.raises(InvalidCredentialsError):
            AlpacaBroker(api_key="", secret_key="")

    def test_initialization_with_whitespace_credentials(self):
        """Test initialization with whitespace-only credentials."""
        with pytest.raises(InvalidCredentialsError):
            AlpacaBroker(api_key="   ", secret_key="   ")

    @patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""})
    def test_initialization_with_empty_env_vars(self):
        """Test initialization with empty environment variables."""
        with pytest.raises(InvalidCredentialsError):
            AlpacaBroker()

    def test_multiple_connect_disconnect_cycles(self):
        """Test multiple connect/disconnect cycles."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with patch("src.infrastructure.brokers.alpaca_broker.TradingClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Multiple cycles
            for _ in range(3):
                broker.connect()
                assert broker.is_connected()

                broker.disconnect()
                assert not broker.is_connected()

    def test_order_with_zero_quantity(self):
        """Test order with zero quantity."""
        # Order entity now validates that quantity must be positive
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("0"),
                order_type=OrderType.MARKET,
            )

    def test_order_with_negative_price(self):
        """Test order with negative limit price."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=Decimal("10"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("-100.00"),  # Negative price
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-negative"
        broker.client.submit_order.return_value = mock_response

        # Broker doesn't validate, just passes through
        submitted = broker.submit_order(order)

        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.limit_price == -100.0

    def test_extremely_large_order_quantity(self):
        """Test order with extremely large quantity."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal("999999999"),
            order_type=OrderType.MARKET,
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-large"
        broker.client.submit_order.return_value = mock_response

        submitted = broker.submit_order(order)

        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.qty == 999999999.0

    def test_unicode_symbol_handling(self):
        """Test handling of unicode characters in symbol."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        order = Order(
            symbol="AAPL₿",  # Unicode character
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
        )

        mock_response = MagicMock()
        mock_response.id = "alpaca-unicode"
        broker.client.submit_order.return_value = mock_response

        submitted = broker.submit_order(order)

        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]
        assert request.symbol == "AAPL₿"


class TestAlpacaBrokerRecovery:
    """Test error recovery scenarios."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_recover_from_temporary_network_error(self, broker):
        """Test recovery from temporary network errors."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-network"

        # First call fails with network error
        broker.client.get_order_by_id.side_effect = [
            APIError("Network error"),
            MagicMock(status=AlpacaOrderStatus.FILLED),  # Second call succeeds
        ]

        # First attempt fails
        with pytest.raises(APIError):
            broker.get_order_status(order_id)

        # Second attempt succeeds
        status = broker.get_order_status(order_id)
        assert status == OrderStatus.FILLED

    def test_handle_alpaca_maintenance_window(self, broker):
        """Test handling Alpaca maintenance window."""
        broker.client.get_clock.side_effect = APIError("Service temporarily unavailable")

        with pytest.raises(APIError) as exc_info:
            broker.is_market_open()

        assert "unavailable" in str(exc_info.value).lower()

    def test_reconnect_after_disconnect(self):
        """Test reconnecting after disconnection."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with patch("src.infrastructure.brokers.alpaca_broker.TradingClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Connect, disconnect, reconnect
            broker.connect()
            assert broker.is_connected()

            broker.disconnect()
            assert not broker.is_connected()

            broker.connect()
            assert broker.is_connected()

            # Should create new client instance
            assert mock_client.call_count == 2
