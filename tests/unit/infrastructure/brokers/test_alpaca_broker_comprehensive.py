"""
Comprehensive unit tests for AlpacaBroker - achieving 90%+ coverage.

Tests all methods, error conditions, edge cases, and thread safety.
"""

import os
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading import LimitOrderRequest, MarketOrderRequest
from alpaca.trading import OrderSide as AlpacaOrderSide
from alpaca.trading import OrderStatus as AlpacaOrderStatus
from alpaca.trading import OrderType as AlpacaOrderType
from alpaca.trading import QueryOrderStatus
from alpaca.trading import TimeInForce as AlpacaTimeInForce
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.models import Position as AlpacaPosition

# Note: Account might be in a different module or named differently
# We'll mock it instead of importing
from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    BrokerError,
    InvalidCredentialsError,
    MarketHours,
    OrderNotFoundError,
)
from src.domain.entities.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.infrastructure.brokers.alpaca_broker import AlpacaBroker


class TestAlpacaBrokerInitialization:
    """Test broker initialization scenarios."""

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicit API credentials."""
        broker = AlpacaBroker(api_key="test_api_key", secret_key="test_secret_key", paper=True)

        assert broker.api_key == "test_api_key"
        assert broker.secret_key == "test_secret_key"
        assert broker.paper is True
        assert broker.client is None
        assert broker._connected is False
        assert broker._order_map == {}
        assert type(broker._order_map_lock).__name__ == "lock"

    def test_init_with_environment_variables(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ, {"ALPACA_API_KEY": "env_api_key", "ALPACA_SECRET_KEY": "env_secret_key"}
        ):
            broker = AlpacaBroker(paper=False)

            assert broker.api_key == "env_api_key"
            assert broker.secret_key == "env_secret_key"
            assert broker.paper is False

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                InvalidCredentialsError, match="Alpaca API credentials not provided"
            ):
                AlpacaBroker(secret_key="secret")

    def test_init_missing_secret_key(self):
        """Test initialization fails without secret key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                InvalidCredentialsError, match="Alpaca API credentials not provided"
            ):
                AlpacaBroker(api_key="key")

    def test_init_credentials_override_environment(self):
        """Test explicit credentials override environment variables."""
        with patch.dict(
            os.environ, {"ALPACA_API_KEY": "env_key", "ALPACA_SECRET_KEY": "env_secret"}
        ):
            broker = AlpacaBroker(api_key="explicit_key", secret_key="explicit_secret")

            assert broker.api_key == "explicit_key"
            assert broker.secret_key == "explicit_secret"


class TestAlpacaBrokerConnection:
    """Test broker connection management."""

    @pytest.fixture
    def broker(self):
        """Create broker instance."""
        return AlpacaBroker(api_key="key", secret_key="secret")

    @patch("src.infrastructure.brokers.alpaca_broker.TradingClient")
    def test_connect(self, mock_client_class, broker):
        """Test successful connection."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        broker.connect()

        assert broker.client == mock_client
        assert broker._connected is True
        mock_client_class.assert_called_once_with(api_key="key", secret_key="secret", paper=True)

    def test_disconnect(self, broker):
        """Test disconnection."""
        broker.client = Mock()
        broker._connected = True

        broker.disconnect()

        assert broker.client is None
        assert broker._connected is False

    def test_is_connected_true(self, broker):
        """Test connection status when connected."""
        broker.client = Mock()
        broker._connected = True

        assert broker.is_connected() is True

    def test_is_connected_false_no_client(self, broker):
        """Test connection status with no client."""
        broker._connected = True
        broker.client = None

        assert broker.is_connected() is False

    def test_is_connected_false_not_connected(self, broker):
        """Test connection status when not connected."""
        broker.client = Mock()
        broker._connected = False

        assert broker.is_connected() is False

    def test_check_connection_raises_when_not_connected(self, broker):
        """Test connection check raises error when not connected."""
        with pytest.raises(BrokerConnectionError, match="Not connected to Alpaca"):
            broker._check_connection()

    def test_check_connection_passes_when_connected(self, broker):
        """Test connection check passes when connected."""
        broker.client = Mock()
        broker._connected = True

        # Should not raise
        broker._check_connection()


class TestAlpacaBrokerOrderSubmission:
    """Test order submission functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    @pytest.fixture
    def market_order(self):
        """Create market order."""
        return Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        )

    @pytest.fixture
    def limit_order(self):
        """Create limit order."""
        return Order.create_limit_order(
            OrderRequest(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.SELL,
                limit_price=Decimal("150.50"),
            )
        )

    def test_submit_market_order(self, broker, market_order):
        """Test submitting market order."""
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca-123"
        mock_alpaca_order.status = AlpacaOrderStatus.NEW
        mock_alpaca_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_alpaca_order

        result = broker.submit_order(market_order)

        # Verify order mapping
        assert broker._order_map[market_order.id] == "alpaca-123"
        assert result.broker_order_id == "alpaca-123"
        assert result.status == OrderStatus.SUBMITTED

        # Verify API call
        broker.client.submit_order.assert_called_once()
        # The submit_order is called with order_data keyword argument
        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert isinstance(request, MarketOrderRequest)
        assert request.symbol == "AAPL"
        assert request.qty == 100
        assert request.side == AlpacaOrderSide.BUY

    def test_submit_limit_order(self, broker, limit_order):
        """Test submitting limit order."""
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca-456"
        mock_alpaca_order.status = AlpacaOrderStatus.NEW
        mock_alpaca_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_alpaca_order

        result = broker.submit_order(limit_order)

        # Verify order mapping
        assert broker._order_map[limit_order.id] == "alpaca-456"
        assert result.broker_order_id == "alpaca-456"

        # Verify API call
        broker.client.submit_order.assert_called_once()
        # The submit_order is called with order_data keyword argument
        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert isinstance(request, LimitOrderRequest)
        assert request.symbol == "AAPL"
        assert request.qty == 100
        assert request.side == AlpacaOrderSide.SELL
        assert request.limit_price == 150.50

    def test_submit_order_with_time_in_force(self, broker, market_order):
        """Test order with different time in force values."""
        market_order.time_in_force = TimeInForce.IOC
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca-789"
        mock_alpaca_order.status = AlpacaOrderStatus.NEW
        mock_alpaca_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_alpaca_order

        broker.submit_order(market_order)

        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert request.time_in_force == AlpacaTimeInForce.IOC

    def test_submit_order_api_error(self, broker, market_order):
        """Test order submission with API error."""
        broker.client.submit_order.side_effect = APIError("Insufficient funds")

        with pytest.raises(BrokerError, match="Failed to submit order"):
            broker.submit_order(market_order)

    def test_submit_order_not_connected(self, market_order):
        """Test order submission when not connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with pytest.raises(BrokerConnectionError):
            broker.submit_order(market_order)

    def test_submit_order_thread_safety(self, broker):
        """Test thread-safe order submission."""
        orders = []
        results = []
        errors = []

        def submit_order(idx):
            try:
                order = Order.create_market_order(
                    OrderRequest(symbol=f"SYM{idx}", quantity=Decimal("10"), side=OrderSide.BUY)
                )
                orders.append(order)

                mock_alpaca_order = Mock()
                mock_alpaca_order.id = f"alpaca-{idx}"
                mock_alpaca_order.status = AlpacaOrderStatus.NEW
                mock_alpaca_order.created_at = datetime.now(UTC)
                broker.client.submit_order.return_value = mock_alpaca_order

                result = broker.submit_order(order)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Submit orders from multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=submit_order, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors and all orders mapped
        assert len(errors) == 0
        assert len(results) == 10
        assert len(broker._order_map) == 10


class TestAlpacaBrokerOrderCancellation:
    """Test order cancellation functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with order."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True

        # Add order to mapping
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-123"

        return broker, order_id

    def test_cancel_order_success(self, broker):
        """Test successful order cancellation."""
        broker, order_id = broker
        broker.client.cancel_order_by_id.return_value = None

        result = broker.cancel_order(order_id)

        assert result is True
        broker.client.cancel_order_by_id.assert_called_once_with("alpaca-123")

    def test_cancel_order_not_found(self, broker):
        """Test cancelling non-existent order."""
        broker, _ = broker
        unknown_id = uuid4()

        with pytest.raises(OrderNotFoundError, match=f"Order {unknown_id} not found"):
            broker.cancel_order(unknown_id)

    def test_cancel_order_api_error(self, broker):
        """Test cancellation with API error."""
        broker, order_id = broker
        broker.client.cancel_order_by_id.side_effect = APIError("Order already filled")

        result = broker.cancel_order(order_id)

        assert result is False

    def test_cancel_order_not_connected(self):
        """Test cancellation when not connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with pytest.raises(BrokerConnectionError):
            broker.cancel_order(uuid4())


class TestAlpacaBrokerOrderRetrieval:
    """Test order retrieval functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    @pytest.fixture
    def mock_alpaca_order(self):
        """Create mock Alpaca order."""
        order = Mock(spec=AlpacaOrder)
        order.id = "alpaca-123"
        order.symbol = "AAPL"
        order.qty = 100
        order.side = AlpacaOrderSide.BUY
        order.order_type = AlpacaOrderType.LIMIT
        order.limit_price = 150.00
        order.stop_price = None
        order.status = AlpacaOrderStatus.FILLED
        order.created_at = datetime.now(UTC)
        order.filled_at = datetime.now(UTC)
        order.filled_qty = 100
        order.filled_avg_price = 149.95
        return order

    def test_get_order_existing(self, broker, mock_alpaca_order):
        """Test getting existing order."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-123"
        broker.client.get_order_by_id.return_value = mock_alpaca_order

        result = broker.get_order(order_id)

        assert result.symbol == "AAPL"
        assert result.quantity == Decimal("100")
        assert result.side == OrderSide.BUY
        assert result.order_type == OrderType.LIMIT
        assert result.limit_price == Decimal("150.00")
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == Decimal("100")
        assert result.average_fill_price == Decimal("149.95")

        broker.client.get_order_by_id.assert_called_once_with("alpaca-123")

    def test_get_order_not_found(self, broker):
        """Test getting non-existent order."""
        order_id = uuid4()

        with pytest.raises(OrderNotFoundError):
            broker.get_order(order_id)

    def test_get_order_api_error(self, broker):
        """Test get order with API error."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-123"
        broker.client.get_order_by_id.side_effect = APIError("Order not found")

        with pytest.raises(OrderNotFoundError):
            broker.get_order(order_id)

    def test_get_recent_orders(self, broker, mock_alpaca_order):
        """Test getting recent orders."""
        broker.client.get_orders.return_value = [mock_alpaca_order]

        results = broker.get_recent_orders(limit=10)

        assert len(results) == 1
        assert results[0].symbol == "AAPL"

        broker.client.get_orders.assert_called_once()
        # Check the filter parameter is a GetOrdersRequest with correct limit
        call_args = broker.client.get_orders.call_args
        request = call_args.kwargs["filter"]
        assert request.limit == 10
        assert request.status == QueryOrderStatus.ALL

    def test_get_recent_orders_empty(self, broker):
        """Test getting recent orders when none exist."""
        broker.client.get_orders.return_value = []

        results = broker.get_recent_orders()

        assert results == []

    def test_get_recent_orders_with_limit(self, broker, mock_alpaca_order):
        """Test getting recent orders with different limit."""
        broker.client.get_orders.return_value = [mock_alpaca_order]

        results = broker.get_recent_orders(limit=50)

        assert len(results) == 1
        broker.client.get_orders.assert_called_once()
        call_args = broker.client.get_orders.call_args
        request = call_args.kwargs["filter"]
        assert request.limit == 50
        assert request.status == QueryOrderStatus.ALL


class TestAlpacaBrokerPositions:
    """Test position management functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    @pytest.fixture
    def mock_alpaca_position(self):
        """Create mock Alpaca position."""
        position = Mock(spec=AlpacaPosition)
        position.symbol = "AAPL"
        position.qty = 100
        position.side = "long"
        position.avg_entry_price = 145.50
        position.current_price = 150.00
        position.market_value = 15000.00
        position.unrealized_pl = 450.00
        position.unrealized_plpc = 0.0309
        return position

    def test_get_positions(self, broker, mock_alpaca_position):
        """Test getting all positions."""
        broker.client.get_all_positions.return_value = [mock_alpaca_position]

        results = broker.get_positions()

        assert len(results) == 1
        position = results[0]
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        # Side is determined by quantity sign (positive = long)
        assert position.quantity > 0  # Long position
        assert position.average_entry_price == Decimal("145.50")
        assert position.current_price == Decimal("150.00")
        # Calculate expected values
        expected_market_value = position.quantity * position.current_price
        expected_unrealized_pnl = (
            position.current_price - position.average_entry_price
        ) * position.quantity
        assert expected_market_value == Decimal("15000.00")
        assert expected_unrealized_pnl == Decimal("450.00")

    def test_get_positions_empty(self, broker):
        """Test getting positions when none exist."""
        broker.client.get_all_positions.return_value = []

        results = broker.get_positions()

        assert results == []

    def test_get_positions_short(self, broker, mock_alpaca_position):
        """Test getting short positions."""
        mock_alpaca_position.qty = -50
        mock_alpaca_position.side = "short"
        broker.client.get_all_positions.return_value = [mock_alpaca_position]

        results = broker.get_positions()

        position = results[0]
        # AlpacaBroker stores short positions as negative quantities
        assert position.quantity == Decimal("-50")
        # Side is determined by quantity sign (negative = short)
        assert position.quantity < 0  # Short position

    def test_get_position_by_symbol(self, broker, mock_alpaca_position):
        """Test getting position by symbol."""
        broker.client.get_position.return_value = mock_alpaca_position

        result = broker.get_position("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        broker.client.get_position.assert_called_once_with("AAPL")

    def test_get_position_not_found(self, broker):
        """Test getting non-existent position."""
        broker.client.get_position.side_effect = APIError("Position not found")

        result = broker.get_position("INVALID")

        assert result is None

    def test_close_position(self, broker):
        """Test closing a position."""
        broker.client.close_position.return_value = None

        result = broker.close_position("AAPL")

        assert result is True
        broker.client.close_position.assert_called_once_with("AAPL")

    def test_close_position_error(self, broker):
        """Test closing position with error."""
        broker.client.close_position.side_effect = APIError("No position")

        result = broker.close_position("INVALID")

        assert result is False

    def test_close_all_positions(self, broker):
        """Test closing all positions."""
        broker.client.close_all_positions.return_value = None

        result = broker.close_all_positions()

        assert result is True
        broker.client.close_all_positions.assert_called_once()


class TestAlpacaBrokerAccount:
    """Test account information functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    @pytest.fixture
    def mock_account(self):
        """Create mock Alpaca account."""
        account = Mock()  # Remove spec since AlpacaAccount not imported
        account.cash = 50000.00
        account.portfolio_value = 100000.00
        account.buying_power = 100000.00
        account.equity = 100000.00
        account.last_equity = 99000.00
        account.long_market_value = 50000.00
        account.short_market_value = 0
        account.initial_margin = 25000.00
        account.maintenance_margin = 25000.00
        account.status = "ACTIVE"
        account.pattern_day_trader = False
        account.trading_blocked = False
        account.transfers_blocked = False
        account.account_blocked = False
        account.trade_suspended_by_user = False
        return account

    def test_get_account_info(self, broker, mock_account):
        """Test getting account information."""
        broker.client.get_account.return_value = mock_account

        result = broker.get_account_info()

        assert isinstance(result, AccountInfo)
        assert result.cash == Decimal("50000.00")
        assert result.equity == Decimal("100000.00")
        assert result.buying_power == Decimal("100000.00")
        assert result.positions_value == Decimal("50000.00")
        # Check that account type is set correctly for paper mode
        assert result.account_type == "paper"

        broker.client.get_account.assert_called_once()

    def test_get_account_info_api_error(self, broker):
        """Test account info with API error."""
        broker.client.get_account.side_effect = APIError("Account suspended")

        with pytest.raises(BrokerConnectionError, match="Failed to get account info"):
            broker.get_account_info()

    def test_get_account_info_not_connected(self):
        """Test account info when not connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with pytest.raises(BrokerConnectionError):
            broker.get_account_info()


class TestAlpacaBrokerMarketHours:
    """Test market hours functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    def test_get_market_hours(self, broker):
        """Test getting market hours."""
        mock_calendar = Mock()
        mock_calendar.open = datetime(2025, 8, 21, 9, 30, tzinfo=UTC)
        mock_calendar.close = datetime(2025, 8, 21, 16, 0, tzinfo=UTC)
        broker.client.get_calendar.return_value = [mock_calendar]

        result = broker.get_market_hours(datetime(2025, 8, 21, tzinfo=UTC))

        assert isinstance(result, MarketHours)
        assert result.is_open is True
        assert result.next_open == datetime(2025, 8, 21, 9, 30, tzinfo=UTC)
        assert result.next_close == datetime(2025, 8, 21, 16, 0, tzinfo=UTC)

        broker.client.get_calendar.assert_called_once()

    def test_get_market_hours_closed(self, broker):
        """Test market hours for closed day."""
        broker.client.get_calendar.return_value = []

        result = broker.get_market_hours(datetime(2025, 8, 23, tzinfo=UTC))  # Saturday

        assert result.is_open is False
        assert result.next_open is None
        assert result.next_close is None

    def test_get_market_hours_api_error(self, broker):
        """Test market hours with API error."""
        broker.client.get_calendar.side_effect = APIError("Calendar unavailable")

        with pytest.raises(BrokerConnectionError, match="Failed to get market hours"):
            broker.get_market_hours(datetime.now(UTC))


class TestAlpacaBrokerMappingFunctions:
    """Test internal mapping functions."""

    @pytest.fixture
    def broker(self):
        """Create broker instance."""
        return AlpacaBroker(api_key="key", secret_key="secret")

    def test_map_order_side(self, broker):
        """Test order side mapping."""
        # The method is actually _map_side, which maps from Alpaca to domain
        assert broker._map_side(AlpacaOrderSide.BUY) == OrderSide.BUY
        assert broker._map_side(AlpacaOrderSide.SELL) == OrderSide.SELL

    def test_map_order_type(self, broker):
        """Test order type mapping."""
        # The method maps from Alpaca to domain
        assert broker._map_order_type(AlpacaOrderType.MARKET) == OrderType.MARKET
        assert broker._map_order_type(AlpacaOrderType.LIMIT) == OrderType.LIMIT
        # Note: current implementation only supports MARKET and LIMIT
        # Other types default to MARKET

    def test_safe_decimal(self, broker):
        """Test safe decimal conversion."""
        # Test valid conversion
        assert broker._safe_decimal("123.45") == Decimal("123.45")
        assert broker._safe_decimal(123.45) == Decimal("123.45")
        # Test None handling
        assert broker._safe_decimal(None) is None

    def test_map_order_status(self, broker):
        """Test order status mapping."""
        assert broker._map_status(AlpacaOrderStatus.NEW) == OrderStatus.SUBMITTED
        assert (
            broker._map_status(AlpacaOrderStatus.PARTIALLY_FILLED) == OrderStatus.PARTIALLY_FILLED
        )
        assert broker._map_status(AlpacaOrderStatus.FILLED) == OrderStatus.FILLED
        assert broker._map_status(AlpacaOrderStatus.CANCELED) == OrderStatus.CANCELLED
        assert broker._map_status(AlpacaOrderStatus.EXPIRED) == OrderStatus.EXPIRED
        assert broker._map_status(AlpacaOrderStatus.REJECTED) == OrderStatus.REJECTED
        assert broker._map_status(AlpacaOrderStatus.PENDING_NEW) == OrderStatus.PENDING
        assert broker._map_status(AlpacaOrderStatus.PENDING_CANCEL) == OrderStatus.PENDING

    def test_map_from_alpaca_order(self, broker):
        """Test mapping from Alpaca order to domain order."""
        alpaca_order = Mock()
        alpaca_order.symbol = "AAPL"
        alpaca_order.qty = 100
        alpaca_order.side = AlpacaOrderSide.BUY
        alpaca_order.order_type = AlpacaOrderType.LIMIT
        alpaca_order.limit_price = 150.00
        alpaca_order.stop_price = None
        alpaca_order.status = AlpacaOrderStatus.FILLED
        alpaca_order.id = "alpaca-123"
        alpaca_order.created_at = datetime.now(UTC)
        alpaca_order.filled_at = datetime.now(UTC)
        alpaca_order.filled_qty = 100
        alpaca_order.filled_avg_price = 149.95

        # The actual method is _map_alpaca_to_domain_order and it doesn't take order_id
        result = broker._map_alpaca_to_domain_order(alpaca_order)

        # The ID is generated automatically by the mapping method
        assert result.id is not None
        assert result.symbol == "AAPL"
        assert result.quantity == Decimal("100")
        assert result.side == OrderSide.BUY
        assert result.order_type == OrderType.LIMIT
        assert result.limit_price == Decimal("150.00")
        assert result.status == OrderStatus.FILLED
        assert result.broker_order_id == "alpaca-123"
        assert result.filled_quantity == Decimal("100")
        assert result.average_fill_price == Decimal("149.95")


class TestAlpacaBrokerThreadSafety:
    """Test thread safety of broker operations."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    def test_concurrent_order_submissions(self, broker):
        """Test concurrent order submissions are thread-safe."""
        num_threads = 20
        orders_submitted = []
        errors = []

        def submit_order(thread_id):
            try:
                order = Order.create_market_order(
                    OrderRequest(
                        symbol=f"SYM{thread_id}", quantity=Decimal("10"), side=OrderSide.BUY
                    )
                )

                mock_order = Mock()
                mock_order.id = f"alpaca-{thread_id}"
                mock_order.status = AlpacaOrderStatus.NEW
                mock_order.created_at = datetime.now(UTC)

                # Simulate random processing time
                time.sleep(0.001 * (thread_id % 3))

                broker.client.submit_order.return_value = mock_order
                result = broker.submit_order(order)
                orders_submitted.append(result)

            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=submit_order, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all orders submitted successfully
        assert len(errors) == 0
        assert len(orders_submitted) == num_threads
        assert len(broker._order_map) == num_threads

        # Verify unique mappings
        alpaca_ids = list(broker._order_map.values())
        assert len(alpaca_ids) == len(set(alpaca_ids))

    def test_concurrent_order_cancellations(self, broker):
        """Test concurrent order cancellations are thread-safe."""
        # Pre-populate order map
        order_ids = [uuid4() for _ in range(10)]
        for i, order_id in enumerate(order_ids):
            broker._order_map[order_id] = f"alpaca-{i}"

        cancelled = []
        errors = []

        def cancel_order(order_id):
            try:
                broker.client.cancel_order_by_id.return_value = None
                result = broker.cancel_order(order_id)
                if result:
                    cancelled.append(order_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for order_id in order_ids:
            thread = threading.Thread(target=cancel_order, args=(order_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all orders cancelled
        assert len(errors) == 0
        assert len(cancelled) == 10

    def test_concurrent_reads_and_writes(self, broker):
        """Test concurrent reads and writes to order map."""
        results = {"reads": [], "writes": [], "errors": []}

        def writer_thread(thread_id):
            try:
                for i in range(10):
                    order_id = uuid4()
                    with broker._order_map_lock:
                        broker._order_map[order_id] = f"alpaca-{thread_id}-{i}"
                    results["writes"].append(order_id)
                    time.sleep(0.0001)
            except Exception as e:
                results["errors"].append(e)

        def reader_thread():
            try:
                for _ in range(50):
                    with broker._order_map_lock:
                        snapshot = dict(broker._order_map)
                    results["reads"].append(len(snapshot))
                    time.sleep(0.0001)
            except Exception as e:
                results["errors"].append(e)

        threads = []

        # Start writer threads
        for i in range(5):
            thread = threading.Thread(target=writer_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Start reader threads
        for _ in range(3):
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors and expected results
        assert len(results["errors"]) == 0
        assert len(results["writes"]) == 50  # 5 threads * 10 writes
        assert len(results["reads"]) == 150  # 3 threads * 50 reads
        assert len(broker._order_map) == 50


class TestAlpacaBrokerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = Mock()
        broker._connected = True
        return broker

    def test_submit_order_with_none_fields(self, broker):
        """Test order with None optional fields."""
        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        )
        order.limit_price = None
        order.stop_price = None

        mock_order = Mock()
        mock_order.id = "alpaca-123"
        mock_order.status = AlpacaOrderStatus.NEW
        mock_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_order

        result = broker.submit_order(order)

        assert result.broker_order_id == "alpaca-123"
        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert isinstance(request, MarketOrderRequest)

    def test_get_position_with_zero_quantity(self, broker):
        """Test position with zero quantity."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = 0
        mock_position.side = "long"
        mock_position.avg_entry_price = 0
        mock_position.current_price = 150.00
        mock_position.market_value = 0
        mock_position.unrealized_pl = 0

        broker.client.get_position.return_value = mock_position

        result = broker.get_position("AAPL")

        # Should return None for zero quantity position
        assert result is None

    def test_map_unknown_order_status(self, broker):
        """Test mapping unknown order status."""
        # Create a mock status that doesn't exist in mapping
        unknown_status = "UNKNOWN_STATUS"

        # Should default to PENDING for unknown statuses
        result = broker._map_status(unknown_status)
        assert result == OrderStatus.PENDING

    def test_submit_order_with_very_large_quantity(self, broker):
        """Test order with very large quantity."""
        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("999999999"), side=OrderSide.BUY)
        )

        mock_order = Mock()
        mock_order.id = "alpaca-large"
        mock_order.status = AlpacaOrderStatus.NEW
        mock_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_order

        result = broker.submit_order(order)

        assert result.broker_order_id == "alpaca-large"
        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert request.qty == 999999999

    def test_submit_order_with_fractional_shares(self, broker):
        """Test order with fractional shares."""
        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("10.5"), side=OrderSide.BUY)
        )

        mock_order = Mock()
        mock_order.id = "alpaca-frac"
        mock_order.status = AlpacaOrderStatus.NEW
        mock_order.created_at = datetime.now(UTC)
        broker.client.submit_order.return_value = mock_order

        result = broker.submit_order(order)

        request = broker.client.submit_order.call_args.kwargs["order_data"]
        assert request.qty == 10.5

    def test_get_account_info_with_negative_values(self, broker):
        """Test account info with negative values (margin call)."""
        mock_account = Mock()
        mock_account.cash = -5000.00
        mock_account.portfolio_value = 45000.00
        mock_account.buying_power = 0
        mock_account.equity = 45000.00
        mock_account.last_equity = 50000.00
        mock_account.long_market_value = 50000.00
        mock_account.short_market_value = 0
        mock_account.initial_margin = 25000.00
        mock_account.maintenance_margin = 30000.00
        mock_account.status = "ACTIVE"
        mock_account.pattern_day_trader = True
        mock_account.trading_blocked = True
        mock_account.transfers_blocked = False
        mock_account.account_blocked = False
        mock_account.trade_suspended_by_user = False

        broker.client.get_account.return_value = mock_account

        result = broker.get_account_info()

        assert result.cash == Decimal("-5000.00")
        assert result.buying_power == Decimal("0")
        # AccountInfo doesn't have trading_blocked field
