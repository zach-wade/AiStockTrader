"""
Comprehensive unit tests for AlpacaBroker implementation.

Tests cover:
- Initialization and configuration
- Connection management
- Order submission and management
- Position retrieval
- Account information
- Market hours
- Error handling
- API mapping functions
"""

import os
import threading
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading import OrderSide as AlpacaOrderSide
from alpaca.trading import OrderStatus as AlpacaOrderStatus
from alpaca.trading import OrderType as AlpacaOrderType
from alpaca.trading.models import Order as AlpacaOrder

from src.application.interfaces.broker import (
    BrokerConnectionError,
    InvalidCredentialsError,
    OrderNotFoundError,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.position import Position
from src.infrastructure.brokers.alpaca_broker import AlpacaBroker


class TestAlpacaBrokerInitialization:
    """Test AlpacaBroker initialization."""

    def test_initialization_with_credentials(self):
        """Test initialization with provided credentials."""
        broker = AlpacaBroker(api_key="test_key", secret_key="test_secret", paper=True)

        assert broker.api_key == "test_key"
        assert broker.secret_key == "test_secret"
        assert broker.paper is True
        assert broker.client is None
        assert not broker._connected
        assert isinstance(broker._order_map, dict)
        assert type(broker._order_map_lock).__name__ == "lock"

    @patch.dict(os.environ, {"ALPACA_API_KEY": "env_key", "ALPACA_SECRET_KEY": "env_secret"})
    def test_initialization_from_environment(self):
        """Test initialization from environment variables."""
        broker = AlpacaBroker()

        assert broker.api_key == "env_key"
        assert broker.secret_key == "env_secret"
        assert broker.paper is True  # Default

    def test_initialization_missing_credentials(self):
        """Test initialization fails with missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                InvalidCredentialsError, match="Alpaca API credentials not provided"
            ):
                AlpacaBroker()

    def test_initialization_paper_mode(self):
        """Test paper mode configuration."""
        broker = AlpacaBroker(api_key="key", secret_key="secret", paper=False)

        assert broker.paper is False


class TestAlpacaBrokerConnection:
    """Test connection management."""

    @patch("src.infrastructure.brokers.alpaca_broker.TradingClient")
    def test_connect(self, mock_client_class):
        """Test connecting to Alpaca API."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.connect()

        assert broker._connected is True
        assert broker.client == mock_client
        mock_client_class.assert_called_once_with(api_key="key", secret_key="secret", paper=True)

    def test_disconnect(self):
        """Test disconnecting from Alpaca API."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        broker.disconnect()

        assert broker.client is None
        assert broker._connected is False

    def test_is_connected(self):
        """Test connection status check."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        assert broker.is_connected() is False

        broker._connected = True
        broker.client = MagicMock()
        assert broker.is_connected() is True

        broker.client = None
        assert broker.is_connected() is False

    def test_check_connection_when_connected(self):
        """Test connection check when connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker._connected = True
        broker.client = MagicMock()

        # Should not raise
        broker._check_connection()

    def test_check_connection_when_disconnected(self):
        """Test connection check when disconnected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with pytest.raises(BrokerConnectionError, match="Not connected to Alpaca"):
            broker._check_connection()


class TestAlpacaBrokerOrderManagement:
    """Test order management functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        return Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("100"), order_type=OrderType.MARKET
        )

    def test_submit_market_order(self, broker, sample_order):
        """Test submitting a market order."""
        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "alpaca-123"
        broker.client.submit_order.return_value = mock_alpaca_order

        submitted_order = broker.submit_order(sample_order)

        # Verify order was submitted to Alpaca
        broker.client.submit_order.assert_called_once()
        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]

        assert request.symbol == "AAPL"
        assert request.qty == 100.0
        assert request.side == AlpacaOrderSide.BUY

        # Verify order mapping
        assert broker._order_map[sample_order.id] == "alpaca-123"
        assert submitted_order.broker_order_id == "alpaca-123"
        assert submitted_order.status == OrderStatus.SUBMITTED

    def test_submit_limit_order(self, broker):
        """Test submitting a limit order."""
        order = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=Decimal("50"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("2500.00"),
        )

        mock_alpaca_order = MagicMock()
        mock_alpaca_order.id = "alpaca-456"
        broker.client.submit_order.return_value = mock_alpaca_order

        submitted_order = broker.submit_order(order)

        # Verify limit order request
        call_args = broker.client.submit_order.call_args
        request = call_args.kwargs["order_data"]

        assert request.symbol == "GOOGL"
        assert request.qty == 50.0
        assert request.side == AlpacaOrderSide.SELL
        assert request.limit_price == 2500.0

    def test_submit_order_not_connected(self, sample_order):
        """Test submitting order when not connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        with pytest.raises(BrokerConnectionError):
            broker.submit_order(sample_order)

    def test_cancel_order_success(self, broker):
        """Test successful order cancellation."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-789"

        result = broker.cancel_order(order_id)

        broker.client.cancel_order_by_id.assert_called_once_with("alpaca-789")
        assert result is True

    def test_cancel_order_not_found(self, broker):
        """Test cancelling non-existent order."""
        order_id = uuid4()

        with pytest.raises(OrderNotFoundError, match=f"Order {order_id} not found"):
            broker.cancel_order(order_id)

    def test_cancel_order_api_error(self, broker):
        """Test cancellation when API returns error."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-error"
        broker.client.cancel_order_by_id.side_effect = APIError("Cannot cancel")

        result = broker.cancel_order(order_id)

        assert result is False

    def test_get_order_status(self, broker):
        """Test getting order status."""
        order_id = uuid4()
        broker._order_map[order_id] = "alpaca-status"

        mock_alpaca_order = MagicMock()
        mock_alpaca_order.status = AlpacaOrderStatus.PARTIALLY_FILLED
        broker.client.get_order_by_id.return_value = mock_alpaca_order

        status = broker.get_order_status(order_id)

        broker.client.get_order_by_id.assert_called_once_with("alpaca-status")
        assert status == OrderStatus.PARTIALLY_FILLED

    def test_get_order_status_not_found(self, broker):
        """Test getting status of non-existent order."""
        order_id = uuid4()

        with pytest.raises(OrderNotFoundError):
            broker.get_order_status(order_id)

    def test_update_order(self, broker, sample_order):
        """Test updating order from Alpaca status."""
        broker._order_map[sample_order.id] = "alpaca-update"

        mock_alpaca_order = MagicMock()
        mock_alpaca_order.status = AlpacaOrderStatus.FILLED
        broker.client.get_order_by_id.return_value = mock_alpaca_order

        updated_order = broker.update_order(sample_order)

        assert updated_order.status == OrderStatus.FILLED

    def test_update_order_not_mapped(self, broker, sample_order):
        """Test updating order that's not in map returns unchanged."""
        updated_order = broker.update_order(sample_order)

        assert updated_order == sample_order
        broker.client.get_order_by_id.assert_not_called()

    def test_get_recent_orders(self, broker):
        """Test getting recent orders from Alpaca."""
        # Create mock Alpaca orders with proper AlpacaOrder type
        mock_orders = []
        for i in range(3):
            mock_order = MagicMock(spec=AlpacaOrder)  # Use spec to make isinstance work
            mock_order.symbol = f"SYM{i}"
            mock_order.side = AlpacaOrderSide.BUY
            mock_order.qty = i + 1
            mock_order.order_type = AlpacaOrderType.MARKET
            mock_order.status = AlpacaOrderStatus.FILLED
            mock_order.limit_price = None
            mock_order.stop_price = None
            mock_order.id = f"alpaca-{i}"
            mock_order.created_at = datetime.now(UTC)
            mock_order.filled_at = datetime.now(UTC)
            mock_order.filled_qty = i + 1
            mock_order.filled_avg_price = 100.0 + i
            mock_orders.append(mock_order)

        broker.client.get_orders.return_value = mock_orders

        orders = broker.get_recent_orders(limit=10)

        assert len(orders) == 3
        assert all(isinstance(order, Order) for order in orders)
        assert orders[0].symbol == "SYM0"
        assert orders[1].symbol == "SYM1"
        assert orders[2].symbol == "SYM2"


class TestAlpacaBrokerPositions:
    """Test position management functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_get_positions(self, broker):
        """Test getting positions from Alpaca."""
        # Create mock positions
        mock_positions = []
        for i, symbol in enumerate(["AAPL", "GOOGL"]):
            mock_pos = MagicMock()
            mock_pos.symbol = symbol
            mock_pos.qty = (i + 1) * 100
            mock_pos.avg_entry_price = 150.0 + i * 100
            mock_pos.current_price = 160.0 + i * 100
            mock_pos.market_value = mock_pos.qty * mock_pos.current_price
            mock_pos.cost_basis = mock_pos.qty * mock_pos.avg_entry_price
            mock_pos.unrealized_pl = mock_pos.market_value - mock_pos.cost_basis
            mock_pos.realized_pl = 0
            mock_positions.append(mock_pos)

        broker.client.get_all_positions.return_value = mock_positions

        positions = broker.get_positions()

        assert len(positions) == 2
        assert all(isinstance(pos, Position) for pos in positions)

        # Check first position
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == Decimal("100")
        assert positions[0].average_entry_price == Decimal("150.0")

        # Check second position
        assert positions[1].symbol == "GOOGL"
        assert positions[1].quantity == Decimal("200")
        assert positions[1].average_entry_price == Decimal("250.0")

    def test_get_positions_with_none_values(self, broker):
        """Test getting positions with None values handled correctly."""
        mock_pos = MagicMock()
        mock_pos.symbol = "TSLA"
        mock_pos.qty = "-50"  # String to avoid decimal conversion issues
        mock_pos.avg_entry_price = "700.0"
        mock_pos.current_price = "650.0"  # Provide a value instead of None
        mock_pos.market_value = None
        mock_pos.cost_basis = None
        mock_pos.unrealized_pl = None
        mock_pos.realized_pl = "100.50"  # String value for realized P&L

        broker.client.get_all_positions.return_value = [mock_pos]

        positions = broker.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "TSLA"
        assert positions[0].quantity == Decimal("-50")
        assert positions[0].current_price == Decimal("650.0")
        assert positions[0].realized_pnl == Decimal("100.50")


class TestAlpacaBrokerAccountInfo:
    """Test account information functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret", paper=True)
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_get_account_info(self, broker):
        """Test getting account information."""
        mock_account = MagicMock()
        mock_account.account_number = "ACC123456"
        mock_account.equity = "100000.00"
        mock_account.cash = "50000.00"
        mock_account.buying_power = "50000.00"
        mock_account.long_market_value = "50000.00"
        mock_account.pattern_day_trader = False

        broker.client.get_account.return_value = mock_account

        account_info = broker.get_account_info()

        assert account_info.account_id == "ACC123456"
        assert account_info.account_type == "paper"
        assert account_info.equity == Decimal("100000.00")
        assert account_info.cash == Decimal("50000.00")
        assert account_info.buying_power == Decimal("50000.00")
        assert account_info.positions_value == Decimal("50000.00")
        assert account_info.pattern_day_trader is False

    def test_get_account_info_live_mode(self):
        """Test account type for live trading."""
        broker = AlpacaBroker(api_key="key", secret_key="secret", paper=False)
        broker.client = MagicMock()
        broker._connected = True

        mock_account = MagicMock()
        mock_account.account_number = "LIVE123"
        mock_account.equity = "250000.00"
        mock_account.cash = "100000.00"
        mock_account.buying_power = "100000.00"
        mock_account.long_market_value = "150000.00"
        mock_account.pattern_day_trader = True

        broker.client.get_account.return_value = mock_account

        account_info = broker.get_account_info()

        assert account_info.account_type == "live"
        assert account_info.pattern_day_trader is True


class TestAlpacaBrokerMarketHours:
    """Test market hours functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True
        return broker

    def test_is_market_open(self, broker):
        """Test checking if market is open."""
        mock_clock = MagicMock()
        mock_clock.is_open = True
        broker.client.get_clock.return_value = mock_clock

        assert broker.is_market_open() is True
        broker.client.get_clock.assert_called_once()

    def test_is_market_closed(self, broker):
        """Test checking if market is closed."""
        mock_clock = MagicMock()
        mock_clock.is_open = False
        broker.client.get_clock.return_value = mock_clock

        assert broker.is_market_open() is False

    def test_get_market_hours(self, broker):
        """Test getting market hours."""
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_clock.next_open = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        mock_clock.next_close = datetime(2024, 1, 1, 21, 0, tzinfo=UTC)
        broker.client.get_clock.return_value = mock_clock

        market_hours = broker.get_market_hours()

        assert market_hours.is_open is True
        assert market_hours.next_open == mock_clock.next_open
        assert market_hours.next_close == mock_clock.next_close


class TestAlpacaBrokerMappers:
    """Test mapping functions between Alpaca and domain objects."""

    @pytest.fixture
    def broker(self):
        """Create broker instance."""
        return AlpacaBroker(api_key="key", secret_key="secret")

    def test_map_status_all_cases(self, broker):
        """Test all status mappings."""
        mappings = {
            AlpacaOrderStatus.NEW: OrderStatus.SUBMITTED,
            AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
            AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
            AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
        }

        for alpaca_status, expected_status in mappings.items():
            assert broker._map_status(alpaca_status) == expected_status

    def test_map_status_unknown(self, broker):
        """Test mapping unknown status defaults to PENDING."""
        unknown_status = MagicMock()
        assert broker._map_status(unknown_status) == OrderStatus.PENDING

    def test_map_side(self, broker):
        """Test side mapping."""
        assert broker._map_side(AlpacaOrderSide.BUY) == OrderSide.BUY
        assert broker._map_side(AlpacaOrderSide.SELL) == OrderSide.SELL

    def test_map_order_type(self, broker):
        """Test order type mapping."""
        assert broker._map_order_type(AlpacaOrderType.LIMIT) == OrderType.LIMIT
        assert broker._map_order_type(AlpacaOrderType.MARKET) == OrderType.MARKET

    def test_safe_decimal_conversion(self, broker):
        """Test safe decimal conversion."""
        assert broker._safe_decimal(100.5) == Decimal("100.5")
        assert broker._safe_decimal("200.75") == Decimal("200.75")
        assert broker._safe_decimal(None) is None
        assert broker._safe_decimal(0) == Decimal("0")

    def test_map_alpaca_to_domain_order_complete(self, broker):
        """Test complete mapping from Alpaca order to domain order."""
        mock_order = MagicMock()
        mock_order.symbol = "AAPL"
        mock_order.side = AlpacaOrderSide.BUY
        mock_order.qty = 100
        mock_order.order_type = AlpacaOrderType.LIMIT
        mock_order.limit_price = 150.50
        mock_order.stop_price = None
        mock_order.status = AlpacaOrderStatus.FILLED
        mock_order.id = "alpaca-123"
        mock_order.created_at = datetime.now(UTC)
        mock_order.filled_at = datetime.now(UTC)
        mock_order.filled_qty = 100
        mock_order.filled_avg_price = 150.45

        domain_order = broker._map_alpaca_to_domain_order(mock_order)

        assert domain_order.symbol == "AAPL"
        assert domain_order.side == OrderSide.BUY
        assert domain_order.quantity == Decimal("100")
        assert domain_order.order_type == OrderType.LIMIT
        assert domain_order.limit_price == Decimal("150.50")
        assert domain_order.stop_price is None
        assert domain_order.status == OrderStatus.FILLED
        assert domain_order.broker_order_id == "alpaca-123"
        assert domain_order.filled_quantity == Decimal("100")
        assert domain_order.average_fill_price == Decimal("150.45")


class TestAlpacaBrokerThreadSafety:
    """Test thread safety of AlpacaBroker operations."""

    @pytest.fixture
    def broker(self):
        """Create connected broker with mock client."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        # Mock submit_order to return unique IDs
        def mock_submit(order_data):
            mock_order = MagicMock()
            mock_order.id = f"alpaca-{uuid4()}"
            return mock_order

        broker.client.submit_order.side_effect = mock_submit
        return broker

    def test_concurrent_order_map_updates(self, broker):
        """Test that order map updates are thread-safe."""
        orders = []
        errors = []

        def submit_order_thread(symbol):
            try:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=Decimal("10"),
                    order_type=OrderType.MARKET,
                )
                submitted = broker.submit_order(order)
                orders.append(submitted)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=submit_order_thread, args=(f"SYM{i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0
        assert len(orders) == 20
        assert len(broker._order_map) == 20

        # Verify all mappings are unique
        mapped_ids = list(broker._order_map.values())
        assert len(mapped_ids) == len(set(mapped_ids))

    def test_concurrent_order_cancellations(self, broker):
        """Test concurrent order cancellations are thread-safe."""
        # Prepare order mappings
        order_ids = [uuid4() for _ in range(10)]
        for i, order_id in enumerate(order_ids):
            broker._order_map[order_id] = f"alpaca-{i}"

        results = []
        errors = []

        def cancel_order_thread(order_id):
            try:
                result = broker.cancel_order(order_id)
                results.append((order_id, result))
            except Exception as e:
                errors.append(e)

        # Create threads for cancellation
        threads = []
        for order_id in order_ids:
            thread = threading.Thread(target=cancel_order_thread, args=(order_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0
        assert len(results) == 10
        assert broker.client.cancel_order_by_id.call_count == 10


class TestAlpacaBrokerErrorHandling:
    """Test error handling in AlpacaBroker."""

    def test_operations_without_connection(self):
        """Test all operations fail without connection."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderType.MARKET
        )

        with pytest.raises(BrokerConnectionError):
            broker.submit_order(order)

        with pytest.raises(BrokerConnectionError):
            broker.cancel_order(uuid4())

        with pytest.raises(BrokerConnectionError):
            broker.get_order_status(uuid4())

        with pytest.raises(BrokerConnectionError):
            broker.update_order(order)

        with pytest.raises(BrokerConnectionError):
            broker.get_recent_orders()

        with pytest.raises(BrokerConnectionError):
            broker.get_positions()

        with pytest.raises(BrokerConnectionError):
            broker.get_account_info()

        with pytest.raises(BrokerConnectionError):
            broker.is_market_open()

        with pytest.raises(BrokerConnectionError):
            broker.get_market_hours()

    def test_submit_order_client_none(self):
        """Test submit_order raises RuntimeError when client is None but connected."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        # Mock _check_connection to pass
        broker._check_connection = MagicMock()
        broker.client = None  # Client is None

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderType.MARKET
        )

        with pytest.raises(RuntimeError, match="Alpaca client not initialized"):
            broker.submit_order(order)

    def test_positions_with_missing_symbol(self):
        """Test get_positions skips positions without symbol attribute."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        # Mock position without symbol attribute
        mock_pos1 = MagicMock()
        del mock_pos1.symbol  # Remove symbol attribute

        # Mock position with symbol
        mock_pos2 = MagicMock()
        mock_pos2.symbol = "AAPL"
        mock_pos2.qty = "100"
        mock_pos2.avg_entry_price = "150.0"
        mock_pos2.current_price = "160.0"
        mock_pos2.realized_pl = "100.0"

        broker.client.get_all_positions.return_value = [mock_pos1, mock_pos2]

        positions = broker.get_positions()

        # Should only include the position with symbol
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    def test_get_recent_orders_with_non_alpaca_order_types(self):
        """Test get_recent_orders filters out non-AlpacaOrder types."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")
        broker.client = MagicMock()
        broker._connected = True

        # Mix of AlpacaOrder and other types
        mock_alpaca_order = MagicMock(spec=AlpacaOrder)
        mock_alpaca_order.symbol = "AAPL"
        mock_alpaca_order.side = AlpacaOrderSide.BUY
        mock_alpaca_order.qty = 100
        mock_alpaca_order.order_type = AlpacaOrderType.MARKET
        mock_alpaca_order.status = AlpacaOrderStatus.FILLED
        mock_alpaca_order.limit_price = None
        mock_alpaca_order.stop_price = None
        mock_alpaca_order.id = "alpaca-123"
        mock_alpaca_order.created_at = datetime.now(UTC)
        mock_alpaca_order.filled_at = datetime.now(UTC)
        mock_alpaca_order.filled_qty = 100
        mock_alpaca_order.filled_avg_price = 150.0

        # Non-AlpacaOrder type (will be filtered out)
        mock_other_order = MagicMock()

        broker.client.get_orders.return_value = [mock_alpaca_order, mock_other_order]

        orders = broker.get_recent_orders()

        # Should only include AlpacaOrder instances
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"

    def test_map_alpaca_to_domain_order_with_defaults(self):
        """Test mapping Alpaca order with None/missing values uses defaults."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        mock_order = MagicMock(spec=AlpacaOrder)
        mock_order.symbol = "TEST"  # Use valid symbol
        mock_order.side = None
        mock_order.qty = "1"  # Use positive quantity
        mock_order.order_type = None
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.status = None
        mock_order.id = None
        mock_order.created_at = None
        mock_order.filled_at = None
        mock_order.filled_qty = None
        mock_order.filled_avg_price = None

        # Remove hasattr check attributes
        del mock_order.status
        del mock_order.id

        domain_order = broker._map_alpaca_to_domain_order(mock_order)

        # Check defaults are applied
        assert domain_order.symbol == "TEST"
        assert domain_order.side == OrderSide.BUY
        assert domain_order.quantity == Decimal("1")
        assert domain_order.order_type == OrderType.MARKET
        assert domain_order.limit_price is None
        assert domain_order.stop_price is None
        assert domain_order.status == OrderStatus.PENDING
        assert domain_order.broker_order_id is None

    def test_order_type_mapping_fallback(self):
        """Test order type mapping falls back to MARKET for unknown types."""
        broker = AlpacaBroker(api_key="key", secret_key="secret")

        # Test unknown order type
        unknown_type = MagicMock()
        result = broker._map_order_type(unknown_type)
        assert result == OrderType.MARKET
