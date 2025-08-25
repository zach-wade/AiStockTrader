"""
Comprehensive unit tests for PaperBroker - achieving 90%+ coverage.

Tests all methods, state management, thread safety, and edge cases.
"""

import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    MarketHours,
    OrderNotFoundError,
)
from src.application.interfaces.exceptions import PositionNotFoundError
from src.domain.entities.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.domain.entities.position import Position
from src.domain.services.trading_calendar import Exchange, TradingCalendar
from src.infrastructure.brokers.paper_broker import PaperBroker, PaperBrokerState


class TestPaperBrokerState:
    """Test PaperBrokerState dataclass."""

    def test_state_initialization_defaults(self):
        """Test state initialization with defaults."""
        state = PaperBrokerState()

        assert state.initial_capital == Decimal("100000")
        assert state.cash_balance == Decimal("100000")
        assert isinstance(state.orders, dict)
        assert isinstance(state.positions, dict)
        assert isinstance(state.market_prices, dict)
        assert len(state.orders) == 0
        assert len(state.positions) == 0
        assert len(state.market_prices) == 0

    def test_state_initialization_custom(self):
        """Test state initialization with custom values."""
        state = PaperBrokerState(initial_capital=Decimal("50000"), cash_balance=Decimal("45000"))

        assert state.initial_capital == Decimal("50000")
        assert state.cash_balance == Decimal("45000")
        assert isinstance(state.orders, dict)
        assert isinstance(state.positions, dict)
        assert isinstance(state.market_prices, dict)

    def test_state_with_existing_data(self):
        """Test state with pre-existing data."""
        orders = {uuid4(): Mock()}
        positions = {"AAPL": Mock()}
        prices = {"AAPL": Decimal("150.00")}

        state = PaperBrokerState(orders=orders, positions=positions, market_prices=prices)

        assert state.orders == orders
        assert state.positions == positions
        assert state.market_prices == prices


class TestPaperBrokerInitialization:
    """Test broker initialization."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        broker = PaperBroker()

        assert broker.state.initial_capital == Decimal("100000")
        assert broker.state.cash_balance == Decimal("100000")
        assert broker._connected is False
        assert type(broker._lock).__name__ == "RLock"
        assert isinstance(broker.trading_calendar, TradingCalendar)
        assert broker.trading_calendar.exchange == Exchange.NYSE

    def test_initialization_custom_capital(self):
        """Test initialization with custom capital."""
        broker = PaperBroker(initial_capital=Decimal("50000"))

        assert broker.state.initial_capital == Decimal("50000")
        assert broker.state.cash_balance == Decimal("50000")

    def test_initialization_custom_exchange(self):
        """Test initialization with custom exchange."""
        broker = PaperBroker(exchange=Exchange.NASDAQ)

        assert broker.trading_calendar.exchange == Exchange.NASDAQ

    def test_initialization_zero_capital(self):
        """Test initialization with zero capital."""
        broker = PaperBroker(initial_capital=Decimal("0"))

        assert broker.state.initial_capital == Decimal("0")
        assert broker.state.cash_balance == Decimal("0")

    def test_initialization_negative_capital(self):
        """Test initialization with negative capital (margin account)."""
        broker = PaperBroker(initial_capital=Decimal("-10000"))

        assert broker.state.initial_capital == Decimal("-10000")
        assert broker.state.cash_balance == Decimal("-10000")


class TestPaperBrokerConnection:
    """Test connection management."""

    @pytest.fixture
    def broker(self):
        """Create broker instance."""
        return PaperBroker()

    def test_connect(self, broker):
        """Test connection."""
        assert broker._connected is False

        broker.connect()

        assert broker._connected is True

    def test_disconnect(self, broker):
        """Test disconnection."""
        broker._connected = True

        broker.disconnect()

        assert broker._connected is False

    def test_is_connected_true(self, broker):
        """Test connection status when connected."""
        broker._connected = True

        assert broker.is_connected() is True

    def test_is_connected_false(self, broker):
        """Test connection status when disconnected."""
        broker._connected = False

        assert broker.is_connected() is False

    def test_check_connection_raises(self, broker):
        """Test connection check raises when not connected."""
        with pytest.raises(BrokerConnectionError, match="Not connected to paper broker"):
            broker._check_connection()

    def test_check_connection_passes(self, broker):
        """Test connection check passes when connected."""
        broker._connected = True

        # Should not raise
        broker._check_connection()

    def test_multiple_connects(self, broker):
        """Test multiple connect calls."""
        broker.connect()
        broker.connect()
        broker.connect()

        assert broker._connected is True

    def test_multiple_disconnects(self, broker):
        """Test multiple disconnect calls."""
        broker._connected = True

        broker.disconnect()
        broker.disconnect()
        broker.disconnect()

        assert broker._connected is False


class TestPaperBrokerOrderSubmission:
    """Test order submission functionality."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = PaperBroker()
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
                quantity=Decimal("50"),
                side=OrderSide.SELL,
                limit_price=Decimal("150.00"),
            )
        )

    def test_submit_market_order(self, broker, market_order):
        """Test submitting market order."""
        result = broker.submit_order(market_order)

        assert result.broker_order_id == f"PAPER-{market_order.id}"
        assert result.status == OrderStatus.SUBMITTED
        assert broker.state.orders[market_order.id] == market_order

    def test_submit_limit_order(self, broker, limit_order):
        """Test submitting limit order."""
        result = broker.submit_order(limit_order)

        assert result.broker_order_id == f"PAPER-{limit_order.id}"
        assert result.status == OrderStatus.SUBMITTED
        assert broker.state.orders[limit_order.id] == limit_order

    def test_submit_order_not_connected(self, market_order):
        """Test order submission when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.submit_order(market_order)

    def test_submit_multiple_orders(self, broker):
        """Test submitting multiple orders."""
        orders = []
        for i in range(10):
            order = Order.create_market_order(
                OrderRequest(
                    symbol=f"SYM{i}",
                    quantity=Decimal("10"),
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                )
            )
            orders.append(order)
            broker.submit_order(order)

        assert len(broker.state.orders) == 10
        for order in orders:
            assert order.id in broker.state.orders
            assert broker.state.orders[order.id] == order

    def test_submit_order_with_stop_price(self, broker):
        """Test submitting stop order."""
        order = Order(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("145.00"),
        )

        result = broker.submit_order(order)

        assert result.stop_price == Decimal("145.00")
        assert result.id in broker.state.orders

    def test_submit_order_with_time_in_force(self, broker):
        """Test order with different time in force values."""
        order = Order.create_limit_order(
            OrderRequest(
                symbol="AAPL",
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                limit_price=Decimal("150.00"),
            )
        )
        order.time_in_force = TimeInForce.IOC

        result = broker.submit_order(order)

        assert result.time_in_force == TimeInForce.IOC
        assert result.id in broker.state.orders


class TestPaperBrokerOrderCancellation:
    """Test order cancellation functionality."""

    @pytest.fixture
    def broker_with_orders(self):
        """Create broker with existing orders."""
        broker = PaperBroker()
        broker._connected = True

        # Add some orders
        orders = []
        for i in range(3):
            order = Order.create_market_order(
                OrderRequest(symbol=f"SYM{i}", quantity=Decimal("100"), side=OrderSide.BUY)
            )
            order.submit(f"PAPER-{order.id}")
            broker.state.orders[order.id] = order
            orders.append(order)

        return broker, orders

    def test_cancel_order_success(self, broker_with_orders):
        """Test successful order cancellation."""
        broker, orders = broker_with_orders
        order_to_cancel = orders[0]

        result = broker.cancel_order(order_to_cancel.id)

        assert result is True
        assert broker.state.orders[order_to_cancel.id].status == OrderStatus.CANCELLED

    def test_cancel_order_not_found(self, broker_with_orders):
        """Test cancelling non-existent order."""
        broker, _ = broker_with_orders
        unknown_id = uuid4()

        with pytest.raises(OrderNotFoundError, match=f"Order {unknown_id} not found"):
            broker.cancel_order(unknown_id)

    def test_cancel_order_already_filled(self, broker_with_orders):
        """Test cancelling already filled order."""
        broker, orders = broker_with_orders
        order = orders[0]

        # Mark order as filled
        order.fill(Decimal("100"), Decimal("150.00"))

        result = broker.cancel_order(order.id)

        assert result is False
        assert order.status == OrderStatus.FILLED

    def test_cancel_order_not_connected(self):
        """Test cancellation when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.cancel_order(uuid4())

    def test_cancel_multiple_orders(self, broker_with_orders):
        """Test cancelling multiple orders."""
        broker, orders = broker_with_orders

        results = []
        for order in orders:
            result = broker.cancel_order(order.id)
            results.append(result)

        assert all(results)
        for order in orders:
            assert broker.state.orders[order.id].status == OrderStatus.CANCELLED


class TestPaperBrokerOrderRetrieval:
    """Test order retrieval functionality."""

    @pytest.fixture
    def broker_with_orders(self):
        """Create broker with various order states."""
        broker = PaperBroker()
        broker._connected = True

        orders = {}

        # Submitted order
        order1 = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        )
        order1.submit(f"PAPER-{order1.id}")
        orders["submitted"] = order1

        # Filled order
        order2 = Order.create_limit_order(
            OrderRequest(
                symbol="GOOGL",
                quantity=Decimal("50"),
                side=OrderSide.SELL,
                limit_price=Decimal("2800.00"),
            )
        )
        order2.submit(f"PAPER-{order2.id}")
        order2.fill(Decimal("50"), Decimal("2805.00"))
        orders["filled"] = order2

        # Cancelled order
        order3 = Order.create_market_order(
            OrderRequest(symbol="MSFT", quantity=Decimal("75"), side=OrderSide.BUY)
        )
        order3.submit(f"PAPER-{order3.id}")
        order3.cancel()
        orders["cancelled"] = order3

        # Add all to broker
        for order in orders.values():
            broker.state.orders[order.id] = order

        return broker, orders

    def test_get_order_existing(self, broker_with_orders):
        """Test getting existing order."""
        broker, orders = broker_with_orders
        submitted_order = orders["submitted"]

        result = broker.get_order(submitted_order.id)

        assert result == submitted_order
        assert result.symbol == "AAPL"
        assert result.status == OrderStatus.SUBMITTED

    def test_get_order_not_found(self, broker_with_orders):
        """Test getting non-existent order."""
        broker, _ = broker_with_orders

        with pytest.raises(OrderNotFoundError):
            broker.get_order(uuid4())

    def test_get_order_not_connected(self):
        """Test get order when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.get_order(uuid4())

    def test_get_recent_orders_all(self, broker_with_orders):
        """Test getting all recent orders."""
        broker, orders = broker_with_orders

        results = broker.get_recent_orders()

        assert len(results) == 3
        symbols = {order.symbol for order in results}
        assert symbols == {"AAPL", "GOOGL", "MSFT"}

    def test_get_recent_orders_with_limit(self, broker_with_orders):
        """Test getting recent orders with limit."""
        broker, _ = broker_with_orders

        results = broker.get_recent_orders(limit=2)

        assert len(results) == 2

    def test_get_recent_orders_by_status(self, broker_with_orders):
        """Test getting orders filtered by status."""
        broker, _ = broker_with_orders

        # Get only filled orders
        results = broker.get_recent_orders(status="filled")

        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED
        assert results[0].symbol == "GOOGL"

    def test_get_recent_orders_open_status(self, broker_with_orders):
        """Test getting open orders."""
        broker, _ = broker_with_orders

        results = broker.get_recent_orders(status="open")

        assert len(results) == 1
        assert results[0].status == OrderStatus.SUBMITTED
        assert results[0].symbol == "AAPL"

    def test_get_recent_orders_empty(self):
        """Test getting orders when none exist."""
        broker = PaperBroker()
        broker._connected = True

        results = broker.get_recent_orders()

        assert results == []


class TestPaperBrokerPositions:
    """Test position management functionality."""

    @pytest.fixture
    def broker_with_positions(self):
        """Create broker with positions."""
        broker = PaperBroker()
        broker._connected = True

        # Add positions
        position1 = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            entry_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
        )

        position2 = Position(
            symbol="GOOGL",
            quantity=Decimal("50"),
            side=PositionSide.SHORT,
            entry_price=Decimal("2850.00"),
            current_price=Decimal("2800.00"),
        )

        broker.state.positions["AAPL"] = position1
        broker.state.positions["GOOGL"] = position2

        # Set market prices
        broker.state.market_prices["AAPL"] = Decimal("150.00")
        broker.state.market_prices["GOOGL"] = Decimal("2800.00")

        return broker

    def test_get_positions(self, broker_with_positions):
        """Test getting all positions."""
        results = broker_with_positions.get_positions()

        assert len(results) == 2
        symbols = {pos.symbol for pos in results}
        assert symbols == {"AAPL", "GOOGL"}

    def test_get_positions_empty(self):
        """Test getting positions when none exist."""
        broker = PaperBroker()
        broker._connected = True

        results = broker.get_positions()

        assert results == []

    def test_get_position_by_symbol(self, broker_with_positions):
        """Test getting position by symbol."""
        result = broker_with_positions.get_position("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.quantity == Decimal("100")
        assert result.side == PositionSide.LONG

    def test_get_position_not_found(self, broker_with_positions):
        """Test getting non-existent position."""
        result = broker_with_positions.get_position("TSLA")

        assert result is None

    def test_get_position_not_connected(self):
        """Test get position when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.get_position("AAPL")

    def test_close_position_success(self, broker_with_positions):
        """Test closing a position."""
        result = broker_with_positions.close_position("AAPL")

        assert result is True
        assert "AAPL" not in broker_with_positions.state.positions

    def test_close_position_not_found(self, broker_with_positions):
        """Test closing non-existent position."""
        with pytest.raises(PositionNotFoundError, match="Position TSLA not found"):
            broker_with_positions.close_position("TSLA")

    def test_close_all_positions(self, broker_with_positions):
        """Test closing all positions."""
        result = broker_with_positions.close_all_positions()

        assert result is True
        assert len(broker_with_positions.state.positions) == 0

    def test_close_all_positions_empty(self):
        """Test closing all when no positions exist."""
        broker = PaperBroker()
        broker._connected = True

        result = broker.close_all_positions()

        assert result is True
        assert len(broker.state.positions) == 0


class TestPaperBrokerAccount:
    """Test account information functionality."""

    @pytest.fixture
    def broker_with_positions(self):
        """Create broker with positions and market prices."""
        broker = PaperBroker(initial_capital=Decimal("100000"))
        broker._connected = True

        # Add positions
        broker.state.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            entry_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
        )

        broker.state.positions["GOOGL"] = Position(
            symbol="GOOGL",
            quantity=Decimal("10"),
            side=PositionSide.LONG,
            entry_price=Decimal("2800.00"),
            current_price=Decimal("2850.00"),
        )

        # Set market prices
        broker.state.market_prices["AAPL"] = Decimal("150.00")
        broker.state.market_prices["GOOGL"] = Decimal("2850.00")

        # Adjust cash (spent on positions)
        broker.state.cash_balance = Decimal("100000") - Decimal("14500") - Decimal("28000")

        return broker

    def test_get_account_info(self, broker_with_positions):
        """Test getting account information."""
        result = broker_with_positions.get_account_info()

        assert isinstance(result, AccountInfo)
        assert result.cash == Decimal("57500")  # 100000 - 14500 - 28000

        # Portfolio value = cash + position values
        # AAPL: 100 * 150 = 15000
        # GOOGL: 10 * 2850 = 28500
        # Total: 57500 + 15000 + 28500 = 101000
        assert result.portfolio_value == Decimal("101000")
        assert result.equity == Decimal("101000")
        assert result.long_market_value == Decimal("43500")  # 15000 + 28500
        assert result.short_market_value == Decimal("0")
        assert result.buying_power == result.cash * 2  # Paper trading default
        assert result.status == "ACTIVE"
        assert result.pattern_day_trader is False
        assert result.trading_blocked is False

    def test_get_account_info_no_positions(self):
        """Test account info with no positions."""
        broker = PaperBroker(initial_capital=Decimal("50000"))
        broker._connected = True

        result = broker.get_account_info()

        assert result.cash == Decimal("50000")
        assert result.portfolio_value == Decimal("50000")
        assert result.long_market_value == Decimal("0")
        assert result.short_market_value == Decimal("0")

    def test_get_account_info_with_short_positions(self):
        """Test account info with short positions."""
        broker = PaperBroker(initial_capital=Decimal("100000"))
        broker._connected = True

        # Add short position
        broker.state.positions["TSLA"] = Position(
            symbol="TSLA",
            quantity=Decimal("50"),
            side=PositionSide.SHORT,
            entry_price=Decimal("200.00"),
            current_price=Decimal("195.00"),
        )

        broker.state.market_prices["TSLA"] = Decimal("195.00")
        broker.state.cash_balance = Decimal("110000")  # Received 10000 from short sale

        result = broker.get_account_info()

        assert result.short_market_value == Decimal("9750")  # 50 * 195

    def test_get_account_info_not_connected(self):
        """Test account info when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.get_account_info()


class TestPaperBrokerMarketOperations:
    """Test market-related operations."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = PaperBroker()
        broker._connected = True
        return broker

    def test_set_market_price(self, broker):
        """Test setting market price."""
        broker.set_market_price("AAPL", Decimal("155.50"))

        assert broker.state.market_prices["AAPL"] == Decimal("155.50")

    def test_set_market_price_update(self, broker):
        """Test updating existing market price."""
        broker.state.market_prices["AAPL"] = Decimal("150.00")

        broker.set_market_price("AAPL", Decimal("155.50"))

        assert broker.state.market_prices["AAPL"] == Decimal("155.50")

    def test_set_market_price_not_connected(self):
        """Test set price when not connected."""
        broker = PaperBroker()

        with pytest.raises(BrokerConnectionError):
            broker.set_market_price("AAPL", Decimal("150.00"))

    def test_get_market_hours_open(self, broker):
        """Test market hours during trading day."""
        # Mock a weekday during market hours
        test_date = datetime(2025, 8, 21, 12, 0, tzinfo=UTC)  # Thursday noon

        with patch.object(broker.trading_calendar, "is_market_open", return_value=True):
            with patch.object(broker.trading_calendar, "get_market_hours") as mock_hours:
                mock_hours.return_value = (
                    datetime(2025, 8, 21, 9, 30, tzinfo=UTC),
                    datetime(2025, 8, 21, 16, 0, tzinfo=UTC),
                )

                result = broker.get_market_hours(test_date)

                assert isinstance(result, MarketHours)
                assert result.is_open is True
                assert result.open_time == datetime(2025, 8, 21, 9, 30, tzinfo=UTC)
                assert result.close_time == datetime(2025, 8, 21, 16, 0, tzinfo=UTC)

    def test_get_market_hours_closed(self, broker):
        """Test market hours on weekend."""
        # Saturday
        test_date = datetime(2025, 8, 23, 12, 0, tzinfo=UTC)

        with patch.object(broker.trading_calendar, "is_market_open", return_value=False):
            with patch.object(broker.trading_calendar, "get_market_hours") as mock_hours:
                mock_hours.return_value = (None, None)

                result = broker.get_market_hours(test_date)

                assert result.is_open is False
                assert result.open_time is None
                assert result.close_time is None

    def test_process_fills(self, broker):
        """Test order fill processing."""
        # Add market order
        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        )
        order.submit(f"PAPER-{order.id}")
        broker.state.orders[order.id] = order

        # Set market price
        broker.state.market_prices["AAPL"] = Decimal("150.00")

        # Process fills
        broker.process_fills()

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("100")
        assert order.average_fill_price == Decimal("150.00")

        # Check position created
        assert "AAPL" in broker.state.positions
        position = broker.state.positions["AAPL"]
        assert position.quantity == Decimal("100")
        assert position.entry_price == Decimal("150.00")

        # Check cash reduced
        assert broker.state.cash_balance == Decimal("100000") - Decimal("15000")

    def test_process_fills_limit_order(self, broker):
        """Test limit order fill processing."""
        # Add limit order
        order = Order.create_limit_order(
            OrderRequest(
                symbol="AAPL",
                quantity=Decimal("50"),
                side=OrderSide.BUY,
                limit_price=Decimal("149.00"),
            )
        )
        order.submit(f"PAPER-{order.id}")
        broker.state.orders[order.id] = order

        # Set market price above limit - should not fill
        broker.state.market_prices["AAPL"] = Decimal("150.00")
        broker.process_fills()

        assert order.status == OrderStatus.SUBMITTED

        # Set market price at or below limit - should fill
        broker.state.market_prices["AAPL"] = Decimal("148.50")
        broker.process_fills()

        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == Decimal("149.00")  # Filled at limit price


class TestPaperBrokerThreadSafety:
    """Test thread safety of broker operations."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = PaperBroker()
        broker._connected = True
        return broker

    def test_concurrent_order_submissions(self, broker):
        """Test concurrent order submissions."""
        num_threads = 20
        orders_submitted = []
        errors = []

        def submit_order(thread_id):
            try:
                order = Order.create_market_order(
                    OrderRequest(
                        symbol=f"SYM{thread_id}",
                        quantity=Decimal("10"),
                        side=OrderSide.BUY if thread_id % 2 == 0 else OrderSide.SELL,
                    )
                )
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

        assert len(errors) == 0
        assert len(orders_submitted) == num_threads
        assert len(broker.state.orders) == num_threads

    def test_concurrent_position_updates(self, broker):
        """Test concurrent position updates."""
        # Pre-populate positions
        for i in range(10):
            broker.state.positions[f"SYM{i}"] = Position(
                symbol=f"SYM{i}",
                quantity=Decimal("100"),
                side=PositionSide.LONG,
                entry_price=Decimal("100.00"),
                current_price=Decimal("100.00"),
            )

        errors = []

        def update_position(symbol):
            try:
                with broker._lock:
                    if symbol in broker.state.positions:
                        position = broker.state.positions[symbol]
                        position.current_price = Decimal("105.00")
                        time.sleep(0.0001)  # Simulate processing
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            for _ in range(5):  # Multiple threads per symbol
                thread = threading.Thread(target=update_position, args=(f"SYM{i}",))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        for i in range(10):
            assert broker.state.positions[f"SYM{i}"].current_price == Decimal("105.00")

    def test_concurrent_market_price_updates(self, broker):
        """Test concurrent market price updates."""
        symbols = [f"SYM{i}" for i in range(10)]
        errors = []

        def update_prices(thread_id):
            try:
                for symbol in symbols:
                    price = Decimal(f"{100 + thread_id}.{thread_id:02d}")
                    broker.set_market_price(symbol, price)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_prices, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(broker.state.market_prices) == 10

        # All symbols should have a price
        for symbol in symbols:
            assert symbol in broker.state.market_prices

    def test_concurrent_reads_and_writes(self, broker):
        """Test concurrent reads and writes."""
        write_errors = []
        read_errors = []

        def writer_thread(thread_id):
            try:
                for i in range(10):
                    order = Order.create_market_order(
                        OrderRequest(
                            symbol=f"W{thread_id}-{i}", quantity=Decimal("10"), side=OrderSide.BUY
                        )
                    )
                    broker.submit_order(order)
                    time.sleep(0.0001)
            except Exception as e:
                write_errors.append(e)

        def reader_thread():
            try:
                for _ in range(50):
                    broker.get_recent_orders()
                    broker.get_positions()
                    broker.get_account_info()
                    time.sleep(0.0001)
            except Exception as e:
                read_errors.append(e)

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

        assert len(write_errors) == 0
        assert len(read_errors) == 0
        assert len(broker.state.orders) == 50  # 5 threads * 10 orders


class TestPaperBrokerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def broker(self):
        """Create connected broker."""
        broker = PaperBroker()
        broker._connected = True
        return broker

    def test_process_fills_insufficient_cash(self, broker):
        """Test order fill with insufficient cash."""
        broker.state.cash_balance = Decimal("1000")

        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        )
        order.submit(f"PAPER-{order.id}")
        broker.state.orders[order.id] = order
        broker.state.market_prices["AAPL"] = Decimal("150.00")  # Needs 15000

        broker.process_fills()

        # Order should remain submitted (not filled)
        assert order.status == OrderStatus.SUBMITTED

    def test_process_fills_sell_without_position(self, broker):
        """Test selling without position (short selling)."""
        order = Order.create_market_order("AAPL", Decimal("50"), OrderSide.SELL)
        order.submit(f"PAPER-{order.id}")
        broker.state.orders[order.id] = order
        broker.state.market_prices["AAPL"] = Decimal("150.00")

        broker.process_fills()

        # Should create short position
        assert order.status == OrderStatus.FILLED
        assert "AAPL" in broker.state.positions
        position = broker.state.positions["AAPL"]
        assert position.side == PositionSide.SHORT
        assert position.quantity == Decimal("50")

        # Cash should increase from short sale
        assert broker.state.cash_balance == Decimal("100000") + Decimal("7500")

    def test_very_large_order(self, broker):
        """Test handling very large order."""
        large_qty = Decimal("999999999")
        order = Order.create_market_order(
            OrderRequest(symbol="AAPL", quantity=large_qty, side=OrderSide.BUY)
        )

        result = broker.submit_order(order)

        assert result.quantity == large_qty
        assert result.id in broker.state.orders

    def test_fractional_shares(self, broker):
        """Test handling fractional shares."""
        order = Order.create_market_order("AAPL", Decimal("10.5"), OrderSide.BUY)
        order.submit(f"PAPER-{order.id}")
        broker.state.orders[order.id] = order
        broker.state.market_prices["AAPL"] = Decimal("150.00")

        broker.process_fills()

        assert order.filled_quantity == Decimal("10.5")
        position = broker.state.positions["AAPL"]
        assert position.quantity == Decimal("10.5")

    def test_zero_quantity_order(self, broker):
        """Test handling zero quantity order."""
        order = Order(
            symbol="AAPL", quantity=Decimal("0"), side=OrderSide.BUY, order_type=OrderType.MARKET
        )

        # Should still accept the order
        result = broker.submit_order(order)
        assert result.id in broker.state.orders

    def test_negative_price(self, broker):
        """Test handling negative price (shouldn't happen but test resilience)."""
        broker.set_market_price("AAPL", Decimal("-10.00"))

        assert broker.state.market_prices["AAPL"] == Decimal("-10.00")

    def test_position_with_zero_quantity(self, broker):
        """Test position with zero quantity."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("0"),
            side=PositionSide.LONG,
            entry_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
        )
        broker.state.positions["AAPL"] = position

        result = broker.get_position("AAPL")

        # Should return the position even with zero quantity
        assert result is not None
        assert result.quantity == Decimal("0")

    def test_unicode_symbol(self, broker):
        """Test handling unicode symbols."""
        order = Order.create_market_order("测试", Decimal("100"), OrderSide.BUY)

        result = broker.submit_order(order)

        assert result.symbol == "测试"
        assert result.id in broker.state.orders

    def test_extremely_long_symbol(self, broker):
        """Test handling very long symbol names."""
        long_symbol = "A" * 1000
        order = Order.create_market_order(long_symbol, Decimal("100"), OrderSide.BUY)

        result = broker.submit_order(order)

        assert result.symbol == long_symbol
        assert result.id in broker.state.orders
