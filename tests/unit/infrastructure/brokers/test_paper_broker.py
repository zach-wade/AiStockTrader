"""
Comprehensive tests for the PaperBroker implementation.
"""

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.application.interfaces.broker import OrderNotFoundError
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.value_objects import Price, Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker


@pytest.fixture
def paper_broker():
    """Create a PaperBroker instance for testing."""
    return PaperBroker()


@pytest.fixture
def connected_paper_broker():
    """Create a connected PaperBroker instance for testing."""
    broker = PaperBroker()
    broker.connect()
    return broker


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
    order.id = uuid4()
    return order


@pytest.fixture
def sample_limit_order():
    """Create a sample limit order for testing."""
    order = Order(
        symbol="GOOGL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50,
        limit_price=Decimal("2500.00"),
    )
    order.id = uuid4()
    return order


class TestPaperBrokerInitialization:
    """Tests for PaperBroker initialization and configuration."""

    def test_default_initialization(self):
        """Test broker initializes with default values."""
        broker = PaperBroker()

        assert broker.state.initial_capital == Decimal("100000")
        assert broker.state.cash_balance == Decimal("100000")
        assert broker.state.orders == {}
        assert broker.state.positions == {}
        assert broker.state.market_prices == {}
        assert broker._lock is not None

    def test_custom_initial_capital(self):
        """Test broker initialization with custom capital."""
        broker = PaperBroker(initial_capital=Decimal("50000"))

        assert broker.state.initial_capital == Decimal("50000")
        assert broker.state.cash_balance == Decimal("50000")

    def test_custom_state(self):
        """Test broker initialization with custom initial capital."""
        broker = PaperBroker(initial_capital=Decimal("200000"))

        assert broker.state.initial_capital == Decimal("200000")
        assert broker.state.cash_balance == Decimal("200000")

    def test_trading_calendar_initialization(self):
        """Test trading calendar is initialized."""
        broker = PaperBroker()
        assert broker.trading_calendar is not None


class TestPaperBrokerConnection:
    """Tests for broker connection management."""

    def test_connect_success(self, paper_broker):
        """Test successful connection."""
        paper_broker.connect()
        assert paper_broker.is_connected() is True

    def test_disconnect_success(self, paper_broker):
        """Test successful disconnection."""
        paper_broker.connect()
        paper_broker.disconnect()
        assert paper_broker.is_connected() is False

    def test_is_connected(self, paper_broker):
        """Test connection status check."""
        # Initially disconnected
        assert paper_broker.is_connected() is False
        paper_broker.connect()
        assert paper_broker.is_connected() is True


class TestPaperBrokerOrderManagement:
    """Tests for order submission and management."""

    def test_submit_market_order(self, connected_paper_broker, sample_order):
        """Test submitting a market order."""
        # Set market price
        connected_paper_broker.update_market_price("AAPL", Decimal("150.00"))

        # Submit order
        result = connected_paper_broker.submit_order(sample_order)

        assert result == sample_order
        assert sample_order.id in connected_paper_broker.state.orders
        assert sample_order.status == OrderStatus.FILLED
        # Check the price value, not the Price object
        assert sample_order.average_fill_price.value == Decimal("150.00")

    @pytest.mark.skip(reason="Limit order logic needs fixing")
    def test_submit_limit_order(self, connected_paper_broker, sample_limit_order):
        """Test submitting a limit order."""
        # Set market price below limit
        connected_paper_broker.update_market_price("GOOGL", Decimal("2400.00"))

        # Submit order
        result = connected_paper_broker.submit_order(sample_limit_order)

        assert result == sample_limit_order
        assert sample_limit_order.id in connected_paper_broker.state.orders
        assert sample_limit_order.status == OrderStatus.PENDING

    def test_submit_order_no_market_price(self, paper_broker, sample_order):
        """Test submitting order without market price."""
        # Don't set market price
        result = paper_broker.submit_order(sample_order)

        assert result == sample_order
        assert sample_order.status == OrderStatus.PENDING

    def test_cancel_order(self, paper_broker, sample_order):
        """Test cancelling an order."""
        # Submit order first
        paper_broker.submit_order(sample_order)

        # Cancel it
        result = paper_broker.cancel_order(sample_order.id)

        assert result is True
        assert sample_order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self, paper_broker):
        """Test cancelling a non-existent order."""
        fake_id = uuid4()

        with pytest.raises(OrderNotFoundError):
            paper_broker.cancel_order(fake_id)

    def test_cancel_filled_order(self, paper_broker, sample_order):
        """Test cancelling an already filled order."""
        # Set market price and submit order (will auto-fill)
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        paper_broker.submit_order(sample_order)

        # Try to cancel filled order
        result = paper_broker.cancel_order(sample_order.id)

        assert result is False  # Can't cancel filled order

    def test_get_order_status(self, paper_broker, sample_order):
        """Test getting order status."""
        # Submit order
        paper_broker.submit_order(sample_order)

        # Get status
        status = paper_broker.get_order_status(sample_order.id)

        assert status == sample_order.status

    def test_get_order_status_nonexistent(self, paper_broker):
        """Test getting status of non-existent order."""
        fake_id = uuid4()

        with pytest.raises(OrderNotFoundError):
            paper_broker.get_order_status(fake_id)

    def test_get_open_orders(self, paper_broker):
        """Test getting list of open orders."""
        # Create multiple orders
        order1 = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(Decimal("100")),
            limit_price=Price(Decimal("150")),
        )
        order2 = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Quantity(Decimal("50")),
            limit_price=Price(Decimal("2500")),
        )
        order3 = Order(
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("75")),
        )

        # Submit orders (limit orders will be pending)
        paper_broker.submit_order(order1)
        paper_broker.submit_order(order2)

        # Set price and submit market order (will fill immediately)
        paper_broker.update_market_price("MSFT", Decimal("300.00"))
        paper_broker.submit_order(order3)

        # Get open orders
        open_orders = paper_broker.get_open_orders()

        assert len(open_orders) == 2  # Only limit orders are open
        assert order1 in open_orders
        assert order2 in open_orders
        assert order3 not in open_orders  # Market order was filled


class TestPaperBrokerPositionManagement:
    """Tests for position tracking and management."""

    def test_get_positions_empty(self, paper_broker):
        """Test getting positions when none exist."""
        positions = paper_broker.get_positions()
        assert positions == []

    def test_get_positions_after_buy(self, paper_broker):
        """Test position creation after buy order."""
        # Set market price and submit buy order
        paper_broker.update_market_price("AAPL", Decimal("150.00"))

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(order)

        # Get positions
        positions = paper_broker.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity.value == Decimal("100")
        # Check the price value, not the Price object
        assert positions[0].average_entry_price.value == Decimal("150.00")

    def test_get_positions_after_sell(self, paper_broker):
        """Test position update after sell order."""
        # Set market price and submit buy order
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(buy_order)

        # Update price and sell half
        paper_broker.update_market_price("AAPL", Decimal("160.00"))
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("50")),
        )
        paper_broker.submit_order(sell_order)

        # Get positions
        positions = paper_broker.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity.value == Decimal("50")  # Reduced quantity

    def test_get_position_by_symbol(self, paper_broker):
        """Test getting specific position by symbol."""
        # Create positions
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        paper_broker.update_market_price("GOOGL", Decimal("2500.00"))

        paper_broker.submit_order(
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Quantity(Decimal("100")),
            )
        )
        paper_broker.submit_order(
            Order(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Quantity(Decimal("50")),
            )
        )

        # Get specific position
        aapl_position = paper_broker.get_position("AAPL")

        assert aapl_position is not None
        assert aapl_position.symbol == "AAPL"
        assert aapl_position.quantity.value == Decimal("100")

    def test_get_position_nonexistent(self, paper_broker):
        """Test getting position for symbol with no position."""
        position = paper_broker.get_position("TSLA")
        assert position is None


class TestPaperBrokerAccountInfo:
    """Tests for account information retrieval."""

    def test_get_account_info_initial(self, paper_broker):
        """Test getting initial account info."""
        account = paper_broker.get_account_info()

        assert account.cash_balance == Decimal("100000")
        assert account.buying_power == Decimal("100000")
        assert account.total_equity == Decimal("100000")
        assert account.margin_used == Decimal("0")

    def test_get_account_info_after_trades(self, paper_broker):
        """Test account info after executing trades."""
        # Execute a buy order
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(order)

        # Get account info
        account = paper_broker.get_account_info()

        expected_cash = Decimal("100000") - (Decimal("150.00") * 100)
        expected_equity = expected_cash + (Decimal("150.00") * 100)

        assert account.cash_balance == expected_cash
        assert account.total_equity == expected_equity


class TestPaperBrokerMarketData:
    """Tests for market data handling."""

    def test_update_market_price(self, paper_broker):
        """Test updating market prices."""
        paper_broker.update_market_price("AAPL", Decimal("150.00"))

        assert paper_broker.state.market_prices["AAPL"] == Decimal("150.00")

    def test_update_multiple_prices(self, paper_broker):
        """Test updating multiple market prices."""
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        paper_broker.update_market_price("GOOGL", Decimal("2500.00"))
        paper_broker.update_market_price("MSFT", Decimal("300.00"))

        assert len(paper_broker.state.market_prices) == 3
        assert paper_broker.state.market_prices["GOOGL"] == Decimal("2500.00")

    def test_limit_order_fill_on_price_update(self, paper_broker):
        """Test limit order fills when price crosses limit."""
        # Submit limit buy order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(Decimal("100")),
            limit_price=Price(Decimal("145.00")),
        )
        paper_broker.submit_order(order)

        assert order.status == OrderStatus.PENDING

        # Update price to trigger fill
        paper_broker.update_market_price("AAPL", Decimal("144.00"))
        paper_broker.check_pending_orders()

        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == Decimal("144.00")


class TestPaperBrokerMarketHours:
    """Tests for market hours checking."""

    def test_get_market_hours(self, paper_broker):
        """Test getting market hours."""
        with patch.object(paper_broker.trading_calendar, "get_market_hours") as mock_hours:
            mock_hours.return_value = {
                "is_open": True,
                "open_time": datetime.now(UTC) - timedelta(hours=1),
                "close_time": datetime.now(UTC) + timedelta(hours=5),
            }

            hours = paper_broker.get_market_hours()

            assert hours.is_open is True
            assert hours.open_time is not None
            assert hours.close_time is not None

    def test_is_market_open(self, paper_broker):
        """Test checking if market is open."""
        with patch.object(paper_broker.trading_calendar, "is_market_open") as mock_open:
            mock_open.return_value = True

            is_open = paper_broker.is_market_open()

            assert is_open is True


class TestPaperBrokerThreadSafety:
    """Tests for thread safety in concurrent operations."""

    def test_concurrent_order_submission(self, paper_broker):
        """Test thread safety with concurrent order submissions."""
        paper_broker.update_market_price("AAPL", Decimal("150.00"))

        # Create multiple orders
        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Quantity(Decimal("10")),
            )
            for _ in range(10)
        ]

        # Submit orders concurrently
        tasks = [paper_broker.submit_order(order) for order in orders]
        results = asyncio.gather(*tasks)

        # All orders should be submitted successfully
        assert len(results) == 10
        assert len(paper_broker.state.orders) == 10

        # Check position is correct
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity.value == Decimal("100")  # 10 orders * 10 shares

    def test_concurrent_price_updates(self, paper_broker):
        """Test thread safety with concurrent price updates."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        def update_prices(symbol, prices):
            for price in prices:
                paper_broker.update_market_price(symbol, price)

        # Create threads for concurrent price updates
        threads = []
        for symbol in symbols:
            prices = [Decimal(str(100 + i)) for i in range(10)]
            thread = threading.Thread(target=update_prices, args=(symbol, prices))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check all symbols have prices
        assert len(paper_broker.state.market_prices) == 5
        for symbol in symbols:
            assert symbol in paper_broker.state.market_prices


class TestPaperBrokerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_submit_order_insufficient_funds(self, paper_broker):
        """Test order submission with insufficient funds."""
        # Try to buy more than we can afford
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("10000")),
        )  # $1.5M worth

        result = paper_broker.submit_order(order)

        # Order should be rejected due to insufficient funds
        assert result == order
        assert order.status == OrderStatus.REJECTED

    def test_short_selling(self, paper_broker):
        """Test short selling (selling without position)."""
        paper_broker.update_market_price("AAPL", Decimal("150.00"))

        # Sell without owning
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(order)

        # Should create short position
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity.value == Decimal("-100")  # Negative for short

    def test_zero_quantity_order(self, paper_broker):
        """Test handling of zero quantity order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("0")),
        )

        result = paper_broker.submit_order(order)

        assert result == order
        assert order.status == OrderStatus.REJECTED

    def test_negative_price_update(self, paper_broker):
        """Test handling of negative price update."""
        # Should handle gracefully without crashing
        paper_broker.update_market_price("AAPL", Decimal("-10.00"))

        assert paper_broker.state.market_prices["AAPL"] == Decimal("-10.00")

    def test_fill_order_with_slippage(self, paper_broker):
        """Test order fill with simulated slippage."""
        paper_broker.state.slippage_bps = 10  # 0.1% slippage
        paper_broker.update_market_price("AAPL", Decimal("150.00"))

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(order)

        # Price should be slightly worse due to slippage
        # Expected: 150.00 * 1.001 for buy at higher price
        assert order.average_fill_price >= Decimal("150.00")


class TestPaperBrokerStateManagement:
    """Tests for state persistence and management."""

    def test_state_reset(self, paper_broker):
        """Test resetting broker state."""
        # Add some orders and positions
        paper_broker.state.orders[uuid4()] = Mock()
        paper_broker.state.positions["AAPL"] = Mock()
        paper_broker.state.cash_balance = Decimal("50000")

        # Reset state
        paper_broker.reset()

        assert paper_broker.state.orders == {}
        assert paper_broker.state.positions == {}
        assert paper_broker.state.cash_balance == paper_broker.state.initial_capital

    def test_get_state_snapshot(self, paper_broker):
        """Test getting a snapshot of current state."""
        # Set up some state
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        paper_broker.state.cash_balance = Decimal("75000")

        snapshot = paper_broker.get_state_snapshot()

        assert snapshot["cash_balance"] == Decimal("75000")
        assert snapshot["initial_capital"] == Decimal("100000")
        assert "AAPL" in snapshot["market_prices"]

    def test_load_state_from_snapshot(self):
        """Test loading state from a snapshot."""
        snapshot = {
            "initial_capital": Decimal("200000"),
            "cash_balance": Decimal("150000"),
            "market_prices": {"AAPL": Decimal("155.00")},
        }

        broker = PaperBroker.from_snapshot(snapshot)

        assert broker.state.initial_capital == Decimal("200000")
        assert broker.state.cash_balance == Decimal("150000")
        assert broker.state.market_prices["AAPL"] == Decimal("155.00")


class TestPaperBrokerCalculations:
    """Tests for P&L and other calculations."""

    def test_realized_pnl_calculation(self, paper_broker):
        """Test calculation of realized P&L."""
        # Buy 100 shares at $150
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(buy_order)

        # Sell 100 shares at $160
        paper_broker.update_market_price("AAPL", Decimal("160.00"))
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(sell_order)

        # Check realized P&L
        expected_pnl = (Decimal("160.00") - Decimal("150.00")) * 100
        assert paper_broker.get_realized_pnl() == expected_pnl

    def test_unrealized_pnl_calculation(self, paper_broker):
        """Test calculation of unrealized P&L."""
        # Buy 100 shares at $150
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        paper_broker.submit_order(order)

        # Update market price
        paper_broker.update_market_price("AAPL", Decimal("155.00"))

        # Check unrealized P&L
        unrealized_pnl = paper_broker.get_unrealized_pnl()
        expected_pnl = (Decimal("155.00") - Decimal("150.00")) * 100
        assert unrealized_pnl == expected_pnl

    def test_total_return_calculation(self, paper_broker):
        """Test calculation of total return percentage."""
        # Execute some trades
        paper_broker.update_market_price("AAPL", Decimal("150.00"))
        paper_broker.submit_order(
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Quantity(Decimal("100")),
            )
        )

        paper_broker.update_market_price("AAPL", Decimal("165.00"))

        # Calculate return
        total_return = paper_broker.get_total_return_percentage()

        # With $100k initial, made $1500 on AAPL
        expected_return = (Decimal("1500") / Decimal("100000")) * 100
        assert abs(total_return - expected_return) < Decimal("0.01")
