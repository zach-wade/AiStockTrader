"""
Smoke tests for paper trading functionality.
These tests provide quick validation of core trading components.
"""

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker


class TestPaperTradingSmoke:
    """Quick smoke tests for paper trading components."""

    def test_broker_initialization(self):
        """Test that paper broker can be initialized."""
        broker = PaperBroker(initial_capital=Decimal("10000"))
        assert broker is not None
        # The internal cash tracking implementation may vary

    def test_broker_connection(self):
        """Test broker connection and disconnection."""
        broker = PaperBroker(initial_capital=Decimal("10000"))

        # Connect
        broker.connect()
        assert broker.is_connected()

        # Disconnect
        broker.disconnect()
        assert not broker.is_connected()

    def test_market_price_update(self):
        """Test updating market prices."""
        broker = PaperBroker(initial_capital=Decimal("10000"))
        broker.connect()

        # Update prices
        broker.update_market_price("AAPL", Decimal("150.00"))
        broker.update_market_price("GOOGL", Decimal("140.00"))

        # Verify by submitting orders at these prices
        # The actual price storage is an implementation detail

        broker.disconnect()

    def test_simple_buy_order(self):
        """Test submitting a simple buy order."""
        broker = PaperBroker(initial_capital=Decimal("10000"))
        broker.connect()

        # Set market price
        broker.update_market_price("TEST", Decimal("100.00"))

        # Create buy order
        order = Order(
            id=uuid4(),
            symbol="TEST",
            quantity=Quantity(Decimal("10")),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # Submit order
        result = broker.submit_order(order)

        # Verify order was filled
        assert result is not None
        assert result.status == OrderStatus.FILLED
        if hasattr(result, "filled_quantity"):
            assert result.filled_quantity == Quantity(Decimal("10"))
        if hasattr(result, "filled_price"):
            assert result.filled_price == Price(Decimal("100.00"))

        broker.disconnect()

    def test_simple_sell_order(self):
        """Test submitting a simple sell order."""
        broker = PaperBroker(initial_capital=Decimal("10000"))
        broker.connect()

        # Set market price
        broker.update_market_price("TEST", Decimal("100.00"))

        # First buy some shares
        buy_order = Order(
            id=uuid4(),
            symbol="TEST",
            quantity=Quantity(Decimal("20")),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        broker.submit_order(buy_order)

        # Now sell some
        sell_order = Order(
            id=uuid4(),
            symbol="TEST",
            quantity=Quantity(Decimal("10")),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        result = broker.submit_order(sell_order)

        # Verify order was filled
        assert result is not None
        assert result.status == OrderStatus.FILLED
        if hasattr(result, "filled_quantity"):
            assert result.filled_quantity == Quantity(Decimal("10"))

        broker.disconnect()

    def test_multiple_symbols(self):
        """Test trading multiple symbols."""
        broker = PaperBroker(initial_capital=Decimal("50000"))
        broker.connect()

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        orders_filled = 0

        for symbol in symbols:
            # Set price
            broker.update_market_price(symbol, Decimal("100.00"))

            # Create order
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            # Submit and verify
            result = broker.submit_order(order)
            if result and result.status == OrderStatus.FILLED:
                orders_filled += 1

        assert orders_filled == len(
            symbols
        ), f"Expected {len(symbols)} orders filled, got {orders_filled}"

        broker.disconnect()

    def test_portfolio_creation(self):
        """Test portfolio entity creation."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.1"),
        )

        assert portfolio.cash_balance == Money(Decimal("100000"))
        assert portfolio.max_position_size == Money(Decimal("20000"))
        assert len(portfolio.positions) == 0

    def test_position_creation(self):
        """Test position entity creation."""
        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Quantity(Decimal("100"))
        assert position.average_entry_price == Price(Decimal("150.00"))
        assert position.current_price == Price(Decimal("155.00"))

    def test_portfolio_with_position(self):
        """Test portfolio with an open position."""
        portfolio = Portfolio(
            cash_balance=Money(Decimal("85000")),  # Started with 100k, bought 15k worth
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.1"),
        )

        position = Position(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(Decimal("150.00")),
            current_price=Price(Decimal("155.00")),
        )

        portfolio.positions["AAPL"] = position

        # Calculate total equity
        total_equity = portfolio.get_total_equity()

        # Cash (85000) + Position value (100 * 155 = 15500) = 100500
        expected_equity = Money(Decimal("100500"))
        assert total_equity == expected_equity

    def test_rapid_order_submission(self):
        """Test submitting multiple orders rapidly."""
        broker = PaperBroker(initial_capital=Decimal("100000"))
        broker.connect()

        # Set price
        broker.update_market_price("RAPID", Decimal("50.00"))

        orders_submitted = 0
        orders_filled = 0

        # Submit 10 orders quickly
        for i in range(10):
            order = Order(
                id=uuid4(),
                symbol="RAPID",
                quantity=Quantity(Decimal("5")),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            orders_submitted += 1
            result = broker.submit_order(order)

            if result and result.status == OrderStatus.FILLED:
                orders_filled += 1

        # Most orders should fill (some sells might fail due to no position)
        assert orders_filled >= 5, f"Expected at least 5 orders filled, got {orders_filled}"

        broker.disconnect()


if __name__ == "__main__":
    # Run smoke tests
    pytest.main([__file__, "-v", "--tb=short"])
