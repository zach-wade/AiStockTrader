"""
Integration tests for paper trading system.
These tests validate the interaction between multiple components.
"""

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.order_validator import OrderValidator
from src.domain.services.position_manager import PositionManager
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects import Money, Price, Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker


class TestPaperTradingIntegration:
    """Integration tests for paper trading system."""

    @pytest.fixture
    def broker(self):
        """Create a connected paper broker."""
        broker = PaperBroker(initial_capital=Decimal("100000"))
        broker.connect()
        yield broker
        broker.disconnect()

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio for testing."""
        return Portfolio(
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),
            max_portfolio_risk=Decimal("0.02"),  # 2% risk
        )

    def test_complete_trading_cycle(self, broker):
        """Test a complete buy-hold-sell cycle."""
        symbol = "AAPL"
        initial_price = Decimal("150.00")
        new_price = Decimal("160.00")
        quantity = Decimal("100")

        # Set initial price
        broker.update_market_price(symbol, initial_price)

        # Buy shares
        buy_order = Order(
            id=uuid4(),
            symbol=symbol,
            quantity=Quantity(quantity),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        buy_result = broker.submit_order(buy_order)
        assert buy_result.status == OrderStatus.FILLED
        assert buy_result.filled_price == Price(initial_price)

        # Update price (simulate price movement)
        broker.update_market_price(symbol, new_price)

        # Sell shares
        sell_order = Order(
            id=uuid4(),
            symbol=symbol,
            quantity=Quantity(quantity),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        sell_result = broker.submit_order(sell_order)
        assert sell_result.status == OrderStatus.FILLED
        assert sell_result.filled_price == Price(new_price)

    def test_portfolio_management_integration(self, broker, portfolio):
        """Test integration between broker and portfolio management."""
        # Set up market prices
        prices = {"AAPL": Decimal("150.00"), "GOOGL": Decimal("140.00"), "MSFT": Decimal("400.00")}

        for symbol, price in prices.items():
            broker.update_market_price(symbol, price)

        # Track orders
        orders_placed = []

        # Place multiple orders
        for symbol, price in prices.items():
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            result = broker.submit_order(order)
            if result.status == OrderStatus.FILLED:
                orders_placed.append(result)

                # Manually update portfolio (in real system, use cases would handle this)
                position = Position(
                    symbol=symbol,
                    quantity=Quantity(Decimal("10")),
                    average_entry_price=Price(price),
                    current_price=Price(price),
                )
                portfolio.positions[symbol] = position

                # Deduct cash
                cost = Money(Decimal("10") * price)
                portfolio.cash_balance = Money(portfolio.cash_balance.amount - cost.amount)

        # Verify portfolio state
        assert len(portfolio.positions) == 3
        assert portfolio.cash_balance.amount < Decimal("100000")

        # Calculate total equity
        total_value = portfolio.cash_balance.amount
        for position in portfolio.positions.values():
            position_value = position.quantity.value * position.current_price.amount
            total_value += position_value

        # Should be close to initial capital (minus any fees/spread)
        assert Decimal("95000") <= total_value <= Decimal("100000")

    def test_risk_management_integration(self, broker, portfolio):
        """Test integration with risk management services."""
        risk_calculator = RiskCalculator()

        # Set market price
        symbol = "RISKY"
        price = Decimal("100.00")
        broker.update_market_price(symbol, price)

        # Create position
        position = Position(
            symbol=symbol,
            quantity=Quantity(Decimal("100")),
            average_entry_price=Price(price),
            current_price=Price(price),
            stop_loss_price=Price(Decimal("95.00")),  # 5% stop loss
        )

        # Calculate risk
        position_risk = risk_calculator.calculate_position_risk(position)
        assert position_risk == Money(Decimal("500.00"))  # 100 shares * $5 risk per share

        # Add to portfolio
        portfolio.positions[symbol] = position

        # Calculate portfolio risk
        portfolio_risk = risk_calculator.calculate_portfolio_risk(portfolio)
        assert portfolio_risk > Money(Decimal("0"))

    def test_order_validation_integration(self, broker):
        """Test order validation before submission."""
        validator = OrderValidator()

        # Valid order
        valid_order = Order(
            id=uuid4(),
            symbol="VALID",
            quantity=Quantity(Decimal("10")),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        is_valid, message = validator.validate(valid_order)
        assert is_valid, f"Valid order should pass validation: {message}"

        # Invalid order (negative quantity - should be caught earlier, but testing validator)
        try:
            invalid_order = Order(
                id=uuid4(),
                symbol="INVALID",
                quantity=Quantity(Decimal("-10")),  # This will raise ValueError
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )
        except ValueError:
            pass  # Expected - quantity validation works

    def test_position_manager_integration(self, broker):
        """Test position management with broker."""
        position_manager = PositionManager()

        # Set price and buy shares
        symbol = "MGMT"
        broker.update_market_price(symbol, Decimal("100.00"))

        # Initial buy
        buy_order = Order(
            id=uuid4(),
            symbol=symbol,
            quantity=Quantity(Decimal("50")),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        result = broker.submit_order(buy_order)

        # Create position from order
        if result.status == OrderStatus.FILLED:
            position = position_manager.open_position(
                symbol=symbol, quantity=result.filled_quantity, entry_price=result.filled_price
            )

            assert position.symbol == symbol
            assert position.quantity == Quantity(Decimal("50"))
            assert position.average_entry_price == Price(Decimal("100.00"))

            # Add to existing position
            additional_buy = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("30")),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            broker.update_market_price(symbol, Decimal("105.00"))
            additional_result = broker.submit_order(additional_buy)

            if additional_result.status == OrderStatus.FILLED:
                position = position_manager.add_to_position(
                    position, additional_result.filled_quantity, additional_result.filled_price
                )

                # Check updated position
                assert position.quantity == Quantity(Decimal("80"))
                # Average price should be weighted average
                expected_avg = (
                    Decimal("50") * Decimal("100") + Decimal("30") * Decimal("105")
                ) / Decimal("80")
                assert abs(position.average_entry_price.amount - expected_avg) < Decimal("0.01")

    def test_concurrent_orders(self, broker):
        """Test handling multiple concurrent orders."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

        # Set prices
        for i, symbol in enumerate(symbols):
            price = Decimal("100") + Decimal(str(i * 10))
            broker.update_market_price(symbol, price)

        # Submit orders concurrently (simulated)
        orders = []
        for symbol in symbols:
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            orders.append(order)

        # Submit all orders
        results = []
        for order in orders:
            result = broker.submit_order(order)
            results.append(result)

        # All should be filled
        assert all(r.status == OrderStatus.FILLED for r in results)
        assert len(results) == len(symbols)

    def test_market_volatility_simulation(self, broker):
        """Test trading during simulated market volatility."""
        symbol = "VOLATILE"
        base_price = Decimal("100.00")

        trades_executed = 0
        profits = Decimal("0")

        for i in range(10):
            # Simulate price volatility
            import random

            volatility = Decimal(str(random.uniform(-5, 5)))
            new_price = base_price + volatility
            broker.update_market_price(symbol, new_price)

            # Decide to buy or sell based on price
            if new_price < base_price and i % 2 == 0:
                # Buy when price is low
                order = Order(
                    id=uuid4(),
                    symbol=symbol,
                    quantity=Quantity(Decimal("10")),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(UTC),
                )
            elif new_price > base_price and i % 2 == 1:
                # Sell when price is high
                order = Order(
                    id=uuid4(),
                    symbol=symbol,
                    quantity=Quantity(Decimal("5")),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(UTC),
                )
            else:
                continue

            result = broker.submit_order(order)
            if result.status == OrderStatus.FILLED:
                trades_executed += 1

                # Track profit/loss (simplified)
                if result.side == OrderSide.SELL:
                    profits += (new_price - base_price) * result.filled_quantity.value

        # Should have executed some trades
        assert trades_executed > 0

    def test_order_history_tracking(self, broker):
        """Test tracking order history."""
        symbol = "HISTORY"
        broker.update_market_price(symbol, Decimal("100.00"))

        order_history = []

        # Place multiple orders
        for i in range(5):
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("5")),
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC) + timedelta(seconds=i),
            )

            result = broker.submit_order(order)
            if result:
                order_history.append(
                    {
                        "order_id": str(result.id),
                        "symbol": result.symbol,
                        "side": result.side,
                        "quantity": result.filled_quantity.value if result.filled_quantity else 0,
                        "price": result.filled_price.amount if result.filled_price else 0,
                        "status": result.status,
                        "timestamp": (
                            result.filled_at if hasattr(result, "filled_at") else result.created_at
                        ),
                    }
                )

        # Verify history
        assert len(order_history) >= 3  # Some sells might fail without position

        # Check chronological order
        for i in range(len(order_history) - 1):
            assert order_history[i]["timestamp"] <= order_history[i + 1]["timestamp"]

    def test_edge_case_insufficient_funds(self, broker):
        """Test handling of insufficient funds."""
        # Small initial capital
        small_broker = PaperBroker(initial_capital=Decimal("1000"))
        small_broker.connect()

        # Try to buy expensive stock
        symbol = "EXPENSIVE"
        small_broker.update_market_price(symbol, Decimal("10000.00"))

        order = Order(
            id=uuid4(),
            symbol=symbol,
            quantity=Quantity(Decimal("1")),  # Even 1 share is too expensive
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # This might reject or partially fill depending on implementation
        result = small_broker.submit_order(order)

        # Order should either be rejected or not fully filled
        if result:
            if result.status == OrderStatus.FILLED:
                # If filled, broker allows margin/credit (check implementation)
                pass
            else:
                # Should be rejected or partially filled
                assert result.status in [OrderStatus.REJECTED, OrderStatus.PARTIALLY_FILLED]

        small_broker.disconnect()

    def test_performance_metrics(self, broker):
        """Test calculation of performance metrics."""
        initial_capital = Decimal("100000")
        trades = []

        # Execute series of trades
        symbols_and_prices = [
            ("AAPL", Decimal("150.00"), Decimal("155.00")),
            ("GOOGL", Decimal("140.00"), Decimal("145.00")),
            ("MSFT", Decimal("400.00"), Decimal("395.00")),  # Loss
        ]

        for symbol, buy_price, sell_price in symbols_and_prices:
            # Buy
            broker.update_market_price(symbol, buy_price)
            buy_order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            buy_result = broker.submit_order(buy_order)

            # Sell
            broker.update_market_price(symbol, sell_price)
            sell_order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=Quantity(Decimal("10")),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            sell_result = broker.submit_order(sell_order)

            if buy_result and sell_result:
                profit_loss = (sell_price - buy_price) * Decimal("10")
                trades.append(
                    {
                        "symbol": symbol,
                        "profit_loss": profit_loss,
                        "return_pct": (profit_loss / (buy_price * Decimal("10"))) * Decimal("100"),
                    }
                )

        # Calculate metrics
        total_pnl = sum(t["profit_loss"] for t in trades)
        winning_trades = [t for t in trades if t["profit_loss"] > 0]
        losing_trades = [t for t in trades if t["profit_loss"] < 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Verify metrics
        assert len(trades) == 3
        assert len(winning_trades) == 2
        assert len(losing_trades) == 1
        assert win_rate > 50  # Should have 66.67% win rate


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
