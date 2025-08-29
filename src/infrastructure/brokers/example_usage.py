"""
Example usage of the broker implementations
"""

# Standard library imports
import logging
from decimal import Decimal
from typing import Literal

# Local imports
from src.domain.entities.order import Order, OrderRequest, OrderSide
from src.domain.value_objects import Price, Quantity
from src.infrastructure.brokers import BrokerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_paper_trading() -> None:
    """Example of using the paper trading broker"""
    print("\n=== Paper Trading Example ===\n")

    # Create paper broker
    factory = BrokerFactory()
    broker = factory.create_broker(
        broker_type="paper",
        initial_capital=Decimal("10000"),
        slippage_pct=Decimal("0.001"),
    )

    # Set some market prices (in production, these would come from market data feed)
    # Type assertion for paper broker methods
    if hasattr(broker, "set_market_price"):
        broker.set_market_price("AAPL", Decimal("150.00"))
        broker.set_market_price("GOOGL", Decimal("2800.00"))

    # Get account info
    account = broker.get_account_info()
    print(f"Account Balance: ${account.cash:.2f}")
    print(f"Buying Power: ${account.buying_power:.2f}")

    # Check market hours
    market_hours = broker.get_market_hours()
    print(f"\nMarket Status: {market_hours}")

    # Create and submit a market order
    request1 = OrderRequest(
        symbol="AAPL", quantity=Quantity("10"), side=OrderSide.BUY, reason="Testing paper trading"
    )
    order1 = Order.create_market_order(request1)

    print(f"\nSubmitting order: {order1}")
    order1 = broker.submit_order(order1)
    print(f"Order submitted: {order1}")

    # Check order status
    status = broker.get_order_status(order1.id)
    print(f"Order status: {status}")

    # Create and submit a limit order
    request2 = OrderRequest(
        symbol="GOOGL",
        quantity=Quantity("1"),
        side=OrderSide.BUY,
        limit_price=Price("2750.00"),
        reason="Limit order test",
    )
    order2 = Order.create_limit_order(request2)

    print(f"\nSubmitting limit order: {order2}")
    order2 = broker.submit_order(order2)
    print(f"Limit order submitted: {order2}")

    # Get all positions
    positions = broker.get_positions()
    print(f"\nPositions: {positions}")

    # Get recent orders
    recent_orders = broker.get_recent_orders(limit=5)
    print(f"\nRecent orders: {recent_orders}")

    # Cancel the limit order
    print(f"\nCancelling order {order2.id}...")
    cancelled = broker.cancel_order(order2.id)
    print(f"Order cancelled: {cancelled}")

    # Update account info
    account = broker.get_account_info()
    print(f"\nFinal Balance: ${account.cash:.2f}")
    print(f"Positions Value: ${account.positions_value:.2f}")


def example_simulated_trading() -> None:
    """Example of a simple trading simulation"""
    print("\n=== Simulated Trading Example ===\n")

    # Create paper broker for simulation
    factory = BrokerFactory()
    broker = factory.create_broker(
        broker_type="paper",
        initial_capital=Decimal("25000"),
        slippage_pct=Decimal("0.0005"),
        fill_delay_seconds=0,  # Instant fills for simulation
    )

    # Simulate price changes and close a position
    print("\nSimulating price increase...")
    if hasattr(broker, "set_market_price"):
        broker.set_market_price("AAPL", Decimal("155.00"))

    # Sell to close position
    request3 = OrderRequest(
        symbol="AAPL", quantity=Quantity("10"), side=OrderSide.SELL, reason="Taking profits"
    )
    order3 = Order.create_market_order(request3)

    print(f"\nClosing position: {order3}")
    order3 = broker.submit_order(order3)
    print(f"Position closed: {order3}")

    # Check P&L
    account = broker.get_account_info()
    print(f"\nFinal Account Balance: ${account.cash:.2f}")


def example_alpaca_integration() -> None:
    """Example of using the Alpaca broker integration"""
    print("\n=== Alpaca Integration Example ===\n")

    try:
        # Create Alpaca broker (paper trading by default)
        factory = BrokerFactory()
        broker = factory.create_broker(
            broker_type="alpaca",
            paper=True,  # Use paper trading
            auto_connect=True,
        )

        # Get account info
        account = broker.get_account_info()
        print(f"Account ID: {account.account_id}")
        print(f"Equity: ${account.equity:.2f}")
        print(f"Cash: ${account.cash:.2f}")
        print(f"Buying Power: ${account.buying_power:.2f}")

        # Check if market is open
        is_open = broker.is_market_open()
        print(f"\nMarket Open: {is_open}")

        # Get market hours
        hours = broker.get_market_hours()
        print(f"Market Hours: {hours}")

        # Get current positions
        positions = broker.get_positions()
        print(f"\nPositions: {len(positions)}")
        for pos in positions[:3]:  # Show first 3
            print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.average_entry_price:.2f}")

        # Get recent orders
        orders = broker.get_recent_orders(limit=5)
        print(f"\nRecent Orders: {len(orders)}")
        for order in orders:
            print(f"  {order.symbol}: {order.side} {order.quantity} shares")

        # NOTE: To submit real orders, uncomment below (be careful with real money!)
        # order = Order.create_market_order(
        #     symbol="AAPL",
        #     quantity=Decimal("1"),
        #     side=OrderSide.BUY
        # )
        # submitted_order = broker.submit_order(order)
        # print(f"Order submitted: {submitted_order}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set")


def example_backtest_mode() -> None:
    """Example of using the backtest broker configuration"""
    print("\n=== Backtest Mode Example ===\n")

    # Create backtest broker (optimized for backtesting)
    factory = BrokerFactory()
    broker = factory.create_broker(
        broker_type="backtest",
        initial_capital=Decimal("50000"),
    )

    # In backtest mode:
    # - No fill delays
    # - Minimal slippage
    # - No partial fills

    print("Backtest broker created with optimized settings")

    # Set historical prices
    if hasattr(broker, "set_market_price"):
        broker.set_market_price("SPY", Decimal("400.00"))

    # Execute a strategy
    request = OrderRequest(
        symbol="SPY", quantity=Quantity("100"), side=OrderSide.BUY, reason="momentum strategy"
    )
    order = Order.create_market_order(request)
    order.tags["strategy"] = "momentum"

    order = broker.submit_order(order)
    print(f"Backtest order: {order}")

    # Check execution
    print(f"Order status: {order.status}")
    print(f"Fill price: ${order.average_fill_price:.2f}")


def example_broker_factory_configs() -> None:
    """Example of using broker factory configurations"""
    print("\n=== Broker Factory Configurations ===\n")

    # Get default configurations
    factory = BrokerFactory()
    for broker_type_str in ["paper", "alpaca", "backtest"]:
        broker_type: Literal["paper", "alpaca", "backtest"] = broker_type_str  # type: ignore
        config = factory.get_default_config(broker_type)
        print(f"\n{broker_type.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Create from configuration dictionary
    config_dict = {
        "type": "paper",
        "initial_capital": "100000",
        "slippage_pct": "0.002",
        "commission_per_share": "0.005",
    }

    factory = BrokerFactory()
    broker = factory.create_from_config(config_dict)
    print(f"\nBroker created from config: {type(broker).__name__}")


def main() -> None:
    """Run all examples"""
    # Choose which examples to run
    example_paper_trading()
    example_simulated_trading()
    example_backtest_mode()
    example_broker_factory_configs()

    # Alpaca example requires API keys
    # example_alpaca_integration()


if __name__ == "__main__":
    main()
