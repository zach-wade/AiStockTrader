"""
Example usage of the broker implementations
"""

# Standard library imports
from decimal import Decimal
import logging

# Local imports
from src.domain.entities.order import Order, OrderSide
from src.infrastructure.brokers import BrokerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_paper_trading():
    """Example of using the paper trading broker"""
    print("\n=== Paper Trading Example ===\n")

    # Create paper broker
    broker = BrokerFactory.create_broker(
        broker_type="paper",
        initial_capital=Decimal("10000"),
        slippage_pct=Decimal("0.001"),
    )

    # Set some market prices (in production, these would come from market data feed)
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
    order1 = Order.create_market_order(
        symbol="AAPL", quantity=Decimal("10"), side=OrderSide.BUY, reason="Testing paper trading"
    )

    print(f"\nSubmitting order: {order1}")
    order1 = broker.submit_order(order1)
    print(f"Order submitted: {order1}")

    # Check order status
    status = broker.get_order_status(order1.id)
    print(f"Order status: {status}")

    # Create and submit a limit order
    order2 = Order.create_limit_order(
        symbol="GOOGL",
        quantity=Decimal("1"),
        side=OrderSide.BUY,
        limit_price=Decimal("2750.00"),
        reason="Limit order test",
    )

    print(f"\nSubmitting limit order: {order2}")
    order2 = broker.submit_order(order2)

    # Simulate price movement that triggers the limit order
    print("\nSimulating price drop to trigger limit order...")
    broker.set_market_price("GOOGL", Decimal("2745.00"))

    # Check positions
    positions = broker.get_positions()
    print(f"\nPositions ({len(positions)}):")
    for pos in positions:
        print(f"  {pos}")

    # Get updated account info
    account = broker.get_account_info()
    print("\nUpdated Account:")
    print(f"  Cash: ${account.cash:.2f}")
    print(f"  Equity: ${account.equity:.2f}")
    print(f"  Positions Value: ${account.positions_value:.2f}")
    print(f"  Unrealized P&L: ${account.unrealized_pnl:.2f}")

    # Simulate price changes and close a position
    print("\nSimulating price increase...")
    broker.set_market_price("AAPL", Decimal("155.00"))

    # Sell to close position
    order3 = Order.create_market_order(
        symbol="AAPL", quantity=Decimal("10"), side=OrderSide.SELL, reason="Taking profits"
    )

    print(f"\nClosing position: {order3}")
    order3 = broker.submit_order(order3)

    # Final account summary
    account = broker.get_account_info()
    positions = broker.get_positions()

    print("\n=== Final Summary ===")
    print(f"Cash: ${account.cash:.2f}")
    print(f"Equity: ${account.equity:.2f}")
    print(f"Realized P&L: ${account.realized_pnl:.2f}")
    print(f"Open Positions: {len(positions)}")

    # Get recent orders
    recent_orders = broker.get_recent_orders(limit=10)
    print(f"\nRecent Orders ({len(recent_orders)}):")
    for order in recent_orders:
        print(f"  {order}")


def example_alpaca_paper():
    """Example of using Alpaca paper trading (requires API credentials)"""
    print("\n=== Alpaca Paper Trading Example ===\n")

    try:
        # Create Alpaca broker (will use environment variables for credentials)
        broker = BrokerFactory.create_broker(
            broker_type="alpaca",
            paper=True,  # Use paper trading
        )

        # Get account info
        account = broker.get_account_info()
        print(f"Alpaca Account: {account.account_id}")
        print(f"  Type: {account.account_type}")
        print(f"  Cash: ${account.cash:.2f}")
        print(f"  Buying Power: ${account.buying_power:.2f}")

        # Check market status
        market_hours = broker.get_market_hours()
        print(f"\nMarket Status: {market_hours}")

        if broker.is_market_open():
            # Create a small test order
            order = Order.create_limit_order(
                symbol="AAPL",
                quantity=Decimal("1"),
                side=OrderSide.BUY,
                limit_price=Decimal("140.00"),  # Below market for safety
                reason="API test order",
            )

            print(f"\nSubmitting test order: {order}")
            order = broker.submit_order(order)
            print(f"Order submitted to Alpaca: {order.broker_order_id}")

            # Check status
            status = broker.get_order_status(order.id)
            print(f"Order status: {status}")

            # Cancel the order
            print("\nCancelling test order...")
            if broker.cancel_order(order.id):
                print("Order cancelled successfully")
            else:
                print("Failed to cancel order")
        else:
            print("\nMarket is closed - skipping order submission")

        # Get positions
        positions = broker.get_positions()
        print(f"\nCurrent Positions ({len(positions)}):")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.quantity} @ ${pos.average_entry_price:.2f}")

        # Get recent orders
        recent_orders = broker.get_recent_orders(limit=5)
        print(f"\nRecent Orders ({len(recent_orders)}):")
        for order in recent_orders:
            print(f"  {order.symbol} {order.side.value} {order.quantity} - {order.status.value}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Alpaca example requires API credentials.")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")


def example_broker_factory():
    """Example of using the broker factory with different configurations"""
    print("\n=== Broker Factory Example ===\n")

    # Create from environment variables
    print("Creating broker from environment...")
    broker = BrokerFactory.create_broker()  # Uses BROKER_TYPE env var
    print(f"Created: {type(broker).__name__}")

    # Create from config dictionary
    config = {
        "type": "paper",
        "initial_capital": "50000",
        "slippage_pct": "0.002",
        "auto_connect": True,
    }

    print("\nCreating broker from config...")
    broker = BrokerFactory.create_from_config(config)
    account = broker.get_account_info()
    print(f"Created paper broker with ${account.cash:.2f} capital")

    # Get default configurations
    print("\nDefault configurations:")
    for broker_type in ["paper", "alpaca", "backtest"]:
        config = BrokerFactory.get_default_config(broker_type)
        print(f"\n{broker_type.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run examples
    example_paper_trading()
    example_broker_factory()

    # Uncomment to test Alpaca (requires API credentials)
    # example_alpaca_paper()
