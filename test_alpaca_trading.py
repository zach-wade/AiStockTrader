#!/usr/bin/env python3
"""
Quick test of Alpaca paper trading - just 3 iterations
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                try:
                    key, value = line.strip().split("=", 1)
                    # Remove any comments after the value
                    value = value.split("#")[0].strip()
                    os.environ[key] = value
                except ValueError:
                    continue

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our components
from paper_trading_alpaca import AlpacaMarketData, AlpacaPaperTrader
from src.infrastructure.brokers.paper_broker import PaperBroker


def quick_test():
    """Quick test with just 3 iterations."""
    print("\n" + "=" * 50)
    print("🧪 QUICK TEST - Alpaca Paper Trading")
    print("=" * 50)

    try:
        # Test 1: Can we connect to Alpaca?
        print("\n1️⃣ Connecting to Alpaca...")
        market_data = AlpacaMarketData()
        print("✅ Connected to Alpaca")

        # Test 2: Can we get a real price?
        print("\n2️⃣ Getting real market price for AAPL...")
        price = market_data.get_latest_price("AAPL")
        print(f"✅ AAPL current price: ${price}")

        # Test 3: Can we run paper trading?
        print("\n3️⃣ Running paper trading for 3 iterations...")

        broker = PaperBroker(initial_capital=Decimal("10000"))
        broker.connect()

        # Just trade 2 symbols for quick test
        symbols = ["AAPL", "MSFT"]

        trader = AlpacaPaperTrader(broker, market_data, symbols)

        # Run for just 3 iterations with 2 second delays
        trader.run(iterations=3, delay_seconds=2)

        print("\n✅ All tests passed! Ready for longer trading sessions.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
