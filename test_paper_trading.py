#!/usr/bin/env python3
"""Quick test of paper trading components."""

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing paper trading components...")

# Test 1: Can we import the broker?
try:
    from src.infrastructure.brokers.paper_broker import PaperBroker

    print("✅ PaperBroker imports successfully")
except Exception as e:
    print(f"❌ Failed to import PaperBroker: {e}")
    sys.exit(1)

# Test 2: Can we create and connect broker?
try:
    broker = PaperBroker(initial_capital=Decimal("10000"))
    broker.connect()
    print("✅ Broker created and connected")
except Exception as e:
    print(f"❌ Failed to create broker: {e}")
    sys.exit(1)

# Test 3: Can we import order components?
try:
    from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
    from src.domain.value_objects import Quantity

    print("✅ Order components import successfully")
except Exception as e:
    print(f"❌ Failed to import order components: {e}")
    sys.exit(1)

# Test 4: Can we create and submit an order?
try:
    order = Order(
        id=uuid4(),
        symbol="AAPL",
        quantity=Quantity(Decimal("10")),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        created_at=datetime.now(UTC),
    )

    # Update market price first
    broker.update_market_price("AAPL", Decimal("150.00"))

    # Submit order
    result = broker.submit_order(order)
    print(f"✅ Order submitted: {result.status}")

    # Check account
    account = broker.get_account_info()
    print(f"✅ Account info: Cash=${account.cash}, Equity=${account.equity}")

except Exception as e:
    print(f"❌ Failed to submit order: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Can we get positions?
try:
    positions = broker.get_positions()
    print(f"✅ Positions retrieved: {len(positions)} positions")

    if positions:
        for symbol, pos in positions.items():
            print(f"  - {symbol}: {pos.quantity.value} shares @ ${pos.current_price}")

except Exception as e:
    print(f"❌ Failed to get positions: {e}")
    sys.exit(1)

# Test 6: Sell the position
try:
    if positions:
        sell_order = Order(
            id=uuid4(),
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # Update price to simulate profit
        broker.update_market_price("AAPL", Decimal("155.00"))

        result = broker.submit_order(sell_order)
        print(f"✅ Sell order submitted: {result.status}")

        # Check final account
        account = broker.get_account_info()
        print(f"✅ Final account: Cash=${account.cash}, Equity=${account.equity}")

        # Calculate P&L
        pnl = account.cash - Decimal("10000")
        print(f"✅ P&L: ${pnl}")

except Exception as e:
    print(f"❌ Failed to sell position: {e}")
    sys.exit(1)

print("\n🎉 All paper trading components work correctly!")
print("Ready to run the full trading system.")

broker.disconnect()
