#!/usr/bin/env python3
"""
Quick test version of paper trading - runs for just 10 iterations
"""

import json
import random
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from uuid import uuid4

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.value_objects import Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker


def run_quick_paper_trading():
    """Synchronous version for quick testing."""

    print("\n" + "=" * 50)
    print("🎯 MINIMAL PAPER TRADING - QUICK TEST")
    print("=" * 50)

    # Initialize broker
    broker = PaperBroker(initial_capital=Decimal("10000"))
    broker.connect()

    symbols = ["AAPL", "GOOGL", "MSFT"]
    positions = {}
    last_prices = {"AAPL": Decimal("150.00"), "GOOGL": Decimal("140.00"), "MSFT": Decimal("380.00")}
    trades = []

    print(f"\n📊 Trading {symbols} with $10,000")
    print("-" * 50)

    # Run 10 quick iterations
    for i in range(10):
        print(f"\n--- Iteration {i+1} ---")

        for symbol in symbols:
            # Generate random price movement
            old_price = last_prices[symbol]
            change = Decimal(random.uniform(-2, 2))
            new_price = max(old_price + change, Decimal("1"))
            last_prices[symbol] = new_price

            # Update broker price
            broker.update_market_price(symbol, new_price)

            # Simple strategy: buy if up, sell if down significantly
            if symbol not in positions:
                # Consider buying
                if new_price > old_price and random.random() > 0.5:  # 50% chance on upward move
                    # Buy
                    quantity = Quantity(Decimal("10"))
                    order = Order(
                        id=uuid4(),
                        symbol=symbol,
                        quantity=quantity,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.PENDING,
                        created_at=datetime.now(UTC),
                    )

                    result = broker.submit_order(order)
                    if result.status == OrderStatus.FILLED:
                        positions[symbol] = {"entry_price": new_price, "quantity": quantity.value}
                        print(f"  📈 BUY {symbol}: 10 shares @ ${new_price:.2f}")
                        trades.append(
                            {
                                "iteration": i + 1,
                                "symbol": symbol,
                                "side": "BUY",
                                "price": float(new_price),
                            }
                        )
            else:
                # Consider selling
                entry_price = positions[symbol]["entry_price"]
                pnl_percent = ((new_price - entry_price) / entry_price) * 100

                # Sell if 2% loss or 3% gain
                if pnl_percent < -2 or pnl_percent > 3:
                    quantity = Quantity(positions[symbol]["quantity"])
                    order = Order(
                        id=uuid4(),
                        symbol=symbol,
                        quantity=quantity,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        status=OrderStatus.PENDING,
                        created_at=datetime.now(UTC),
                    )

                    result = broker.submit_order(order)
                    if result.status == OrderStatus.FILLED:
                        pnl = (new_price - entry_price) * quantity.value
                        emoji = "💰" if pnl > 0 else "💸"
                        print(
                            f"  {emoji} SELL {symbol}: 10 shares @ ${new_price:.2f} (P&L: ${pnl:.2f})"
                        )
                        trades.append(
                            {
                                "iteration": i + 1,
                                "symbol": symbol,
                                "side": "SELL",
                                "price": float(new_price),
                                "pnl": float(pnl),
                            }
                        )
                        del positions[symbol]

    # Final report
    print("\n" + "=" * 50)
    print("📊 FINAL REPORT")
    print("=" * 50)

    account = broker.get_account_info()
    broker_positions = broker.get_positions()

    print("💰 Starting Capital: $10,000.00")
    print(f"💵 Final Cash: ${account.cash:.2f}")
    print(f"📈 Final Equity: ${account.equity:.2f}")

    total_pnl = account.equity - Decimal("10000")
    emoji = "🎉" if total_pnl >= 0 else "😢"
    print(f"{emoji} Total P&L: ${total_pnl:.2f}")

    print("\n📊 Trading Statistics:")
    print(f"  Total Trades: {len(trades)}")

    sells = [t for t in trades if t["side"] == "SELL"]
    if sells:
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        losses = sum(1 for t in sells if t.get("pnl", 0) < 0)
        win_rate = wins / len(sells) * 100 if sells else 0
        print(f"  Completed Trades: {len(sells)}")
        print(f"  Win Rate: {win_rate:.1f}%")

    if broker_positions:
        print("\n📊 Open Positions:")
        for symbol, pos in broker_positions.items():
            print(f"  {symbol}: {pos.quantity.value} shares @ ${pos.current_price:.2f}")

    # Save trades
    with open("quick_test_trades.json", "w") as f:
        json.dump(trades, f, indent=2)

    print("\n💾 Trades saved to quick_test_trades.json")

    broker.disconnect()
    print("\n✅ Paper trading test complete!")

    return total_pnl


if __name__ == "__main__":
    try:
        pnl = run_quick_paper_trading()
        sys.exit(0 if pnl >= 0 else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
