#!/usr/bin/env python3
"""
Minimal Paper Trading System - Just Make It Work
No complexity, no perfection, just trading.
"""

import asyncio
import json
import random
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Add the src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

# These already work - verified!
from uuid import uuid4

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.value_objects import Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker


class SimpleMomentumTrader:
    """Dead simple momentum trader - buy if price goes up, sell if it goes down."""

    def __init__(self, broker: PaperBroker, symbols: list[str]):
        self.broker = broker
        self.symbols = symbols
        self.positions: dict[str, dict] = {}
        self.last_prices: dict[str, Decimal] = {}
        self.trades = []
        self.iteration = 0

    async def run(self, iterations: int = 100):
        """Main trading loop - keep it simple."""
        print(f"\n🚀 Starting paper trading with {self.symbols}")
        print("Initial capital: $10,000")
        print("-" * 50)

        for i in range(iterations):
            self.iteration = i

            for symbol in self.symbols:
                # Get price (mock for now, real data later)
                price = self.get_mock_price(symbol)

                # Update broker's internal price
                self.broker.update_market_price(symbol, price)

                # Make trading decision
                signal = self.get_signal(symbol, price)

                # Execute trade if we have a signal
                if signal:
                    self.execute_trade(symbol, signal, price)

            # Show progress every 10 iterations
            if i % 10 == 0:
                self.print_status()

            # Small delay to simulate real trading
            await asyncio.sleep(0.5)

        self.print_final_report()

    def get_mock_price(self, symbol: str) -> Decimal:
        """
        Simple random walk for prices.
        Replace this with real data later (Yahoo Finance or Alpaca).
        """
        # Initialize price if first time
        if symbol not in self.last_prices:
            # Different starting prices for each symbol
            base_prices = {
                "AAPL": Decimal("150.00"),
                "GOOGL": Decimal("140.00"),
                "MSFT": Decimal("380.00"),
                "TSLA": Decimal("250.00"),
                "SPY": Decimal("450.00"),
            }
            self.last_prices[symbol] = base_prices.get(symbol, Decimal("100.00"))

        # Random walk with slight upward bias (bull market simulation)
        change_percent = Decimal(random.uniform(-0.02, 0.025))  # -2% to +2.5%
        change = self.last_prices[symbol] * change_percent
        new_price = max(self.last_prices[symbol] + change, Decimal("1"))

        return new_price

    def get_signal(self, symbol: str, current_price: Decimal) -> str | None:
        """
        Ultra simple momentum strategy:
        - Buy if price went up and we don't have a position
        - Sell if we hit stop loss (-2%) or take profit (+3%)
        """
        # First price seen - no signal
        if symbol not in self.last_prices:
            return None

        last_price = self.last_prices[symbol]

        if symbol not in self.positions:
            # We don't have a position - look for entry
            if current_price > last_price:
                return "BUY"  # Momentum up
        else:
            # We have a position - check exit conditions
            position = self.positions[symbol]
            entry_price = position["entry_price"]

            # Stop loss at -2%
            if current_price < entry_price * Decimal("0.98"):
                return "SELL"

            # Take profit at +3%
            if current_price > entry_price * Decimal("1.03"):
                return "SELL"

        # Update last price for next iteration
        self.last_prices[symbol] = current_price
        return None

    def execute_trade(self, symbol: str, signal: str, price: Decimal):
        """Execute the trade through our paper broker."""
        quantity = Quantity(Decimal("10"))  # Fixed 10 shares for simplicity

        if signal == "BUY":
            # Create and submit buy order
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=quantity,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            result = self.broker.submit_order(order)

            # Track position
            self.positions[symbol] = {
                "entry_price": price,
                "quantity": quantity.value,
                "entry_time": datetime.now(UTC),
                "order_id": str(order.id),
            }

            # Log trade
            trade_record = {
                "iteration": self.iteration,
                "time": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "side": "BUY",
                "price": float(price),
                "quantity": float(quantity.value),
                "order_id": str(order.id),
            }
            self.trades.append(trade_record)

            print(f"📈 BUY {symbol}: {quantity.value} shares @ ${price:.2f}")

        elif signal == "SELL" and symbol in self.positions:
            # Create and submit sell order
            order = Order(
                id=uuid4(),
                symbol=symbol,
                quantity=quantity,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                created_at=datetime.now(UTC),
            )

            result = self.broker.submit_order(order)

            # Calculate P&L
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            pnl = (price - entry_price) * quantity.value
            pnl_percent = ((price - entry_price) / entry_price) * 100

            # Log trade with P&L
            trade_record = {
                "iteration": self.iteration,
                "time": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "side": "SELL",
                "price": float(price),
                "quantity": float(quantity.value),
                "pnl": float(pnl),
                "pnl_percent": float(pnl_percent),
                "order_id": str(order.id),
            }
            self.trades.append(trade_record)

            # Remove position
            del self.positions[symbol]

            # Use emoji to show profit/loss
            emoji = "💰" if pnl > 0 else "💸"
            print(
                f"{emoji} SELL {symbol}: {quantity.value} shares @ ${price:.2f} (P&L: ${pnl:.2f} / {pnl_percent:.1f}%)"
            )

    def print_status(self):
        """Show current trading status."""
        account = self.broker.get_account_info()

        print(f"\n--- Iteration {self.iteration} Status ---")
        print(f"💵 Cash: ${account.cash:.2f}")
        print(f"📊 Positions: {len(self.positions)}")
        print(f"📝 Total Trades: {len(self.trades)}")

        if self.positions:
            print("Current positions:")
            for symbol, pos in self.positions.items():
                current_price = self.last_prices[symbol]
                entry_price = pos["entry_price"]
                unrealized_pnl = (current_price - entry_price) * pos["quantity"]
                print(
                    f"  {symbol}: Entry ${entry_price:.2f} → Current ${current_price:.2f} (P&L: ${unrealized_pnl:.2f})"
                )

    def print_final_report(self):
        """Generate final performance report."""
        print("\n" + "=" * 50)
        print("📊 FINAL TRADING REPORT")
        print("=" * 50)

        account = self.broker.get_account_info()

        # Calculate metrics
        total_trades = len(self.trades)
        sells = [t for t in self.trades if t["side"] == "SELL"]
        winning_trades = sum(1 for t in sells if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in sells if t.get("pnl", 0) < 0)

        if winning_trades + losing_trades > 0:
            win_rate = winning_trades / (winning_trades + losing_trades) * 100
        else:
            win_rate = 0

        total_pnl = sum(t.get("pnl", 0) for t in sells)

        print("💰 Starting Capital: $10,000.00")
        print(f"💵 Final Cash: ${account.cash:.2f}")
        print(f"📈 Total Return: ${total_pnl:.2f} ({total_pnl/100:.1f}%)")
        print("\n📊 Trading Statistics:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Completed Trades: {len(sells)}")
        print(f"  Winning Trades: {winning_trades}")
        print(f"  Losing Trades: {losing_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")

        if sells:
            avg_win = sum(t["pnl"] for t in sells if t["pnl"] > 0) / max(winning_trades, 1)
            avg_loss = sum(t["pnl"] for t in sells if t["pnl"] < 0) / max(losing_trades, 1)
            print(f"  Avg Win: ${avg_win:.2f}")
            print(f"  Avg Loss: ${avg_loss:.2f}")

        # Save trades to JSON file
        output_file = "paper_trades.json"
        with open(output_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

        print(f"\n💾 Trades saved to {output_file}")
        print("=" * 50)


async def main():
    """Main entry point - keep it simple."""
    print("\n" + "=" * 50)
    print("🎯 MINIMAL PAPER TRADING SYSTEM")
    print("Simple momentum strategy - no complexity!")
    print("=" * 50)

    # Initialize broker with $10,000
    initial_capital = Decimal("10000.00")
    broker = PaperBroker(initial_capital=initial_capital)
    broker.connect()

    # Trade these popular symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]

    # Create and run trader
    trader = SimpleMomentumTrader(broker, symbols)

    try:
        # Run for 100 iterations (about 50 seconds with 0.5s delay)
        await trader.run(iterations=100)
    except KeyboardInterrupt:
        print("\n\n⚠️ Trading interrupted by user")
        trader.print_final_report()
    except Exception as e:
        print(f"\n❌ Error during trading: {e}")
        trader.print_final_report()
    finally:
        broker.disconnect()
        print("\n✅ Paper trading session complete!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
