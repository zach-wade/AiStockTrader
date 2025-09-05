#!/usr/bin/env python3
"""
Paper Trading with Real Market Data from Alpaca
Uses your Alpaca account for live/historical prices
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal

# Load .env file if it exists
from pathlib import Path

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print("Loading .env file...")
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our working components
from uuid import uuid4

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.value_objects import Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker

# Import Alpaca client
try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("Installing alpaca-py package...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-py"])
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame


class AlpacaMarketData:
    """Get real market data from Alpaca."""

    def __init__(self, api_key: str = None, secret_key: str = None):
        """Initialize Alpaca client."""
        # Use environment variables if not provided
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
            )

        # Create Alpaca data client
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        print("✅ Connected to Alpaca market data")

    def get_latest_price(self, symbol: str) -> Decimal:
        """Get the latest price for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.client.get_stock_latest_quote(request)

            if symbol in quote:
                # Use ask price if available, otherwise bid, otherwise last trade
                price = quote[symbol].ask_price or quote[symbol].bid_price or 0
                if price > 0:
                    return Decimal(str(price))
                else:
                    # If no quote, get last bar
                    return self.get_last_close(symbol)
            else:
                print(f"⚠️ No quote for {symbol}, using last close")
                return self.get_last_close(symbol)

        except Exception as e:
            print(f"⚠️ Error getting price for {symbol}: {e}")
            return Decimal("100.00")  # Fallback price

    def get_last_close(self, symbol: str) -> Decimal:
        """Get the last closing price."""
        try:
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
            bars = self.client.get_stock_bars(request)

            if symbol in bars:
                bar_list = bars[symbol]
                if bar_list:
                    return Decimal(str(bar_list[-1].close))

            return Decimal("100.00")  # Fallback

        except Exception as e:
            print(f"⚠️ Error getting close for {symbol}: {e}")
            return Decimal("100.00")


class AlpacaPaperTrader:
    """Paper trader using real Alpaca market data."""

    def __init__(self, broker: PaperBroker, market_data: AlpacaMarketData, symbols: list[str]):
        self.broker = broker
        self.market_data = market_data
        self.symbols = symbols
        self.positions: dict[str, dict] = {}
        self.last_prices: dict[str, Decimal] = {}
        self.trades = []
        self.iteration = 0

    def run(self, iterations: int = 100, delay_seconds: int = 5):
        """
        Run paper trading with real market data.

        Args:
            iterations: Number of iterations to run
            delay_seconds: Seconds between each iteration (be respectful of API limits)
        """
        print("\n🚀 Starting paper trading with REAL Alpaca data")
        print(f"📊 Trading: {self.symbols}")
        print("💰 Initial capital: $10,000")
        print(f"⏱️ Checking prices every {delay_seconds} seconds")
        print("-" * 50)

        for i in range(iterations):
            self.iteration = i

            # Check if market is open (simple check - improve later)
            now = datetime.now()
            if now.weekday() >= 5:  # Weekend
                print("🏖️ Market closed (weekend), using last known prices")

            for symbol in self.symbols:
                # Get REAL price from Alpaca
                current_price = self.market_data.get_latest_price(symbol)

                # Update broker's internal price
                self.broker.update_market_price(symbol, current_price)

                # Show price update
                if i % 5 == 0:  # Every 5 iterations
                    print(f"📊 {symbol}: ${current_price:.2f}")

                # Make trading decision
                signal = self.get_signal(symbol, current_price)

                # Execute trade if we have a signal
                if signal:
                    self.execute_trade(symbol, signal, current_price)

            # Show status every 10 iterations
            if i % 10 == 0:
                self.print_status()

            # Don't spam the API
            time.sleep(delay_seconds)

        self.print_final_report()

    def get_signal(self, symbol: str, current_price: Decimal) -> str | None:
        """
        Simple momentum strategy with real prices.
        Buy on 1% up move, sell on 2% loss or 3% gain.
        """
        if symbol not in self.last_prices:
            self.last_prices[symbol] = current_price
            return None

        last_price = self.last_prices[symbol]
        price_change = (current_price - last_price) / last_price * 100

        if symbol not in self.positions:
            # Look for entry - buy on momentum
            if price_change > 0.5:  # 0.5% up move
                return "BUY"
        else:
            # We have a position - check exit conditions
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            pnl_percent = ((current_price - entry_price) / entry_price) * 100

            # Stop loss at -2%
            if pnl_percent <= -2:
                print(f"🛑 Stop loss triggered for {symbol}")
                return "SELL"

            # Take profit at +3%
            if pnl_percent >= 3:
                print(f"🎯 Take profit triggered for {symbol}")
                return "SELL"

        # Update last price
        self.last_prices[symbol] = current_price
        return None

    def execute_trade(self, symbol: str, signal: str, price: Decimal):
        """Execute trade through paper broker."""
        # Calculate position size (risk 1% of capital per trade)
        account = self.broker.get_account_info()
        position_size = (account.equity * Decimal("0.01")) / price
        quantity = Quantity(Decimal(int(position_size)))  # Round to whole shares

        if quantity.value < 1:
            quantity = Quantity(Decimal("1"))  # Minimum 1 share

        if signal == "BUY":
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

            if result.status == OrderStatus.FILLED:
                self.positions[symbol] = {
                    "entry_price": price,
                    "quantity": quantity.value,
                    "entry_time": datetime.now(UTC),
                }

                self.trades.append(
                    {
                        "iteration": self.iteration,
                        "time": datetime.now(UTC).isoformat(),
                        "symbol": symbol,
                        "side": "BUY",
                        "price": float(price),
                        "quantity": float(quantity.value),
                        "value": float(price * quantity.value),
                    }
                )

                print(
                    f"📈 BUY {symbol}: {quantity.value} shares @ ${price:.2f} (${price * quantity.value:.2f})"
                )

        elif signal == "SELL" and symbol in self.positions:
            position = self.positions[symbol]
            quantity = Quantity(position["quantity"])

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

            if result.status == OrderStatus.FILLED:
                entry_price = position["entry_price"]
                pnl = (price - entry_price) * quantity.value
                pnl_percent = ((price - entry_price) / entry_price) * 100

                self.trades.append(
                    {
                        "iteration": self.iteration,
                        "time": datetime.now(UTC).isoformat(),
                        "symbol": symbol,
                        "side": "SELL",
                        "price": float(price),
                        "quantity": float(quantity.value),
                        "pnl": float(pnl),
                        "pnl_percent": float(pnl_percent),
                        "value": float(price * quantity.value),
                    }
                )

                del self.positions[symbol]

                emoji = "💰" if pnl > 0 else "💸"
                print(
                    f"{emoji} SELL {symbol}: {quantity.value} shares @ ${price:.2f} (P&L: ${pnl:.2f} / {pnl_percent:.1f}%)"
                )

    def print_status(self):
        """Show current status."""
        account = self.broker.get_account_info()

        print(f"\n--- Iteration {self.iteration} Status ---")
        print(f"💵 Cash: ${account.cash:.2f}")
        print(f"📈 Equity: ${account.equity:.2f}")
        print(f"📊 Positions: {len(self.positions)}")
        print(f"📝 Total Trades: {len(self.trades)}")

        if self.positions:
            print("Current positions:")
            for symbol, pos in self.positions.items():
                current_price = self.last_prices.get(symbol, pos["entry_price"])
                entry_price = pos["entry_price"]
                quantity = pos["quantity"]
                unrealized_pnl = (current_price - entry_price) * quantity
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                print(
                    f"  {symbol}: {quantity} shares @ ${entry_price:.2f} → ${current_price:.2f} (P&L: ${unrealized_pnl:.2f} / {pnl_percent:.1f}%)"
                )

    def print_final_report(self):
        """Generate final report."""
        print("\n" + "=" * 50)
        print("📊 FINAL TRADING REPORT (with REAL market data)")
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

        total_pnl = account.equity - Decimal("10000")
        total_pnl_percent = (total_pnl / Decimal("10000")) * 100

        print("💰 Starting Capital: $10,000.00")
        print(f"💵 Final Cash: ${account.cash:.2f}")
        print(f"📈 Final Equity: ${account.equity:.2f}")
        print(
            f"{'🎉' if total_pnl >= 0 else '😢'} Total P&L: ${total_pnl:.2f} ({total_pnl_percent:.2f}%)"
        )

        print("\n📊 Trading Statistics:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Completed Trades: {len(sells)}")
        print(f"  Winning Trades: {winning_trades}")
        print(f"  Losing Trades: {losing_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")

        if sells:
            total_profit = sum(t["pnl"] for t in sells if t["pnl"] > 0)
            total_loss = sum(abs(t["pnl"]) for t in sells if t["pnl"] < 0)
            avg_win = total_profit / max(winning_trades, 1)
            avg_loss = total_loss / max(losing_trades, 1)
            print(f"  Avg Win: ${avg_win:.2f}")
            print(f"  Avg Loss: ${avg_loss:.2f}")

            if avg_loss > 0:
                profit_factor = total_profit / total_loss
                print(f"  Profit Factor: {profit_factor:.2f}")

        # Save trades
        output_file = "alpaca_paper_trades.json"
        with open(output_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

        print(f"\n💾 Trades saved to {output_file}")
        print("=" * 50)


def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("🎯 PAPER TRADING WITH REAL ALPACA DATA")
    print("=" * 50)

    try:
        # Initialize Alpaca market data
        market_data = AlpacaMarketData()

        # Initialize paper broker
        broker = PaperBroker(initial_capital=Decimal("10000"))
        broker.connect()

        # Select symbols to trade (liquid stocks)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Create trader
        trader = AlpacaPaperTrader(broker, market_data, symbols)

        # Run for 20 iterations with 10 second delays (about 3 minutes)
        # Adjust iterations and delay as needed
        trader.run(iterations=20, delay_seconds=10)

    except KeyboardInterrupt:
        print("\n\n⚠️ Trading interrupted by user")
        if "trader" in locals():
            trader.print_final_report()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "broker" in locals():
            broker.disconnect()
        print("\n✅ Paper trading session complete!")


if __name__ == "__main__":
    main()
