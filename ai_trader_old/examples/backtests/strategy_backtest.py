"""
Strategy Backtest Example

This example demonstrates how to backtest trading strategies
using historical data and analyze performance metrics.
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Third-party imports
import structlog

# Local imports
from main.config.config_manager import get_config
from main.data_providers.alpaca.market_data import AlpacaMarketClient
from main.utils.resilience.recovery_manager import get_global_recovery_manager

logger = structlog.get_logger(__name__)


class StrategyBacktest:
    """
    Backtesting framework for trading strategies.

    Features:
    - Historical data simulation
    - Performance metrics calculation
    - Risk analysis
    - Trade tracking
    """

    def __init__(
        self,
        config,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        self.config = config
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # Initialize market data client
        recovery_manager = get_global_recovery_manager()
        self.market_client = AlpacaMarketClient(config, recovery_manager)

        # Track performance
        self.reset()

    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.dates = []

    async def run_backtest(
        self, strategy, symbols: list[str], start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Run backtest for given strategy and symbols."""
        logger.info(
            "Starting backtest",
            strategy=strategy.__class__.__name__,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        self.reset()

        # Get historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            bars = await self.market_client.get_bars(
                symbol=symbol, start=start_date, end=end_date, timeframe="1Day"
            )

            if bars:
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": bar.timestamp,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                        }
                        for bar in bars
                    ]
                )
                df.set_index("timestamp", inplace=True)
                historical_data[symbol] = df

        if not historical_data:
            return self._empty_results()

        # Get all unique dates
        all_dates = sorted(set(date for df in historical_data.values() for date in df.index))

        # Simulate trading for each date
        for date in all_dates:
            daily_prices = {}

            # Get prices for all symbols on this date
            for symbol, df in historical_data.items():
                if date in df.index:
                    daily_prices[symbol] = df.loc[date]

            # Process signals and execute trades
            await self._process_day(strategy, date, daily_prices, historical_data)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(daily_prices)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)

        # Calculate performance metrics
        metrics = self._calculate_metrics()

        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_values,
            "dates": self.dates,
            "final_capital": self.capital,
            "final_positions": self.positions,
        }

    async def _process_day(
        self,
        strategy,
        date: datetime,
        daily_prices: dict[str, pd.Series],
        historical_data: dict[str, pd.DataFrame],
    ):
        """Process a single trading day."""
        # Check existing positions for exit signals
        for symbol in list(self.positions.keys()):
            if symbol in daily_prices:
                # Get historical data up to current date
                hist_df = historical_data[symbol][historical_data[symbol].index <= date]

                # Analyze with strategy
                signal = await self._get_signal(strategy, symbol, hist_df)

                if signal == -1:  # Sell signal
                    self._execute_sell(symbol, daily_prices[symbol]["close"], date)

        # Check for new entry signals
        for symbol, price_data in daily_prices.items():
            if symbol not in self.positions:
                # Get historical data up to current date
                hist_df = historical_data[symbol][historical_data[symbol].index <= date]

                # Analyze with strategy
                signal = await self._get_signal(strategy, symbol, hist_df)

                if signal == 1:  # Buy signal
                    # Determine position size (simplified - equal weight)
                    position_size = min(
                        self.capital * 0.1,  # Max 10% per position
                        self.capital / max(1, len(daily_prices) - len(self.positions)),
                    )

                    if position_size > 100:  # Minimum position size
                        self._execute_buy(symbol, price_data["close"], position_size, date)

    async def _get_signal(self, strategy, symbol: str, historical_data: pd.DataFrame) -> int:
        """Get trading signal from strategy."""
        try:
            # Mock the market data client response
            strategy.market_client.get_bars = lambda **kwargs: self._mock_bars(historical_data)

            # Get analysis from strategy
            analysis = await strategy.analyze(symbol)
            return analysis.get("signal", 0)
        except Exception as e:
            logger.error(f"Error getting signal for {symbol}: {e}")
            return 0

    def _mock_bars(self, df: pd.DataFrame):
        """Convert DataFrame back to bar format for strategy."""

        class MockBar:
            def __init__(self, row):
                self.timestamp = row.name
                self.open = row["open"]
                self.high = row["high"]
                self.low = row["low"]
                self.close = row["close"]
                self.volume = row["volume"]

        return [MockBar(row) for _, row in df.iterrows()]

    def _execute_buy(self, symbol: str, price: float, size: float, date: datetime):
        """Execute buy order."""
        # Apply slippage
        execution_price = price * (1 + self.slippage)

        # Calculate shares
        shares = int(size / execution_price)
        if shares == 0:
            return

        # Calculate cost including commission
        cost = shares * execution_price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost

        if total_cost > self.capital:
            return

        # Update positions and capital
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": execution_price,
            "entry_date": date,
        }
        self.capital -= total_cost

        # Record trade
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "action": "BUY",
                "shares": shares,
                "price": execution_price,
                "commission": commission_cost,
                "value": cost,
            }
        )

    def _execute_sell(self, symbol: str, price: float, date: datetime):
        """Execute sell order."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Apply slippage
        execution_price = price * (1 - self.slippage)

        # Calculate proceeds
        proceeds = position["shares"] * execution_price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost

        # Calculate profit/loss
        entry_cost = position["shares"] * position["entry_price"]
        pnl = net_proceeds - entry_cost
        pnl_pct = pnl / entry_cost

        # Update capital and remove position
        self.capital += net_proceeds
        del self.positions[symbol]

        # Record trade
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "action": "SELL",
                "shares": position["shares"],
                "price": execution_price,
                "commission": commission_cost,
                "value": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "holding_days": (date - position["entry_date"]).days,
            }
        )

    def _calculate_portfolio_value(self, daily_prices: dict[str, pd.Series]) -> float:
        """Calculate total portfolio value."""
        value = self.capital

        for symbol, position in self.positions.items():
            if symbol in daily_prices:
                value += position["shares"] * daily_prices[symbol]["close"]

        return value

    def _calculate_metrics(self) -> dict[str, float]:
        """Calculate performance metrics."""
        if not self.portfolio_values:
            return self._empty_metrics()

        # Convert to numpy array for calculations
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        # Basic metrics
        total_return = (values[-1] - self.initial_capital) / self.initial_capital

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        avg_return = np.mean(returns) * 252 if len(returns) > 0 else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        winning_trades = len(trades_df[trades_df.get("pnl", 0) > 0])
        losing_trades = len(trades_df[trades_df.get("pnl", 0) < 0])
        total_trades = winning_trades + losing_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average win/loss
        wins = trades_df[trades_df.get("pnl", 0) > 0]["pnl"]
        losses = trades_df[trades_df.get("pnl", 0) < 0]["pnl"]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_value": values[-1] if len(values) > 0 else self.initial_capital,
        }

    def _empty_results(self) -> dict[str, Any]:
        """Return empty results."""
        return {
            "metrics": self._empty_metrics(),
            "trades": [],
            "portfolio_values": [],
            "dates": [],
            "final_capital": self.initial_capital,
            "final_positions": {},
        }

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty metrics."""
        return {
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "final_value": self.initial_capital,
        }


async def main():
    """Example backtest of mean reversion strategy."""

    # Load configuration
    config = get_config(config_name="prod", environment="dev")

    # Import mean reversion strategy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Third-party imports
    from strategies.mean_reversion import MeanReversionStrategy

    # Create strategy
    strategy = MeanReversionStrategy(
        config, lookback_period=20, entry_threshold=-2.0, exit_threshold=0.5
    )

    # Create backtest engine
    backtest = StrategyBacktest(config, initial_capital=100000, commission=0.001, slippage=0.0005)

    # Test parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months backtest

    print("=== Strategy Backtest Example ===\n")
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${backtest.initial_capital:,.2f}")

    print("\nRunning backtest...")

    # Run backtest
    results = await backtest.run_backtest(strategy, symbols, start_date, end_date)

    # Display results
    metrics = results["metrics"]

    print("\n=== Performance Metrics ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    print("\n=== Trade Statistics ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    print("\n=== Final Results ===")
    print(f"Initial Capital: ${backtest.initial_capital:,.2f}")
    print(f"Final Value: ${metrics['final_value']:,.2f}")
    print(f"Net Profit/Loss: ${metrics['final_value'] - backtest.initial_capital:,.2f}")

    # Show recent trades
    if results["trades"]:
        print("\n=== Recent Trades ===")
        recent_trades = results["trades"][-5:]
        for trade in recent_trades:
            action = trade["action"]
            symbol = trade["symbol"]
            shares = trade["shares"]
            price = trade["price"]
            date = trade["date"].date()

            print(f"{date}: {action} {shares} {symbol} @ ${price:.2f}", end="")
            if "pnl" in trade:
                print(f" (P&L: ${trade['pnl']:.2f} / {trade['pnl_pct']:.2%})")
            else:
                print()


if __name__ == "__main__":
    asyncio.run(main())
