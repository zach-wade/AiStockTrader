"""
Strategy Analytics Service

Handles all trading strategy performance calculations and analytics in the domain layer.
This service contains the business logic for calculating strategy metrics,
win rates, profit factors, and other strategy-specific KPIs.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class StrategyPerformanceMetrics:
    """Strategy performance metrics result."""

    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: float
    total_pnl: float
    avg_pnl_per_trade: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_trade_duration_seconds: float
    symbols_traded: int
    last_trade_time: float | None = None
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class StrategyTradeRecord:
    """Record of a strategy trade."""

    timestamp: float
    order_id: str
    symbol: str
    strategy: str
    pnl: float | None = None
    duration_seconds: float | None = None
    side: str | None = None  # buy/sell
    quantity: float | None = None
    price: float | None = None


@dataclass
class StrategyComparison:
    """Comparison metrics between strategies."""

    strategies: list[str]
    best_performer: str
    worst_performer: str
    most_active: str
    most_consistent: str
    highest_win_rate: str
    highest_profit_factor: str
    comparison_period: tuple[float, float]


class StrategyAnalyticsService:
    """
    Service for calculating trading strategy performance metrics and analytics.

    This service encapsulates all business logic related to strategy
    performance calculations, separating it from infrastructure concerns.
    """

    def __init__(self) -> None:
        """Initialize strategy analytics service."""
        self.min_trades_for_statistics = 10

    def calculate_strategy_performance(
        self, strategy_name: str, trades: list[StrategyTradeRecord]
    ) -> StrategyPerformanceMetrics:
        """
        Calculate comprehensive strategy performance metrics.

        Args:
            strategy_name: Name of the strategy
            trades: List of trade records for the strategy

        Returns:
            Calculated strategy performance metrics
        """
        if not trades:
            return self._create_empty_metrics(strategy_name)

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        neutral_trades = [t for t in trades if t.pnl and t.pnl == 0]

        # Calculate basic metrics
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # Calculate P&L metrics
        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Calculate average win and loss
        avg_win = (
            (sum(t.pnl for t in winning_trades if t.pnl is not None) / win_count)
            if win_count > 0
            else 0
        )
        avg_loss = (
            (sum(abs(t.pnl) for t in losing_trades if t.pnl is not None) / loss_count)
            if loss_count > 0
            else 0
        )

        # Calculate profit factor
        gross_profit = (
            sum(t.pnl for t in winning_trades if t.pnl is not None) if winning_trades else 0
        )
        gross_loss = (
            sum(abs(t.pnl) for t in losing_trades if t.pnl is not None) if losing_trades else 0
        )
        profit_factor = (
            (gross_profit / gross_loss)
            if gross_loss > 0
            else float("inf")
            if gross_profit > 0
            else 0
        )

        # Calculate consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(
            trades, lambda t: bool(t.pnl and t.pnl > 0)
        )
        max_consecutive_losses = self._calculate_max_consecutive(
            trades, lambda t: bool(t.pnl and t.pnl < 0)
        )

        # Calculate average trade duration
        durations = [t.duration_seconds for t in trades if t.duration_seconds is not None]
        avg_duration = statistics.mean(durations) if durations else 0

        # Get unique symbols traded
        symbols_traded = len(set(t.symbol for t in trades if t.symbol))

        # Get last trade time
        last_trade_time = max(t.timestamp for t in trades) if trades else None

        # Calculate expectancy
        expectancy = self._calculate_expectancy(win_rate / 100, avg_win, avg_loss)

        # Calculate max drawdown
        max_drawdown = self._calculate_strategy_drawdown(trades)

        # Calculate recovery factor
        recovery_factor = (total_pnl / max_drawdown) if max_drawdown > 0 else 0

        return StrategyPerformanceMetrics(
            strategy_name=strategy_name,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate_percent=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_trade_duration_seconds=avg_duration,
            symbols_traded=symbols_traded,
            last_trade_time=last_trade_time,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            recovery_factor=recovery_factor,
        )

    def _create_empty_metrics(self, strategy_name: str) -> StrategyPerformanceMetrics:
        """Create empty metrics for a strategy with no trades."""
        return StrategyPerformanceMetrics(
            strategy_name=strategy_name,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_percent=0,
            total_pnl=0,
            avg_pnl_per_trade=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            avg_trade_duration_seconds=0,
            symbols_traded=0,
            last_trade_time=None,
        )

    def _calculate_max_consecutive(
        self,
        trades: list[StrategyTradeRecord],
        condition_func: Callable[[StrategyTradeRecord], bool],
    ) -> int:
        """
        Calculate maximum consecutive occurrences based on condition.

        Args:
            trades: List of trades
            condition_func: Function to check if trade meets condition

        Returns:
            Maximum consecutive count
        """
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if condition_func(trade):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate trading expectancy.

        Expectancy = (Win Rate * Average Win) - (Loss Rate * Average Loss)

        Args:
            win_rate: Win rate as decimal (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)

        Returns:
            Expectancy value
        """
        if avg_loss == 0:
            return avg_win * win_rate

        loss_rate = 1 - win_rate
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def _calculate_strategy_drawdown(self, trades: list[StrategyTradeRecord]) -> float:
        """
        Calculate maximum drawdown for a strategy.

        Args:
            trades: List of trades sorted by timestamp

        Returns:
            Maximum drawdown amount
        """
        if not trades:
            return 0

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Calculate cumulative P&L
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for trade in sorted_trades:
            if trade.pnl is not None:
                cumulative_pnl = cumulative_pnl + trade.pnl

                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl

                drawdown = peak_pnl - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_risk_reward_ratio(self, trades: list[StrategyTradeRecord]) -> float:
        """
        Calculate risk-reward ratio.

        Args:
            trades: List of trade records

        Returns:
            Risk-reward ratio
        """
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]

        if not winning_trades or not losing_trades:
            return 0

        avg_win = sum(t.pnl for t in winning_trades if t.pnl is not None) / len(winning_trades)
        avg_loss = sum(abs(t.pnl) for t in losing_trades if t.pnl is not None) / len(losing_trades)

        return avg_win / avg_loss if avg_loss > 0 else 0

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing.

        Kelly % = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of win to loss

        Args:
            win_rate: Win rate as percentage
            avg_win: Average winning amount
            avg_loss: Average losing amount

        Returns:
            Kelly percentage (0-1)
        """
        if avg_loss <= 0:
            return 0

        p = win_rate / 100  # Convert to probability
        q = 1 - p
        b = avg_win / avg_loss

        if b <= 0:
            return 0

        kelly = (p * b - q) / b

        # Cap Kelly at 25% for safety (common practice)
        return min(max(kelly, 0), 0.25)

    def analyze_trade_distribution(self, trades: list[StrategyTradeRecord]) -> dict[str, Any]:
        """
        Analyze the distribution of trades.

        Args:
            trades: List of trade records

        Returns:
            Dictionary with distribution analysis
        """
        if not trades:
            return {}

        pnls = [t.pnl for t in trades if t.pnl is not None]

        if not pnls:
            return {}

        analysis = {
            "count": len(pnls),
            "mean": statistics.mean(pnls),
            "median": statistics.median(pnls),
            "std_dev": statistics.stdev(pnls) if len(pnls) > 1 else 0,
            "min": min(pnls),
            "max": max(pnls),
            "skewness": self._calculate_skewness(pnls),
            "kurtosis": self._calculate_kurtosis(pnls),
        }

        # Calculate percentiles
        sorted_pnls = sorted(pnls)
        n = len(sorted_pnls)

        percentiles = {
            "p25": sorted_pnls[int(n * 0.25)] if n > 0 else 0,
            "p50": sorted_pnls[int(n * 0.50)] if n > 0 else 0,
            "p75": sorted_pnls[int(n * 0.75)] if n > 0 else 0,
            "p95": sorted_pnls[int(n * 0.95)] if n > 0 else 0,
        }

        analysis["percentiles"] = percentiles  # type: ignore[assignment]

        return analysis

    def _calculate_skewness(self, values: list[float]) -> float:
        """
        Calculate skewness of distribution.

        Args:
            values: List of values

        Returns:
            Skewness value
        """
        if len(values) < 3:
            return 0

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            return 0

        n = len(values)
        skewness = (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in values)

        return skewness

    def _calculate_kurtosis(self, values: list[float]) -> float:
        """
        Calculate kurtosis of distribution.

        Args:
            values: List of values

        Returns:
            Kurtosis value
        """
        if len(values) < 4:
            return 0

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            return 0

        n = len(values)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(
            ((x - mean) / std) ** 4 for x in values
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return kurtosis

    def compare_strategies(
        self,
        strategies_data: dict[str, list[StrategyTradeRecord]],
        period: tuple[float, float] | None = None,
    ) -> StrategyComparison:
        """
        Compare multiple strategies.

        Args:
            strategies_data: Dictionary mapping strategy names to trade records
            period: Optional time period tuple (start, end)

        Returns:
            Strategy comparison results
        """
        if not strategies_data:
            raise ValueError("No strategies to compare")

        # Calculate metrics for each strategy
        metrics = {}
        for name, trades in strategies_data.items():
            # Filter by period if specified
            if period:
                start_time, end_time = period
                filtered_trades = [t for t in trades if start_time <= t.timestamp <= end_time]
            else:
                filtered_trades = trades

            metrics[name] = self.calculate_strategy_performance(name, filtered_trades)

        # Find best performers
        best_pnl = max(metrics.items(), key=lambda x: x[1].total_pnl)[0]
        worst_pnl = min(metrics.items(), key=lambda x: x[1].total_pnl)[0]
        most_active = max(metrics.items(), key=lambda x: x[1].total_trades)[0]

        # Most consistent (lowest drawdown relative to profit)
        consistency_scores = {}
        for name, m in metrics.items():
            if m.max_drawdown > 0:
                consistency_scores[name] = m.total_pnl / m.max_drawdown
            else:
                consistency_scores[name] = m.total_pnl

        most_consistent = (
            max(consistency_scores.items(), key=lambda x: x[1])[0]
            if consistency_scores
            else best_pnl
        )

        highest_win_rate = max(metrics.items(), key=lambda x: x[1].win_rate_percent)[0]
        highest_profit_factor = max(
            metrics.items(),
            key=lambda x: x[1].profit_factor if x[1].profit_factor != float("inf") else 0,
        )[0]

        return StrategyComparison(
            strategies=list(strategies_data.keys()),
            best_performer=best_pnl,
            worst_performer=worst_pnl,
            most_active=most_active,
            most_consistent=most_consistent,
            highest_win_rate=highest_win_rate,
            highest_profit_factor=highest_profit_factor,
            comparison_period=period or (0, float("inf")),
        )
