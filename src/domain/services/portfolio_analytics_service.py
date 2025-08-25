"""
Portfolio Analytics Service

Handles all portfolio performance calculations and analytics in the domain layer.
This service contains the business logic for calculating portfolio metrics,
performance indicators, and risk analytics.
"""

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class PortfolioPerformanceMetrics:
    """Portfolio performance metrics result."""

    portfolio_id: str
    period_days: int
    start_value: float
    end_value: float
    total_return_percent: float
    annualized_volatility_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float
    total_trades: int
    current_positions: int
    data_points: int
    analysis_timestamp: float


@dataclass
class PortfolioValue:
    """Portfolio value at a point in time."""

    timestamp: float
    portfolio_id: str
    value: float
    metadata: dict[str, Any] | None = None


@dataclass
class TradeRecord:
    """Trade execution record."""

    timestamp: float
    order_id: str
    symbol: str
    portfolio_id: str
    strategy: str | None = None
    operation: str = ""
    status: str = ""
    duration_ms: float | None = None
    context: dict[str, Any] | None = None
    submit_time: float | None = None


@dataclass
class PositionInfo:
    """Position information."""

    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    last_update: float


class PortfolioAnalyticsService:
    """
    Service for calculating portfolio performance metrics and analytics.

    This service encapsulates all business logic related to portfolio
    performance calculations, separating it from infrastructure concerns.
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize portfolio analytics service.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def calculate_performance_metrics(
        self,
        portfolio_values: list[PortfolioValue],
        trades: list[TradeRecord],
        current_positions_count: int,
        period_days: int,
    ) -> PortfolioPerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics.

        Args:
            portfolio_values: Historical portfolio values
            trades: Trade records for the period
            current_positions_count: Number of current positions
            period_days: Analysis period in days

        Returns:
            Calculated performance metrics

        Raises:
            ValueError: If insufficient data for calculations
        """
        if not portfolio_values:
            raise ValueError("No portfolio values provided")

        if len(portfolio_values) < 2:
            raise ValueError("Insufficient data points for calculation")

        # Extract values and ensure they're sorted by timestamp
        sorted_values = sorted(portfolio_values, key=lambda x: x.timestamp)
        values = [v.value for v in sorted_values]

        # Basic performance metrics
        start_value = values[0]
        end_value = values[-1]

        if start_value <= 0:
            raise ValueError("Invalid start value: must be positive")

        total_return = self._calculate_total_return(start_value, end_value)

        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(values)

        # Risk metrics
        volatility = self._calculate_annualized_volatility(daily_returns)
        max_drawdown = self._calculate_max_drawdown(values)
        sharpe_ratio = self._calculate_sharpe_ratio(total_return, volatility)

        # Count trades in period
        period_trades_count = len(trades) if trades else 0

        return PortfolioPerformanceMetrics(
            portfolio_id=sorted_values[0].portfolio_id,
            period_days=period_days,
            start_value=start_value,
            end_value=end_value,
            total_return_percent=total_return,
            annualized_volatility_percent=volatility,
            max_drawdown_percent=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=period_trades_count,
            current_positions=current_positions_count,
            data_points=len(portfolio_values),
            analysis_timestamp=sorted_values[-1].timestamp,
        )

    def _calculate_total_return(self, start_value: float, end_value: float) -> float:
        """
        Calculate total return percentage.

        Args:
            start_value: Starting portfolio value
            end_value: Ending portfolio value

        Returns:
            Total return as percentage
        """
        if start_value <= 0:
            return 0.0
        return ((end_value - start_value) / start_value) * 100

    def _calculate_daily_returns(self, values: list[float]) -> list[float]:
        """
        Calculate daily returns from portfolio values.

        Args:
            values: List of portfolio values

        Returns:
            List of daily returns
        """
        daily_returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                daily_return = (values[i] - values[i - 1]) / values[i - 1]
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0.0)
        return daily_returns

    def _calculate_annualized_volatility(self, daily_returns: list[float]) -> float:
        """
        Calculate annualized volatility from daily returns.

        Args:
            daily_returns: List of daily returns

        Returns:
            Annualized volatility as percentage
        """
        if len(daily_returns) <= 1:
            return 0.0

        try:
            daily_std = statistics.stdev(daily_returns)
            annualized_vol = daily_std * (self.trading_days_per_year**0.5) * 100
            return float(annualized_vol)
        except statistics.StatisticsError:
            return 0.0

    def _calculate_max_drawdown(self, values: list[float]) -> float:
        """
        Calculate maximum drawdown percentage.

        Args:
            values: List of portfolio values

        Returns:
            Maximum drawdown as percentage
        """
        if not values:
            return 0.0

        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value

            if peak > 0:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100

    def _calculate_sharpe_ratio(self, total_return: float, volatility: float) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            total_return: Total return as percentage
            volatility: Annualized volatility as percentage

        Returns:
            Sharpe ratio
        """
        if volatility <= 0:
            return 0.0

        # Convert percentages to decimals
        annual_return = total_return / 100
        annual_volatility = volatility / 100

        # Calculate excess return over risk-free rate
        excess_return = annual_return - self.risk_free_rate

        # Calculate Sharpe ratio
        return excess_return / annual_volatility

    def calculate_position_weights(
        self, positions: dict[str, PositionInfo]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate position weights and exposure metrics.

        Args:
            positions: Dictionary of positions by symbol

        Returns:
            Dictionary with position weights and metrics
        """
        if not positions:
            return {}

        total_value = sum(pos.market_value for pos in positions.values())

        if total_value <= 0:
            return {}

        weights = {}
        for symbol, position in positions.items():
            weight_percent = (position.market_value / total_value) * 100

            weights[symbol] = {
                "market_value": position.market_value,
                "weight_percent": weight_percent,
                "quantity": position.quantity,
                "avg_cost": position.avg_cost,
                "unrealized_pnl": position.unrealized_pnl,
            }

        return weights

    def calculate_portfolio_pnl(
        self, positions: dict[str, PositionInfo], realized_pnl: float = 0.0
    ) -> dict[str, float]:
        """
        Calculate portfolio P&L metrics.

        Args:
            positions: Dictionary of positions
            realized_pnl: Realized P&L from closed positions

        Returns:
            P&L metrics dictionary
        """
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_pnl = realized_pnl + unrealized_pnl

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
        }

    def calculate_risk_adjusted_metrics(
        self, returns: list[float], benchmark_returns: list[float] | None = None
    ) -> dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics: dict[str, float] = {}

        if not returns or len(returns) < 2:
            return metrics

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
            avg_return = statistics.mean(returns)
            sortino_ratio = (
                (avg_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            )
            metrics["sortino_ratio"] = sortino_ratio

        # Calmar ratio (return / max drawdown)
        if returns:
            cumulative_returns = []
            cumulative = 1.0
            for r in returns:
                cumulative *= 1 + r
                cumulative_returns.append(cumulative)

            max_dd = self._calculate_max_drawdown(cumulative_returns)
            total_return = (cumulative_returns[-1] - 1) * 100 if cumulative_returns else 0
            calmar_ratio = total_return / max_dd if max_dd > 0 else 0
            metrics["calmar_ratio"] = calmar_ratio

        # Information ratio (if benchmark provided)
        if benchmark_returns and len(benchmark_returns) == len(returns):
            excess_returns = [r - b for r, b in zip(returns, benchmark_returns)]
            if len(excess_returns) > 1:
                tracking_error = statistics.stdev(excess_returns)
                avg_excess = statistics.mean(excess_returns)
                information_ratio = avg_excess / tracking_error if tracking_error > 0 else 0
                metrics["information_ratio"] = information_ratio

        return metrics
