"""Performance Calculator domain service for risk-adjusted performance metrics.

This module provides the PerformanceCalculator service which focuses specifically
on performance analysis and risk-adjusted return calculations. It handles Sharpe
ratio calculations and comprehensive risk-adjusted return analysis.

The service follows Single Responsibility Principle by focusing solely on performance
calculations, extracted from the original RiskCalculator to improve maintainability.
"""

# Standard library imports
from decimal import Decimal

from ...constants import MIN_DATA_POINTS_FOR_STATS
from ...entities import Portfolio
from ...value_objects import Money

# Trading days constants
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_FACTOR = Decimal(str(TRADING_DAYS_PER_YEAR**0.5))


class PerformanceCalculator:
    """Domain service for calculating risk-adjusted performance metrics.

    This service provides comprehensive risk-adjusted performance analysis
    functionality including Sharpe ratio calculations and comprehensive
    risk-adjusted return metrics for portfolio evaluation.

    The service is stateless and thread-safe, with all methods operating as pure
    functions on provided entities.
    """

    def calculate_sharpe_ratio(
        self, returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal | None:
        """Calculate Sharpe ratio from returns.

        Computes the risk-adjusted return metric that measures excess return per
        unit of risk. The Sharpe ratio is the industry standard for comparing
        investment strategies on a risk-adjusted basis.

        Args:
            returns: List of period returns as decimals (e.g., 0.01 for 1%).
                Assumed to be daily returns. Requires at least MIN_DATA_POINTS_FOR_STATS
                values for statistical significance.
            risk_free_rate: Annual risk-free rate as decimal (default 0.02 for 2%).
                Typically the Treasury bill rate or similar benchmark.

        Returns:
            Decimal | None: Annualized Sharpe ratio. Returns None if:
                - Insufficient data points
                - Standard deviation is zero (no volatility)
                Higher values indicate better risk-adjusted returns.
                - < 0: Losing money relative to risk-free rate
                - 0-1: Positive but subpar risk-adjusted returns
                - 1-2: Good risk-adjusted returns
                - > 2: Excellent risk-adjusted returns

        Formula:
            Sharpe = (Annual Return - Risk-Free Rate) / Annual Volatility
            - Returns are annualized by multiplying by 252 (trading days)
            - Volatility is annualized by multiplying by √252

        Note:
            The Sharpe ratio assumes returns are normally distributed, which may
            not hold for all trading strategies. Consider supplementing with other
            metrics like Sortino ratio for downside risk assessment.
        """
        if not returns or len(returns) < MIN_DATA_POINTS_FOR_STATS:
            return None

        # Calculate average return
        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((Decimal(str(r)) - Decimal(str(avg_return))) ** 2 for r in returns) / len(
            returns
        )
        std_dev = (
            variance.sqrt() if hasattr(variance, "sqrt") else Decimal(str(float(variance) ** 0.5))
        )

        if std_dev == 0:
            return None

        # Annualize (assuming daily returns)
        annual_return = Decimal(str(avg_return)) * Decimal(str(TRADING_DAYS_PER_YEAR))
        annual_std = Decimal(str(std_dev)) * ANNUALIZATION_FACTOR

        # Calculate Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_std

        return sharpe

    def calculate_risk_adjusted_return(
        self, portfolio: Portfolio, _time_period_days: int = 30
    ) -> dict[str, Decimal | Money | None]:
        """Calculate comprehensive risk-adjusted return metrics.

        Computes a suite of risk-adjusted performance metrics that provide a
        holistic view of portfolio performance considering both returns and risk.
        These metrics are essential for strategy evaluation and comparison.

        Args:
            portfolio: Portfolio to analyze.
            _time_period_days: Period for calculations (currently unused but reserved
                for future time-based filtering). Defaults to 30 days.

        Returns:
            dict[str, Decimal | Money | None]: Dictionary containing metrics:
                - total_return: Overall portfolio return percentage (Decimal)
                - win_rate: Percentage of winning trades (Decimal)
                - profit_factor: Ratio of gross profit to gross loss (Decimal)
                - average_win: Average profit on winning trades (Money)
                - average_loss: Average loss on losing trades (Money)
                - max_drawdown: Maximum peak-to-trough decline (Decimal)
                - sharpe_ratio: Risk-adjusted return metric (Decimal)
                - calmar_ratio: Return relative to maximum drawdown (Decimal)
                - expectancy: Expected value per trade (Money)
            Values are None when insufficient data or calculation not possible.

        Metrics Interpretation:
            - Profit Factor > 1.5: Good profitability
            - Win Rate > 50%: More winners than losers (not always necessary)
            - Sharpe Ratio > 1: Good risk-adjusted returns
            - Calmar Ratio > 1: Return exceeds maximum drawdown
            - Positive Expectancy: Positive expected value per trade

        Note:
            Some metrics may be None if the portfolio lacks sufficient trading
            history or if certain calculations are undefined (e.g., Calmar ratio
            when max drawdown is zero).
        """
        # Import here to avoid circular dependency
        from .portfolio_var_calculator import PortfolioVaRCalculator

        portfolio_var_calculator = PortfolioVaRCalculator()

        metrics = {
            "total_return": portfolio.get_return_percentage(),
            "win_rate": portfolio.get_win_rate(),
            "profit_factor": portfolio.get_profit_factor(),
            "average_win": portfolio.get_average_win(),
            "average_loss": portfolio.get_average_loss(),
            "max_drawdown": portfolio_var_calculator.calculate_max_drawdown(
                [portfolio.get_total_value()]
            ),
            "sharpe_ratio": portfolio.get_sharpe_ratio(),
        }

        # Calculate risk-adjusted return (Calmar ratio)
        if metrics["total_return"] and metrics["max_drawdown"]:
            if metrics["max_drawdown"] > 0:
                metrics["calmar_ratio"] = metrics["total_return"] / metrics["max_drawdown"]
            else:
                metrics["calmar_ratio"] = None
        else:
            metrics["calmar_ratio"] = None

        # Calculate expectancy
        avg_win = metrics["average_win"]
        avg_loss = metrics["average_loss"]
        if (
            metrics["win_rate"] is not None
            and avg_win is not None
            and avg_loss is not None
            and isinstance(avg_win, Money)
            and isinstance(avg_loss, Money)
            and isinstance(metrics["win_rate"], Decimal)
        ):
            win_rate = metrics["win_rate"] / Decimal("100")
            loss_rate = 1 - win_rate
            expectancy_amount = (win_rate * avg_win.amount) - (loss_rate * avg_loss.amount)
            metrics["expectancy"] = Money(expectancy_amount)
        else:
            metrics["expectancy"] = None

        return metrics
