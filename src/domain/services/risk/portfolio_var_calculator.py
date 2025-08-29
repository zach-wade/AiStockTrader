"""Portfolio VaR Calculator domain service for portfolio-level risk analysis.

This module provides the PortfolioVaRCalculator service which focuses specifically
on portfolio-level risk calculations including Value at Risk (VaR) and maximum
drawdown analysis. It handles comprehensive risk assessment at the portfolio level.

The service follows Single Responsibility Principle by focusing solely on portfolio
risk calculations, extracted from the original RiskCalculator to improve maintainability.
"""

# Standard library imports
import math
from decimal import Decimal

from ...constants import MIN_DATA_POINTS_FOR_STATS
from ...entities import Portfolio
from ...value_objects import Money

# Trading days constants
TRADING_DAYS_PER_YEAR = 252


class PortfolioVaRCalculator:
    """Domain service for calculating portfolio-level risk metrics.

    This service provides comprehensive Value at Risk (VaR) and drawdown analysis
    functionality for portfolios, focusing on portfolio-level risk assessment
    and portfolio-wide risk metrics.

    The service is stateless and thread-safe, with all methods operating as pure
    functions on provided entities.
    """

    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Money:
        """Calculate Value at Risk (VaR) for portfolio.

        Computes the potential loss in portfolio value that will not be exceeded
        with a given confidence level over a specified time horizon. This implementation
        uses a parametric (variance-covariance) approach with assumed volatility.

        Args:
            portfolio: Portfolio to analyze.
            confidence_level: Confidence level as decimal (e.g., 0.95 for 95%).
                Common values: 0.90, 0.95, 0.99. Must be between 0 and 1.
            time_horizon: Time horizon in trading days. Typically 1, 5, or 10 days.

        Returns:
            Money: VaR amount in USD. This represents the maximum expected loss
                at the given confidence level.

        Raises:
            ValueError: If confidence_level is not between 0 and 1.

        Algorithm:
            VaR = Portfolio Value × Daily Volatility × Z-Score × √Time Horizon
            - Uses 2% daily volatility assumption (would use historical data in production)
            - Z-scores: 90%=1.28, 95%=1.65, 99%=2.33 (normal distribution)
            - Scales by square root of time for multi-day horizons

        Limitations:
            - Assumes normal distribution of returns (often violated in practice)
            - Uses fixed volatility rather than historical or implied volatility
            - Doesn't account for correlation between positions
            - Simplified for demonstration; production would use historical simulation

        Note:
            VaR has known limitations including failure to capture tail risk.
            Consider supplementing with stress testing and scenario analysis.
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        # Simplified VaR: use portfolio value * assumed volatility * z-score
        portfolio_value = portfolio.get_total_value()

        # Z-scores for common confidence levels
        z_scores = {
            Decimal("0.90"): Decimal("1.28"),
            Decimal("0.95"): Decimal("1.65"),
            Decimal("0.99"): Decimal("2.33"),
        }

        # Get closest z-score
        z_score = z_scores.get(confidence_level, Decimal("1.65"))

        # Assume 2% daily volatility (would calculate from historical data)
        daily_volatility = Decimal("0.02")

        # Calculate VaR
        var_amount = (
            portfolio_value.amount
            * daily_volatility
            * z_score
            * Decimal(math.sqrt(float(time_horizon)))
        )

        return Money(var_amount, "USD")

    def calculate_max_drawdown(self, portfolio_history: list[Money]) -> Decimal:
        """Calculate maximum drawdown from portfolio value history.

        Computes the largest peak-to-trough decline in portfolio value over the
        given history. Maximum drawdown is a key risk metric that measures the
        worst-case historical loss from a peak value.

        Args:
            portfolio_history: List of portfolio values over time, ordered chronologically.
                Each Money value represents the total portfolio value at a point in time.
                Requires at least MIN_DATA_POINTS_FOR_STATS values.

        Returns:
            Decimal: Maximum drawdown as a percentage (0-100). Returns 0 if insufficient
                data or empty list. A value of 25 means the portfolio experienced a
                maximum 25% decline from peak.

        Algorithm:
            1. Track running maximum (peak) value
            2. Calculate drawdown from peak for each subsequent value
            3. Return the maximum drawdown observed

        Note:
            Drawdown is always calculated from the historical peak, not from
            the initial value. This captures the psychological impact of losses
            from the highest point achieved.
        """
        if not portfolio_history or len(portfolio_history) < MIN_DATA_POINTS_FOR_STATS:
            return Decimal("0")

        max_value = portfolio_history[0].amount
        max_drawdown = Decimal("0")

        for money_value in portfolio_history:
            value = money_value.amount
            if value > max_value:
                max_value = value

            drawdown = (max_value - value) / max_value if max_value > 0 else Decimal("0")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown * Decimal("100")
