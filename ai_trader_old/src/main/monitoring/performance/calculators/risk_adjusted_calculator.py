"""
Risk-Adjusted Calculator

Handles all risk-adjusted return calculations for performance analysis.
"""

# Third-party imports
import numpy as np

# Local imports
from main.utils.math_utils import safe_divide

from .return_calculator import ReturnCalculator
from .risk_calculator import RiskCalculator


class RiskAdjustedCalculator:
    """Risk-adjusted return calculations."""

    @staticmethod
    def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0

        excess_returns = [r - risk_free_rate / 252 for r in returns]  # Daily risk-free rate
        avg_excess = np.mean(excess_returns)
        vol = np.std(excess_returns, ddof=1)

        if vol == 0:
            return 0.0

        return safe_divide(avg_excess, vol, default_value=0.0) * np.sqrt(252)  # Annualized

    @staticmethod
    def sortino_ratio(
        returns: list[float], target_return: float = 0.0, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if not returns:
            return 0.0

        excess_returns = [r - risk_free_rate / 252 for r in returns]
        avg_excess = np.mean(excess_returns)
        downside_vol = RiskCalculator.downside_volatility(
            returns, target_return / 252, annualize=True
        )

        if downside_vol == 0:
            return 0.0

        return safe_divide(avg_excess * 252, downside_vol, default_value=0.0)  # Annualized

    @staticmethod
    def calmar_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Calmar ratio."""
        if not returns:
            return 0.0

        cumulative_returns = ReturnCalculator.cumulative_returns(returns)
        max_dd = RiskCalculator.max_drawdown(cumulative_returns)
        annualized_ret = np.mean(returns) * 252

        if max_dd == 0:
            return 0.0

        return safe_divide(annualized_ret, max_dd, default_value=0.0)

    @staticmethod
    def information_ratio(portfolio_returns: list[float], benchmark_returns: list[float]) -> float:
        """Calculate Information ratio."""
        if len(portfolio_returns) != len(benchmark_returns) or not portfolio_returns:
            return 0.0

        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        avg_excess = np.mean(excess_returns)
        tracking_error = np.std(excess_returns, ddof=1)

        if tracking_error == 0:
            return 0.0

        return safe_divide(avg_excess, tracking_error, default_value=0.0) * np.sqrt(
            252
        )  # Annualized

    @staticmethod
    def treynor_ratio(returns: list[float], beta: float, risk_free_rate: float = 0.02) -> float:
        """Calculate Treynor ratio."""
        if not returns or beta == 0:
            return 0.0

        excess_returns = [r - risk_free_rate / 252 for r in returns]
        avg_excess = np.mean(excess_returns)

        return safe_divide(avg_excess * 252, beta, default_value=0.0)  # Annualized
