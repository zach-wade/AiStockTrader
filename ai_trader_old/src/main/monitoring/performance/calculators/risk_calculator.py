"""
Risk Calculator

Handles all risk-related calculations for performance analysis.
"""

# Third-party imports
import numpy as np

# Local imports
from main.utils.math_utils import safe_divide


class RiskCalculator:
    """Consolidated risk calculations."""

    @staticmethod
    def volatility(returns: list[float], annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0

        vol = np.std(returns, ddof=1)
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        return vol

    @staticmethod
    def downside_volatility(
        returns: list[float], target_return: float = 0.0, annualize: bool = True
    ) -> float:
        """Calculate downside volatility."""
        downside_returns = [min(0, r - target_return) for r in returns]
        downside_var = np.mean([r**2 for r in downside_returns])
        downside_vol = np.sqrt(downside_var)

        if annualize:
            downside_vol *= np.sqrt(252)
        return downside_vol

    @staticmethod
    def max_drawdown(cumulative_returns: list[float]) -> float:
        """Calculate maximum drawdown."""
        if not cumulative_returns:
            return 0.0

        peak = cumulative_returns[0]
        max_dd = 0.0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = safe_divide(peak - value, 1 + peak, default_value=0.0)
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def current_drawdown(cumulative_returns: list[float]) -> float:
        """Calculate current drawdown."""
        if not cumulative_returns:
            return 0.0

        peak = max(cumulative_returns)
        current = cumulative_returns[-1]

        if peak == -1:  # Avoid division by zero
            return 0.0

        return safe_divide(peak - current, 1 + peak, default_value=0.0)

    @staticmethod
    def var(returns: list[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if not returns:
            return 0.0

        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0

    @staticmethod
    def cvar(returns: list[float], confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not returns:
            return 0.0

        var_value = RiskCalculator.var(returns, confidence_level)
        tail_returns = [r for r in returns if r <= -var_value]

        return abs(np.mean(tail_returns)) if tail_returns else 0.0
