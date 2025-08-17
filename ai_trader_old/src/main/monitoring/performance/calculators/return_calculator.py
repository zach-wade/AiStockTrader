"""
Return Calculator

Handles all return-related calculations for performance analysis.
"""


class ReturnCalculator:
    """Consolidated return calculations."""

    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """Calculate total return."""
        if initial_value == 0:
            return 0.0
        return (final_value - initial_value) / initial_value

    @staticmethod
    def annualized_return(total_return: float, days: int) -> float:
        """Calculate annualized return."""
        if days <= 0:
            return 0.0
        years = days / 365.25
        return ((1 + total_return) ** (1 / years)) - 1

    @staticmethod
    def cagr(initial_value: float, final_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial_value <= 0 or final_value <= 0 or years <= 0:
            return 0.0
        return ((final_value / initial_value) ** (1 / years)) - 1

    @staticmethod
    def daily_returns(values: list[float]) -> list[float]:
        """Calculate daily returns from value series."""
        if len(values) < 2:
            return []

        returns = []
        for i in range(1, len(values)):
            if values[i - 1] != 0:
                ret = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(ret)
        return returns

    @staticmethod
    def cumulative_returns(daily_returns: list[float]) -> list[float]:
        """Calculate cumulative returns from daily returns."""
        cumulative = []
        cum_ret = 1.0

        for ret in daily_returns:
            cum_ret *= 1 + ret
            cumulative.append(cum_ret - 1)

        return cumulative
