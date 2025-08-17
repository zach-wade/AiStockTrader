"""Advanced performance metrics calculation."""

# Third-party imports
import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """Calculate comprehensive performance metrics."""

    @staticmethod
    def calculate_metrics(
        equity_curve: pd.Series, trades: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> dict[str, float]:
        """Calculate all performance metrics."""
        returns = equity_curve.pct_change().dropna()

        metrics = {
            # Basic metrics
            "total_return": PerformanceAnalyzer.total_return(equity_curve),
            "cagr": PerformanceAnalyzer.cagr(equity_curve),
            "volatility": PerformanceAnalyzer.volatility(returns),
            # Risk-adjusted metrics
            "sharpe_ratio": PerformanceAnalyzer.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": PerformanceAnalyzer.sortino_ratio(returns, risk_free_rate),
            "calmar_ratio": PerformanceAnalyzer.calmar_ratio(equity_curve),
            # Drawdown metrics
            "max_drawdown": PerformanceAnalyzer.max_drawdown(equity_curve),
            "avg_drawdown": PerformanceAnalyzer.avg_drawdown(equity_curve),
            "max_drawdown_duration": PerformanceAnalyzer.max_drawdown_duration(equity_curve),
            # Trade metrics
            "win_rate": PerformanceAnalyzer.win_rate(trades),
            "profit_factor": PerformanceAnalyzer.profit_factor(trades),
            "avg_win_loss_ratio": PerformanceAnalyzer.avg_win_loss_ratio(trades),
            # Risk metrics
            "var_95": PerformanceAnalyzer.value_at_risk(returns, 0.95),
            "cvar_95": PerformanceAnalyzer.conditional_value_at_risk(returns, 0.95),
            "tail_ratio": PerformanceAnalyzer.tail_ratio(returns),
            # Other metrics
            "kelly_criterion": PerformanceAnalyzer.kelly_criterion(trades),
            "ulcer_index": PerformanceAnalyzer.ulcer_index(equity_curve),
        }

        return metrics

    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """Calculate total return."""
        return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        years = len(equity_curve) / 252  # Assuming daily data
        return (np.power(equity_curve.iloc[-1] / equity_curve.iloc[0], 1 / years) - 1) * 100

    @staticmethod
    def volatility(returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(periods) * 100

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods: int = 252) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / periods
        return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float, periods: int = 252) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        excess_returns = returns - risk_free_rate / periods
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(periods) * excess_returns.mean() / downside_std if downside_std > 0 else 0

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min() * 100

    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        """Calculate win rate from trades."""
        if trades.empty:
            return 0
        profitable = trades["pnl"] > 0
        return (profitable.sum() / len(trades)) * 100

    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if trades.empty:
            return 0
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100) * 100

    @staticmethod
    def kelly_criterion(trades: pd.DataFrame) -> float:
        """Calculate Kelly Criterion for position sizing."""
        if trades.empty:
            return 0
        win_rate = (trades["pnl"] > 0).mean()
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean()
        avg_loss = abs(trades[trades["pnl"] < 0]["pnl"].mean())

        if avg_loss == 0:
            return 0

        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
