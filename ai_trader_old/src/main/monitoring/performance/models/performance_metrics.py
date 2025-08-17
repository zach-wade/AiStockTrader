"""
Performance Metrics Data Models

Core data structures for performance tracking and analysis.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class PerformanceMetricType(Enum):
    """Types of performance metrics."""

    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    TRADING = "trading"
    SYSTEM = "system"
    COST = "cost"


class TimeFrame(Enum):
    """Time frames for performance analysis."""

    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""

    # Time period
    start_date: datetime
    end_date: datetime
    period_days: int

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    cumulative_returns: List[float] = field(default_factory=list)

    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Portfolio metrics
    positions_count: int = 0
    portfolio_value: float = 0.0
    cash_balance: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Cost metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_fees: float = 0.0
    execution_cost_bps: float = 0.0

    # System metrics
    avg_execution_time: float = 0.0
    system_uptime_pct: float = 0.0
    api_success_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0

    # Metadata
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_points: int = 0
    benchmark_return: Optional[float] = None
    risk_free_rate: float = 0.02  # 2% default

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "period_days": self.period_days,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "positions_count": self.positions_count,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_fees": self.total_fees,
            "execution_cost_bps": self.execution_cost_bps,
            "avg_execution_time": self.avg_execution_time,
            "system_uptime_pct": self.system_uptime_pct,
            "api_success_rate": self.api_success_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_pct": self.cpu_usage_pct,
            "calculated_at": self.calculated_at.isoformat(),
            "data_points": self.data_points,
            "benchmark_return": self.benchmark_return,
            "risk_free_rate": self.risk_free_rate,
        }
