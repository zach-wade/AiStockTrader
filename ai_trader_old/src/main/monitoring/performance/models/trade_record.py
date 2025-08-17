"""
Trade Record Models

Data structures for individual trade tracking and analysis.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TradeStatus(Enum):
    """Trade execution status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradeSide(Enum):
    """Trade side."""

    BUY = "buy"
    SELL = "sell"


class TradeType(Enum):
    """Trade type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradeRecord:
    """Individual trade record for performance analysis."""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    fees: float = 0.0
    execution_time_ms: float = 0.0
    strategy: str = ""
    is_closed: bool = False

    @property
    def duration_minutes(self) -> Optional[float]:
        """Get trade duration in minutes."""
        if not self.exit_time:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 60

    @property
    def return_pct(self) -> float:
        """Get trade return percentage."""
        if not self.exit_price or self.entry_price == 0:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100


@dataclass
class ExecutionDetails:
    """Detailed execution information for a trade."""

    execution_id: str = ""
    venue: str = ""
    liquidity_flag: str = ""  # maker/taker
    market_impact: float = 0.0
    timing_stats: dict = field(default_factory=dict)


@dataclass
class TradeStatistics:
    """Aggregated statistics for a group of trades."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
