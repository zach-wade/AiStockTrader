"""
Unified Limit Checker Types Module

This module contains all enum definitions and type constants used throughout
the unified limit checker system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass


class LimitType(Enum):
    """Types of limits that can be checked."""
    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    RISK_METRIC = "risk_metric"
    TRADING_VELOCITY = "trading_velocity"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    SECTOR_EXPOSURE = "sector_exposure"
    CURRENCY_EXPOSURE = "currency_exposure"
    LEVERAGE = "leverage"
    VAR_UTILIZATION = "var_utilization"
    PERFORMANCE = "performance"
    OPERATIONAL = "operational"


class LimitScope(Enum):
    """Scope of limit application."""
    GLOBAL = "global"               # Applies to entire portfolio
    SECTOR = "sector"               # Applies to sector grouping
    STRATEGY = "strategy"           # Applies to specific strategy
    POSITION = "position"           # Applies to individual position
    COUNTERPARTY = "counterparty"   # Applies to counterparty exposure
    GEOGRAPHIC = "geographic"       # Applies to geographic region
    CURRENCY = "currency"           # Applies to currency exposure
    ASSET_CLASS = "asset_class"     # Applies to asset class


class ViolationSeverity(Enum):
    """Severity levels for limit violations."""
    INFO = "info"                   # Informational, no action required
    WARNING = "warning"             # Warning level, monitor closely
    SOFT_BREACH = "soft_breach"     # Soft breach, consider action
    HARD_BREACH = "hard_breach"     # Hard breach, immediate action required
    CRITICAL = "critical"           # Critical breach, emergency action


class LimitAction(Enum):
    """Actions to take on limit violations."""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    BLOCK_TRADE = "block_trade"
    REDUCE_POSITION = "reduce_position"
    LIQUIDATE = "liquidate"
    PAUSE_STRATEGY = "pause_strategy"
    EMERGENCY_STOP = "emergency_stop"


class ComparisonOperator(Enum):
    """Comparison operators for limit checking."""
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    BETWEEN = "between"
    OUTSIDE = "outside"


@dataclass
class PortfolioState:
    """
    Represents the current state of a portfolio for risk checking.
    
    This class contains all the portfolio information needed to perform
    comprehensive risk checks before trade execution.
    """
    portfolio_id: Optional[str] = None
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions: Dict[str, Any] = None
    value_history: List[Tuple[datetime, float]] = None
    last_updated: Optional[datetime] = None
    
    # Risk metrics
    current_leverage: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    sector_exposures: Dict[str, float] = None
    
    # Performance tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.positions is None:
            self.positions = {}
        if self.value_history is None:
            self.value_history = []
        if self.sector_exposures is None:
            self.sector_exposures = {}
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


@dataclass
class CheckContext:
    """
    Context information provided to limit checkers.
    
    This class contains all the context information that risk checkers
    need to make informed decisions about trade approval.
    """
    portfolio_state: PortfolioState
    market_conditions: Dict[str, Any] = None
    strategy_id: Optional[str] = None
    
    # Trading session info
    session_id: Optional[str] = None
    trading_session: Optional[str] = None  # e.g., "pre_market", "regular", "after_hours"
    
    # Risk configuration
    risk_profile: Optional[str] = None  # e.g., "conservative", "moderate", "aggressive"
    
    # Timestamp for check
    check_timestamp: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.market_conditions is None:
            self.market_conditions = {}
        if self.metadata is None:
            self.metadata = {}
        if self.check_timestamp is None:
            self.check_timestamp = datetime.utcnow()