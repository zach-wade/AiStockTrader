"""
Core types and enums for the risk management system.

This module defines the fundamental data types, enums, and dataclasses
used throughout the risk management framework.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RiskLevel(Enum):
    """Risk severity levels."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


# Alias for backward compatibility
RiskAlertLevel = RiskLevel


class RiskEventType(Enum):
    """Types of risk events."""

    # Limit breaches
    POSITION_LIMIT_BREACH = "position_limit_breach"
    EXPOSURE_LIMIT_BREACH = "exposure_limit_breach"
    CONCENTRATION_LIMIT_BREACH = "concentration_limit_breach"
    LIQUIDITY_LIMIT_BREACH = "liquidity_limit_breach"

    # Market events
    VOLATILITY_SPIKE = "volatility_spike"
    PRICE_GAP = "price_gap"
    VOLUME_ANOMALY = "volume_anomaly"
    CORRELATION_BREAKDOWN = "correlation_breakdown"

    # Portfolio events
    DRAWDOWN_ALERT = "drawdown_alert"
    VAR_BREACH = "var_breach"
    MARGIN_CALL = "margin_call"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"

    # System events
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    ANOMALY_DETECTED = "anomaly_detected"
    REGIME_CHANGE = "regime_change"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"


class RiskMetric(Enum):
    """Risk metrics tracked by the system."""

    # Value at Risk metrics
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"

    # Portfolio metrics
    PORTFOLIO_BETA = "portfolio_beta"
    PORTFOLIO_VOLATILITY = "portfolio_volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"

    # Exposure metrics
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    SECTOR_EXPOSURE = "sector_exposure"
    CONCENTRATION_RISK = "concentration_risk"

    # Drawdown metrics
    MAX_DRAWDOWN = "max_drawdown"
    CURRENT_DRAWDOWN = "current_drawdown"
    DRAWDOWN_DURATION = "drawdown_duration"

    # Liquidity metrics
    LIQUIDITY_RATIO = "liquidity_ratio"
    DAYS_TO_LIQUIDATE = "days_to_liquidate"
    MARKET_IMPACT = "market_impact"


class RiskStatus(Enum):
    """Overall risk status of the system."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


@dataclass
class RiskCheckResult:
    """Result of a risk check operation."""

    passed: bool
    check_name: str
    metric: RiskMetric
    current_value: float
    limit_value: float
    utilization: float  # Percentage of limit used
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> RiskLevel:
        """Determine severity based on utilization."""
        if self.utilization < 50:
            return RiskLevel.MINIMAL
        elif self.utilization < 70:
            return RiskLevel.LOW
        elif self.utilization < 85:
            return RiskLevel.MODERATE
        elif self.utilization < 95:
            return RiskLevel.HIGH
        elif self.utilization < 100:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EXTREME


@dataclass
class RiskEvent:
    """Represents a risk event that occurred."""

    event_id: str
    event_type: RiskEventType
    severity: RiskLevel
    title: str
    description: str
    symbol: str | None = None
    portfolio_impact: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        """Mark the event as resolved."""
        self.resolved = True
        self.resolution_time = datetime.utcnow()

    @property
    def duration(self) -> float | None:
        """Get event duration in seconds."""
        if self.resolution_time:
            return (self.resolution_time - self.timestamp).total_seconds()
        return None


@dataclass
class RiskLimitBreach:
    """Represents a breach of a risk limit."""

    limit_name: str
    limit_type: str
    current_value: float
    limit_value: float
    breach_amount: float
    breach_percentage: float
    symbol: str | None = None
    portfolio_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actions_taken: list[str] = field(default_factory=list)

    @property
    def severity(self) -> RiskLevel:
        """Determine breach severity."""
        if self.breach_percentage < 5:
            return RiskLevel.LOW
        elif self.breach_percentage < 10:
            return RiskLevel.MODERATE
        elif self.breach_percentage < 20:
            return RiskLevel.HIGH
        elif self.breach_percentage < 50:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EXTREME


@dataclass
class RiskAlert:
    """Risk alert for notification systems."""

    alert_id: str
    alert_type: str
    severity: RiskLevel
    title: str
    message: str
    source: str  # Component that generated the alert
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    requires_action: bool = False
    suggested_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def acknowledge(self, user: str):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""

    symbol: str
    quantity: float
    market_value: float
    var_95: float
    var_99: float
    beta: float
    volatility: float
    liquidity_score: float  # 0-1, higher is more liquid
    concentration: float  # Percentage of portfolio
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioRisk:
    """Aggregate portfolio risk metrics."""

    total_value: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    concentration_score: float  # 0-1, lower is better diversified
    liquidity_score: float  # 0-1, higher is more liquid
    position_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        # Simple weighted average of key metrics
        var_score = min(self.var_95 / self.total_value * 100, 100) * 0.25
        dd_score = min(abs(self.current_drawdown) * 100, 100) * 0.25
        conc_score = self.concentration_score * 100 * 0.25
        lev_score = min(self.leverage * 20, 100) * 0.25

        return var_score + dd_score + conc_score + lev_score


@dataclass
class RiskLimit:
    """Definition of a risk limit."""

    name: str
    metric: RiskMetric
    limit_value: float
    warning_threshold: float  # Percentage of limit to trigger warning
    enabled: bool = True
    applies_to: str = "portfolio"  # "portfolio", "position", or specific symbol
    check_frequency: int = 60  # Seconds between checks
    actions_on_breach: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskProfile:
    """Risk profile configuration."""

    name: str
    description: str
    max_position_size: float  # As percentage of portfolio
    max_sector_exposure: float  # As percentage of portfolio
    max_leverage: float
    max_drawdown: float
    var_limit_95: float  # As percentage of portfolio
    var_limit_99: float
    stop_loss_pct: float
    take_profit_pct: float
    correlation_limit: float
    liquidity_requirement: float  # Minimum liquidity score
    position_limits: dict[str, float] = field(default_factory=dict)  # Symbol-specific limits
    enabled_checks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
