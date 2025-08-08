"""
Performance Alert Models

Data structures for performance alerts and notifications.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class PerformanceAlert(Enum):
    """Performance alert types."""
    HIGH_DRAWDOWN = "high_drawdown"
    LOW_SHARPE = "low_sharpe"
    HIGH_VAR_BREACH = "high_var_breach"
    SLOW_EXECUTION = "slow_execution"
    HIGH_COSTS = "high_costs"
    SYSTEM_PERFORMANCE = "system_performance"


@dataclass
class PerformanceAlertData:
    """Performance alert data."""
    alert_id: str
    alert_type: PerformanceAlert
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types."""
    PERFORMANCE = "performance"
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"


class AlertCondition(Enum):
    """Alert condition types."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    THRESHOLD_BELOW = "threshold_below"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ERROR = "system_error"


class AlertAction(Enum):
    """Alert actions."""
    NOTIFY = "notify"
    EMAIL = "email"
    SMS = "sms"
    LOG = "log"
    ESCALATE = "escalate"


# Alias for compatibility
AlertModel = PerformanceAlertData


@dataclass
class AlertHistory:
    """Alert history record."""
    alert_id: str = ""
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime = None
    action_taken: AlertAction = AlertAction.LOG
    notes: str = ""