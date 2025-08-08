"""
Circuit Breaker Types and Data Classes

Core data structures and enums for the circuit breaker system.
Extracted from monolithic circuit_breaker.py for better organization.

Created: 2025-07-15
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np


class BreakerType(Enum):
    """Types of circuit breakers."""
    VOLATILITY = "volatility"          # Market volatility exceeds threshold
    DRAWDOWN = "drawdown"              # Portfolio drawdown limit
    LOSS_RATE = "loss_rate"           # Rapid loss in short time
    POSITION_LIMIT = "position_limit"  # Too many positions or size
    CORRELATION = "correlation"        # Correlation breakdown
    TECHNICAL = "technical"            # Technical/system issues
    MANUAL = "manual"                  # Manual override
    TIME_BASED = "time_based"         # Time-of-day restrictions
    KILL_SWITCH = "kill_switch"        # Emergency kill switch
    MARKET_CIRCUIT_BREAKER = "market_circuit_breaker"  # External market halts
    ANOMALY_DETECTION = "anomaly_detection"  # Statistical anomaly detection
    LIQUIDATION_REQUIRED = "liquidation_required"  # Emergency liquidation needed
    PREMARKET_VALIDATION = "premarket_validation"  # Pre-market checks failed
    DATA_FEED_INTEGRITY = "data_feed_integrity"  # Data feed issues
    MODEL_PERFORMANCE = "model_performance"  # Model performance degradation


class BreakerStatus(Enum):
    """Circuit breaker states."""
    ACTIVE = "active"      # Normal trading
    WARNING = "warning"    # Close to triggering
    TRIPPED = "tripped"    # Trading halted
    COOLDOWN = "cooldown"  # Waiting before resume
    EMERGENCY_HALT = "emergency_halt"  # Emergency halt activated
    LIQUIDATING = "liquidating"  # Emergency liquidation in progress
    MAINTENANCE = "maintenance"  # System maintenance mode


@dataclass
class BreakerEvent:
    """Circuit breaker event record."""
    timestamp: datetime
    breaker_type: BreakerType
    status: BreakerStatus
    message: str
    metrics: Dict[str, float]
    auto_reset_time: Optional[datetime] = None


@dataclass
class MarketConditions:
    """Current market conditions."""
    timestamp: datetime
    volatility: float
    correlation_matrix: Optional[np.ndarray] = None
    volume_ratio: float = 1.0  # Current vs average volume
    spread_widening: float = 1.0  # Spread vs normal
    # Enhanced anomaly detection fields
    price_anomaly_score: float = 0.0  # Statistical price anomaly score
    volume_anomaly_score: float = 0.0  # Statistical volume anomaly score
    correlation_breakdown_score: float = 0.0  # Correlation breakdown severity
    market_regime_change: bool = False  # Detected market regime change
    # External market data
    nyse_circuit_breaker: bool = False  # NYSE circuit breaker status
    nasdaq_circuit_breaker: bool = False  # NASDAQ circuit breaker status
    vix_level: float = 0.0  # VIX level
    # Data integrity indicators
    data_feed_latency_ms: float = 0.0  # Data feed latency
    missing_data_pct: float = 0.0  # Percentage of missing data points
    data_quality_score: float = 100.0  # Overall data quality score (0-100)


@dataclass
class BreakerMetrics:
    """Circuit breaker metrics."""
    portfolio_peak: float = 0.0
    current_drawdown: float = 0.0
    recent_volatility: float = 0.0
    loss_rate: float = 0.0
    position_count: int = 0
    max_position_size: float = 0.0
    anomaly_score: float = 0.0
    data_quality_score: float = 100.0
    model_accuracy: float = 0.0
    external_market_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class BreakerConfiguration:
    """Configuration for a specific breaker type."""
    breaker_type: BreakerType
    enabled: bool = True
    threshold: float = 0.0
    cooldown_minutes: int = 15
    auto_reset: bool = True
    severity_weight: float = 1.0
    custom_config: Dict[str, Any] = field(default_factory=dict)


# Helper function to convert numpy types to Python types
def to_python_float(val):
    """Convert numpy float to Python float."""
    return float(val) if hasattr(val, "item") else float(val)