"""
Real-time risk management module.

This module provides real-time risk monitoring, anomaly detection,
circuit breakers, and automated risk controls for live trading.
"""

# Import anomaly detection
from .anomaly_detector import AnomalyDetector  # Backward compatibility alias
from .anomaly_detector import RealTimeAnomalyDetector
from .anomaly_models import AnomalyDetectionConfig, AnomalyEvent, MarketRegime, RegimeType
from .anomaly_types import AnomalySeverity, AnomalyType

# LiquidationOrder,  # TODO: Need to implement
# EmergencyLiquidation  # TODO: Need to implement
# Import circuit breaker system
from .circuit_breaker import (
    BaseBreaker,
    BreakerConfig,
    BreakerRegistry,
    BreakerType,
    CircuitBreakerEvent,
    CircuitBreakerFacade,
    DrawdownBreaker,
    LossRateBreaker,
    PositionLimitBreaker,
    VolatilityBreaker,
)
from .correlation_detector import CorrelationAnomalyDetector
from .drawdown_control import DrawdownController

# Import risk monitors
from .live_risk_monitor import LiveRiskMonitor, MonitoringAlert, RiskMonitorConfig

# StopLossConfig,  # TODO: Need to implement
# TrailingStopLoss  # TODO: Need to implement
from .position_liquidator import LiquidationStrategy, PositionLiquidator
from .regime_detector import MarketRegimeDetector
from .statistical_detector import (
    IsolationForestDetector,
    StatisticalAnomalyDetector,
    StatisticalConfig,
    ZScoreDetector,
)

# DrawdownConfig,  # TODO: Need to implement
# DrawdownAction   # TODO: Need to implement
from .stop_loss import DynamicStopLossManager, StopLoss

__all__ = [
    # Anomaly detection
    "RealTimeAnomalyDetector",
    "AnomalyDetector",
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyEvent",
    "MarketRegime",
    "RegimeType",
    "AnomalyDetectionConfig",
    # Statistical detection
    "StatisticalAnomalyDetector",
    "StatisticalConfig",
    "ZScoreDetector",
    "IsolationForestDetector",
    # Correlation detection
    "CorrelationAnomalyDetector",
    # Regime detection
    "MarketRegimeDetector",
    # Risk monitors
    "LiveRiskMonitor",
    "RiskMonitorConfig",
    "MonitoringAlert",
    # Drawdown control
    "DrawdownController",
    # Stop loss management
    "DynamicStopLossManager",
    "StopLoss",
    "StopLossType",
    # Position liquidation
    "PositionLiquidator",
    "LiquidationStrategy",
    # Circuit breakers
    "CircuitBreakerFacade",
    "BreakerConfig",
    "BreakerType",
    "CircuitBreakerEvent",
    "BreakerRegistry",
    "BaseBreaker",
    "DrawdownBreaker",
    "VolatilityBreaker",
    "LossRateBreaker",
    "PositionLimitBreaker",
]
