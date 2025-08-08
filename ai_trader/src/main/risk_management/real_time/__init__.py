"""
Real-time risk management module.

This module provides real-time risk monitoring, anomaly detection,
circuit breakers, and automated risk controls for live trading.
"""

# Import anomaly detection
from .anomaly_detector import (
    RealTimeAnomalyDetector,
    AnomalyDetector  # Backward compatibility alias
)

from .anomaly_types import (
    AnomalyType,
    AnomalySeverity
)

from .anomaly_models import (
    AnomalyEvent,
    MarketRegime,
    RegimeType,
    AnomalyDetectionConfig
)

from .statistical_detector import (
    StatisticalAnomalyDetector,
    StatisticalConfig,
    ZScoreDetector,
    IsolationForestDetector
)

from .correlation_detector import (
    CorrelationAnomalyDetector
)

from .regime_detector import (
    MarketRegimeDetector
)

# Import risk monitors
from .live_risk_monitor import (
    LiveRiskMonitor,
    RiskMonitorConfig,
    MonitoringAlert
)

from .drawdown_control import (
    DrawdownController,
    # DrawdownConfig,  # TODO: Need to implement
    # DrawdownAction   # TODO: Need to implement
)

from .stop_loss import (
    DynamicStopLossManager,
    StopLoss,
    # StopLossConfig,  # TODO: Need to implement
    # TrailingStopLoss  # TODO: Need to implement
)

from .position_liquidator import (
    PositionLiquidator,
    LiquidationStrategy,
    # LiquidationOrder,  # TODO: Need to implement
    # EmergencyLiquidation  # TODO: Need to implement
)

# Import circuit breaker system
from .circuit_breaker import (
    CircuitBreakerFacade,
    BreakerConfig,
    BreakerType,
    CircuitBreakerEvent,
    BreakerRegistry,
    BaseBreaker,
    DrawdownBreaker,
    VolatilityBreaker,
    LossRateBreaker,
    PositionLimitBreaker
)

__all__ = [
    # Anomaly detection
    'RealTimeAnomalyDetector',
    'AnomalyDetector',
    'AnomalyType',
    'AnomalySeverity',
    'AnomalyEvent',
    'MarketRegime',
    'RegimeType',
    'AnomalyDetectionConfig',
    
    # Statistical detection
    'StatisticalAnomalyDetector',
    'StatisticalConfig',
    'ZScoreDetector',
    'IsolationForestDetector',
    
    # Correlation detection
    'CorrelationAnomalyDetector',
    
    # Regime detection
    'MarketRegimeDetector',
    
    # Risk monitors
    'LiveRiskMonitor',
    'RiskMonitorConfig',
    'MonitoringAlert',
    
    # Drawdown control
    'DrawdownController',
    
    # Stop loss management
    'DynamicStopLossManager',
    'StopLoss',
    'StopLossType',
    
    # Position liquidation
    'PositionLiquidator',
    'LiquidationStrategy',
    
    # Circuit breakers
    'CircuitBreakerFacade',
    'BreakerConfig',
    'BreakerType',
    'CircuitBreakerEvent',
    'BreakerRegistry',
    'BaseBreaker',
    'DrawdownBreaker',
    'VolatilityBreaker',
    'LossRateBreaker',
    'PositionLimitBreaker'
]