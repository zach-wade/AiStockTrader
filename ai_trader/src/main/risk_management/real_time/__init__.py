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
    CorrelationAnomalyDetector,
    CorrelationBreakdownEvent,
    CorrelationConfig
)

from .regime_detector import (
    MarketRegimeDetector,
    RegimeChangeEvent,
    RegimeConfig
)

# Import risk monitors
from .live_risk_monitor import (
    LiveRiskMonitor,
    RiskMonitorConfig,
    MonitoringAlert
)

from .drawdown_control import (
    DrawdownController,
    DrawdownConfig,
    DrawdownAction
)

from .stop_loss import (
    StopLossManager,
    StopLossOrder,
    StopLossConfig,
    TrailingStopLoss
)

from .position_liquidator import (
    PositionLiquidator,
    LiquidationStrategy,
    LiquidationOrder,
    EmergencyLiquidation
)

# Import circuit breaker system
from .circuit_breaker import (
    CircuitBreakerFacade,
    CircuitBreakerConfig,
    CircuitBreakerType,
    CircuitBreakerEvent,
    CircuitBreakerRegistry,
    BaseCircuitBreaker,
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
    'CorrelationBreakdownEvent',
    'CorrelationConfig',
    
    # Regime detection
    'MarketRegimeDetector',
    'RegimeChangeEvent',
    'RegimeConfig',
    
    # Risk monitors
    'LiveRiskMonitor',
    'RiskMonitorConfig',
    'MonitoringAlert',
    
    # Drawdown control
    'DrawdownController',
    'DrawdownConfig',
    'DrawdownAction',
    
    # Stop loss management
    'StopLossManager',
    'StopLossOrder',
    'StopLossConfig',
    'TrailingStopLoss',
    
    # Position liquidation
    'PositionLiquidator',
    'LiquidationStrategy',
    'LiquidationOrder',
    'EmergencyLiquidation',
    
    # Circuit breakers
    'CircuitBreakerFacade',
    'CircuitBreakerConfig',
    'CircuitBreakerType',
    'CircuitBreakerEvent',
    'CircuitBreakerRegistry',
    'BaseCircuitBreaker',
    'DrawdownBreaker',
    'VolatilityBreaker',
    'LossRateBreaker',
    'PositionLimitBreaker'
]