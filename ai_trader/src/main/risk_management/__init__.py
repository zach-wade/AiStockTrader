"""
Risk Management module for comprehensive trading risk control.

This module provides a complete risk management framework including:
- Pre-trade risk checks and position limits
- Real-time risk monitoring and anomaly detection
- Circuit breakers and emergency controls
- Position sizing and portfolio risk management
- Post-trade analysis and reporting
"""

# Import core types
from .types import (
    RiskLevel,
    RiskEventType,
    RiskMetric,
    RiskStatus,
    RiskCheckResult,
    RiskEvent,
    RiskLimitBreach,
    RiskAlert
)

# Import pre-trade risk management
from .pre_trade import (
    UnifiedLimitChecker,
    PositionLimitChecker,
    ExposureLimitsChecker,
    LiquidityChecker
)

# Import real-time risk management
from .real_time import (
    RealTimeAnomalyDetector,
    LiveRiskMonitor,
    DynamicStopLossManager,
    DrawdownController,
    PositionLiquidator,
    CircuitBreakerFacade
)

# Import position sizing
from .position_sizing import (
    VaRPositionSizer,
    # KellyPositionSizer,  # TODO: Need to implement
    # VolatilityPositionSizer,  # TODO: Need to implement
    # OptimalFPositionSizer  # TODO: Need to implement
)

# Import risk metrics
from .metrics import (
    RiskMetricsCalculator,
    PortfolioRiskMetrics,
    # PositionRiskMetrics,  # TODO: Need to implement
    # VaRCalculator,  # TODO: Need to implement
    # CVaRCalculator,  # TODO: Need to implement
    # SharpRatioCalculator  # TODO: Need to implement
)

# Import integration
from .integration import (
    TradingEngineRiskIntegration,
    RiskEventBridge,
    RiskDashboardIntegration
)

# Import post-trade analysis
from .post_trade import (
    PostTradeAnalyzer,
    TradeReview,
    SlippageAnalyzer,
    # RiskPerformanceAnalyzer,  # TODO: Need to implement
    # ComplianceChecker  # TODO: Need to implement
)

__all__ = [
    # Core types
    'RiskLevel',
    'RiskEventType',
    'RiskMetric',
    'RiskStatus',
    'RiskCheckResult',
    'RiskEvent',
    'RiskLimitBreach',
    'RiskAlert',
    
    # Pre-trade
    'UnifiedLimitChecker',
    'PositionLimitChecker',
    'ExposureLimitsChecker',
    'LiquidityChecker',
    
    # Real-time
    'RealTimeAnomalyDetector',
    'LiveRiskMonitor',
    'DynamicStopLossManager',
    'DrawdownController',
    'PositionLiquidator',
    'CircuitBreakerFacade',
    
    # Position sizing
    'VaRPositionSizer',
    
    # Metrics
    'RiskMetricsCalculator',
    'PortfolioRiskMetrics',
    
    # Integration
    'TradingEngineRiskIntegration',
    'RiskEventBridge',
    'RiskDashboardIntegration',
    
    # Post-trade
    'PostTradeAnalyzer',
    'TradeReview',
    'SlippageAnalyzer'
]