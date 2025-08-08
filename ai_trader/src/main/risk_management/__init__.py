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
    ExposureLimitChecker,
    LiquidityChecker
)

# Import real-time risk management
from .real_time import (
    RealTimeAnomalyDetector,
    LiveRiskMonitor,
    StopLossManager,
    DrawdownController,
    PositionLiquidator,
    CircuitBreakerFacade
)

# Import position sizing
from .position_sizing import (
    VaRPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
    OptimalFPositionSizer
)

# Import risk metrics
from .metrics import (
    RiskMetricsCalculator,
    PortfolioRiskMetrics,
    PositionRiskMetrics,
    VaRCalculator,
    CVaRCalculator,
    SharpRatioCalculator
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
    RiskPerformanceAnalyzer,
    ComplianceChecker
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
    'ExposureLimitChecker',
    'LiquidityChecker',
    
    # Real-time
    'RealTimeAnomalyDetector',
    'LiveRiskMonitor',
    'StopLossManager',
    'DrawdownController',
    'PositionLiquidator',
    'CircuitBreakerFacade',
    
    # Position sizing
    'VaRPositionSizer',
    'KellyPositionSizer',
    'VolatilityPositionSizer',
    'OptimalFPositionSizer',
    
    # Metrics
    'RiskMetricsCalculator',
    'PortfolioRiskMetrics',
    'PositionRiskMetrics',
    'VaRCalculator',
    'CVaRCalculator',
    'SharpRatioCalculator',
    
    # Integration
    'TradingEngineRiskIntegration',
    'RiskEventBridge',
    'RiskDashboardIntegration',
    
    # Post-trade
    'PostTradeAnalyzer',
    'TradeReview',
    'RiskPerformanceAnalyzer',
    'ComplianceChecker'
]