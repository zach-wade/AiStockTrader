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
# PositionRiskMetrics,  # TODO: Need to implement
# VaRCalculator,  # TODO: Need to implement
# CVaRCalculator,  # TODO: Need to implement
# SharpRatioCalculator  # TODO: Need to implement
# Import integration
from .integration import RiskDashboardIntegration, RiskEventBridge, TradingEngineRiskIntegration

# KellyPositionSizer,  # TODO: Need to implement
# VolatilityPositionSizer,  # TODO: Need to implement
# OptimalFPositionSizer  # TODO: Need to implement
# Import risk metrics
from .metrics import PortfolioRiskMetrics, RiskMetricsCalculator

# Import position sizing
from .position_sizing import VaRPositionSizer

# Import post-trade analysis
from .post_trade import PostTradeAnalyzer, SlippageAnalyzer, TradeReview

# Import pre-trade risk management
from .pre_trade import (
    ExposureLimitsChecker,
    LiquidityChecker,
    PositionLimitChecker,
    UnifiedLimitChecker,
)

# Import real-time risk management
from .real_time import (
    CircuitBreakerFacade,
    DrawdownController,
    DynamicStopLossManager,
    LiveRiskMonitor,
    PositionLiquidator,
    RealTimeAnomalyDetector,
)
from .types import (
    RiskAlert,
    RiskCheckResult,
    RiskEvent,
    RiskEventType,
    RiskLevel,
    RiskLimitBreach,
    RiskMetric,
    RiskStatus,
)

__all__ = [
    # Core types
    "RiskLevel",
    "RiskEventType",
    "RiskMetric",
    "RiskStatus",
    "RiskCheckResult",
    "RiskEvent",
    "RiskLimitBreach",
    "RiskAlert",
    # Pre-trade
    "UnifiedLimitChecker",
    "PositionLimitChecker",
    "ExposureLimitsChecker",
    "LiquidityChecker",
    # Real-time
    "RealTimeAnomalyDetector",
    "LiveRiskMonitor",
    "DynamicStopLossManager",
    "DrawdownController",
    "PositionLiquidator",
    "CircuitBreakerFacade",
    # Position sizing
    "VaRPositionSizer",
    # Metrics
    "RiskMetricsCalculator",
    "PortfolioRiskMetrics",
    # Integration
    "TradingEngineRiskIntegration",
    "RiskEventBridge",
    "RiskDashboardIntegration",
    # Post-trade
    "PostTradeAnalyzer",
    "TradeReview",
    "SlippageAnalyzer",
]
