"""
Performance Monitoring Package

Modular performance tracking and analysis system.
"""

from .alerts.alert_manager import AlertManager
from .calculators import (
    ReturnCalculator,
    RiskAdjustedCalculator,
    RiskCalculator,
    TradingMetricsCalculator,
)
from .models import (
    PerformanceAlert,
    PerformanceAlertData,
    PerformanceMetrics,
    PerformanceMetricType,
    SystemPerformanceRecord,
    TimeFrame,
    TradeRecord,
)
from .performance_tracker import PerformanceTracker, create_performance_tracker

# Backward compatibility alias
UnifiedPerformanceTracker = PerformanceTracker

__all__ = [
    # Main classes
    "PerformanceTracker",
    "create_performance_tracker",
    # Models
    "PerformanceMetrics",
    "PerformanceMetricType",
    "TimeFrame",
    "TradeRecord",
    "SystemPerformanceRecord",
    "PerformanceAlert",
    "PerformanceAlertData",
    # Calculators
    "ReturnCalculator",
    "RiskCalculator",
    "RiskAdjustedCalculator",
    "TradingMetricsCalculator",
    # Alerts
    "AlertManager",
    # Backward compatibility
    "UnifiedPerformanceTracker",
]
