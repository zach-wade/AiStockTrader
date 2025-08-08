"""
Performance Monitoring Package

Modular performance tracking and analysis system.
"""

from .performance_tracker import PerformanceTracker, create_performance_tracker
from .models import (
    PerformanceMetrics, PerformanceMetricType, TimeFrame,
    TradeRecord, SystemPerformanceRecord, PerformanceAlert, PerformanceAlertData
)
from .calculators import (
    ReturnCalculator, RiskCalculator, RiskAdjustedCalculator, TradingMetricsCalculator
)
from .alerts.alert_manager import AlertManager

# Backward compatibility alias
UnifiedPerformanceTracker = PerformanceTracker

__all__ = [
    # Main classes
    'PerformanceTracker',
    'create_performance_tracker',
    
    # Models
    'PerformanceMetrics',
    'PerformanceMetricType',
    'TimeFrame',
    'TradeRecord',
    'SystemPerformanceRecord',
    'PerformanceAlert',
    'PerformanceAlertData',
    
    # Calculators
    'ReturnCalculator',
    'RiskCalculator',
    'RiskAdjustedCalculator',
    'TradingMetricsCalculator',
    
    # Alerts
    'AlertManager',
    
    # Backward compatibility
    'UnifiedPerformanceTracker'
]