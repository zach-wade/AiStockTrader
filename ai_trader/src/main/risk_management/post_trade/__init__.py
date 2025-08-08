"""
Post-trade analysis and compliance module.

This module provides tools for analyzing executed trades,
ensuring compliance, and generating risk performance reports.
"""

from .post_trade_analyzer import (
    PostTradeAnalyzer,
    TradeAnalysis,
    ExecutionQuality,
    SlippageAnalysis
)

from .trade_review import (
    TradeReview,
    ReviewStatus,
    ReviewAction,
    ComplianceFlag
)

from .risk_performance import (
    RiskPerformanceAnalyzer,
    RiskPerformanceMetrics,
    RiskAttribution,
    PerformancePeriod
)

from .compliance_checker import (
    ComplianceChecker,
    ComplianceRule,
    ComplianceResult,
    ViolationType,
    RegulatoryFramework
)

from .reconciliation import (
    TradeReconciliation,
    ReconciliationStatus,
    PositionReconciliation,
    ReconciliationReport
)

from .reporting import (
    PostTradeReporter,
    ReportType,
    ReportFormat,
    ReportSchedule,
    RiskReport
)

from .analytics import (
    PostTradeAnalytics,
    TradingPatternAnalysis,
    BehavioralMetrics,
    PerformanceAttribution
)

__all__ = [
    # Post-trade analysis
    'PostTradeAnalyzer',
    'TradeAnalysis',
    'ExecutionQuality',
    'SlippageAnalysis',
    
    # Trade review
    'TradeReview',
    'ReviewStatus',
    'ReviewAction',
    'ComplianceFlag',
    
    # Risk performance
    'RiskPerformanceAnalyzer',
    'RiskPerformanceMetrics',
    'RiskAttribution',
    'PerformancePeriod',
    
    # Compliance
    'ComplianceChecker',
    'ComplianceRule',
    'ComplianceResult',
    'ViolationType',
    'RegulatoryFramework',
    
    # Reconciliation
    'TradeReconciliation',
    'ReconciliationStatus',
    'PositionReconciliation',
    'ReconciliationReport',
    
    # Reporting
    'PostTradeReporter',
    'ReportType',
    'ReportFormat',
    'ReportSchedule',
    'RiskReport',
    
    # Analytics
    'PostTradeAnalytics',
    'TradingPatternAnalysis',
    'BehavioralMetrics',
    'PerformanceAttribution'
]