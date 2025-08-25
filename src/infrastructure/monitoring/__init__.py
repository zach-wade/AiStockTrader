"""
Infrastructure Monitoring Module

Comprehensive monitoring infrastructure for the AI trading system including:
- OpenTelemetry distributed tracing
- Structured logging with correlation IDs
- Business and technical metrics collection
- Health check endpoints
- Performance monitoring and APM

This module provides production-ready observability for trading operations,
ensuring visibility into system performance and trading activities.
"""

from .health import MarketHoursHealthCheck, TradingHealthChecker
from .logging import get_correlation_id, mask_sensitive_data, setup_structured_logging
from .metrics import TradingMetrics, get_trading_metrics
from .performance import PerformanceMonitor, trading_performance
from .telemetry import TradingTelemetry, get_current_span, trading_tracer

__all__ = [
    "TradingTelemetry",
    "trading_tracer",
    "get_current_span",
    "setup_structured_logging",
    "get_correlation_id",
    "mask_sensitive_data",
    "TradingMetrics",
    "get_trading_metrics",
    "TradingHealthChecker",
    "MarketHoursHealthCheck",
    "PerformanceMonitor",
    "trading_performance",
]
