"""
Performance Monitoring and APM for AI Trading System

Application Performance Monitoring with:
- Request tracing and timing
- Database query performance
- API response times
- Memory and resource usage
- Bottleneck identification
- Trading-specific performance metrics

This module has been refactored into focused components in the performance/ subdirectory.
The classes here are maintained for backward compatibility.
"""

# Import refactored components
from .performance import (
    APIPerformanceTracker,
    CPUProfiler,
    DatabaseQueryProfiler,
    MemoryProfiler,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
    get_performance_monitor,
    initialize_performance_monitor,
    profile_api_endpoint,
    profile_database_query,
    profile_performance,
    trading_performance,
)

# Re-export for backward compatibility
__all__ = [
    "PerformanceMetric",
    "PerformanceReport",
    "MemoryProfiler",
    "CPUProfiler",
    "DatabaseQueryProfiler",
    "APIPerformanceTracker",
    "PerformanceMonitor",
    "get_performance_monitor",
    "initialize_performance_monitor",
    "trading_performance",
    "profile_performance",
    "profile_database_query",
    "profile_api_endpoint",
]
