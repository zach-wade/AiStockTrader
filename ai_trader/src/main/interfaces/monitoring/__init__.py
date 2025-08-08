"""
Monitoring interfaces for dashboards and related components.

This module defines the protocols and contracts for monitoring system components
including dashboards, dashboard managers, and monitoring services.
"""

from .dashboard import (
    IDashboard,
    IDashboardManager,
    DashboardStatus,
    DashboardConfig,
    IMetricsCollector,
    IArchiveMetricsCollector
)

__all__ = [
    'IDashboard',
    'IDashboardManager',
    'DashboardStatus',
    'DashboardConfig',
    'IMetricsCollector',
    'IArchiveMetricsCollector'
]