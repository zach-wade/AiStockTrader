"""
Monitoring interfaces for dashboards and related components.

This module defines the protocols and contracts for monitoring system components
including dashboards, dashboard managers, and monitoring services.
"""

from .dashboard import (
    DashboardConfig,
    DashboardStatus,
    IArchiveMetricsCollector,
    IDashboard,
    IDashboardManager,
    IMetricsCollector,
)

__all__ = [
    "IDashboard",
    "IDashboardManager",
    "DashboardStatus",
    "DashboardConfig",
    "IMetricsCollector",
    "IArchiveMetricsCollector",
]
