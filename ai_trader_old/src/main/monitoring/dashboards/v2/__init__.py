"""
V2 Dashboard System - Modern, non-blocking dashboard implementation.

This module provides the new dashboard system with:
- TradingDashboardV2: Comprehensive trading monitoring
- SystemDashboardV2: System health and infrastructure monitoring
- DashboardManager: Process-based lifecycle management
"""

from .dashboard_manager import DashboardInfo, DashboardManager, DashboardState
from .system_dashboard_v2 import SystemDashboardV2
from .trading_dashboard_v2 import TradingDashboardV2

__all__ = [
    "TradingDashboardV2",
    "SystemDashboardV2",
    "DashboardManager",
    "DashboardState",
    "DashboardInfo",
]
