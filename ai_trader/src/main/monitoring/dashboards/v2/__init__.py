"""
V2 Dashboard System - Modern, non-blocking dashboard implementation.

This module provides the new dashboard system with:
- TradingDashboardV2: Comprehensive trading monitoring
- SystemDashboardV2: System health and infrastructure monitoring
- DashboardManager: Process-based lifecycle management
"""

from .trading_dashboard_v2 import TradingDashboardV2
from .system_dashboard_v2 import SystemDashboardV2
from .dashboard_manager import DashboardManager, DashboardState, DashboardInfo

__all__ = [
    'TradingDashboardV2',
    'SystemDashboardV2',
    'DashboardManager',
    'DashboardState',
    'DashboardInfo'
]