"""
AI Trader Monitoring System

Real-time monitoring and observability framework providing:
- Trading dashboards
- Performance metrics
- Alert systems
- Logging infrastructure
- System health monitoring
"""

# V2 Dashboards
try:
    from .dashboards.v2 import DashboardManager, SystemDashboardV2, TradingDashboardV2
except ImportError:
    TradingDashboardV2 = None
    SystemDashboardV2 = None
    DashboardManager = None

# Import only existing modules to avoid import errors
try:
    from .metrics.collector import MetricsCollector
    from .metrics.unified_metrics import UnifiedMetrics
except ImportError:
    UnifiedMetrics = None
    MetricsCollector = None

# Import timer and global monitor from utils.monitoring
try:
    # Local imports
    from main.utils.monitoring import timer
    from main.utils.monitoring.global_monitor import get_global_monitor
except ImportError:
    timer = None
    get_global_monitor = None

try:
    from .alerts.unified_alerts import UnifiedAlertSystem as UnifiedAlerts
except ImportError:
    UnifiedAlerts = None

try:
    from .performance import PerformanceTracker, UnifiedPerformanceTracker
except ImportError:
    UnifiedPerformanceTracker = None
    PerformanceTracker = None

__all__ = [
    # V2 Dashboards
    "TradingDashboardV2",
    "SystemDashboardV2",
    "DashboardManager",
    # Optional modules (may be None if not available)
    "UnifiedMetrics",
    "MetricsCollector",
    "UnifiedAlerts",
    "UnifiedPerformanceTracker",
    "PerformanceTracker",
    "timer",
    "get_global_monitor",
]
