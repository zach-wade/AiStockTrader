"""
Global Performance Monitor

Global performance monitor instance and convenience functions.
"""

import logging
from typing import Optional, Dict, Callable, Any

from .monitor import PerformanceMonitor
from .migration import MigrationMonitor, create_monitor
from .types import MetricType

logger = logging.getLogger(__name__)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        # Try to get database pool if available
        db_pool = None
        try:
            from main.utils.database import get_default_db_pool
            db_pool = get_default_db_pool()
        except Exception:
            pass  # Database not available
        
        # Create migration monitor that can use enhanced features
        _global_monitor = create_monitor(db_pool=db_pool)
        logger.info(f"Created global performance monitor (enhanced={'enabled' if db_pool else 'disabled'})")
    return _global_monitor


def set_global_monitor(monitor: PerformanceMonitor):
    """Set the global performance monitor instance."""
    global _global_monitor
    _global_monitor = monitor
    logger.info("Set global performance monitor")


def reset_global_monitor():
    """Reset the global performance monitor instance."""
    global _global_monitor
    _global_monitor = None
    logger.info("Reset global performance monitor")


def is_global_monitor_initialized() -> bool:
    """Check if global performance monitor is initialized."""
    return _global_monitor is not None


# Convenience functions using global monitor
def record_metric(name: str, 
                 value: float, 
                 metric_type: MetricType = MetricType.GAUGE,
                 tags: Optional[Dict[str, str]] = None):
    """Record a metric using the global monitor."""
    monitor = get_global_monitor()
    monitor.record_metric(name, value, metric_type, tags)


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution using global monitor."""
    monitor = get_global_monitor()
    return monitor.time_function(func)


async def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Async timer context manager using global monitor."""
    monitor = get_global_monitor()
    async with monitor.timer(name, tags):
        yield


def sync_timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Synchronous timer context manager using global monitor."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _timer():
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            record_metric(f"{name}.duration", duration, MetricType.HISTOGRAM, tags)
    
    return _timer()


def start_monitoring():
    """Start global performance monitoring."""
    monitor = get_global_monitor()
    return monitor.start_monitoring()


def stop_monitoring():
    """Stop global performance monitoring."""
    monitor = get_global_monitor()
    return monitor.stop_monitoring()


def get_system_summary() -> Dict[str, Any]:
    """Get system performance summary from global monitor."""
    monitor = get_global_monitor()
    return monitor.get_system_summary()


def get_function_summary() -> Dict[str, Any]:
    """Get function performance summary from global monitor."""
    monitor = get_global_monitor()
    return monitor.get_function_summary()


def get_alerts_summary() -> Dict[str, Any]:
    """Get alerts summary from global monitor."""
    monitor = get_global_monitor()
    return monitor.get_alerts_summary()


def set_default_thresholds():
    """Set default alert thresholds on global monitor."""
    monitor = get_global_monitor()
    monitor.set_default_thresholds()


def clear_metrics():
    """Clear all metrics from global monitor."""
    monitor = get_global_monitor()
    monitor.clear_metrics()


def export_metrics(format: str = 'json') -> str:
    """Export metrics from global monitor."""
    monitor = get_global_monitor()
    return monitor.export_metrics(format)