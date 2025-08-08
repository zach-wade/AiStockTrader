"""
Metrics Adapter

Adapts the global monitor's record_metric function to the IMetricsRecorder interface.
"""

from typing import Dict, Optional
from datetime import datetime

from main.interfaces.metrics import IMetricsRecorder, MetricType as IMetricType
from .types import MetricType
from .global_monitor import get_global_monitor


class GlobalMetricsAdapter:
    """
    Adapter that implements IMetricsRecorder using the global monitor.
    
    This allows components to use the interface without directly
    depending on the global monitor implementation.
    """
    
    def __init__(self):
        """Initialize the adapter."""
        self._monitor = None
    
    def _get_monitor(self):
        """Lazy load the monitor to avoid initialization issues."""
        if self._monitor is None:
            self._monitor = get_global_monitor()
        return self._monitor
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: IMetricType = IMetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric using the global monitor."""
        # Convert interface metric type to internal type
        internal_type = self._convert_metric_type(metric_type)
        
        # The global monitor's record_metric doesn't support timestamp
        # but we can ignore it for now as it uses current time by default
        monitor = self._get_monitor()
        monitor.record_metric(name, value, internal_type, tags)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, IMetricType.COUNTER, tags)
    
    def update_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Update a gauge metric."""
        self.record_metric(name, value, IMetricType.GAUGE, tags)
    
    def record_duration(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a duration metric."""
        self.record_metric(name, duration_ms, IMetricType.TIMER, tags)
    
    def _convert_metric_type(self, metric_type: IMetricType) -> MetricType:
        """Convert interface metric type to internal metric type."""
        mapping = {
            IMetricType.COUNTER: MetricType.COUNTER,
            IMetricType.GAUGE: MetricType.GAUGE,
            IMetricType.HISTOGRAM: MetricType.HISTOGRAM,
            IMetricType.TIMER: MetricType.TIMER,
        }
        return mapping.get(metric_type, MetricType.GAUGE)


# Global instance for convenience
_global_metrics_adapter = GlobalMetricsAdapter()


def get_metrics_recorder() -> IMetricsRecorder:
    """Get the global metrics recorder instance."""
    return _global_metrics_adapter