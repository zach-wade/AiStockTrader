"""
Performance Monitor

Main performance monitoring system coordinating all monitoring components.
"""

# Standard library imports
import asyncio
from collections import defaultdict, deque
from datetime import datetime
import json
import logging
from typing import Any

from .collectors import SystemMetricsCollector

# AlertManager import removed - not used in this module
from .function_tracker import FunctionTracker
from .types import AlertLevel, MetricType, PerformanceMetric

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.

    Coordinates system resource monitoring, function timing, custom metrics,
    and alerting capabilities.
    """

    def __init__(
        self,
        history_size: int = 1000,
        monitoring_interval: float = 5.0,
        enable_system_monitoring: bool = True,
    ):
        """
        Initialize performance monitor.

        Args:
            history_size: Number of historical metrics to keep
            monitoring_interval: System monitoring interval in seconds
            enable_system_monitoring: Whether to monitor system resources
        """
        self.history_size = history_size
        self.monitoring_interval = monitoring_interval
        self.enable_system_monitoring = enable_system_monitoring

        # Components
        self.collector = SystemMetricsCollector()
        # AlertManager removed - not used in this module
        # self.alert_manager = AlertManager()
        self.function_tracker = FunctionTracker()

        # Set up function tracker with metric recorder
        self.function_tracker.set_metric_recorder(self.record_metric)

        # Metrics storage
        self.metrics: deque = deque(maxlen=history_size)
        self.system_metrics: deque = deque(maxlen=history_size)
        self.custom_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))

        # Monitoring control
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_active = False
        self._lock = asyncio.Lock()

        logger.info("Performance monitor initialized")

    async def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self._monitoring_active = True
        if self.enable_system_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started system performance monitoring")

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped system performance monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._collect_and_check_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _collect_and_check_system_metrics(self):
        """Collect system metrics and check for alerts."""
        try:
            # Collect system metrics
            system_resources = self.collector.collect_system_metrics()

            if system_resources is None:
                return  # Skip if no data (establishing baseline)

            # Store metrics
            async with self._lock:
                self.system_metrics.append(system_resources)

            # Check for alerts
            alerts = await self.alert_manager.check_system_alerts(system_resources)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
    ):
        """
        Record a custom metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for the metric
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
        )

        self.metrics.append(metric)
        self.custom_metrics[name].append(metric)

        # Alert manager functionality removed to avoid circular dependencies
        # Custom metric alerts should be handled at a higher level
        # if name in self.alert_manager.alert_thresholds:
        #     asyncio.create_task(self.alert_manager.check_custom_metric_alerts(name, value))

    def set_alert_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """Set alert threshold for a metric."""
        # Alert manager functionality removed
        # self.alert_manager.set_alert_threshold(metric_name, level, threshold)
        pass

    def add_alert_callback(self, callback):
        """Add alert callback function."""
        self.alert_manager.add_alert_callback(callback)

    def remove_alert_callback(self, callback):
        """Remove alert callback function."""
        self.alert_manager.remove_alert_callback(callback)

    def time_function(self, func):
        """Decorator to time function execution."""
        return self.function_tracker.time_function(func)

    async def timer(self, name: str, tags: dict[str, str] | None = None):
        """Timer context manager."""
        async with self.function_tracker.timer(name, tags):
            yield

    def get_system_summary(self) -> dict[str, Any]:
        """Get current system performance summary."""
        if not self.system_metrics:
            return {"status": "no_data"}

        latest = self.system_metrics[-1]

        return {
            "timestamp": latest.timestamp.isoformat(),
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "memory_used_mb": latest.memory_used_mb,
            "memory_available_mb": latest.memory_available_mb,
            "disk_usage_percent": latest.disk_usage_percent,
            "disk_free_gb": latest.disk_free_gb,
            "network_bytes_sent": latest.network_bytes_sent,
            "network_bytes_recv": latest.network_bytes_recv,
            "monitoring_active": self._monitoring_active,
        }

    def get_function_summary(self) -> dict[str, Any]:
        """Get function performance summary."""
        return self.function_tracker.get_function_summary()

    def get_alerts_summary(self) -> dict[str, Any]:
        """Get alerts summary."""
        return self.alert_manager.get_alerts_summary()

    def get_metrics_by_name(self, name: str) -> list[dict[str, Any]]:
        """Get all metrics for a specific name."""
        if name not in self.custom_metrics:
            return []

        return [metric.to_dict() for metric in self.custom_metrics[name]]

    def get_system_metrics_history(self, count: int = None) -> list[dict[str, Any]]:
        """Get system metrics history."""
        metrics = list(self.system_metrics)
        if count:
            metrics = metrics[-count:]
        return [metric.to_dict() for metric in metrics]

    def get_custom_metrics_summary(self) -> dict[str, Any]:
        """Get summary of custom metrics."""
        summary = {}

        for name, metrics in self.custom_metrics.items():
            if not metrics:
                continue

            values = [m.value for m in metrics]
            summary[name] = {
                "count": len(metrics),
                "latest_value": values[-1] if values else None,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None,
                "avg_value": sum(values) / len(values) if values else None,
                "latest_timestamp": metrics[-1].timestamp.isoformat() if metrics else None,
            }

        return summary

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "monitoring_active": self._monitoring_active,
            "monitoring_interval": self.monitoring_interval,
            "history_size": self.history_size,
            "system_metrics_count": len(self.system_metrics),
            "custom_metrics_count": len(self.custom_metrics),
            "total_metrics_count": len(self.metrics),
            "tracked_functions_count": self.function_tracker.get_function_count(),
            "alerts_count": len(self.alert_manager.alerts_history),
            "alert_thresholds": self.alert_manager.alert_thresholds,
            "system_summary": self.get_system_summary(),
            "function_summary": self.get_function_summary(),
            "alerts_summary": self.get_alerts_summary(),
        }

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in specified format.

        Args:
            format: Export format ('json', 'csv', 'html')

        Returns:
            Exported metrics as string
        """
        if format == "json":
            return json.dumps(
                {
                    "system_metrics": [m.to_dict() for m in self.system_metrics],
                    "custom_metrics": {
                        name: [m.to_dict() for m in metrics]
                        for name, metrics in self.custom_metrics.items()
                    },
                    "function_metrics": self.get_function_summary(),
                    "alerts": [a.to_dict() for a in self.alert_manager.alerts_history],
                    "export_timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )

        elif format == "csv":
            # Standard library imports
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # System metrics
            writer.writerow(
                ["Metric Type", "Timestamp", "CPU %", "Memory %", "Disk %", "Network MB/s"]
            )
            for metric in self.system_metrics:
                writer.writerow(
                    [
                        "System",
                        metric.timestamp.isoformat(),
                        f"{metric.cpu_percent:.2f}",
                        f"{metric.memory_percent:.2f}",
                        f"{metric.disk_percent:.2f}",
                        f"{metric.network_bytes_sent / 1024 / 1024:.2f}",
                    ]
                )

            # Custom metrics
            writer.writerow([])  # Empty row
            writer.writerow(["Custom Metrics"])
            writer.writerow(["Name", "Timestamp", "Value"])
            for name, metrics in self.custom_metrics.items():
                for metric in metrics:
                    writer.writerow([name, metric.timestamp.isoformat(), metric.value])

            return output.getvalue()

        elif format == "html":
            html_parts = ["<!DOCTYPE html><html><head><title>AI Trader Metrics Report</title>"]
            html_parts.append("<style>")
            html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
            html_parts.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
            html_parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_parts.append("th { background-color: #4CAF50; color: white; }")
            html_parts.append("tr:nth-child(even) { background-color: #f2f2f2; }")
            html_parts.append("h2 { color: #333; }")
            html_parts.append(".alert { padding: 10px; margin: 10px 0; border-radius: 4px; }")
            html_parts.append(
                ".alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }"
            )
            html_parts.append(
                ".alert-critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }"
            )
            html_parts.append("</style></head><body>")

            html_parts.append("<h1>AI Trader Metrics Report</h1>")
            html_parts.append(f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

            # System metrics
            html_parts.append("<h2>System Metrics</h2>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Timestamp</th><th>CPU %</th><th>Memory %</th><th>Disk %</th></tr>"
            )
            for metric in list(self.system_metrics)[-10:]:  # Last 10 entries
                html_parts.append("<tr>")
                html_parts.append(f'<td>{metric.timestamp.strftime("%H:%M:%S")}</td>')
                html_parts.append(f"<td>{metric.cpu_percent:.1f}</td>")
                html_parts.append(f"<td>{metric.memory_percent:.1f}</td>")
                html_parts.append(f"<td>{metric.disk_percent:.1f}</td>")
                html_parts.append("</tr>")
            html_parts.append("</table>")

            # Function metrics
            func_summary = self.get_function_summary()
            if func_summary:
                html_parts.append("<h2>Function Performance</h2>")
                html_parts.append("<table>")
                html_parts.append(
                    "<tr><th>Function</th><th>Calls</th><th>Avg Time (s)</th><th>Total Time (s)</th></tr>"
                )
                for name, stats in func_summary.items():
                    html_parts.append("<tr>")
                    html_parts.append(f"<td>{name}</td>")
                    html_parts.append(f'<td>{stats["calls"]}</td>')
                    html_parts.append(f'<td>{stats["avg_time"]:.3f}</td>')
                    html_parts.append(f'<td>{stats["total_time"]:.3f}</td>')
                    html_parts.append("</tr>")
                html_parts.append("</table>")

            # Recent alerts
            recent_alerts = list(self.alert_manager.alerts_history)[-10:]
            if recent_alerts:
                html_parts.append("<h2>Recent Alerts</h2>")
                for alert in recent_alerts:
                    alert_class = (
                        "alert-critical"
                        if "critical" in alert.severity.lower()
                        else "alert-warning"
                    )
                    html_parts.append(f'<div class="alert {alert_class}">')
                    html_parts.append(
                        f'<strong>{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</strong> - '
                    )
                    html_parts.append(f"{alert.metric_name}: {alert.message}")
                    html_parts.append("</div>")

            html_parts.append("</body></html>")
            return "".join(html_parts)

        else:
            raise NotImplementedError(
                f"Export format '{format}' not implemented. Supported formats: json, csv, html"
            )

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        self.system_metrics.clear()
        self.custom_metrics.clear()
        self.function_tracker.reset_function_metrics()
        self.alert_manager.clear_alerts()
        logger.info("Cleared all performance metrics")

    def set_default_thresholds(self):
        """Set default alert thresholds."""
        self.alert_manager.set_default_system_thresholds()

    def get_collector(self) -> SystemMetricsCollector:
        """Get system metrics collector."""
        return self.collector

    # AlertManager methods removed - not used in this module
    # def get_alert_manager(self) -> AlertManager:
    #     """Get alert manager."""
    #     return self.alert_manager

    def get_function_tracker(self) -> FunctionTracker:
        """Get function tracker."""
        return self.function_tracker
