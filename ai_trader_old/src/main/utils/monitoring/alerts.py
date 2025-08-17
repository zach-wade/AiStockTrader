"""
Alert Management

Performance alert system with thresholds and callbacks.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
import logging
from typing import Any

from .types import Alert, AlertLevel, SystemResources

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages performance alerts and thresholds."""

    def __init__(self):
        """Initialize alert manager."""
        self.alert_thresholds: dict[str, dict[str, float]] = {}
        self.alert_callbacks: list[Callable[[Alert], None]] = []
        self.alerts_history: list[Alert] = []
        self.max_alerts_history = 1000

        logger.debug("Alert manager initialized")

    def set_alert_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """
        Set alert threshold for a metric.

        Args:
            metric_name: Name of the metric
            level: Alert level
            threshold: Threshold value
        """
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}

        self.alert_thresholds[metric_name][level.value] = threshold
        logger.debug(f"Set {level.value} threshold for {metric_name}: {threshold}")

    def remove_alert_threshold(self, metric_name: str, level: AlertLevel = None):
        """
        Remove alert threshold(s) for a metric.

        Args:
            metric_name: Name of the metric
            level: Specific alert level to remove (None removes all)
        """
        if metric_name not in self.alert_thresholds:
            return

        if level is None:
            del self.alert_thresholds[metric_name]
            logger.debug(f"Removed all thresholds for {metric_name}")
        elif level.value in self.alert_thresholds[metric_name]:
            del self.alert_thresholds[metric_name][level.value]
            logger.debug(f"Removed {level.value} threshold for {metric_name}")

            # Remove metric entry if no thresholds remain
            if not self.alert_thresholds[metric_name]:
                del self.alert_thresholds[metric_name]

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
        logger.debug("Added alert callback")

    def remove_alert_callback(self, callback: Callable[[Alert], None]):
        """Remove alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.debug("Removed alert callback")

    async def check_system_alerts(self, resources: SystemResources) -> list[Alert]:
        """
        Check system metrics against alert thresholds.

        Args:
            resources: System resources snapshot

        Returns:
            List of triggered alerts
        """
        alerts = []

        checks = [
            ("cpu_percent", resources.cpu_percent),
            ("memory_percent", resources.memory_percent),
            ("disk_usage_percent", resources.disk_usage_percent),
        ]

        for metric_name, value in checks:
            if metric_name in self.alert_thresholds:
                alert = await self._check_metric_threshold(metric_name, value)
                if alert:
                    alerts.append(alert)

        return alerts

    async def check_custom_metric_alerts(self, metric_name: str, value: float) -> Alert:
        """
        Check custom metric against alert thresholds.

        Args:
            metric_name: Name of the metric
            value: Metric value

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        return await self._check_metric_threshold(metric_name, value)

    async def _check_metric_threshold(self, metric_name: str, value: float) -> Alert:
        """
        Check a metric value against its thresholds.

        Args:
            metric_name: Name of the metric
            value: Metric value

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if metric_name not in self.alert_thresholds:
            return None

        thresholds = self.alert_thresholds[metric_name]

        # Check thresholds in order of severity (critical first)
        for level_name in ["critical", "error", "warning", "info"]:
            if level_name in thresholds:
                threshold = thresholds[level_name]
                if value >= threshold:
                    alert = Alert(
                        metric_name=metric_name,
                        level=AlertLevel(level_name),
                        message=f"{metric_name} is {value:.2f} (threshold: {threshold})",
                        value=value,
                        threshold=threshold,
                        timestamp=datetime.now(),
                    )

                    await self._fire_alert(alert)
                    return alert

        return None

    async def _fire_alert(self, alert: Alert):
        """
        Fire an alert to all registered callbacks.

        Args:
            alert: Alert to fire
        """
        # Add to history
        self.alerts_history.append(alert)

        # Trim history if needed
        if len(self.alerts_history) > self.max_alerts_history:
            self.alerts_history = self.alerts_history[-self.max_alerts_history :]

        # Log alert
        log_level = logging.INFO
        if alert.level == AlertLevel.WARNING:
            log_level = logging.WARNING
        elif alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            log_level = logging.ERROR

        logger.log(log_level, f"Performance alert: {alert.message}")

        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_alerts_summary(self) -> dict[str, Any]:
        """Get alerts summary."""
        if not self.alerts_history:
            return {
                "total_alerts": 0,
                "recent_alerts": [],
                "level_counts": {},
                "alert_thresholds": self.alert_thresholds,
            }

        # Count by level
        level_counts = defaultdict(int)
        for alert in self.alerts_history:
            level_counts[alert.level.value] += 1

        # Recent alerts (last 10)
        recent_alerts = [alert.to_dict() for alert in self.alerts_history[-10:]]

        return {
            "total_alerts": len(self.alerts_history),
            "level_counts": dict(level_counts),
            "recent_alerts": recent_alerts,
            "alert_thresholds": self.alert_thresholds,
        }

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts_history if alert.level == level]

    def get_recent_alerts(self, count: int = 10) -> list[Alert]:
        """Get recent alerts."""
        return self.alerts_history[-count:] if self.alerts_history else []

    def clear_alerts(self):
        """Clear all alerts history."""
        self.alerts_history.clear()
        logger.info("Cleared all alerts history")

    def get_metrics_with_thresholds(self) -> list[str]:
        """Get list of metrics that have thresholds configured."""
        return list(self.alert_thresholds.keys())

    def get_threshold_info(self, metric_name: str) -> dict[str, float]:
        """Get threshold information for a metric."""
        return self.alert_thresholds.get(metric_name, {})

    def set_default_system_thresholds(self):
        """Set default thresholds for system metrics."""
        # CPU thresholds
        self.set_alert_threshold("cpu_percent", AlertLevel.WARNING, 80.0)
        self.set_alert_threshold("cpu_percent", AlertLevel.ERROR, 90.0)
        self.set_alert_threshold("cpu_percent", AlertLevel.CRITICAL, 95.0)

        # Memory thresholds
        self.set_alert_threshold("memory_percent", AlertLevel.WARNING, 80.0)
        self.set_alert_threshold("memory_percent", AlertLevel.ERROR, 90.0)
        self.set_alert_threshold("memory_percent", AlertLevel.CRITICAL, 95.0)

        # Disk thresholds
        self.set_alert_threshold("disk_usage_percent", AlertLevel.WARNING, 80.0)
        self.set_alert_threshold("disk_usage_percent", AlertLevel.ERROR, 90.0)
        self.set_alert_threshold("disk_usage_percent", AlertLevel.CRITICAL, 95.0)

        logger.info("Set default system alert thresholds")
