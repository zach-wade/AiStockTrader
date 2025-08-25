"""
Monitoring and metrics integration for rate limiting system.

Provides comprehensive monitoring, alerting, and metrics collection
for rate limiting operations.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .config import RateLimitConfig

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RateLimitMetric:
    """Rate limiting metric data point."""

    timestamp: datetime
    limiter_id: str
    identifier: str
    action: str  # "allowed", "denied", "reset", "cleanup"
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitAlert:
    """Rate limiting alert."""

    level: AlertLevel
    message: str
    limiter_id: str
    identifier: str | None
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates rate limiting metrics."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._metrics: dict[str, list[RateLimitMetric]] = defaultdict(list)
        self._aggregated_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
        self._last_aggregation = time.time()
        self._lock = threading.RLock()

        # Metrics retention (keep last 1 hour by default)
        self.metrics_retention_seconds = 3600

        # Performance counters
        self._counters: dict[str, int] = defaultdict(int)
        self._timing_windows: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )  # Last 1000 measurements

    def record_metric(self, metric: RateLimitMetric) -> None:
        """Record a rate limiting metric."""
        if not self.config.enable_monitoring:
            return

        with self._lock:
            key = f"{metric.limiter_id}:{metric.action}"
            self._metrics[key].append(metric)

            # Update counters
            self._counters[f"{metric.limiter_id}:total"] += metric.count
            self._counters[f"{metric.limiter_id}:{metric.action}"] += metric.count
            self._counters["global:total"] += metric.count
            self._counters[f"global:{metric.action}"] += metric.count

            # Clean up old metrics
            self._cleanup_old_metrics()

    def record_timing(
        self, operation: str, duration_ms: float, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record timing information for operations."""
        if not self.config.enable_monitoring:
            return

        with self._lock:
            self._timing_windows[operation].append(
                {"duration_ms": duration_ms, "timestamp": time.time(), "metadata": metadata or {}}
            )

    def record_rate_limit_check(
        self,
        limiter_id: str,
        identifier: str,
        allowed: bool,
        current_count: int,
        limit: int,
        duration_ms: float,
    ) -> None:
        """Record a rate limit check with all relevant metrics."""
        # Record basic metric
        action = "allowed" if allowed else "denied"
        metric = RateLimitMetric(
            timestamp=datetime.utcnow(),
            limiter_id=limiter_id,
            identifier=identifier,
            action=action,
            metadata={
                "current_count": current_count,
                "limit": limit,
                "utilization": current_count / limit if limit > 0 else 0,
            },
        )
        self.record_metric(metric)

        # Record timing
        self.record_timing(
            f"rate_limit_check:{limiter_id}",
            duration_ms,
            {"allowed": allowed, "utilization": current_count / limit if limit > 0 else 0},
        )

        # Check for alerting thresholds
        utilization = current_count / limit if limit > 0 else 0
        if utilization > self.config.alert_threshold:
            self._trigger_utilization_alert(limiter_id, identifier, utilization)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get aggregated metrics summary."""
        with self._lock:
            current_time = time.time()

            # Calculate rates (requests per minute)
            window_minutes = 5  # 5-minute window
            window_start = current_time - (window_minutes * 60)

            summary: dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "counters": dict(self._counters),
                "rates": {},
                "timing": {},
                "utilization": {},
                "health": self._calculate_health_metrics(),
            }

            # Calculate rates for different time windows
            for key, metrics in self._metrics.items():
                recent_metrics = [m for m in metrics if m.timestamp.timestamp() > window_start]

                if recent_metrics:
                    rate = len(recent_metrics) / window_minutes
                    summary["rates"][f"{key}_per_minute"] = rate

            # Calculate timing statistics
            for operation, timings in self._timing_windows.items():
                if timings:
                    durations = [t["duration_ms"] for t in timings]
                    summary["timing"][operation] = {
                        "count": len(durations),
                        "avg_ms": sum(durations) / len(durations),
                        "min_ms": min(durations),
                        "max_ms": max(durations),
                        "p95_ms": self._percentile(durations, 95),
                        "p99_ms": self._percentile(durations, 99),
                    }

            return summary

    def get_limiter_metrics(self, limiter_id: str) -> dict[str, Any]:
        """Get metrics for a specific rate limiter."""
        with self._lock:
            metrics: dict[str, Any] = {}

            # Get counters for this limiter
            for key, value in self._counters.items():
                if key.startswith(f"{limiter_id}:"):
                    action = key.split(":", 1)[1]
                    metrics[action] = value

            # Get recent metrics
            recent_metrics = []
            for key, metric_list in self._metrics.items():
                if key.startswith(f"{limiter_id}:"):
                    recent_metrics.extend(metric_list[-100:])  # Last 100 metrics

            # Calculate utilization statistics
            utilizations = []
            for metric in recent_metrics:
                if "utilization" in metric.metadata:
                    utilizations.append(metric.metadata["utilization"])

            if utilizations:
                metrics["utilization"] = {
                    "avg": sum(utilizations) / len(utilizations),
                    "max": max(utilizations),
                    "current": utilizations[-1] if utilizations else 0,
                }

            return metrics

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory growth."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.metrics_retention_seconds)

        for key in list(self._metrics.keys()):
            self._metrics[key] = [m for m in self._metrics[key] if m.timestamp > cutoff_time]

            # Remove empty entries
            if not self._metrics[key]:
                del self._metrics[key]

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _calculate_health_metrics(self) -> dict[str, Any]:
        """Calculate overall health metrics."""
        current_time = time.time()
        window_start = current_time - 300  # 5-minute window

        total_requests = 0
        denied_requests = 0

        for key, metrics in self._metrics.items():
            recent_metrics = [m for m in metrics if m.timestamp.timestamp() > window_start]

            for metric in recent_metrics:
                total_requests += metric.count
                if metric.action == "denied":
                    denied_requests += metric.count

        error_rate = denied_requests / total_requests if total_requests > 0 else 0

        return {
            "total_requests_5min": total_requests,
            "denied_requests_5min": denied_requests,
            "error_rate_5min": error_rate,
            "healthy": error_rate < 0.1,  # Less than 10% error rate
        }

    def _trigger_utilization_alert(
        self, limiter_id: str, identifier: str, utilization: float
    ) -> None:
        """Trigger alert for high utilization."""
        # This would integrate with your alerting system
        logger.warning(
            f"High rate limit utilization: {limiter_id} for {identifier} at {utilization:.1%}"
        )


class AlertManager:
    """Manages rate limiting alerts and notifications."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._alerts: list[RateLimitAlert] = []
        self._alert_handlers: list[Callable[[RateLimitAlert], None]] = []
        self._alert_counts: dict[str, int] = defaultdict(int)
        self._last_alert_times: dict[str, float] = {}

        # Alert throttling (don't spam alerts)
        self.alert_throttle_seconds = 300  # 5 minutes

    def add_alert_handler(self, handler: Callable[[RateLimitAlert], None]) -> None:
        """Add an alert handler function."""
        self._alert_handlers.append(handler)

    def trigger_alert(self, alert: RateLimitAlert) -> None:
        """Trigger an alert with throttling."""
        alert_key = f"{alert.limiter_id}:{alert.level.value}"
        current_time = time.time()

        # Check throttling
        if alert_key in self._last_alert_times:
            time_since_last = current_time - self._last_alert_times[alert_key]
            if time_since_last < self.alert_throttle_seconds:
                return  # Throttled

        self._last_alert_times[alert_key] = current_time
        self._alert_counts[alert_key] += 1
        self._alerts.append(alert)

        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        # Log the alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(alert.level, logging.INFO)

        logger.log(log_level, f"Rate limit alert: {alert.message}")

    def get_recent_alerts(self, limit: int = 100) -> list[RateLimitAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics."""
        return {
            "total_alerts": len(self._alerts),
            "alert_counts_by_type": dict(self._alert_counts),
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "limiter_id": alert.limiter_id,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self._alerts[-10:]  # Last 10 alerts
            ],
        }


class RateLimitMonitor:
    """Main monitoring class that orchestrates metrics and alerts."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)

        # Setup default alert handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default alert handlers."""

        def log_alert_handler(alert: RateLimitAlert) -> None:
            """Default handler that logs alerts."""
            logger.info(f"Rate limit alert [{alert.level.value}]: {alert.message}")

        self.alert_manager.add_alert_handler(log_alert_handler)

    def record_rate_limit_check(
        self,
        limiter_id: str,
        identifier: str,
        allowed: bool,
        current_count: int,
        limit: int,
        duration_ms: float,
        retry_after: int | None = None,
    ) -> None:
        """Record a rate limit check with comprehensive monitoring."""
        # Record metrics
        self.metrics_collector.record_rate_limit_check(
            limiter_id, identifier, allowed, current_count, limit, duration_ms
        )

        # Check for alerts
        utilization = current_count / limit if limit > 0 else 0

        if not allowed:
            # Rate limit exceeded
            self.alert_manager.trigger_alert(
                RateLimitAlert(
                    level=AlertLevel.WARNING,
                    message=f"Rate limit exceeded for {limiter_id} (identifier: {identifier})",
                    limiter_id=limiter_id,
                    identifier=identifier,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "current_count": current_count,
                        "limit": limit,
                        "utilization": utilization,
                        "retry_after": retry_after,
                    },
                )
            )

        elif utilization > 0.9:  # 90% utilization
            # High utilization warning
            self.alert_manager.trigger_alert(
                RateLimitAlert(
                    level=AlertLevel.INFO,
                    message=f"High utilization ({utilization:.1%}) for {limiter_id}",
                    limiter_id=limiter_id,
                    identifier=identifier,
                    timestamp=datetime.utcnow(),
                    metadata={"utilization": utilization},
                )
            )

    def record_storage_operation(self, operation: str, duration_ms: float, success: bool) -> None:
        """Record storage operation metrics."""
        self.metrics_collector.record_timing(
            f"storage:{operation}", duration_ms, {"success": success}
        )

        # Alert on storage failures
        if not success:
            self.alert_manager.trigger_alert(
                RateLimitAlert(
                    level=AlertLevel.ERROR,
                    message=f"Storage operation failed: {operation}",
                    limiter_id="storage",
                    identifier=operation,
                    timestamp=datetime.utcnow(),
                    metadata={"operation": operation, "duration_ms": duration_ms},
                )
            )

    def record_cleanup_operation(
        self, limiter_id: str, cleaned_count: int, duration_ms: float
    ) -> None:
        """Record cleanup operation metrics."""
        self.metrics_collector.record_metric(
            RateLimitMetric(
                timestamp=datetime.utcnow(),
                limiter_id=limiter_id,
                identifier="cleanup",
                action="cleanup",
                count=cleaned_count,
                metadata={"duration_ms": duration_ms},
            )
        )

        self.metrics_collector.record_timing(
            f"cleanup:{limiter_id}", duration_ms, {"cleaned_count": cleaned_count}
        )

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "metrics": self.metrics_collector.get_metrics_summary(),
            "alerts": self.alert_manager.get_alert_summary(),
            "config": {
                "storage_backend": self.config.storage_backend,
                "monitoring_enabled": self.config.enable_monitoring,
                "alert_threshold": self.config.alert_threshold,
            },
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for monitoring systems."""
        metrics = self.metrics_collector.get_metrics_summary()
        alerts = self.alert_manager.get_alert_summary()

        # Determine overall health
        health_metrics = metrics.get("health", {})
        recent_critical_alerts = sum(
            1 for alert in alerts["recent_alerts"] if alert["level"] == "critical"
        )

        healthy = health_metrics.get("healthy", True) and recent_critical_alerts == 0

        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "degraded",
            "checks": {
                "error_rate": health_metrics.get("error_rate_5min", 0),
                "recent_critical_alerts": recent_critical_alerts,
                "storage_healthy": True,  # Would check storage health
            },
            "metrics_summary": {
                "total_requests_5min": health_metrics.get("total_requests_5min", 0),
                "error_rate_5min": health_metrics.get("error_rate_5min", 0),
            },
        }

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.metrics_collector.get_metrics_summary()

        prometheus_output = []

        # Counters
        for key, value in metrics.get("counters", {}).items():
            metric_name = f"rate_limit_{key.replace(':', '_')}"
            prometheus_output.append(f"{metric_name} {value}")

        # Rates
        for key, value in metrics.get("rates", {}).items():
            metric_name = f"rate_limit_rate_{key.replace(':', '_')}"
            prometheus_output.append(f"{metric_name} {value}")

        # Timing metrics
        for operation, timing_data in metrics.get("timing", {}).items():
            base_name = f"rate_limit_timing_{operation.replace(':', '_')}"
            for stat, value in timing_data.items():
                prometheus_output.append(f"{base_name}_{stat} {value}")

        return "\n".join(prometheus_output)


# Global monitor instance
_monitor: RateLimitMonitor | None = None


def initialize_monitoring(config: RateLimitConfig) -> RateLimitMonitor:
    """Initialize global monitoring."""
    global _monitor
    _monitor = RateLimitMonitor(config)
    return _monitor


def get_monitor() -> RateLimitMonitor | None:
    """Get global monitor instance."""
    return _monitor


def record_rate_limit_check(
    limiter_id: str,
    identifier: str,
    allowed: bool,
    current_count: int,
    limit: int,
    duration_ms: float,
    retry_after: int | None = None,
) -> None:
    """Convenience function to record rate limit check."""
    if _monitor:
        _monitor.record_rate_limit_check(
            limiter_id, identifier, allowed, current_count, limit, duration_ms, retry_after
        )
