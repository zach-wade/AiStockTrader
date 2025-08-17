"""
Monitoring System Migration Layer

This module provides transparent migration from basic to enhanced monitoring
while maintaining full backward compatibility.
"""

# Standard library imports
import os
from typing import Any

# Local imports
from main.utils.core import get_logger
from main.utils.database import DatabasePool

from .enhanced import EnhancedMonitor, get_enhanced_monitor
from .monitor import PerformanceMonitor
from .types import AlertLevel, MetricType

logger = get_logger(__name__)


class MigrationMonitor(PerformanceMonitor):
    """
    Migration monitor that transparently uses enhanced features when available.

    This class extends PerformanceMonitor to add enhanced features while
    maintaining full backward compatibility with existing code.
    """

    def __init__(
        self,
        history_size: int = 1000,
        monitoring_interval: float = 5.0,
        enable_system_monitoring: bool = True,
        db_pool: DatabasePool | None = None,
        enable_enhanced: bool | None = None,
    ):
        """
        Initialize migration monitor.

        Args:
            history_size: Number of historical metrics to keep
            monitoring_interval: System monitoring interval in seconds
            enable_system_monitoring: Whether to monitor system resources
            db_pool: Optional database pool for persistence
            enable_enhanced: Force enable/disable enhanced features (auto-detect if None)
        """
        # Initialize base monitor
        super().__init__(history_size, monitoring_interval, enable_system_monitoring)

        # Determine if we should use enhanced features
        if enable_enhanced is None:
            # Auto-detect based on database availability
            self.use_enhanced = db_pool is not None
        else:
            self.use_enhanced = enable_enhanced and db_pool is not None

        # Initialize enhanced monitor if enabled
        self.enhanced_monitor: EnhancedMonitor | None = None
        if self.use_enhanced:
            self.enhanced_monitor = get_enhanced_monitor(db_pool, create_if_missing=True)
            logger.info("Migration monitor initialized with enhanced features")
        else:
            logger.info("Migration monitor initialized with basic features only")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
    ):
        """
        Record a metric value.

        Uses enhanced monitor if available, otherwise falls back to base implementation.
        """
        # Always record to base monitor for in-memory access
        super().record_metric(name, value, metric_type, tags)

        # Also record to enhanced monitor if available
        if self.enhanced_monitor:
            self.enhanced_monitor.record_metric(
                name=name, value=value, metric_type=metric_type, tags=tags
            )

    async def get_metric_value(
        self,
        name: str,
        aggregation: str = "last",
        period_minutes: int = 5,
        tags: dict[str, str] | None = None,
    ) -> float | None:
        """
        Get aggregated metric value.

        Uses enhanced monitor for aggregation if available.
        """
        if self.enhanced_monitor:
            return await self.enhanced_monitor.get_metric_value(
                name, aggregation, period_minutes, tags
            )
        else:
            # Fallback to basic in-memory lookup
            metrics = self.get_metric_history(name, minutes=period_minutes)
            if not metrics:
                return None

            values = [m.value for m in metrics]

            if aggregation == "last":
                return values[-1] if values else None
            elif aggregation == "avg":
                return sum(values) / len(values) if values else None
            elif aggregation == "min":
                return min(values) if values else None
            elif aggregation == "max":
                return max(values) if values else None
            elif aggregation == "sum":
                return sum(values) if values else None
            else:
                return None

    async def get_metric_series(
        self, name: str, period_minutes: int = 60, tags: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Get time series data for a metric.

        Returns data from enhanced monitor if available, otherwise from memory.
        """
        if self.enhanced_monitor:
            return await self.enhanced_monitor.get_metric_series(name, period_minutes, tags)
        else:
            # Fallback to in-memory data
            metrics = self.get_metric_history(name, minutes=period_minutes)
            return [
                {"timestamp": m.timestamp.isoformat(), "value": m.value, "tags": m.tags or {}}
                for m in metrics
            ]

    def register_metric_with_thresholds(
        self,
        name: str,
        metric_type: MetricType = MetricType.GAUGE,
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
        description: str = "",
    ):
        """
        Register a metric with alert thresholds.

        This is an enhanced feature that's only available with enhanced monitor.
        """
        if self.enhanced_monitor:
            from .enhanced import EnhancedMetricDefinition

            definition = EnhancedMetricDefinition(
                name=name,
                metric_type=metric_type,
                description=description,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
            )

            self.enhanced_monitor.register_metric(definition)
            logger.info(f"Registered metric {name} with thresholds")
        else:
            logger.warning(
                f"Cannot register metric {name} with thresholds - "
                "enhanced features not available"
            )

    async def start_monitoring(self):
        """Start monitoring with both basic and enhanced features."""
        # Start base monitoring
        await super().start_monitoring()

        # Start enhanced monitoring if available
        if self.enhanced_monitor:
            await self.enhanced_monitor.start()

    async def stop_monitoring(self):
        """Stop monitoring for both basic and enhanced features."""
        # Stop base monitoring
        await super().stop_monitoring()

        # Stop enhanced monitoring if available
        if self.enhanced_monitor:
            await self.enhanced_monitor.stop()

    def get_system_health_score(self) -> dict[str, Any]:
        """
        Get comprehensive system health score.

        Enhanced feature that combines multiple metrics.
        """
        if self.enhanced_monitor:
            # Use enhanced health scoring
            active_alerts = self.enhanced_monitor.get_active_alerts()
            critical_count = sum(1 for a in active_alerts if a.level == AlertLevel.CRITICAL)
            warning_count = sum(1 for a in active_alerts if a.level == AlertLevel.WARNING)

            # Calculate health score (0-100)
            score = 100
            score -= critical_count * 20  # -20 per critical alert
            score -= warning_count * 10  # -10 per warning alert
            score = max(0, score)

            return {
                "overall_score": score,
                "status": "healthy" if score > 80 else "warning" if score > 50 else "critical",
                "active_alerts": len(active_alerts),
                "critical_alerts": critical_count,
                "warning_alerts": warning_count,
            }
        else:
            # Fallback to basic health check
            alerts = self.alert_manager.get_active_alerts()
            return {
                "overall_score": 100 - len(alerts) * 10,
                "status": "healthy" if not alerts else "warning",
                "active_alerts": len(alerts),
                "critical_alerts": 0,
                "warning_alerts": 0,
            }

    def cleanup_old_metrics(self, retention_hours: int = 168):
        """Clean up old metrics from both basic and enhanced storage."""
        # Clean basic in-memory metrics
        # (Basic monitor doesn't have cleanup, so we'll add it here)

        # Clean enhanced metrics if available
        if self.enhanced_monitor:
            self.enhanced_monitor.cleanup_old_metrics(retention_hours)


def create_monitor(db_pool: DatabasePool | None = None, **kwargs) -> MigrationMonitor:
    """
    Create a monitor instance with appropriate features.

    Args:
        db_pool: Optional database pool for enhanced features
        **kwargs: Additional arguments for monitor initialization

    Returns:
        MigrationMonitor instance with basic or enhanced features
    """
    return MigrationMonitor(db_pool=db_pool, **kwargs)


# Environment variable to control migration
USE_ENHANCED_MONITORING = os.getenv("USE_ENHANCED_MONITORING", "auto").lower()


def should_use_enhanced(db_pool: DatabasePool | None = None) -> bool:
    """Determine if enhanced monitoring should be used."""
    if USE_ENHANCED_MONITORING == "true":
        return True
    elif USE_ENHANCED_MONITORING == "false":
        return False
    else:  # auto
        return db_pool is not None
