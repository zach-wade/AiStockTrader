"""
Data Pipeline Monitoring Interfaces

Interfaces for monitoring, health checking, and alerting components
with layer-aware monitoring and comprehensive metrics collection.
"""

# Standard library imports
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IHealthMonitor(ABC):
    """Interface for monitoring system health."""

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        pass

    @abstractmethod
    async def check_component_health(self, component: str) -> dict[str, Any]:
        """Check health of a specific component."""
        pass

    @abstractmethod
    async def check_layer_health(self, layer: DataLayer) -> dict[str, Any]:
        """Check health of a specific layer."""
        pass

    @abstractmethod
    async def check_system_health(self) -> dict[str, Any]:
        """Check overall system health."""
        pass

    @abstractmethod
    async def get_health_history(
        self, component: str | None = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get health check history."""
        pass

    @abstractmethod
    async def register_health_check(
        self, component: str, check_function: Callable, interval_seconds: int = 60
    ) -> str:
        """Register a custom health check."""
        pass

    @abstractmethod
    async def unregister_health_check(self, check_id: str) -> bool:
        """Unregister a health check."""
        pass


class IMetricsCollector(ABC):
    """Interface for collecting and managing metrics."""

    @abstractmethod
    async def start_collection(self) -> None:
        """Start metrics collection."""
        pass

    @abstractmethod
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        pass

    @abstractmethod
    async def record_operation_metric(
        self,
        operation: str,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record an operation metric."""
        pass

    @abstractmethod
    async def record_layer_metric(
        self, layer: DataLayer, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a layer-specific metric."""
        pass

    @abstractmethod
    async def increment_counter(
        self, counter_name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    async def record_histogram(
        self, histogram_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        pass

    @abstractmethod
    async def record_gauge(
        self, gauge_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a gauge metric."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        metric_names: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get collected metrics."""
        pass

    @abstractmethod
    async def get_layer_metrics(
        self, layer: DataLayer, metric_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Get metrics for a specific layer."""
        pass


class IPerformanceMonitor(ABC):
    """Interface for monitoring performance metrics."""

    @abstractmethod
    async def start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        pass

    @abstractmethod
    async def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring."""
        pass

    @abstractmethod
    async def track_operation_performance(
        self,
        operation: str,
        duration_ms: float,
        layer: DataLayer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track performance of an operation."""
        pass

    @abstractmethod
    async def track_throughput(
        self,
        component: str,
        records_processed: int,
        duration_ms: float,
        layer: DataLayer | None = None,
    ) -> None:
        """Track throughput metrics."""
        pass

    @abstractmethod
    async def track_resource_usage(
        self, component: str, cpu_percent: float, memory_mb: float, layer: DataLayer | None = None
    ) -> None:
        """Track resource usage."""
        pass

    @abstractmethod
    async def get_performance_summary(
        self, component: str | None = None, layer: DataLayer | None = None, hours: int = 24
    ) -> dict[str, Any]:
        """Get performance summary."""
        pass

    @abstractmethod
    async def detect_performance_anomalies(
        self, component: str, threshold_multiplier: float = 2.0
    ) -> list[dict[str, Any]]:
        """Detect performance anomalies."""
        pass


class IAlertManager(ABC):
    """Interface for managing alerts and notifications."""

    @abstractmethod
    async def start_alert_manager(self) -> None:
        """Start the alert manager."""
        pass

    @abstractmethod
    async def stop_alert_manager(self) -> None:
        """Stop the alert manager."""
        pass

    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        component: str,
        layer: DataLayer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send an alert."""
        pass

    @abstractmethod
    async def create_alert_rule(
        self,
        rule_name: str,
        condition: str,
        severity: AlertSeverity,
        component: str,
        notification_channels: list[str],
        layer: DataLayer | None = None,
    ) -> str:
        """Create an alert rule."""
        pass

    @abstractmethod
    async def update_alert_rule(self, rule_id: str, updates: dict[str, Any]) -> bool:
        """Update an alert rule."""
        pass

    @abstractmethod
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        pass

    @abstractmethod
    async def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        component: str | None = None,
        layer: DataLayer | None = None,
    ) -> list[dict[str, Any]]:
        """Get active alerts."""
        pass

    @abstractmethod
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        pass

    @abstractmethod
    async def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        pass


class IDashboardProvider(ABC):
    """Interface for providing monitoring dashboard data."""

    @abstractmethod
    async def get_system_overview(self) -> dict[str, Any]:
        """Get system overview dashboard data."""
        pass

    @abstractmethod
    async def get_layer_dashboard(self, layer: DataLayer) -> dict[str, Any]:
        """Get layer-specific dashboard data."""
        pass

    @abstractmethod
    async def get_component_dashboard(self, component: str) -> dict[str, Any]:
        """Get component-specific dashboard data."""
        pass

    @abstractmethod
    async def get_performance_dashboard(self, time_range_hours: int = 24) -> dict[str, Any]:
        """Get performance dashboard data."""
        pass

    @abstractmethod
    async def get_alerts_dashboard(self) -> dict[str, Any]:
        """Get alerts dashboard data."""
        pass

    @abstractmethod
    async def get_data_quality_dashboard(self, layer: DataLayer | None = None) -> dict[str, Any]:
        """Get data quality dashboard data."""
        pass

    @abstractmethod
    async def create_custom_dashboard(self, dashboard_config: dict[str, Any]) -> str:
        """Create a custom dashboard."""
        pass


class ILogAggregator(ABC):
    """Interface for aggregating and analyzing logs."""

    @abstractmethod
    async def start_log_aggregation(self) -> None:
        """Start log aggregation."""
        pass

    @abstractmethod
    async def stop_log_aggregation(self) -> None:
        """Stop log aggregation."""
        pass

    @abstractmethod
    async def aggregate_logs(
        self,
        component: str,
        level: str,
        start_time: datetime,
        end_time: datetime,
        layer: DataLayer | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate logs for analysis."""
        pass

    @abstractmethod
    async def search_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        components: list[str] | None = None,
        layers: list[DataLayer] | None = None,
    ) -> list[dict[str, Any]]:
        """Search logs with query."""
        pass

    @abstractmethod
    async def analyze_error_patterns(
        self, component: str | None = None, layer: DataLayer | None = None, hours: int = 24
    ) -> dict[str, Any]:
        """Analyze error patterns in logs."""
        pass

    @abstractmethod
    async def generate_log_insights(self, time_range_hours: int = 24) -> list[dict[str, Any]]:
        """Generate insights from log analysis."""
        pass


class IServiceMonitor(ABC):
    """Interface for monitoring individual services."""

    @abstractmethod
    async def register_service(self, service_name: str, service_config: dict[str, Any]) -> str:
        """Register a service for monitoring."""
        pass

    @abstractmethod
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from monitoring."""
        pass

    @abstractmethod
    async def check_service_health(self, service_id: str) -> dict[str, Any]:
        """Check health of a specific service."""
        pass

    @abstractmethod
    async def get_service_metrics(
        self, service_id: str, metric_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Get metrics for a specific service."""
        pass

    @abstractmethod
    async def restart_service(self, service_id: str) -> dict[str, Any]:
        """Restart a service."""
        pass

    @abstractmethod
    async def get_service_dependencies(self, service_id: str) -> list[str]:
        """Get service dependencies."""
        pass

    @abstractmethod
    async def check_service_dependencies(self, service_id: str) -> dict[str, Any]:
        """Check health of service dependencies."""
        pass
