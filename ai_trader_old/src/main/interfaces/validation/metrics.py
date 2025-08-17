"""
Validation Framework - Metrics Interfaces

Metrics-specific validation interfaces for comprehensive validation
metrics collection, analysis, and reporting.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Local imports
from main.interfaces.data_pipeline.validation import (
    IValidationContext,
    IValidationMetrics,
    IValidationResult,
    ValidationSeverity,
    ValidationStage,
)


class MetricType(Enum):
    """Types of validation metrics."""

    COUNTER = "counter"  # Incrementing counters
    GAUGE = "gauge"  # Point-in-time values
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summaries
    RATE = "rate"  # Rate calculations


class MetricAggregation(Enum):
    """Metric aggregation methods."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    MEDIAN = "median"
    STD_DEV = "std_dev"


class IValidationMetricsCollector(IValidationMetrics):
    """Extended interface for validation metrics collection."""

    @abstractmethod
    async def collect_stage_metrics(
        self, stage: ValidationStage, results: list[IValidationResult], context: IValidationContext
    ) -> dict[str, Any]:
        """Collect metrics for a validation stage."""
        pass

    @abstractmethod
    async def collect_validator_metrics(
        self, validator_name: str, results: list[IValidationResult], execution_times: list[float]
    ) -> dict[str, Any]:
        """Collect metrics for individual validators."""
        pass

    @abstractmethod
    async def collect_data_quality_metrics(
        self, quality_scores: dict[str, float], context: IValidationContext
    ) -> dict[str, Any]:
        """Collect data quality metrics."""
        pass

    @abstractmethod
    async def collect_performance_metrics(
        self,
        operation_name: str,
        execution_time: float,
        memory_usage: float | None = None,
        cpu_usage: float | None = None,
    ) -> None:
        """Collect performance metrics."""
        pass

    @abstractmethod
    async def collect_error_metrics(
        self,
        error_type: str,
        error_message: str,
        context: IValidationContext,
        severity: ValidationSeverity,
    ) -> None:
        """Collect error metrics."""
        pass


class IMetricsAggregator(ABC):
    """Interface for metrics aggregation operations."""

    @abstractmethod
    async def aggregate_metrics(
        self,
        metrics: list[dict[str, Any]],
        aggregation_method: MetricAggregation,
        group_by: list[str] | None = None,
        time_window: timedelta | None = None,
    ) -> dict[str, Any]:
        """Aggregate metrics using specified method."""
        pass

    @abstractmethod
    async def aggregate_by_time_window(
        self,
        metrics: list[dict[str, Any]],
        window_size: timedelta,
        aggregation_methods: dict[str, MetricAggregation],
    ) -> list[dict[str, Any]]:
        """Aggregate metrics by time windows."""
        pass

    @abstractmethod
    async def aggregate_by_dimensions(
        self,
        metrics: list[dict[str, Any]],
        dimensions: list[str],
        metric_fields: dict[str, MetricAggregation],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate metrics by specified dimensions."""
        pass

    @abstractmethod
    async def calculate_derived_metrics(
        self,
        base_metrics: dict[str, Any],
        derivation_rules: dict[str, str],  # metric_name -> formula
    ) -> dict[str, Any]:
        """Calculate derived metrics from base metrics."""
        pass


class IMetricsAnalyzer(ABC):
    """Interface for metrics analysis operations."""

    @abstractmethod
    async def analyze_trends(
        self, metrics_history: list[dict[str, Any]], metric_name: str, trend_period: timedelta
    ) -> dict[str, Any]:
        """Analyze trends in metrics over time."""
        pass

    @abstractmethod
    async def detect_anomalies(
        self,
        metrics: list[dict[str, Any]],
        metric_name: str,
        detection_method: str = "statistical",
        sensitivity: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in metrics."""
        pass

    @abstractmethod
    async def analyze_correlations(
        self, metrics: list[dict[str, Any]], metric_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], float]:
        """Analyze correlations between metrics."""
        pass

    @abstractmethod
    async def calculate_sla_compliance(
        self, metrics: list[dict[str, Any]], sla_definitions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Calculate SLA compliance metrics."""
        pass

    @abstractmethod
    async def forecast_metrics(
        self,
        historical_metrics: list[dict[str, Any]],
        metric_name: str,
        forecast_horizon: timedelta,
        method: str = "linear",
    ) -> list[dict[str, Any]]:
        """Forecast future metric values."""
        pass


class IMetricsStorage(ABC):
    """Interface for metrics storage operations."""

    @abstractmethod
    async def store_metrics(
        self, metrics: list[dict[str, Any]], retention_policy: dict[str, Any] | None = None
    ) -> None:
        """Store metrics data."""
        pass

    @abstractmethod
    async def query_metrics(
        self,
        metric_names: list[str],
        time_range: tuple[datetime, datetime],
        filters: dict[str, Any] | None = None,
        aggregation: MetricAggregation | None = None,
    ) -> list[dict[str, Any]]:
        """Query stored metrics."""
        pass

    @abstractmethod
    async def delete_metrics(
        self,
        metric_names: list[str],
        time_range: tuple[datetime, datetime],
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Delete metrics data."""
        pass

    @abstractmethod
    async def get_metric_metadata(self, metric_name: str) -> dict[str, Any]:
        """Get metadata for a metric."""
        pass

    @abstractmethod
    async def compact_metrics(
        self, older_than: datetime, aggregation_method: MetricAggregation
    ) -> None:
        """Compact old metrics data."""
        pass


class IMetricsExporter(ABC):
    """Interface for metrics export operations."""

    @abstractmethod
    async def export_prometheus_metrics(
        self, metrics: list[dict[str, Any]], labels: dict[str, str] | None = None
    ) -> str:
        """Export metrics in Prometheus format."""
        pass

    @abstractmethod
    async def export_json_metrics(
        self, metrics: list[dict[str, Any]], include_metadata: bool = True
    ) -> str:
        """Export metrics in JSON format."""
        pass

    @abstractmethod
    async def export_csv_metrics(
        self, metrics: list[dict[str, Any]], columns: list[str] | None = None
    ) -> str:
        """Export metrics in CSV format."""
        pass

    @abstractmethod
    async def export_to_external_system(
        self, metrics: list[dict[str, Any]], system_config: dict[str, Any]
    ) -> bool:
        """Export metrics to external monitoring system."""
        pass


class IMetricsDashboard(ABC):
    """Interface for metrics dashboard operations."""

    @abstractmethod
    async def create_dashboard_config(
        self,
        dashboard_name: str,
        metric_definitions: list[dict[str, Any]],
        layout_config: dict[str, Any],
    ) -> str:
        """Create dashboard configuration."""
        pass

    @abstractmethod
    async def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: tuple[datetime, datetime],
        refresh_interval: int | None = None,
    ) -> dict[str, Any]:
        """Get data for dashboard rendering."""
        pass

    @abstractmethod
    async def create_alert_rules(
        self,
        metric_name: str,
        alert_conditions: list[dict[str, Any]],
        notification_config: dict[str, Any],
    ) -> str:
        """Create alert rules for metrics."""
        pass

    @abstractmethod
    async def get_active_alerts(
        self, severity_filter: ValidationSeverity | None = None
    ) -> list[dict[str, Any]]:
        """Get currently active alerts."""
        pass


class IMetricsReporter(ABC):
    """Interface for metrics reporting operations."""

    @abstractmethod
    async def generate_metrics_report(
        self,
        report_type: str,
        time_range: tuple[datetime, datetime],
        metric_filters: dict[str, Any] | None = None,
        report_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive metrics report."""
        pass

    @abstractmethod
    async def generate_summary_report(
        self, metrics: list[dict[str, Any]], summary_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate summary metrics report."""
        pass

    @abstractmethod
    async def generate_comparison_report(
        self,
        current_metrics: list[dict[str, Any]],
        baseline_metrics: list[dict[str, Any]],
        comparison_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate metrics comparison report."""
        pass

    @abstractmethod
    async def schedule_recurring_report(
        self,
        report_config: dict[str, Any],
        schedule_expression: str,  # cron expression
        delivery_config: dict[str, Any],
    ) -> str:
        """Schedule recurring metrics report."""
        pass


class IValidationInsights(ABC):
    """Interface for validation insights generation."""

    @abstractmethod
    async def generate_validation_insights(
        self, validation_metrics: list[dict[str, Any]], time_range: tuple[datetime, datetime]
    ) -> dict[str, Any]:
        """Generate insights from validation metrics."""
        pass

    @abstractmethod
    async def identify_performance_bottlenecks(
        self, performance_metrics: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify validation performance bottlenecks."""
        pass

    @abstractmethod
    async def analyze_failure_patterns(
        self, failure_metrics: list[dict[str, Any]], pattern_detection_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze patterns in validation failures."""
        pass

    @abstractmethod
    async def recommend_optimizations(
        self, validation_metrics: list[dict[str, Any]], performance_targets: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Recommend validation optimizations."""
        pass

    @abstractmethod
    async def predict_validation_load(
        self, historical_metrics: list[dict[str, Any]], prediction_horizon: timedelta
    ) -> dict[str, Any]:
        """Predict future validation load."""
        pass
