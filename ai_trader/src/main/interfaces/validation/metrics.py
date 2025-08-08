"""
Validation Framework - Metrics Interfaces

Metrics-specific validation interfaces for comprehensive validation
metrics collection, analysis, and reporting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncIterator
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import (
    ValidationStage, 
    ValidationSeverity,
    IValidationResult, 
    IValidationContext,
    IValidationMetrics
)


class MetricType(Enum):
    """Types of validation metrics."""
    COUNTER = "counter"        # Incrementing counters
    GAUGE = "gauge"           # Point-in-time values
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Statistical summaries
    RATE = "rate"            # Rate calculations


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
        self,
        stage: ValidationStage,
        results: List[IValidationResult],
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Collect metrics for a validation stage."""
        pass
    
    @abstractmethod
    async def collect_validator_metrics(
        self,
        validator_name: str,
        results: List[IValidationResult],
        execution_times: List[float]
    ) -> Dict[str, Any]:
        """Collect metrics for individual validators."""
        pass
    
    @abstractmethod
    async def collect_data_quality_metrics(
        self,
        quality_scores: Dict[str, float],
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Collect data quality metrics."""
        pass
    
    @abstractmethod
    async def collect_performance_metrics(
        self,
        operation_name: str,
        execution_time: float,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ) -> None:
        """Collect performance metrics."""
        pass
    
    @abstractmethod
    async def collect_error_metrics(
        self,
        error_type: str,
        error_message: str,
        context: IValidationContext,
        severity: ValidationSeverity
    ) -> None:
        """Collect error metrics."""
        pass


class IMetricsAggregator(ABC):
    """Interface for metrics aggregation operations."""
    
    @abstractmethod
    async def aggregate_metrics(
        self,
        metrics: List[Dict[str, Any]],
        aggregation_method: MetricAggregation,
        group_by: Optional[List[str]] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Aggregate metrics using specified method."""
        pass
    
    @abstractmethod
    async def aggregate_by_time_window(
        self,
        metrics: List[Dict[str, Any]],
        window_size: timedelta,
        aggregation_methods: Dict[str, MetricAggregation]
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics by time windows."""
        pass
    
    @abstractmethod
    async def aggregate_by_dimensions(
        self,
        metrics: List[Dict[str, Any]],
        dimensions: List[str],
        metric_fields: Dict[str, MetricAggregation]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by specified dimensions."""
        pass
    
    @abstractmethod
    async def calculate_derived_metrics(
        self,
        base_metrics: Dict[str, Any],
        derivation_rules: Dict[str, str]  # metric_name -> formula
    ) -> Dict[str, Any]:
        """Calculate derived metrics from base metrics."""
        pass


class IMetricsAnalyzer(ABC):
    """Interface for metrics analysis operations."""
    
    @abstractmethod
    async def analyze_trends(
        self,
        metrics_history: List[Dict[str, Any]],
        metric_name: str,
        trend_period: timedelta
    ) -> Dict[str, Any]:
        """Analyze trends in metrics over time."""
        pass
    
    @abstractmethod
    async def detect_anomalies(
        self,
        metrics: List[Dict[str, Any]],
        metric_name: str,
        detection_method: str = "statistical",
        sensitivity: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        pass
    
    @abstractmethod
    async def analyze_correlations(
        self,
        metrics: List[Dict[str, Any]],
        metric_pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], float]:
        """Analyze correlations between metrics."""
        pass
    
    @abstractmethod
    async def calculate_sla_compliance(
        self,
        metrics: List[Dict[str, Any]],
        sla_definitions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate SLA compliance metrics."""
        pass
    
    @abstractmethod
    async def forecast_metrics(
        self,
        historical_metrics: List[Dict[str, Any]],
        metric_name: str,
        forecast_horizon: timedelta,
        method: str = "linear"
    ) -> List[Dict[str, Any]]:
        """Forecast future metric values."""
        pass


class IMetricsStorage(ABC):
    """Interface for metrics storage operations."""
    
    @abstractmethod
    async def store_metrics(
        self,
        metrics: List[Dict[str, Any]],
        retention_policy: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store metrics data."""
        pass
    
    @abstractmethod
    async def query_metrics(
        self,
        metric_names: List[str],
        time_range: Tuple[datetime, datetime],
        filters: Optional[Dict[str, Any]] = None,
        aggregation: Optional[MetricAggregation] = None
    ) -> List[Dict[str, Any]]:
        """Query stored metrics."""
        pass
    
    @abstractmethod
    async def delete_metrics(
        self,
        metric_names: List[str],
        time_range: Tuple[datetime, datetime],
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete metrics data."""
        pass
    
    @abstractmethod
    async def get_metric_metadata(
        self,
        metric_name: str
    ) -> Dict[str, Any]:
        """Get metadata for a metric."""
        pass
    
    @abstractmethod
    async def compact_metrics(
        self,
        older_than: datetime,
        aggregation_method: MetricAggregation
    ) -> None:
        """Compact old metrics data."""
        pass


class IMetricsExporter(ABC):
    """Interface for metrics export operations."""
    
    @abstractmethod
    async def export_prometheus_metrics(
        self,
        metrics: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Export metrics in Prometheus format."""
        pass
    
    @abstractmethod
    async def export_json_metrics(
        self,
        metrics: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """Export metrics in JSON format."""
        pass
    
    @abstractmethod
    async def export_csv_metrics(
        self,
        metrics: List[Dict[str, Any]],
        columns: Optional[List[str]] = None
    ) -> str:
        """Export metrics in CSV format."""
        pass
    
    @abstractmethod
    async def export_to_external_system(
        self,
        metrics: List[Dict[str, Any]],
        system_config: Dict[str, Any]
    ) -> bool:
        """Export metrics to external monitoring system."""
        pass


class IMetricsDashboard(ABC):
    """Interface for metrics dashboard operations."""
    
    @abstractmethod
    async def create_dashboard_config(
        self,
        dashboard_name: str,
        metric_definitions: List[Dict[str, Any]],
        layout_config: Dict[str, Any]
    ) -> str:
        """Create dashboard configuration."""
        pass
    
    @abstractmethod
    async def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Tuple[datetime, datetime],
        refresh_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get data for dashboard rendering."""
        pass
    
    @abstractmethod
    async def create_alert_rules(
        self,
        metric_name: str,
        alert_conditions: List[Dict[str, Any]],
        notification_config: Dict[str, Any]
    ) -> str:
        """Create alert rules for metrics."""
        pass
    
    @abstractmethod
    async def get_active_alerts(
        self,
        severity_filter: Optional[ValidationSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        pass


class IMetricsReporter(ABC):
    """Interface for metrics reporting operations."""
    
    @abstractmethod
    async def generate_metrics_report(
        self,
        report_type: str,
        time_range: Tuple[datetime, datetime],
        metric_filters: Optional[Dict[str, Any]] = None,
        report_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        pass
    
    @abstractmethod
    async def generate_summary_report(
        self,
        metrics: List[Dict[str, Any]],
        summary_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary metrics report."""
        pass
    
    @abstractmethod
    async def generate_comparison_report(
        self,
        current_metrics: List[Dict[str, Any]],
        baseline_metrics: List[Dict[str, Any]],
        comparison_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metrics comparison report."""
        pass
    
    @abstractmethod
    async def schedule_recurring_report(
        self,
        report_config: Dict[str, Any],
        schedule_expression: str,  # cron expression
        delivery_config: Dict[str, Any]
    ) -> str:
        """Schedule recurring metrics report."""
        pass


class IValidationInsights(ABC):
    """Interface for validation insights generation."""
    
    @abstractmethod
    async def generate_validation_insights(
        self,
        validation_metrics: List[Dict[str, Any]],
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Generate insights from validation metrics."""
        pass
    
    @abstractmethod
    async def identify_performance_bottlenecks(
        self,
        performance_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify validation performance bottlenecks."""
        pass
    
    @abstractmethod
    async def analyze_failure_patterns(
        self,
        failure_metrics: List[Dict[str, Any]],
        pattern_detection_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in validation failures."""
        pass
    
    @abstractmethod
    async def recommend_optimizations(
        self,
        validation_metrics: List[Dict[str, Any]],
        performance_targets: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recommend validation optimizations."""
        pass
    
    @abstractmethod
    async def predict_validation_load(
        self,
        historical_metrics: List[Dict[str, Any]],
        prediction_horizon: timedelta
    ) -> Dict[str, Any]:
        """Predict future validation load."""
        pass