"""
Unified Metrics Module for AI Trader System

This module consolidates all metric-related types and definitions,
eliminating duplicates and providing a single source of truth for
metrics throughout the system.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Core Metric Types
# ============================================================================


class MetricType(Enum):
    """
    Unified metric types combining standard and validation-specific types.
    """

    # Standard metric types (from interfaces/validation/metrics.py)
    COUNTER = "counter"  # Incrementing counters
    GAUGE = "gauge"  # Point-in-time values
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summaries
    RATE = "rate"  # Rate calculations

    # Validation-specific types (from exporters/metric_types.py)
    VALIDATION_RESULT = "validation_result"
    QUALITY_SCORE = "quality_score"
    DATA_STALENESS = "data_staleness"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_METRICS = "error_metrics"
    COVERAGE_METRICS = "coverage_metrics"
    PIPELINE_STATUS = "pipeline_status"
    CRITICAL_FAILURE = "critical_failure"


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


class ExportFormat(Enum):
    """Supported export formats."""

    PROMETHEUS = "prometheus"
    JSON = "json"
    CSV = "csv"
    INFLUXDB = "influxdb"
    DATADOG = "datadog"


class CollectorStatus(Enum):
    """Status of the metrics collector."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    INITIALIZING = "initializing"


class PipelineStatus:
    """Pipeline status codes."""

    STOPPED = 0
    RUNNING = 1
    ERROR = 2
    WARNING = 3
    PAUSED = 4


# ============================================================================
# Metric Data Classes
# ============================================================================


@dataclass
class ValidationMetric:
    """Definition of a validation metric."""

    name: str
    type: MetricType
    description: str
    labels: list[str]


@dataclass
class MetricRecord:
    """
    Unified metric record supporting both collector and exporter use cases.

    This combines fields from both implementations to support all scenarios.
    """

    # Common fields
    timestamp: datetime

    # From exporters/metric_types.py
    type: MetricType | None = None
    value: Any | None = None
    labels: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None

    # From collectors/metric_types.py
    stage: str | None = None
    source: str | None = None
    data_type: str | None = None
    metrics: dict[str, Any] | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"timestamp": self.timestamp.isoformat()}

        # Add fields that are present
        if self.type is not None:
            result["type"] = self.type.value
        if self.value is not None:
            result["value"] = self.value
        if self.labels:
            result["labels"] = self.labels
        if self.metadata:
            result["metadata"] = self.metadata
        if self.stage:
            result["stage"] = self.stage
        if self.source:
            result["source"] = self.source
        if self.data_type:
            result["data_type"] = self.data_type
        if self.metrics:
            result["metrics"] = self.metrics
        if self.errors:
            result["errors"] = self.errors
        if self.warnings:
            result["warnings"] = self.warnings

        return result

    @classmethod
    def create_collector_record(
        cls,
        timestamp: datetime,
        stage: str,
        source: str,
        data_type: str,
        metrics: dict[str, Any],
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> "MetricRecord":
        """Factory method for creating collector-style metric records."""
        return cls(
            timestamp=timestamp,
            stage=stage,
            source=source,
            data_type=data_type,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def create_exporter_record(
        cls,
        type: MetricType,
        timestamp: datetime,
        value: Any,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MetricRecord":
        """Factory method for creating exporter-style metric records."""
        return cls(timestamp=timestamp, type=type, value=value, labels=labels, metadata=metadata)


# ============================================================================
# Metric Buckets and Constants
# ============================================================================


class MetricBuckets:
    """Standard histogram buckets for different metric types."""

    DURATION_SECONDS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    BATCH_SIZE = (10, 50, 100, 500, 1000, 5000, 10000)
    PERCENTAGE = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    SCORE = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    LATENCY_MS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


# ============================================================================
# Export All Public Classes
# ============================================================================

__all__ = [
    # Enums
    "MetricType",
    "MetricAggregation",
    "ExportFormat",
    "CollectorStatus",
    "PipelineStatus",
    # Data classes
    "ValidationMetric",
    "MetricRecord",
    # Constants
    "MetricBuckets",
]
