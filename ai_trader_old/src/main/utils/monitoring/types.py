"""
Monitoring Data Types

Data structures and enums for performance monitoring system.
"""

# Standard library imports
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics that can be monitored."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class SystemResources:
    """System resource utilization snapshot."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FunctionMetrics:
    """Metrics for function execution."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_duration(self) -> float:
        """Average execution duration."""
        return self.total_duration / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        return (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0.0

    @property
    def recent_avg_duration(self) -> float:
        """Recent average duration (last 100 calls)."""
        return (
            sum(self.recent_durations) / len(self.recent_durations)
            if self.recent_durations
            else 0.0
        )


@dataclass
class Alert:
    """Performance alert."""

    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "level": self.level.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }
