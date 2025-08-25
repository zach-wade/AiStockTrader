"""
Performance metrics data structures.

Contains data classes and types for performance metrics and reports.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceMetric:
    """Performance metric data."""

    name: str
    value: float
    timestamp: float
    unit: str = "ms"
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance analysis report."""

    operation: str
    total_time: float
    cpu_time: float
    memory_peak: int
    memory_current: int
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    error_count: int
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
