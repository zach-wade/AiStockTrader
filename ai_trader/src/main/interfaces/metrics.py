"""
Metrics Recording Interfaces

Defines contracts for metrics recording to break circular dependencies
and provide clean abstractions for performance monitoring.
"""

from abc import abstractmethod
from typing import Protocol, Dict, Optional, Any, runtime_checkable
from enum import Enum
from datetime import datetime


class MetricType(Enum):
    """Types of metrics that can be recorded."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"          # Duration measurements
    

@runtime_checkable
class IMetricsRecorder(Protocol):
    """
    Interface for recording metrics.
    
    This protocol defines the contract for recording various types of
    metrics throughout the application without creating dependencies
    on specific monitoring implementations.
    """
    
    @abstractmethod
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name (e.g., "api.request.duration")
            value: Numeric value to record
            metric_type: Type of metric being recorded
            tags: Optional tags for metric categorization
            timestamp: Optional timestamp (defaults to current time)
        """
        ...
    
    @abstractmethod
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Amount to increment by (default: 1.0)
            tags: Optional tags
        """
        ...
    
    @abstractmethod
    def update_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Update a gauge metric with current value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
        """
        ...
    
    @abstractmethod
    def record_duration(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a duration/timer metric.
        
        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            tags: Optional tags
        """
        ...


@runtime_checkable
class IMetricsProvider(Protocol):
    """
    Interface for providing metrics recorder instances.
    
    This protocol is useful for components that need to provide
    metrics recording capabilities to their sub-components.
    """
    
    @abstractmethod
    def get_metrics_recorder(self) -> IMetricsRecorder:
        """
        Get a metrics recorder instance.
        
        Returns:
            IMetricsRecorder implementation
        """
        ...