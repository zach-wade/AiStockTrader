"""
System Performance Record Models

Data structures for system performance monitoring and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class SystemMetricType(Enum):
    """Types of system metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemPerformanceRecord:
    """System performance record for monitoring."""
    timestamp: datetime
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_pct: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for compatibility with imports
SystemRecord = SystemPerformanceRecord


@dataclass
class ResourceUsage:
    """System resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: ComponentStatus = ComponentStatus.UNKNOWN
    uptime_hours: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    last_check: datetime = field(default_factory=datetime.now)