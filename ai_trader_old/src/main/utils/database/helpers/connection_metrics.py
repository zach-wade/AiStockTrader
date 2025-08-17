"""
Database Connection Pool Metrics

Focused module for connection pool metrics collection and analysis.
"""

# Standard library imports
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Any


@dataclass
class ConnectionPoolMetrics:
    """Metrics for database connection pool monitoring"""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    checked_out_connections: int = 0
    overflow_connections: int = 0
    invalid_connections: int = 0

    # Performance metrics
    query_count: int = 0
    slow_query_count: int = 0
    total_query_time: float = 0.0
    avg_query_time: float = 0.0

    # Connection metrics
    connection_errors: int = 0
    connection_timeouts: int = 0
    pool_exhaustions: int = 0

    # Recent query times (last 100 queries)
    recent_query_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_query_time(self, execution_time: float):
        """Add a query execution time"""
        self.query_count += 1
        self.total_query_time += execution_time
        self.avg_query_time = self.total_query_time / self.query_count
        self.recent_query_times.append(execution_time)

        # Track slow queries (>100ms)
        if execution_time > 0.1:
            self.slow_query_count += 1

    def get_recent_avg_time(self) -> float:
        """Get average query time for recent queries"""
        if not self.recent_query_times:
            return 0.0
        return sum(self.recent_query_times) / len(self.recent_query_times)


@dataclass
class ConnectionHealthStatus:
    """Health status of database connections"""

    is_healthy: bool
    pool_utilization: float
    avg_response_time: float
    error_rate: float
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class MetricsCollector:
    """Thread-safe metrics collection for database operations"""

    def __init__(self):
        self.metrics = ConnectionPoolMetrics()
        self._lock = threading.Lock()
        self._start_time = datetime.now()

    def record_query(self, execution_time: float):
        """Record a query execution"""
        with self._lock:
            self.metrics.add_query_time(execution_time)

    def record_connection_event(self, event_type: str):
        """Record connection pool events"""
        with self._lock:
            if event_type == "connect":
                self.metrics.total_connections += 1
            elif event_type == "checkout":
                self.metrics.checked_out_connections += 1
            elif event_type == "checkin":
                if self.metrics.checked_out_connections > 0:
                    self.metrics.checked_out_connections -= 1
            elif event_type == "error":
                self.metrics.connection_errors += 1
            elif event_type == "timeout":
                self.metrics.connection_timeouts += 1
            elif event_type == "exhaustion":
                self.metrics.pool_exhaustions += 1

    def update_pool_status(self, pool_info: dict[str, int]):
        """Update current pool status"""
        with self._lock:
            self.metrics.active_connections = pool_info.get("active", 0)
            self.metrics.idle_connections = pool_info.get("idle", 0)
            self.metrics.overflow_connections = pool_info.get("overflow", 0)
            self.metrics.invalid_connections = pool_info.get("invalid", 0)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """Get current metrics snapshot"""
        with self._lock:
            return {
                "pool_status": {
                    "total_connections": self.metrics.total_connections,
                    "active_connections": self.metrics.active_connections,
                    "idle_connections": self.metrics.idle_connections,
                    "overflow_connections": self.metrics.overflow_connections,
                    "invalid_connections": self.metrics.invalid_connections,
                },
                "performance_metrics": {
                    "total_queries": self.metrics.query_count,
                    "slow_queries": self.metrics.slow_query_count,
                    "avg_query_time": self.metrics.avg_query_time,
                    "recent_avg_time": self.metrics.get_recent_avg_time(),
                    "slow_query_rate": (
                        self.metrics.slow_query_count / max(self.metrics.query_count, 1) * 100
                    ),
                },
                "error_metrics": {
                    "connection_errors": self.metrics.connection_errors,
                    "connection_timeouts": self.metrics.connection_timeouts,
                    "pool_exhaustions": self.metrics.pool_exhaustions,
                },
                "uptime_info": {
                    "start_time": self._start_time.isoformat(),
                    "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                },
            }

    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics = ConnectionPoolMetrics()
            self._start_time = datetime.now()
