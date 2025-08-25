"""
Thread Safety Monitoring for Trading System

Provides real-time monitoring of concurrent operations, lock contention,
and performance metrics for the thread-safe trading system.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LockMetrics:
    """Metrics for a specific lock."""

    lock_name: str
    acquisition_count: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    contention_count: int = 0
    deadlock_count: int = 0

    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time for lock acquisition."""
        if self.acquisition_count == 0:
            return 0.0
        return self.total_wait_time / self.acquisition_count


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type."""

    operation_name: str
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    total_duration: float = 0.0
    max_duration: float = 0.0
    concurrent_executions: int = 0
    max_concurrent: int = 0

    @property
    def average_duration(self) -> float:
        """Calculate average operation duration."""
        total_ops = self.success_count + self.failure_count
        if total_ops == 0:
            return 0.0
        return self.total_duration / total_ops

    @property
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        total_ops = self.success_count + self.failure_count
        if total_ops == 0:
            return 0.0
        return (self.success_count / total_ops) * 100


@dataclass
class ThreadSafetyMetrics:
    """Aggregated thread safety metrics."""

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    lock_metrics: dict[str, LockMetrics] = field(default_factory=dict)
    operation_metrics: dict[str, OperationMetrics] = field(default_factory=dict)
    race_conditions_detected: int = 0
    data_corruption_incidents: int = 0
    optimistic_lock_conflicts: int = 0
    deadlocks_resolved: int = 0

    @property
    def uptime_seconds(self) -> float:
        """Calculate system uptime in seconds."""
        return (datetime.now(UTC) - self.start_time).total_seconds()

    @property
    def total_operations(self) -> int:
        """Calculate total operations performed."""
        return sum(m.success_count + m.failure_count for m in self.operation_metrics.values())

    @property
    def operations_per_second(self) -> float:
        """Calculate average operations per second."""
        if self.uptime_seconds == 0:
            return 0.0
        return self.total_operations / self.uptime_seconds


class ThreadSafetyMonitor:
    """
    Monitor thread safety and concurrent operation performance.

    Tracks lock contention, operation success rates, and detects
    potential race conditions or data corruption.
    """

    def __init__(self) -> None:
        """Initialize the monitor."""
        self.metrics = ThreadSafetyMetrics()
        self._operation_timers: dict[str, float] = {}
        self._lock_timers: dict[str, float] = {}
        self._monitoring_lock = asyncio.Lock()

    async def record_lock_acquisition(
        self, lock_name: str, wait_time: float, was_contended: bool = False
    ) -> None:
        """
        Record metrics for a lock acquisition.

        Args:
            lock_name: Name of the lock
            wait_time: Time spent waiting for the lock
            was_contended: Whether there was contention for the lock
        """
        async with self._monitoring_lock:
            if lock_name not in self.metrics.lock_metrics:
                self.metrics.lock_metrics[lock_name] = LockMetrics(lock_name)

            metrics = self.metrics.lock_metrics[lock_name]
            metrics.acquisition_count += 1
            metrics.total_wait_time += wait_time
            metrics.max_wait_time = max(metrics.max_wait_time, wait_time)

            if was_contended:
                metrics.contention_count += 1

    async def record_operation(
        self, operation_name: str, duration: float, success: bool, retries: int = 0
    ) -> None:
        """
        Record metrics for an operation.

        Args:
            operation_name: Name of the operation
            duration: Time taken to complete the operation
            success: Whether the operation succeeded
            retries: Number of retries performed
        """
        async with self._monitoring_lock:
            if operation_name not in self.metrics.operation_metrics:
                self.metrics.operation_metrics[operation_name] = OperationMetrics(operation_name)

            metrics = self.metrics.operation_metrics[operation_name]

            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1

            metrics.retry_count += retries
            metrics.total_duration += duration
            metrics.max_duration = max(metrics.max_duration, duration)

    async def start_operation(self, operation_name: str) -> str:
        """
        Start tracking an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{time.time()}"

        async with self._monitoring_lock:
            if operation_name not in self.metrics.operation_metrics:
                self.metrics.operation_metrics[operation_name] = OperationMetrics(operation_name)

            metrics = self.metrics.operation_metrics[operation_name]
            metrics.concurrent_executions += 1
            metrics.max_concurrent = max(metrics.max_concurrent, metrics.concurrent_executions)

            self._operation_timers[operation_id] = time.time()

        return operation_id

    async def end_operation(
        self, operation_id: str, success: bool = True, retries: int = 0
    ) -> None:
        """
        End tracking an operation.

        Args:
            operation_id: Operation ID from start_operation
            success: Whether the operation succeeded
            retries: Number of retries performed
        """
        if operation_id not in self._operation_timers:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        start_time = self._operation_timers.pop(operation_id)
        duration = time.time() - start_time
        operation_name = operation_id.rsplit("_", 1)[0]

        async with self._monitoring_lock:
            if operation_name in self.metrics.operation_metrics:
                self.metrics.operation_metrics[operation_name].concurrent_executions -= 1

        await self.record_operation(operation_name, duration, success, retries)

    async def record_optimistic_lock_conflict(self) -> None:
        """Record an optimistic lock conflict."""
        async with self._monitoring_lock:
            self.metrics.optimistic_lock_conflicts += 1

    async def record_race_condition(self) -> None:
        """Record a detected race condition."""
        async with self._monitoring_lock:
            self.metrics.race_conditions_detected += 1

    async def record_data_corruption(self) -> None:
        """Record a data corruption incident."""
        async with self._monitoring_lock:
            self.metrics.data_corruption_incidents += 1

        logger.error("Data corruption detected!")

    async def record_deadlock_resolution(self) -> None:
        """Record a deadlock resolution."""
        async with self._monitoring_lock:
            self.metrics.deadlocks_resolved += 1

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary containing metrics summary
        """
        return {
            "uptime_seconds": self.metrics.uptime_seconds,
            "total_operations": self.metrics.total_operations,
            "operations_per_second": self.metrics.operations_per_second,
            "lock_metrics": {
                name: {
                    "acquisitions": m.acquisition_count,
                    "avg_wait_time": m.average_wait_time,
                    "max_wait_time": m.max_wait_time,
                    "contention_rate": (
                        (m.contention_count / m.acquisition_count * 100)
                        if m.acquisition_count > 0
                        else 0
                    ),
                }
                for name, m in self.metrics.lock_metrics.items()
            },
            "operation_metrics": {
                name: {
                    "success_rate": m.success_rate,
                    "avg_duration": m.average_duration,
                    "max_duration": m.max_duration,
                    "max_concurrent": m.max_concurrent,
                    "total_retries": m.retry_count,
                }
                for name, m in self.metrics.operation_metrics.items()
            },
            "safety_incidents": {
                "race_conditions": self.metrics.race_conditions_detected,
                "data_corruption": self.metrics.data_corruption_incidents,
                "optimistic_conflicts": self.metrics.optimistic_lock_conflicts,
                "deadlocks_resolved": self.metrics.deadlocks_resolved,
            },
        }

    def print_report(self) -> None:
        """Print a formatted metrics report."""
        summary = self.get_metrics_summary()

        print("\n" + "=" * 60)
        print("THREAD SAFETY METRICS REPORT")
        print("=" * 60)

        print(f"\nSystem Uptime: {summary['uptime_seconds']:.2f} seconds")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Operations/Second: {summary['operations_per_second']:.2f}")

        print("\n--- Lock Metrics ---")
        for name, metrics in summary["lock_metrics"].items():
            print(f"\n{name}:")
            print(f"  Acquisitions: {metrics['acquisitions']}")
            print(f"  Avg Wait Time: {metrics['avg_wait_time']:.4f}s")
            print(f"  Max Wait Time: {metrics['max_wait_time']:.4f}s")
            print(f"  Contention Rate: {metrics['contention_rate']:.2f}%")

        print("\n--- Operation Metrics ---")
        for name, metrics in summary["operation_metrics"].items():
            print(f"\n{name}:")
            print(f"  Success Rate: {metrics['success_rate']:.2f}%")
            print(f"  Avg Duration: {metrics['avg_duration']:.4f}s")
            print(f"  Max Duration: {metrics['max_duration']:.4f}s")
            print(f"  Max Concurrent: {metrics['max_concurrent']}")
            print(f"  Total Retries: {metrics['total_retries']}")

        print("\n--- Safety Incidents ---")
        incidents = summary["safety_incidents"]
        print(f"Race Conditions: {incidents['race_conditions']}")
        print(f"Data Corruption: {incidents['data_corruption']}")
        print(f"Optimistic Conflicts: {incidents['optimistic_conflicts']}")
        print(f"Deadlocks Resolved: {incidents['deadlocks_resolved']}")

        print("\n" + "=" * 60)


class MonitoredLock:
    """
    A wrapper around asyncio.Lock that reports metrics to ThreadSafetyMonitor.
    """

    def __init__(self, name: str, monitor: ThreadSafetyMonitor) -> None:
        """
        Initialize monitored lock.

        Args:
            name: Name for the lock (for monitoring)
            monitor: ThreadSafetyMonitor instance
        """
        self.name = name
        self.monitor = monitor
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Any:
        """Async context manager entry with monitoring."""
        start_time = time.time()
        was_locked = self._lock.locked()

        await self._lock.__aenter__()

        wait_time = time.time() - start_time
        await self.monitor.record_lock_acquisition(self.name, wait_time, was_contended=was_locked)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Async context manager exit."""
        return await self._lock.__aexit__(exc_type, exc_val, exc_tb)


# Global monitor instance
_global_monitor = ThreadSafetyMonitor()


def get_global_monitor() -> ThreadSafetyMonitor:
    """Get the global thread safety monitor instance."""
    return _global_monitor


async def monitor_operation(operation_name: str) -> Any:
    """
    Decorator to monitor async operations.

    Usage:
        @monitor_operation("open_position")
        async def open_position(...) -> Any:
            ...
    """

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor = get_global_monitor()
            op_id = await monitor.start_operation(operation_name)

            try:
                result = await func(*args, **kwargs)
                await monitor.end_operation(op_id, success=True)
                return result
            except Exception:
                await monitor.end_operation(op_id, success=False)
                raise

        return wrapper

    return decorator
