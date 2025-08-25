"""
Comprehensive performance monitoring system.

Provides memory profiling, CPU usage tracking, database query performance,
API endpoint monitoring, resource usage analysis, and bottleneck identification.
"""

import asyncio
import gc
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any

from .api_tracker import APIPerformanceTracker
from .bottleneck_detector import BottleneckDetector
from .cpu_profiler import CPUProfiler
from .database_profiler import DatabaseQueryProfiler
from .memory_profiler import MemoryProfiler
from .metrics import PerformanceMetric, PerformanceReport
from .system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.

    Provides:
    - Memory profiling with leak detection
    - CPU usage tracking
    - Database query performance
    - API endpoint monitoring
    - Resource usage analysis
    - Bottleneck identification
    """

    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_cpu_profiling: bool = True,
        enable_query_profiling: bool = True,
        enable_api_profiling: bool = True,
    ):
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_query_profiling = enable_query_profiling
        self.enable_api_profiling = enable_api_profiling

        # Initialize profilers
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        self.cpu_profiler = CPUProfiler() if enable_cpu_profiling else None
        self.db_profiler = DatabaseQueryProfiler() if enable_query_profiling else None
        self.api_profiler = APIPerformanceTracker() if enable_api_profiling else None

        # Initialize bottleneck detector
        self.bottleneck_detector = BottleneckDetector(
            self.memory_profiler, self.db_profiler, self.api_profiler
        )

        # Performance metrics storage
        self._performance_metrics: deque[PerformanceMetric] = deque(maxlen=10000)
        self._lock = threading.Lock()

        # Background monitoring
        self._monitoring_task: asyncio.Task[None] | None = None
        self._stop_monitoring = False

        # Resource baselines
        self._baseline_metrics = SystemMonitor.capture_baseline_metrics()

    @contextmanager
    def profile_operation(self, operation_name: str, **tags: Any) -> Generator[None, None, None]:
        """Context manager for profiling operations."""
        start_time = time.perf_counter()

        # Start profiling
        if self.memory_profiler:
            self.memory_profiler.start_profiling()
            self.memory_profiler.take_snapshot(f"start_{operation_name}")

        if self.cpu_profiler:
            self.cpu_profiler.start_profiling(operation_name)

        try:
            yield
        finally:
            # End profiling and collect metrics
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Memory metrics
            memory_metrics = {}
            if self.memory_profiler:
                memory_snapshot = self.memory_profiler.take_snapshot(f"end_{operation_name}")
                memory_metrics = {
                    "memory_delta": memory_snapshot.get("memory_delta", 0),
                    "peak_memory": memory_snapshot.get("peak_memory", 0),
                }

            # CPU metrics
            cpu_metrics = {}
            if self.cpu_profiler:
                cpu_metrics = self.cpu_profiler.end_profiling(operation_name)

            # Record performance metric
            self._record_performance_metric(
                name=operation_name,
                duration=duration,
                memory_metrics=memory_metrics,
                cpu_metrics=cpu_metrics,
                tags=tags,
            )

    @asynccontextmanager
    async def async_profile_operation(
        self, operation_name: str, **tags: Any
    ) -> AsyncGenerator[None, None]:
        """Async context manager for profiling operations."""
        start_time = time.perf_counter()

        # Start profiling
        if self.memory_profiler:
            self.memory_profiler.start_profiling()
            self.memory_profiler.take_snapshot(f"start_{operation_name}")

        if self.cpu_profiler:
            self.cpu_profiler.start_profiling(operation_name)

        try:
            yield
        finally:
            # End profiling and collect metrics
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Memory metrics
            memory_metrics = {}
            if self.memory_profiler:
                memory_snapshot = self.memory_profiler.take_snapshot(f"end_{operation_name}")
                memory_metrics = {
                    "memory_delta": memory_snapshot.get("memory_delta", 0),
                    "peak_memory": memory_snapshot.get("peak_memory", 0),
                }

            # CPU metrics
            cpu_metrics = {}
            if self.cpu_profiler:
                cpu_metrics = self.cpu_profiler.end_profiling(operation_name)

            # Record performance metric
            self._record_performance_metric(
                name=operation_name,
                duration=duration,
                memory_metrics=memory_metrics,
                cpu_metrics=cpu_metrics,
                tags=tags,
            )

    def _record_performance_metric(
        self,
        name: str,
        duration: float,
        memory_metrics: dict[str, Any],
        cpu_metrics: dict[str, Any],
        tags: dict[str, str],
    ) -> None:
        """Record performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=duration * 1000,  # Convert to milliseconds
            timestamp=time.time(),
            unit="ms",
            tags=tags,
            metadata={"memory_metrics": memory_metrics, "cpu_metrics": cpu_metrics},
        )

        with self._lock:
            self._performance_metrics.append(metric)

    def record_database_query(
        self,
        query: str,
        duration: float,
        parameters: dict[str, Any] | None = None,
        result_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record database query performance."""
        if self.db_profiler:
            self.db_profiler.record_query(query, duration, parameters, result_count, error)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_size: int | None = None,
        response_size: int | None = None,
    ) -> None:
        """Record API request performance."""
        if self.api_profiler:
            self.api_profiler.record_request(
                endpoint, method, status_code, duration, request_size, response_size
            )

    def get_performance_report(self, operation: str | None = None) -> PerformanceReport:
        """Generate performance report."""
        with self._lock:
            metrics = list(self._performance_metrics)

        if operation:
            metrics = [m for m in metrics if m.name == operation]

        if not metrics:
            return PerformanceReport(
                operation=operation or "all",
                total_time=0,
                cpu_time=0,
                memory_peak=0,
                memory_current=0,
                call_count=0,
                avg_time=0,
                min_time=0,
                max_time=0,
                p95_time=0,
                p99_time=0,
                error_count=0,
                timestamp=time.time(),
            )

        # Calculate statistics
        durations = [m.value for m in metrics]
        durations.sort()

        total_time = sum(durations)
        call_count = len(durations)
        avg_time = total_time / call_count

        # CPU and memory aggregation
        cpu_times = []
        memory_peaks = []

        for metric in metrics:
            cpu_metrics = metric.metadata.get("cpu_metrics", {})
            memory_metrics = metric.metadata.get("memory_metrics", {})

            if "cpu_time" in cpu_metrics:
                cpu_times.append(cpu_metrics["cpu_time"] * 1000)  # Convert to ms

            if "peak_memory" in memory_metrics:
                memory_peaks.append(memory_metrics["peak_memory"])

        return PerformanceReport(
            operation=operation or "all",
            total_time=total_time,
            cpu_time=sum(cpu_times) if cpu_times else 0,
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            memory_current=SystemMonitor.get_current_memory_usage(),
            call_count=call_count,
            avg_time=avg_time,
            min_time=min(durations),
            max_time=max(durations),
            p95_time=durations[int(len(durations) * 0.95)],
            p99_time=durations[int(len(durations) * 0.99)],
            error_count=0,  # TODO: Track errors separately
            timestamp=time.time(),
        )

    def get_bottlenecks(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""
        return self.bottleneck_detector.get_bottlenecks(top_n)

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "system_info": SystemMonitor.get_system_info(),
            "memory_analysis": {},
            "cpu_analysis": {},
            "database_analysis": {},
            "api_analysis": {},
            "bottlenecks": self.get_bottlenecks(),
            "recommendations": [],
        }

        # Memory analysis
        if self.memory_profiler:
            report["memory_analysis"] = {
                "current_usage": SystemMonitor.get_current_memory_usage(),
                "growth_analysis": self.memory_profiler.analyze_memory_growth(),
                "snapshots_count": len(self.memory_profiler.get_snapshots()),
            }

        # CPU analysis
        if self.cpu_profiler:
            cpu_history = self.cpu_profiler.get_cpu_history()
            if cpu_history:
                avg_cpu_usage = sum(h["cpu_usage_percent"] for h in cpu_history) / len(cpu_history)
                report["cpu_analysis"] = {
                    "avg_cpu_usage_percent": avg_cpu_usage,
                    "measurements_count": len(cpu_history),
                }

        # Database analysis
        if self.db_profiler:
            report["database_analysis"] = {
                "query_statistics": self.db_profiler.get_query_statistics(),
                "slow_queries": self.db_profiler.get_slow_queries(),
            }

        # API analysis
        if self.api_profiler:
            report["api_analysis"] = self.api_profiler.get_endpoint_statistics()

        # Generate recommendations
        report["recommendations"] = self.bottleneck_detector.generate_recommendations(report)

        return report

    async def start_background_monitoring(self, interval: float = 60.0) -> None:
        """Start background performance monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Background monitoring already running")
            return

        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started background performance monitoring (interval: {interval}s)")

    async def stop_background_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._stop_monitoring = True

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background performance monitoring")

    async def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Take system snapshots
                if self.memory_profiler:
                    self.memory_profiler.take_snapshot(f"background_{int(time.time())}")

                # Force garbage collection periodically
                if time.time() % 300 < interval:  # Every 5 minutes
                    gc.collect()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5)


# Global performance monitor
_performance_monitor: PerformanceMonitor | None = None


def initialize_performance_monitor(**kwargs: Any) -> PerformanceMonitor:
    """Initialize global performance monitor."""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(**kwargs)
    return _performance_monitor


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    if not _performance_monitor:
        raise RuntimeError(
            "Performance monitor not initialized. Call initialize_performance_monitor() first."
        )
    return _performance_monitor


# Convenience alias
trading_performance = get_performance_monitor


# Decorators for automatic performance tracking
def profile_performance(operation_name: str | None = None, **tags: Any) -> Callable[[Any], Any]:
    """Decorator for automatic performance profiling."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()

            with monitor.profile_operation(name, **tags):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()

            async with monitor.async_profile_operation(name, **tags):
                return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def profile_database_query(func: Any) -> Any:
    """Decorator for database query profiling."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        monitor = get_performance_monitor()

        start_time = time.perf_counter()
        error = None
        result_count = None

        try:
            result = await func(*args, **kwargs)

            # Try to get result count
            if hasattr(result, "__len__"):
                result_count = len(result)

            return result

        except Exception as e:
            error = str(e)
            raise

        finally:
            duration = time.perf_counter() - start_time

            # Extract query from function or args
            query = getattr(func, "__name__", "unknown_query")
            if args and isinstance(args[0], str):
                query = args[0]

            monitor.record_database_query(
                query=query, duration=duration, result_count=result_count, error=error
            )

    return wrapper


def profile_api_endpoint(endpoint: str, method: str = "GET") -> Any:
    """Decorator for API endpoint profiling."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor = get_performance_monitor()

            start_time = time.perf_counter()
            status_code = 200

            try:
                result = await func(*args, **kwargs)
                return result

            except Exception:
                status_code = 500
                raise

            finally:
                duration = time.perf_counter() - start_time

                monitor.record_api_request(
                    endpoint=endpoint, method=method, status_code=status_code, duration=duration
                )

        return wrapper

    return decorator
