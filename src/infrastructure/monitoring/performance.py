"""
Performance Monitoring and APM for AI Trading System

Application Performance Monitoring with:
- Request tracing and timing
- Database query performance
- API response times
- Memory and resource usage
- Bottleneck identification
- Trading-specific performance metrics
"""

import asyncio
import gc
import logging
import resource
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import psutil

logger = logging.getLogger(__name__)


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


class MemoryProfiler:
    """Memory usage profiler."""

    def __init__(self, enable_tracemalloc: bool = True) -> None:
        self.enable_tracemalloc = enable_tracemalloc
        self._baseline_memory = 0
        self._peak_memory = 0
        self._snapshots: deque[Any] = deque(maxlen=100)

        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()

    def start_profiling(self) -> None:
        """Start memory profiling session."""
        if self.enable_tracemalloc:
            self._baseline_memory = tracemalloc.get_traced_memory()[0]
        else:
            process = psutil.Process()
            self._baseline_memory = process.memory_info().rss

        self._peak_memory = self._baseline_memory

    def get_current_memory(self) -> int:
        """Get current memory usage."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()[0]
        else:
            process = psutil.Process()
            return int(process.memory_info().rss)

    def get_memory_delta(self) -> int:
        """Get memory usage delta from baseline."""
        current = self.get_current_memory()
        return current - self._baseline_memory

    def get_peak_memory(self) -> int:
        """Get peak memory usage."""
        current = self.get_current_memory()
        self._peak_memory = max(self._peak_memory, current)
        return self._peak_memory - self._baseline_memory

    def take_snapshot(self, label: str = "") -> dict[str, Any]:
        """Take memory snapshot."""
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "current_memory": self.get_current_memory(),
            "memory_delta": self.get_memory_delta(),
            "peak_memory": self.get_peak_memory(),
        }

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            snapshot["traced_memory"] = tracemalloc.get_traced_memory()

            # Get top memory consumers
            try:
                snapshot_obj = tracemalloc.take_snapshot()
                top_stats = snapshot_obj.statistics("lineno")[:10]
                snapshot["top_memory_consumers"] = [
                    {
                        "filename": stat.traceback[0].filename,
                        "lineno": stat.traceback[0].lineno,
                        "size": stat.size,
                        "count": stat.count,
                    }
                    for stat in top_stats
                ]
            except Exception as e:
                logger.warning(f"Failed to get memory statistics: {e}")

        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshots(self) -> list[dict[str, Any]]:
        """Get all memory snapshots."""
        return list(self._snapshots)

    def analyze_memory_growth(self) -> dict[str, Any]:
        """Analyze memory growth patterns."""
        if len(self._snapshots) < 2:
            return {"error": "Insufficient snapshots for analysis"}

        snapshots = list(self._snapshots)

        # Calculate growth rate
        first = snapshots[0]
        last = snapshots[-1]
        time_delta = last["timestamp"] - first["timestamp"]
        memory_delta = last["current_memory"] - first["current_memory"]

        growth_rate = memory_delta / time_delta if time_delta > 0 else 0

        # Find largest growth periods
        max_growth = 0
        max_growth_period = None

        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            growth = curr["current_memory"] - prev["current_memory"]

            if growth > max_growth:
                max_growth = growth
                max_growth_period = {
                    "from_label": prev["label"],
                    "to_label": curr["label"],
                    "growth_bytes": growth,
                    "time_delta": curr["timestamp"] - prev["timestamp"],
                }

        return {
            "total_growth_bytes": memory_delta,
            "growth_rate_bytes_per_sec": growth_rate,
            "max_growth_period": max_growth_period,
            "peak_memory_bytes": max(s["peak_memory"] for s in snapshots),
            "snapshot_count": len(snapshots),
        }


class CPUProfiler:
    """CPU usage profiler."""

    def __init__(self) -> None:
        self._start_times: dict[str, dict[str, Any]] = {}
        self._cpu_times: deque[dict[str, Any]] = deque(maxlen=1000)
        self._process = psutil.Process()

    def start_profiling(self, operation: str) -> None:
        """Start CPU profiling for an operation."""
        self._start_times[operation] = {
            "wall_time": time.perf_counter(),
            "cpu_time": self._get_cpu_time(),
            "process_cpu_time": self._process.cpu_times(),
        }

    def end_profiling(self, operation: str) -> dict[str, float]:
        """End CPU profiling and return metrics."""
        if operation not in self._start_times:
            return {"error": 0.0}

        start_data = self._start_times[operation]

        wall_time_delta = time.perf_counter() - start_data["wall_time"]
        cpu_time_delta = self._get_cpu_time() - start_data["cpu_time"]

        current_process_cpu = self._process.cpu_times()
        process_cpu_delta = (
            current_process_cpu.user
            - start_data["process_cpu_time"].user
            + current_process_cpu.system
            - start_data["process_cpu_time"].system
        )

        cpu_usage_percent = (cpu_time_delta / wall_time_delta * 100) if wall_time_delta > 0 else 0

        metrics = {
            "wall_time": wall_time_delta,
            "cpu_time": cpu_time_delta,
            "process_cpu_time": process_cpu_delta,
            "cpu_usage_percent": cpu_usage_percent,
            "efficiency": (cpu_time_delta / wall_time_delta) if wall_time_delta > 0 else 0,
        }

        self._cpu_times.append({"operation": operation, "timestamp": time.time(), **metrics})

        del self._start_times[operation]
        return metrics

    def _get_cpu_time(self) -> float:
        """Get current CPU time."""
        try:
            times = resource.getrusage(resource.RUSAGE_SELF)
            return times.ru_utime + times.ru_stime
        except Exception:
            return time.process_time()

    def get_cpu_history(self, operation: str | None = None) -> list[dict[str, Any]]:
        """Get CPU usage history."""
        history = list(self._cpu_times)
        if operation:
            history = [h for h in history if h.get("operation") == operation]
        return history


class DatabaseQueryProfiler:
    """Database query performance profiler."""

    def __init__(self) -> None:
        self._query_stats: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self._slow_query_threshold = 1.0  # 1 second
        self._lock = threading.Lock()

    def record_query(
        self,
        query: str,
        duration: float,
        parameters: dict[str, Any] | None = None,
        result_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record query execution statistics."""

        # Normalize query (remove specific values)
        normalized_query = self._normalize_query(query)

        query_stat = {
            "timestamp": time.time(),
            "duration": duration,
            "parameters_count": len(parameters) if parameters else 0,
            "result_count": result_count,
            "error": error,
            "is_slow": duration > self._slow_query_threshold,
        }

        with self._lock:
            self._query_stats[normalized_query].append(query_stat)

            # Keep only last 100 executions per query
            if len(self._query_stats[normalized_query]) > 100:
                self._query_stats[normalized_query] = self._query_stats[normalized_query][-100:]

    def _normalize_query(self, query: str) -> str:
        """Normalize query by removing specific values."""
        import re

        # Remove string literals
        normalized = re.sub(r"'[^']*'", "'?'", query)

        # Remove numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        # Remove whitespace variations
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def get_query_statistics(self) -> dict[str, dict[str, Any]]:
        """Get comprehensive query statistics."""
        with self._lock:
            statistics = {}

            for query, stats in self._query_stats.items():
                if not stats:
                    continue

                durations = [s["duration"] for s in stats if s["error"] is None]
                error_count = sum(1 for s in stats if s["error"] is not None)
                slow_count = sum(1 for s in stats if s["is_slow"])

                if durations:
                    durations.sort()

                    statistics[query] = {
                        "execution_count": len(stats),
                        "error_count": error_count,
                        "slow_query_count": slow_count,
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "median_duration": durations[len(durations) // 2],
                        "p95_duration": durations[int(len(durations) * 0.95)],
                        "p99_duration": durations[int(len(durations) * 0.99)],
                        "total_time": sum(durations),
                        "last_executed": max(s["timestamp"] for s in stats),
                    }

            return statistics

    def get_slow_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get slowest queries."""
        all_stats = []

        with self._lock:
            for query, stats in self._query_stats.items():
                slow_stats = [s for s in stats if s["is_slow"]]
                if slow_stats:
                    max_duration = max(s["duration"] for s in slow_stats)
                    all_stats.append(
                        {
                            "query": query,
                            "max_duration": max_duration,
                            "slow_count": len(slow_stats),
                            "total_count": len(stats),
                        }
                    )

        # Sort by max duration and return top queries
        all_stats.sort(key=lambda x: x["max_duration"], reverse=True)
        return all_stats[:limit]


class APIPerformanceTracker:
    """Track API endpoint performance."""

    def __init__(self) -> None:
        self._endpoint_stats: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_size: int | None = None,
        response_size: int | None = None,
    ) -> None:
        """Record API request statistics."""

        request_stat = {
            "timestamp": time.time(),
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "request_size": request_size,
            "response_size": response_size,
            "is_error": status_code >= 400,
            "is_slow": duration > 5.0,  # 5 second threshold
        }

        endpoint_key = f"{method} {endpoint}"

        with self._lock:
            self._endpoint_stats[endpoint_key].append(request_stat)

            # Keep only last 1000 requests per endpoint
            if len(self._endpoint_stats[endpoint_key]) > 1000:
                self._endpoint_stats[endpoint_key] = self._endpoint_stats[endpoint_key][-1000:]

    def get_endpoint_statistics(self) -> dict[str, dict[str, Any]]:
        """Get API endpoint statistics."""
        with self._lock:
            statistics = {}

            for endpoint, stats in self._endpoint_stats.items():
                if not stats:
                    continue

                durations = [s["duration"] for s in stats]
                error_count = sum(1 for s in stats if s["is_error"])
                slow_count = sum(1 for s in stats if s["is_slow"])

                durations.sort()

                # Calculate throughput (requests per minute)
                if len(stats) >= 2:
                    time_span = stats[-1]["timestamp"] - stats[0]["timestamp"]
                    throughput = len(stats) / (time_span / 60) if time_span > 0 else 0
                else:
                    throughput = 0

                statistics[endpoint] = {
                    "request_count": len(stats),
                    "error_count": error_count,
                    "error_rate": error_count / len(stats),
                    "slow_request_count": slow_count,
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p95_duration": durations[int(len(durations) * 0.95)],
                    "p99_duration": durations[int(len(durations) * 0.99)],
                    "throughput_rpm": throughput,
                    "last_request": max(s["timestamp"] for s in stats),
                }

            return statistics


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

        # Performance metrics storage
        self._performance_metrics: deque[PerformanceMetric] = deque(maxlen=10000)
        self._lock = threading.Lock()

        # Background monitoring
        self._monitoring_task: asyncio.Task[None] | None = None
        self._stop_monitoring = False

        # Resource baselines
        self._baseline_metrics = self._capture_baseline_metrics()

    def _capture_baseline_metrics(self) -> dict[str, Any]:
        """Capture baseline system metrics."""
        try:
            process = psutil.Process()
            return {
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.warning(f"Failed to capture baseline metrics: {e}")
            return {"timestamp": time.time()}

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
            memory_current=self.get_current_memory_usage(),
            call_count=call_count,
            avg_time=avg_time,
            min_time=min(durations),
            max_time=max(durations),
            p95_time=durations[int(len(durations) * 0.95)],
            p99_time=durations[int(len(durations) * 0.99)],
            error_count=0,  # TODO: Track errors separately
            timestamp=time.time(),
        )

    def get_current_memory_usage(self) -> int:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            return int(process.memory_info().rss)
        except Exception:
            return 0

    def get_bottlenecks(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Database bottlenecks
        if self.db_profiler:
            slow_queries = self.db_profiler.get_slow_queries(top_n)
            for query in slow_queries:
                bottlenecks.append(
                    {
                        "type": "database_query",
                        "description": f"Slow query: {query['query'][:100]}...",
                        "max_duration": query["max_duration"],
                        "slow_count": query["slow_count"],
                        "severity": "high" if query["max_duration"] > 5.0 else "medium",
                    }
                )

        # API bottlenecks
        if self.api_profiler:
            endpoint_stats = self.api_profiler.get_endpoint_statistics()
            slow_endpoints = sorted(
                endpoint_stats.items(), key=lambda x: x[1]["p99_duration"], reverse=True
            )[:top_n]

            for endpoint, stats in slow_endpoints:
                if stats["p99_duration"] > 2.0:  # 2 second threshold
                    bottlenecks.append(
                        {
                            "type": "api_endpoint",
                            "description": f"Slow endpoint: {endpoint}",
                            "p99_duration": stats["p99_duration"],
                            "error_rate": stats["error_rate"],
                            "severity": "high" if stats["p99_duration"] > 5.0 else "medium",
                        }
                    )

        # Memory bottlenecks
        if self.memory_profiler:
            memory_analysis = self.memory_profiler.analyze_memory_growth()
            if memory_analysis.get("growth_rate_bytes_per_sec", 0) > 1024 * 1024:  # 1MB/sec
                bottlenecks.append(
                    {
                        "type": "memory_growth",
                        "description": "High memory growth rate detected",
                        "growth_rate_mb_per_sec": memory_analysis["growth_rate_bytes_per_sec"]
                        / (1024 * 1024),
                        "severity": "high",
                    }
                )

        # Sort by severity and duration
        bottlenecks.sort(
            key=lambda x: (
                x["severity"] == "high",
                x.get("max_duration", x.get("p99_duration", 0)),
            ),
            reverse=True,
        )

        return bottlenecks[:top_n]

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
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
                "current_usage": self.get_current_memory_usage(),
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
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _get_system_info(self) -> dict[str, Any]:
        """Get current system information."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "python_version": sys.version,
                "process_memory_rss": process.memory_info().rss,
                "process_cpu_percent": process.cpu_percent(),
                "process_threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Memory recommendations
        memory_analysis = report.get("memory_analysis", {})
        growth_rate = memory_analysis.get("growth_analysis", {}).get("growth_rate_bytes_per_sec", 0)

        if growth_rate > 1024 * 1024:  # 1MB/sec
            recommendations.append(
                "High memory growth rate detected. Consider implementing object pooling "
                "or reviewing memory-intensive operations."
            )

        # CPU recommendations
        cpu_analysis = report.get("cpu_analysis", {})
        avg_cpu = cpu_analysis.get("avg_cpu_usage_percent", 0)

        if avg_cpu > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing computational operations "
                "or implementing async processing."
            )

        # Database recommendations
        db_analysis = report.get("database_analysis", {})
        slow_queries = db_analysis.get("slow_queries", [])

        if slow_queries:
            recommendations.append(
                f"Found {len(slow_queries)} slow database queries. "
                "Consider adding indexes or optimizing query structure."
            )

        # API recommendations
        api_analysis = report.get("api_analysis", {})
        slow_endpoints = [
            endpoint
            for endpoint, stats in api_analysis.items()
            if stats.get("p99_duration", 0) > 2.0
        ]

        if slow_endpoints:
            recommendations.append(
                f"Found {len(slow_endpoints)} slow API endpoints. "
                "Consider caching, pagination, or request optimization."
            )

        return recommendations

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
