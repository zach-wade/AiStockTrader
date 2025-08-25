"""
Comprehensive unit tests for performance monitoring module.

Tests performance metrics collection, memory profiling, CPU tracking,
database query performance, API monitoring, and bottleneck identification.
"""

import asyncio
import time
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.infrastructure.monitoring.performance import (
    APIPerformanceTracker,
    CPUProfiler,
    DatabaseQueryProfiler,
    MemoryProfiler,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
    get_performance_monitor,
    initialize_performance_monitor,
    profile_api_endpoint,
    profile_database_query,
    profile_performance,
)


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""

    def test_metric_creation(self):
        """Test creating performance metric."""
        metric = PerformanceMetric(
            name="test_operation",
            value=100.5,
            timestamp=1234567890.0,
            unit="ms",
            tags={"env": "test"},
            metadata={"additional": "data"},
        )

        assert metric.name == "test_operation"
        assert metric == 100.5
        assert metric.timestamp == 1234567890.0
        assert metric.unit == "ms"
        assert metric.tags == {"env": "test"}
        assert metric.metadata == {"additional": "data"}

    def test_metric_defaults(self):
        """Test performance metric defaults."""
        metric = PerformanceMetric(name="test", value=50.0, timestamp=time.time())

        assert metric.unit == "ms"
        assert metric.tags == {}
        assert metric.metadata == {}


class TestPerformanceReport:
    """Test PerformanceReport dataclass."""

    def test_report_creation(self):
        """Test creating performance report."""
        report = PerformanceReport(
            operation="test_op",
            total_time=1000.0,
            cpu_time=800.0,
            memory_peak=1024000,
            memory_current=512000,
            call_count=10,
            avg_time=100.0,
            min_time=50.0,
            max_time=200.0,
            p95_time=180.0,
            p99_time=195.0,
            error_count=1,
            timestamp=1234567890.0,
            details={"extra": "info"},
        )

        assert report.operation == "test_op"
        assert report.total_time == 1000.0
        assert report.cpu_time == 800.0
        assert report.memory_peak == 1024000
        assert report.call_count == 10
        assert report.avg_time == 100.0
        assert report.error_count == 1
        assert report.details == {"extra": "info"}


class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""

    @patch("tracemalloc.is_tracing")
    @patch("tracemalloc.start")
    def test_initialization_with_tracemalloc(self, mock_start, mock_is_tracing):
        """Test initialization with tracemalloc enabled."""
        mock_is_tracing.return_value = False

        profiler = MemoryProfiler(enable_tracemalloc=True)

        assert profiler.enable_tracemalloc is True
        mock_start.assert_called_once()

    @patch("psutil.Process")
    def test_initialization_without_tracemalloc(self, mock_process):
        """Test initialization without tracemalloc."""
        profiler = MemoryProfiler(enable_tracemalloc=False)

        assert profiler.enable_tracemalloc is False
        assert profiler._baseline_memory == 0
        assert profiler._peak_memory == 0

    @patch("tracemalloc.get_traced_memory")
    @patch("tracemalloc.is_tracing")
    def test_start_profiling_with_tracemalloc(self, mock_is_tracing, mock_get_memory):
        """Test starting profiling with tracemalloc."""
        mock_is_tracing.return_value = True
        mock_get_memory.return_value = (1024000, 2048000)

        profiler = MemoryProfiler(enable_tracemalloc=True)
        profiler.start_profiling()

        assert profiler._baseline_memory == 1024000
        assert profiler._peak_memory == 1024000

    @patch("psutil.Process")
    def test_start_profiling_without_tracemalloc(self, mock_process_class):
        """Test starting profiling without tracemalloc."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 2048000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        profiler = MemoryProfiler(enable_tracemalloc=False)
        profiler.start_profiling()

        assert profiler._baseline_memory == 2048000
        assert profiler._peak_memory == 2048000

    @patch("tracemalloc.get_traced_memory")
    @patch("tracemalloc.is_tracing")
    def test_get_current_memory(self, mock_is_tracing, mock_get_memory):
        """Test getting current memory usage."""
        mock_is_tracing.return_value = True
        mock_get_memory.return_value = (3072000, 4096000)

        profiler = MemoryProfiler(enable_tracemalloc=True)
        current = profiler.get_current_memory()

        assert current == 3072000

    @patch("psutil.Process")
    def test_get_memory_delta(self, mock_process_class):
        """Test getting memory delta."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 3072000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        profiler = MemoryProfiler(enable_tracemalloc=False)
        profiler._baseline_memory = 2048000

        delta = profiler.get_memory_delta()
        assert delta == 1024000

    @patch("psutil.Process")
    def test_get_peak_memory(self, mock_process_class):
        """Test getting peak memory usage."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 4096000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        profiler = MemoryProfiler(enable_tracemalloc=False)
        profiler._baseline_memory = 2048000
        profiler._peak_memory = 3072000

        peak = profiler.get_peak_memory()
        assert peak == 2048000  # 4096000 - 2048000

    @patch("time.time")
    @patch("psutil.Process")
    def test_take_snapshot(self, mock_process_class, mock_time):
        """Test taking memory snapshot."""
        mock_time.return_value = 1234567890.0

        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 3072000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        profiler = MemoryProfiler(enable_tracemalloc=False)
        profiler._baseline_memory = 2048000

        snapshot = profiler.take_snapshot("test_point")

        assert snapshot["timestamp"] == 1234567890.0
        assert snapshot["label"] == "test_point"
        assert snapshot["current_memory"] == 3072000
        assert snapshot["memory_delta"] == 1024000
        assert len(profiler._snapshots) == 1

    @patch("tracemalloc.take_snapshot")
    @patch("tracemalloc.get_traced_memory")
    @patch("tracemalloc.is_tracing")
    def test_take_snapshot_with_tracemalloc(
        self, mock_is_tracing, mock_get_memory, mock_take_snapshot
    ):
        """Test taking snapshot with tracemalloc details."""
        mock_is_tracing.return_value = True
        mock_get_memory.return_value = (3072000, 4096000)

        mock_stat = Mock()
        mock_stat.traceback.filename = "test.py"
        mock_stat.traceback.lineno = 100
        mock_stat.size = 1024
        mock_stat.count = 10

        mock_snapshot_obj = Mock()
        mock_snapshot_obj.statistics.return_value = [mock_stat]
        mock_take_snapshot.return_value = mock_snapshot_obj

        profiler = MemoryProfiler(enable_tracemalloc=True)
        profiler._baseline_memory = 2048000

        snapshot = profiler.take_snapshot("test")

        assert "traced_memory" in snapshot
        assert "top_memory_consumers" in snapshot
        assert len(snapshot["top_memory_consumers"]) == 1
        assert snapshot["top_memory_consumers"][0]["filename"] == "test.py"

    def test_analyze_memory_growth(self):
        """Test analyzing memory growth patterns."""
        profiler = MemoryProfiler(enable_tracemalloc=False)

        # Add test snapshots
        profiler._snapshots.append(
            {
                "timestamp": 1000.0,
                "label": "start",
                "current_memory": 1024000,
                "peak_memory": 1024000,
            }
        )
        profiler._snapshots.append(
            {
                "timestamp": 1010.0,
                "label": "middle",
                "current_memory": 2048000,
                "peak_memory": 2048000,
            }
        )
        profiler._snapshots.append(
            {"timestamp": 1020.0, "label": "end", "current_memory": 3072000, "peak_memory": 3072000}
        )

        analysis = profiler.analyze_memory_growth()

        assert analysis["total_growth_bytes"] == 2048000
        assert analysis["growth_rate_bytes_per_sec"] == 102400.0  # 2048000 / 20
        assert analysis["max_growth_period"]["from_label"] == "start"
        assert analysis["max_growth_period"]["to_label"] == "middle"
        assert analysis["max_growth_period"]["growth_bytes"] == 1024000
        assert analysis["peak_memory_bytes"] == 3072000
        assert analysis["snapshot_count"] == 3

    def test_analyze_memory_growth_insufficient_data(self):
        """Test analyzing memory growth with insufficient data."""
        profiler = MemoryProfiler(enable_tracemalloc=False)

        analysis = profiler.analyze_memory_growth()
        assert "error" in analysis


class TestCPUProfiler:
    """Test CPUProfiler functionality."""

    @patch("psutil.Process")
    def test_initialization(self, mock_process_class):
        """Test CPU profiler initialization."""
        profiler = CPUProfiler()

        assert profiler._start_times == {}
        assert isinstance(profiler._cpu_times, deque)
        assert profiler._process is not None

    @patch("time.perf_counter")
    @patch("resource.getrusage")
    @patch("psutil.Process")
    def test_start_profiling(self, mock_process_class, mock_getrusage, mock_perf_counter):
        """Test starting CPU profiling."""
        mock_perf_counter.return_value = 1000.0

        mock_rusage = Mock()
        mock_rusage.ru_utime = 10.0
        mock_rusage.ru_stime = 5.0
        mock_getrusage.return_value = mock_rusage

        mock_process = Mock()
        mock_cpu_times = Mock()
        mock_cpu_times.user = 8.0
        mock_cpu_times.system = 4.0
        mock_process.cpu_times.return_value = mock_cpu_times
        mock_process_class.return_value = mock_process

        profiler = CPUProfiler()
        profiler.start_profiling("test_op")

        assert "test_op" in profiler._start_times
        assert profiler._start_times["test_op"]["wall_time"] == 1000.0
        assert profiler._start_times["test_op"]["cpu_time"] == 15.0

    @patch("time.perf_counter")
    @patch("resource.getrusage")
    @patch("psutil.Process")
    def test_end_profiling(self, mock_process_class, mock_getrusage, mock_perf_counter):
        """Test ending CPU profiling."""
        mock_process = Mock()
        mock_cpu_times = Mock()
        mock_cpu_times.user = 12.0
        mock_cpu_times.system = 6.0
        mock_process.cpu_times.return_value = mock_cpu_times
        mock_process_class.return_value = mock_process

        profiler = CPUProfiler()

        # Start profiling
        profiler._start_times["test_op"] = {
            "wall_time": 1000.0,
            "cpu_time": 15.0,
            "process_cpu_time": Mock(user=8.0, system=4.0),
        }

        # End profiling
        mock_perf_counter.return_value = 1002.0
        mock_rusage = Mock()
        mock_rusage.ru_utime = 11.5
        mock_rusage.ru_stime = 5.5
        mock_getrusage.return_value = mock_rusage

        metrics = profiler.end_profiling("test_op")

        assert metrics["wall_time"] == 2.0
        assert metrics["cpu_time"] == 2.0  # (11.5 + 5.5) - 15.0
        assert metrics["process_cpu_time"] == 6.0  # (12 - 8) + (6 - 4)
        assert metrics["cpu_usage_percent"] == 100.0
        assert metrics["efficiency"] == 1.0
        assert "test_op" not in profiler._start_times

    def test_end_profiling_not_started(self):
        """Test ending profiling for operation not started."""
        profiler = CPUProfiler()
        metrics = profiler.end_profiling("not_started")
        assert metrics == {"error": "Operation not started"}

    @patch("time.process_time")
    def test_get_cpu_time_fallback(self, mock_process_time):
        """Test CPU time retrieval with fallback."""
        mock_process_time.return_value = 100.0

        profiler = CPUProfiler()

        with patch("resource.getrusage", side_effect=Exception("Not available")):
            cpu_time = profiler._get_cpu_time()
            assert cpu_time == 100.0

    def test_get_cpu_history(self):
        """Test getting CPU usage history."""
        profiler = CPUProfiler()

        # Add test data
        profiler._cpu_times.append(
            {"operation": "op1", "timestamp": 1000.0, "cpu_usage_percent": 50.0}
        )
        profiler._cpu_times.append(
            {"operation": "op2", "timestamp": 1001.0, "cpu_usage_percent": 75.0}
        )
        profiler._cpu_times.append(
            {"operation": "op1", "timestamp": 1002.0, "cpu_usage_percent": 60.0}
        )

        # Get all history
        history = profiler.get_cpu_history()
        assert len(history) == 3

        # Get filtered history
        op1_history = profiler.get_cpu_history("op1")
        assert len(op1_history) == 2
        assert all(h["operation"] == "op1" for h in op1_history)


class TestDatabaseQueryProfiler:
    """Test DatabaseQueryProfiler functionality."""

    def test_initialization(self):
        """Test database query profiler initialization."""
        profiler = DatabaseQueryProfiler()

        assert profiler._query_stats == {}
        assert profiler._slow_query_threshold == 1.0

    @patch("time.time")
    def test_record_query(self, mock_time):
        """Test recording query execution."""
        mock_time.return_value = 1234567890.0

        profiler = DatabaseQueryProfiler()

        profiler.record_query(
            query="SELECT * FROM users WHERE id = 123",
            duration=0.5,
            parameters={"id": 123},
            result_count=1,
        )

        # Query should be normalized
        normalized = "SELECT * FROM users WHERE id = ?"
        assert normalized in profiler._query_stats
        assert len(profiler._query_stats[normalized]) == 1

        stat = profiler._query_stats[normalized][0]
        assert stat["duration"] == 0.5
        assert stat["parameters_count"] == 1
        assert stat["result_count"] == 1
        assert stat["is_slow"] is False

    def test_record_slow_query(self):
        """Test recording slow query."""
        profiler = DatabaseQueryProfiler()

        profiler.record_query(query="SELECT * FROM large_table", duration=2.5, result_count=10000)

        normalized = "SELECT * FROM large_table"
        stat = profiler._query_stats[normalized][0]
        assert stat["is_slow"] is True

    def test_record_query_with_error(self):
        """Test recording query with error."""
        profiler = DatabaseQueryProfiler()

        profiler.record_query(
            query="SELECT * FROM non_existent", duration=0.1, error="Table not found"
        )

        normalized = "SELECT * FROM non_existent"
        stat = profiler._query_stats[normalized][0]
        assert stat["error"] == "Table not found"

    def test_normalize_query(self):
        """Test query normalization."""
        profiler = DatabaseQueryProfiler()

        # Test string literal removal
        query1 = "SELECT * FROM users WHERE name = 'John Doe'"
        normalized1 = profiler._normalize_query(query1)
        assert normalized1 == "SELECT * FROM users WHERE name = '?'"

        # Test numeric literal removal
        query2 = "SELECT * FROM orders WHERE id = 12345 AND amount > 100.50"
        normalized2 = profiler._normalize_query(query2)
        assert "?" in normalized2
        assert "12345" not in normalized2
        assert "100.50" not in normalized2

        # Test whitespace normalization
        query3 = "SELECT   *   FROM   users   WHERE   active   =   true"
        normalized3 = profiler._normalize_query(query3)
        assert "  " not in normalized3

    def test_query_statistics(self):
        """Test getting query statistics."""
        profiler = DatabaseQueryProfiler()

        # Add test queries
        for i in range(10):
            profiler.record_query(
                query="SELECT * FROM users WHERE id = ?", duration=0.1 * (i + 1), result_count=1
            )

        # Add one error
        profiler.record_query(
            query="SELECT * FROM users WHERE id = ?", duration=0.05, error="Connection lost"
        )

        stats = profiler.get_query_statistics()
        query_stats = stats["SELECT * FROM users WHERE id = ?"]

        assert query_stats["execution_count"] == 11
        assert query_stats["error_count"] == 1
        assert query_stats["avg_duration"] == 0.55  # Average of 0.1 to 1.0
        assert query_stats["min_duration"] == 0.1
        assert query_stats["max_duration"] == 1.0
        assert query_stats["median_duration"] == 0.5
        assert query_stats["p95_duration"] == 0.9
        assert query_stats["p99_duration"] == 0.9
        assert query_stats["total_time"] == 5.5

    def test_get_slow_queries(self):
        """Test getting slow queries."""
        profiler = DatabaseQueryProfiler()

        # Add regular query
        profiler.record_query("SELECT * FROM small_table", duration=0.5)

        # Add slow queries
        profiler.record_query("SELECT * FROM large_table1", duration=3.0)
        profiler.record_query("SELECT * FROM large_table2", duration=2.0)
        profiler.record_query("SELECT * FROM large_table1", duration=2.5)

        slow_queries = profiler.get_slow_queries(limit=2)

        assert len(slow_queries) == 2
        assert slow_queries[0]["query"] == "SELECT * FROM large_table1"
        assert slow_queries[0]["max_duration"] == 3.0
        assert slow_queries[0]["slow_count"] == 2
        assert slow_queries[1]["query"] == "SELECT * FROM large_table2"
        assert slow_queries[1]["max_duration"] == 2.0

    def test_query_limit_enforcement(self):
        """Test that query history is limited."""
        profiler = DatabaseQueryProfiler()

        # Add more than 100 queries
        for i in range(150):
            profiler.record_query(query="SELECT * FROM test", duration=0.1)

        # Should only keep last 100
        assert len(profiler._query_stats["SELECT * FROM test"]) == 100


class TestAPIPerformanceTracker:
    """Test APIPerformanceTracker functionality."""

    def test_initialization(self):
        """Test API performance tracker initialization."""
        tracker = APIPerformanceTracker()
        assert tracker._endpoint_stats == {}

    @patch("time.time")
    def test_record_request(self, mock_time):
        """Test recording API request."""
        mock_time.return_value = 1234567890.0

        tracker = APIPerformanceTracker()

        tracker.record_request(
            endpoint="/api/users",
            method="GET",
            status_code=200,
            duration=0.5,
            request_size=100,
            response_size=1024,
        )

        key = "GET /api/users"
        assert key in tracker._endpoint_stats
        assert len(tracker._endpoint_stats[key]) == 1

        stat = tracker._endpoint_stats[key][0]
        assert stat["method"] == "GET"
        assert stat["status_code"] == 200
        assert stat["duration"] == 0.5
        assert stat["is_error"] is False
        assert stat["is_slow"] is False

    def test_record_error_request(self):
        """Test recording error request."""
        tracker = APIPerformanceTracker()

        tracker.record_request(endpoint="/api/users", method="POST", status_code=500, duration=1.0)

        key = "POST /api/users"
        stat = tracker._endpoint_stats[key][0]
        assert stat["is_error"] is True

    def test_record_slow_request(self):
        """Test recording slow request."""
        tracker = APIPerformanceTracker()

        tracker.record_request(endpoint="/api/data", method="GET", status_code=200, duration=6.0)

        key = "GET /api/data"
        stat = tracker._endpoint_stats[key][0]
        assert stat["is_slow"] is True

    def test_get_endpoint_statistics(self):
        """Test getting endpoint statistics."""
        tracker = APIPerformanceTracker()

        # Add test requests with different timestamps
        with patch("time.time") as mock_time:
            # First request at t=1000
            mock_time.return_value = 1000.0
            tracker.record_request("/api/users", "GET", 200, 0.5)

            # More requests
            for i in range(9):
                mock_time.return_value = 1000.0 + (i + 1) * 10
                status = 500 if i == 5 else 200
                duration = 6.0 if i == 7 else 0.3 + i * 0.1
                tracker.record_request("/api/users", "GET", status, duration)

        stats = tracker.get_endpoint_statistics()
        endpoint_stats = stats["GET /api/users"]

        assert endpoint_stats["request_count"] == 10
        assert endpoint_stats["error_count"] == 1
        assert endpoint_stats["error_rate"] == 0.1
        assert endpoint_stats["slow_request_count"] == 1
        assert endpoint_stats["throughput_rpm"] > 0
        assert "avg_duration" in endpoint_stats
        assert "min_duration" in endpoint_stats
        assert "max_duration" in endpoint_stats
        assert "p95_duration" in endpoint_stats
        assert "p99_duration" in endpoint_stats

    def test_request_limit_enforcement(self):
        """Test that request history is limited."""
        tracker = APIPerformanceTracker()

        # Add more than 1000 requests
        for i in range(1500):
            tracker.record_request(
                endpoint="/api/test", method="GET", status_code=200, duration=0.1
            )

        # Should only keep last 1000
        assert len(tracker._endpoint_stats["GET /api/test"]) == 1000


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""

    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(
            enable_memory_profiling=True,
            enable_cpu_profiling=True,
            enable_query_profiling=True,
            enable_api_profiling=True,
        )

        assert monitor.memory_profiler is not None
        assert monitor.cpu_profiler is not None
        assert monitor.db_profiler is not None
        assert monitor.api_profiler is not None
        assert isinstance(monitor._performance_metrics, deque)

    def test_initialization_partial(self):
        """Test partial initialization."""
        monitor = PerformanceMonitor(enable_memory_profiling=False, enable_cpu_profiling=False)

        assert monitor.memory_profiler is None
        assert monitor.cpu_profiler is None
        assert monitor.db_profiler is not None
        assert monitor.api_profiler is not None

    @patch("psutil.Process")
    def test_capture_baseline_metrics(self, mock_process_class):
        """Test capturing baseline metrics."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024000
        mock_memory_info.vms = 2048000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 25.0
        mock_process.open_files.return_value = [1, 2, 3]
        mock_process.num_threads.return_value = 4
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        baseline = monitor._baseline_metrics

        assert baseline["memory_rss"] == 1024000
        assert baseline["memory_vms"] == 2048000
        assert baseline["cpu_percent"] == 25.0
        assert baseline["open_files"] == 3
        assert baseline["threads"] == 4

    @patch("time.perf_counter")
    def test_profile_operation_context(self, mock_perf_counter):
        """Test profile_operation context manager."""
        mock_perf_counter.side_effect = [0.0, 1.5]

        monitor = PerformanceMonitor(enable_memory_profiling=False, enable_cpu_profiling=False)

        with monitor.profile_operation("test_op", env="test"):
            # Simulate operation
            pass

        assert len(monitor._performance_metrics) == 1
        metric = monitor._performance_metrics[0]
        assert metric.name == "test_op"
        assert metric == 1500.0  # 1.5 seconds in ms
        assert metric.tags == {"env": "test"}

    @pytest.mark.asyncio
    @patch("time.perf_counter")
    async def test_async_profile_operation_context(self, mock_perf_counter):
        """Test async profile_operation context manager."""
        mock_perf_counter.side_effect = [0.0, 2.0]

        monitor = PerformanceMonitor(enable_memory_profiling=False, enable_cpu_profiling=False)

        async with monitor.async_profile_operation("async_op"):
            await asyncio.sleep(0.01)

        assert len(monitor._performance_metrics) == 1
        metric = monitor._performance_metrics[0]
        assert metric.name == "async_op"
        assert metric == 2000.0  # 2 seconds in ms

    def test_record_database_query(self):
        """Test recording database query."""
        monitor = PerformanceMonitor(enable_query_profiling=True)

        monitor.record_database_query(query="SELECT * FROM users", duration=0.5, result_count=10)

        stats = monitor.db_profiler.get_query_statistics()
        assert "SELECT * FROM users" in stats

    def test_record_api_request(self):
        """Test recording API request."""
        monitor = PerformanceMonitor(enable_api_profiling=True)

        monitor.record_api_request(
            endpoint="/api/test", method="GET", status_code=200, duration=0.3
        )

        stats = monitor.api_profiler.get_endpoint_statistics()
        assert "GET /api/test" in stats

    def test_get_performance_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor(enable_memory_profiling=False, enable_cpu_profiling=False)

        # Add test metrics
        for i in range(10):
            monitor._performance_metrics.append(
                PerformanceMetric(
                    name="test_op",
                    value=50.0 + i * 10,
                    timestamp=time.time(),
                    metadata={
                        "cpu_metrics": {"cpu_time": 0.04 + i * 0.01},
                        "memory_metrics": {"peak_memory": 1024000 + i * 1000},
                    },
                )
            )

        report = monitor.get_performance_report("test_op")

        assert report.operation == "test_op"
        assert report.call_count == 10
        assert report.total_time == 950.0  # Sum of 50 to 140
        assert report.avg_time == 95.0
        assert report.min_time == 50.0
        assert report.max_time == 140.0
        assert report.p95_time == 130.0
        assert report.p99_time == 140.0
        assert report.cpu_time > 0
        assert report.memory_peak > 0

    def test_get_performance_report_empty(self):
        """Test generating report with no data."""
        monitor = PerformanceMonitor()

        report = monitor.get_performance_report("nonexistent")

        assert report.operation == "nonexistent"
        assert report.call_count == 0
        assert report.total_time == 0

    @patch("psutil.Process")
    def test_get_current_memory_usage(self, mock_process_class):
        """Test getting current memory usage."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 3072000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        memory = monitor.get_current_memory_usage()

        assert memory == 3072000

    def test_get_bottlenecks_database(self):
        """Test identifying database bottlenecks."""
        monitor = PerformanceMonitor(enable_query_profiling=True)

        # Add slow queries
        monitor.db_profiler.record_query("SELECT * FROM huge_table", duration=10.0)
        monitor.db_profiler.record_query("SELECT * FROM huge_table", duration=8.0)

        bottlenecks = monitor.get_bottlenecks(top_n=5)

        assert len(bottlenecks) > 0
        db_bottleneck = bottlenecks[0]
        assert db_bottleneck["type"] == "database_query"
        assert db_bottleneck["severity"] == "high"
        assert "Slow query" in db_bottleneck["description"]

    def test_get_bottlenecks_api(self):
        """Test identifying API bottlenecks."""
        monitor = PerformanceMonitor(enable_api_profiling=True)

        # Add slow API calls
        for i in range(5):
            monitor.api_profiler.record_request("/api/slow", "GET", 200, 6.0)

        bottlenecks = monitor.get_bottlenecks(top_n=5)

        api_bottlenecks = [b for b in bottlenecks if b["type"] == "api_endpoint"]
        assert len(api_bottlenecks) > 0
        assert api_bottlenecks[0]["severity"] == "high"

    def test_get_bottlenecks_memory(self):
        """Test identifying memory bottlenecks."""
        monitor = PerformanceMonitor(enable_memory_profiling=True)

        # Simulate memory growth
        monitor.memory_profiler._snapshots.append(
            {"timestamp": 1000.0, "current_memory": 1024000000, "peak_memory": 1024000000}
        )
        monitor.memory_profiler._snapshots.append(
            {"timestamp": 1001.0, "current_memory": 2048000000, "peak_memory": 2048000000}
        )

        bottlenecks = monitor.get_bottlenecks()

        memory_bottlenecks = [b for b in bottlenecks if b["type"] == "memory_growth"]
        if memory_bottlenecks:
            assert memory_bottlenecks[0]["severity"] == "high"

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.Process")
    def test_generate_comprehensive_report(
        self, mock_process_class, mock_disk, mock_memory, mock_cpu_count
    ):
        """Test generating comprehensive report."""
        # Setup mocks
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.memory_info.return_value = Mock(rss=1024000)
        mock_process.cpu_percent.return_value = 50.0
        mock_process.num_threads.return_value = 10
        mock_process.open_files.return_value = []
        mock_process_class.return_value = mock_process

        mock_memory.return_value = Mock(total=8192000000, available=4096000000)
        mock_disk.return_value = Mock(percent=60.0)
        mock_cpu_count.return_value = 4

        monitor = PerformanceMonitor()

        # Add some test data
        monitor._performance_metrics.append(PerformanceMetric("test", 100.0, time.time()))

        report = monitor.generate_comprehensive_report()

        assert "timestamp" in report
        assert "system_info" in report
        assert "memory_analysis" in report
        assert "cpu_analysis" in report
        assert "database_analysis" in report
        assert "api_analysis" in report
        assert "bottlenecks" in report
        assert "recommendations" in report

    def test_generate_recommendations(self):
        """Test generating performance recommendations."""
        monitor = PerformanceMonitor()

        # Test memory recommendation
        report = {
            "memory_analysis": {"growth_analysis": {"growth_rate_bytes_per_sec": 2000000}},
            "cpu_analysis": {},
            "database_analysis": {},
            "api_analysis": {},
        }

        recommendations = monitor._generate_recommendations(report)
        assert any("memory growth" in r.lower() for r in recommendations)

        # Test CPU recommendation
        report["cpu_analysis"] = {"avg_cpu_usage_percent": 85}
        recommendations = monitor._generate_recommendations(report)
        assert any("cpu usage" in r.lower() for r in recommendations)

        # Test database recommendation
        report["database_analysis"] = {"slow_queries": [1, 2, 3]}
        recommendations = monitor._generate_recommendations(report)
        assert any("slow database queries" in r.lower() for r in recommendations)

        # Test API recommendation
        report["api_analysis"] = {
            "endpoint1": {"p99_duration": 3.0},
            "endpoint2": {"p99_duration": 2.5},
        }
        recommendations = monitor._generate_recommendations(report)
        assert any("slow api endpoints" in r.lower() for r in recommendations)

    @pytest.mark.asyncio
    async def test_background_monitoring(self):
        """Test background monitoring."""
        monitor = PerformanceMonitor()

        # Start monitoring
        await monitor.start_background_monitoring(interval=0.1)
        assert monitor._monitoring_task is not None
        assert not monitor._monitoring_task.done()

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_background_monitoring()
        assert monitor._stop_monitoring is True

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self):
        """Test error handling in monitoring loop."""
        monitor = PerformanceMonitor()

        # Mock memory profiler to raise error
        with patch.object(
            monitor.memory_profiler, "take_snapshot", side_effect=Exception("Test error")
        ):
            await monitor.start_background_monitoring(interval=0.05)
            await asyncio.sleep(0.1)
            await monitor.stop_background_monitoring()

        # Should handle error and continue


class TestGlobalPerformanceMonitor:
    """Test global performance monitor functions."""

    def test_initialize_performance_monitor(self):
        """Test initializing global monitor."""
        monitor = initialize_performance_monitor(
            enable_memory_profiling=True, enable_cpu_profiling=False
        )

        assert monitor is not None
        assert monitor.memory_profiler is not None
        assert monitor.cpu_profiler is None

    def test_get_performance_monitor(self):
        """Test getting global monitor."""
        # Initialize first
        initialize_performance_monitor()

        monitor = get_performance_monitor()
        assert monitor is not None

    def test_get_performance_monitor_not_initialized(self):
        """Test getting monitor when not initialized."""
        # Reset global
        import src.infrastructure.monitoring.performance as perf_module

        perf_module._performance_monitor = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_performance_monitor()


class TestPerformanceDecorators:
    """Test performance tracking decorators."""

    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    def test_profile_performance_sync(self, mock_get_monitor):
        """Test profile_performance decorator on sync function."""
        mock_monitor = Mock()
        mock_profile_context = MagicMock()
        mock_monitor.profile_operation.return_value = mock_profile_context
        mock_get_monitor.return_value = mock_monitor

        @profile_performance("custom_op", env="test")
        def test_func(x):
            return x * 2

        result = test_func(5)

        assert result == 10
        mock_monitor.profile_operation.assert_called_once_with("custom_op", env="test")

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    async def test_profile_performance_async(self, mock_get_monitor):
        """Test profile_performance decorator on async function."""
        mock_monitor = Mock()
        mock_profile_context = MagicMock()
        mock_monitor.async_profile_operation.return_value = mock_profile_context
        mock_get_monitor.return_value = mock_monitor

        @profile_performance(env="prod")
        async def test_func(x):
            await asyncio.sleep(0.01)
            return x + 10

        result = await test_func(5)

        assert result == 15
        mock_monitor.async_profile_operation.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    async def test_profile_database_query_decorator(self, mock_get_monitor):
        """Test profile_database_query decorator."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @profile_database_query
        async def execute_query(query):
            await asyncio.sleep(0.01)
            return [1, 2, 3]

        result = await execute_query("SELECT * FROM test")

        assert result == [1, 2, 3]
        mock_monitor.record_database_query.assert_called_once()
        call_args = mock_monitor.record_database_query.call_args
        assert call_args[1]["query"] == "SELECT * FROM test"
        assert call_args[1]["result_count"] == 3
        assert call_args[1]["error"] is None

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    async def test_profile_database_query_error(self, mock_get_monitor):
        """Test profile_database_query decorator with error."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @profile_database_query
        async def failing_query():
            raise ValueError("Query failed")

        with pytest.raises(ValueError):
            await failing_query()

        mock_monitor.record_database_query.assert_called_once()
        call_args = mock_monitor.record_database_query.call_args
        assert call_args[1]["error"] == "Query failed"

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    async def test_profile_api_endpoint_decorator(self, mock_get_monitor):
        """Test profile_api_endpoint decorator."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @profile_api_endpoint("/api/test", "POST")
        async def api_handler():
            await asyncio.sleep(0.01)
            return {"status": "ok"}

        result = await api_handler()

        assert result == {"status": "ok"}
        mock_monitor.record_api_request.assert_called_once()
        call_args = mock_monitor.record_api_request.call_args
        assert call_args[1]["endpoint"] == "/api/test"
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["status_code"] == 200

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.performance.get_performance_monitor")
    async def test_profile_api_endpoint_error(self, mock_get_monitor):
        """Test profile_api_endpoint decorator with error."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @profile_api_endpoint("/api/error", "GET")
        async def failing_handler():
            raise Exception("API error")

        with pytest.raises(Exception):
            await failing_handler()

        mock_monitor.record_api_request.assert_called_once()
        call_args = mock_monitor.record_api_request.call_args
        assert call_args[1]["status_code"] == 500
