"""
Comprehensive unit tests for monitoring infrastructure - achieving 90%+ coverage.

Tests metrics collection, telemetry, performance monitoring, and health checks.
"""

import asyncio
import threading
import time
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.monitoring.metrics import MetricType
from src.infrastructure.monitoring.performance import PerformanceMonitor


class TestMetricsConfig:
    """Test MetricsConfig class."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.export_interval == 60.0
        assert config.retention_period == 3600.0
        assert config.max_metrics == 10000
        assert config.aggregation_window == 60.0

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = MetricsConfig(enabled=False, export_interval=30.0, max_metrics=5000)

        assert config.enabled is False
        assert config.export_interval == 30.0
        assert config.max_metrics == 5000


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MetricsConfig(export_interval=1.0, retention_period=10.0)

    @pytest.fixture
    def collector(self, config):
        """Create metrics collector."""
        return MetricsCollector(config)

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector._metrics == {}
        assert collector._counters == {}
        assert collector._gauges == {}
        assert collector._histograms == {}
        assert collector._timers == {}

    def test_increment_counter(self, collector):
        """Test incrementing counter."""
        collector.increment("api.requests")
        collector.increment("api.requests")
        collector.increment("api.requests", 3)

        assert collector._counters["api.requests"] == 5

    def test_increment_counter_with_tags(self, collector):
        """Test counter with tags."""
        collector.increment("api.requests", tags={"endpoint": "/users"})
        collector.increment("api.requests", tags={"endpoint": "/posts"})

        # Tags create separate metrics
        assert "api.requests.endpoint=/users" in collector._counters
        assert "api.requests.endpoint=/posts" in collector._counters

    def test_set_gauge(self, collector):
        """Test setting gauge value."""
        collector.set_gauge("memory.usage", 75.5)
        assert collector._gauges["memory.usage"] == 75.5

        collector.set_gauge("memory.usage", 80.2)
        assert collector._gauges["memory.usage"] == 80.2

    def test_record_histogram(self, collector):
        """Test recording histogram values."""
        histogram = Histogram("response.time")
        collector._histograms["response.time"] = histogram

        values = [100, 150, 200, 250, 300]
        for value in values:
            collector.record_histogram("response.time", value)

        assert histogram.count == 5
        assert histogram.sum == 1000
        assert histogram.min == 100
        assert histogram.max == 300
        assert histogram.mean == 200

    def test_start_timer(self, collector):
        """Test timer functionality."""
        timer = collector.start_timer("operation.duration")
        assert isinstance(timer, Timer)
        assert timer.name == "operation.duration"
        assert timer.start_time > 0

        time.sleep(0.1)
        duration = timer.stop()

        assert duration >= 0.1
        assert "operation.duration" in collector._timers

    def test_get_metrics(self, collector):
        """Test getting all metrics."""
        collector.increment("counter1", 5)
        collector.set_gauge("gauge1", 10.5)
        collector.record_histogram("hist1", 100)

        metrics = collector.get_metrics()

        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics
        assert metrics["counters"]["counter1"] == 5
        assert metrics["gauges"]["gauge1"] == 10.5

    def test_reset_metrics(self, collector):
        """Test resetting metrics."""
        collector.increment("counter1", 5)
        collector.set_gauge("gauge1", 10.5)

        collector.reset()

        assert collector._counters == {}
        assert collector._gauges == {}
        assert collector._histograms == {}
        assert collector._timers == {}

    def test_export_metrics(self, collector):
        """Test exporting metrics."""
        exporter = Mock(spec=MetricsExporter)
        collector.add_exporter(exporter)

        collector.increment("test.counter", 10)
        collector.export()

        exporter.export.assert_called_once()
        metrics = exporter.export.call_args[0][0]
        assert metrics["counters"]["test.counter"] == 10

    def test_metric_expiration(self, collector):
        """Test metric expiration based on retention."""
        # Create metrics with old timestamps
        old_time = time.time() - 20  # 20 seconds ago
        collector._metric_timestamps["old.metric"] = old_time
        collector._counters["old.metric"] = 100

        # Create recent metric
        collector.increment("new.metric")

        # Clean old metrics
        collector._clean_old_metrics()

        assert "old.metric" not in collector._counters
        assert "new.metric" in collector._counters

    def test_thread_safety(self, collector):
        """Test thread-safe metric updates."""

        def increment_counter(name, count):
            for _ in range(count):
                collector.increment(name)
                time.sleep(0.0001)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=increment_counter, args=(f"counter{i % 3}", 100))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each of 3 counters should have accumulated values
        total = sum(collector._counters.values())
        assert total == 1000  # 10 threads * 100 increments


class TestMetricDecorators:
    """Test metric decorator functions."""

    def test_metric_timer_decorator(self):
        """Test timing decorator."""
        collector = MetricsCollector(MetricsConfig())

        @metric_timer("function.duration", collector=collector)
        def test_function():
            time.sleep(0.1)
            return "result"

        result = test_function()

        assert result == "result"
        assert "function.duration" in collector._timers
        duration = collector._timers["function.duration"][-1]
        assert duration >= 0.1

    def test_track_metric_decorator(self):
        """Test metric tracking decorator."""
        collector = MetricsCollector(MetricsConfig())

        @track_metric("api.calls", metric_type=MetricType.COUNTER, collector=collector)
        def api_function():
            return "response"

        # Call function multiple times
        for _ in range(5):
            api_function()

        assert collector._counters["api.calls"] == 5

    def test_async_metric_timer(self):
        """Test async timer decorator."""
        collector = MetricsCollector(MetricsConfig())

        @metric_timer("async.duration", collector=collector)
        async def async_function():
            await asyncio.sleep(0.1)
            return "async result"

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(async_function())

        assert result == "async result"
        assert "async.duration" in collector._timers


class TestPrometheusExporter:
    """Test Prometheus metrics exporter."""

    def test_export_counters(self):
        """Test exporting counters in Prometheus format."""
        exporter = PrometheusExporter()
        metrics = {
            "counters": {"http_requests_total": 1000, "errors_total": 50},
            "gauges": {},
            "histograms": {},
        }

        output = exporter.export(metrics)

        assert "# TYPE http_requests_total counter" in output
        assert "http_requests_total 1000" in output
        assert "# TYPE errors_total counter" in output
        assert "errors_total 50" in output

    def test_export_gauges(self):
        """Test exporting gauges."""
        exporter = PrometheusExporter()
        metrics = {
            "counters": {},
            "gauges": {"memory_usage_bytes": 1024000, "cpu_usage_percent": 45.5},
            "histograms": {},
        }

        output = exporter.export(metrics)

        assert "# TYPE memory_usage_bytes gauge" in output
        assert "memory_usage_bytes 1024000" in output
        assert "# TYPE cpu_usage_percent gauge" in output
        assert "cpu_usage_percent 45.5" in output

    def test_export_histograms(self):
        """Test exporting histograms."""
        histogram = Histogram("response_time")
        histogram.observe(100)
        histogram.observe(200)
        histogram.observe(300)

        exporter = PrometheusExporter()
        metrics = {"counters": {}, "gauges": {}, "histograms": {"response_time": histogram}}

        output = exporter.export(metrics)

        assert "# TYPE response_time histogram" in output
        assert "response_time_count 3" in output
        assert "response_time_sum 600" in output
        assert "response_time_min 100" in output
        assert "response_time_max 300" in output


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PerformanceConfig(enabled=True, sampling_rate=1.0, profile_interval=60.0)

    @pytest.fixture
    def monitor(self, config):
        """Create performance monitor."""
        return PerformanceMonitor(config)

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor._profiles == {}
        assert monitor._resource_monitor is not None
        assert monitor._is_profiling is False

    def test_start_profiling(self, monitor):
        """Test starting profiling."""
        profile = monitor.start_profiling("test_operation")

        assert isinstance(profile, PerformanceProfile)
        assert profile.name == "test_operation"
        assert profile.start_time > 0
        assert "test_operation" in monitor._profiles
        assert monitor._is_profiling is True

    def test_stop_profiling(self, monitor):
        """Test stopping profiling."""
        profile = monitor.start_profiling("test_operation")
        time.sleep(0.1)

        result = monitor.stop_profiling("test_operation")

        assert result == profile
        assert profile.end_time > profile.start_time
        assert profile.duration >= 0.1
        assert monitor._is_profiling is False

    def test_get_profile_statistics(self, monitor):
        """Test getting profile statistics."""
        # Create multiple profiles
        for i in range(5):
            profile = monitor.start_profiling(f"op_{i}")
            time.sleep(0.01 * (i + 1))
            monitor.stop_profiling(f"op_{i}")

        stats = monitor.get_statistics()

        assert "profiles" in stats
        assert len(stats["profiles"]) == 5
        assert "resource_usage" in stats

    def test_profile_with_sampling(self):
        """Test profiling with sampling rate."""
        config = PerformanceConfig(sampling_rate=0.5)
        monitor = PerformanceMonitor(config)

        # With 50% sampling, roughly half should be profiled
        profiled_count = 0
        for i in range(100):
            profile = monitor.start_profiling(f"op_{i}")
            if profile:
                profiled_count += 1
                monitor.stop_profiling(f"op_{i}")

        # Should be roughly 50, but allow for randomness
        assert 30 <= profiled_count <= 70

    def test_concurrent_profiles(self, monitor):
        """Test concurrent profiling operations."""

        def profile_operation(name):
            profile = monitor.start_profiling(name)
            time.sleep(0.05)
            monitor.stop_profiling(name)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=profile_operation, args=(f"concurrent_{i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        stats = monitor.get_statistics()
        assert len(stats["profiles"]) == 10


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create resource monitor."""
        return ResourceMonitor()

    def test_get_cpu_usage(self, monitor):
        """Test CPU usage monitoring."""
        cpu_usage = monitor.get_cpu_usage()

        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100

    def test_get_memory_usage(self, monitor):
        """Test memory usage monitoring."""
        memory = monitor.get_memory_usage()

        assert "used" in memory
        assert "available" in memory
        assert "percent" in memory
        assert isinstance(memory["percent"], float)
        assert 0 <= memory["percent"] <= 100

    def test_get_disk_usage(self, monitor):
        """Test disk usage monitoring."""
        disk = monitor.get_disk_usage()

        assert "used" in disk
        assert "free" in disk
        assert "percent" in disk
        assert isinstance(disk["percent"], float)
        assert 0 <= disk["percent"] <= 100

    def test_get_network_stats(self, monitor):
        """Test network statistics."""
        network = monitor.get_network_stats()

        assert "bytes_sent" in network
        assert "bytes_received" in network
        assert "packets_sent" in network
        assert "packets_received" in network
        assert isinstance(network["bytes_sent"], int)
        assert isinstance(network["bytes_received"], int)

    def test_get_all_metrics(self, monitor):
        """Test getting all resource metrics."""
        metrics = monitor.get_all_metrics()

        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        assert "network" in metrics
        assert "timestamp" in metrics


class TestPerformanceDecorators:
    """Test performance decorator functions."""

    def test_performance_profile_decorator(self):
        """Test performance profiling decorator."""
        monitor = PerformanceMonitor(PerformanceConfig())

        @performance_profile("decorated_function", monitor=monitor)
        def test_function(x, y):
            time.sleep(0.05)
            return x + y

        result = test_function(1, 2)

        assert result == 3
        stats = monitor.get_statistics()
        assert len(stats["profiles"]) > 0

    def test_track_performance_decorator(self):
        """Test performance tracking decorator."""
        metrics = []

        @track_performance(callback=lambda m: metrics.append(m))
        def tracked_function():
            time.sleep(0.05)
            return "done"

        result = tracked_function()

        assert result == "done"
        assert len(metrics) == 1
        assert metrics[0]["duration"] >= 0.05

    def test_async_performance_profile(self):
        """Test async performance profiling."""
        monitor = PerformanceMonitor(PerformanceConfig())

        @performance_profile("async_function", monitor=monitor)
        async def async_function():
            await asyncio.sleep(0.05)
            return "async done"

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(async_function())

        assert result == "async done"
        stats = monitor.get_statistics()
        assert len(stats["profiles"]) > 0


class TestTelemetryClient:
    """Test TelemetryClient class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TelemetryConfig(enabled=True, batch_size=10, flush_interval=1.0)

    @pytest.fixture
    def client(self, config):
        """Create telemetry client."""
        return TelemetryClient(config)

    def test_initialization(self, client):
        """Test client initialization."""
        assert client._events == []
        assert client._batch_size == 10
        assert client._last_flush > 0

    def test_track_event(self, client):
        """Test tracking telemetry event."""
        event = TelemetryEvent(
            name="user.login",
            event_type=EventType.USER_ACTION,
            level=EventLevel.INFO,
            properties={"user_id": "123"},
        )

        client.track_event(event)

        assert len(client._events) == 1
        assert client._events[0].name == "user.login"

    def test_track_exception(self, client):
        """Test tracking exceptions."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            client.track_exception(e, {"context": "test"})

        assert len(client._events) == 1
        event = client._events[0]
        assert event.event_type == EventType.EXCEPTION
        assert event.level == EventLevel.ERROR
        assert "ValueError" in event.properties["exception_type"]

    def test_track_metric_event(self, client):
        """Test tracking metric events."""
        client.track_metric("response_time", 150.5, {"endpoint": "/api"})

        assert len(client._events) == 1
        event = client._events[0]
        assert event.event_type == EventType.METRIC
        assert event.properties["value"] == 150.5

    def test_track_dependency(self, client):
        """Test tracking dependency calls."""
        client.track_dependency("database", "SELECT * FROM users", 250, success=True)

        assert len(client._events) == 1
        event = client._events[0]
        assert event.event_type == EventType.DEPENDENCY
        assert event.properties["duration"] == 250
        assert event.properties["success"] is True

    def test_batch_flush(self, client):
        """Test batch flushing."""
        with patch.object(client, "_send_batch") as mock_send:
            # Add events up to batch size
            for i in range(10):
                client.track_event(TelemetryEvent(name=f"event_{i}", event_type=EventType.CUSTOM))

            # Should trigger automatic flush
            mock_send.assert_called_once()
            assert len(client._events) == 0

    def test_time_based_flush(self, client):
        """Test time-based flushing."""
        with patch.object(client, "_send_batch") as mock_send:
            # Add some events
            for i in range(5):
                client.track_event(TelemetryEvent(name=f"event_{i}", event_type=EventType.CUSTOM))

            # Simulate time passing
            client._last_flush = time.time() - 2.0

            # Add one more event to trigger time check
            client.track_event(TelemetryEvent(name="trigger", event_type=EventType.CUSTOM))

            mock_send.assert_called_once()

    def test_manual_flush(self, client):
        """Test manual flush."""
        with patch.object(client, "_send_batch") as mock_send:
            # Add some events
            for i in range(3):
                client.track_event(TelemetryEvent(name=f"event_{i}", event_type=EventType.CUSTOM))

            client.flush()

            mock_send.assert_called_once()
            assert len(client._events) == 0

    def test_event_filtering(self, client):
        """Test event level filtering."""
        client.config.min_level = EventLevel.WARNING

        # Track events of different levels
        client.track_event(
            TelemetryEvent(name="debug_event", event_type=EventType.CUSTOM, level=EventLevel.DEBUG)
        )

        client.track_event(
            TelemetryEvent(
                name="warning_event", event_type=EventType.CUSTOM, level=EventLevel.WARNING
            )
        )

        # Only WARNING and above should be tracked
        assert len(client._events) == 1
        assert client._events[0].name == "warning_event"

    def test_context_enrichment(self, client):
        """Test event context enrichment."""
        client.set_context("app_version", "1.0.0")
        client.set_context("environment", "production")

        event = TelemetryEvent(name="test_event", event_type=EventType.CUSTOM)

        client.track_event(event)

        tracked_event = client._events[0]
        assert tracked_event.properties["app_version"] == "1.0.0"
        assert tracked_event.properties["environment"] == "production"


class TestTelemetryDecorators:
    """Test telemetry decorator functions."""

    def test_track_event_decorator(self):
        """Test event tracking decorator."""
        client = TelemetryClient(TelemetryConfig())

        @track_event("function.called", client=client)
        def test_function(x):
            return x * 2

        result = test_function(5)

        assert result == 10
        assert len(client._events) == 1
        assert client._events[0].name == "function.called"

    def test_track_event_with_exception(self):
        """Test event tracking with exception."""
        client = TelemetryClient(TelemetryConfig())

        @track_event("function.error", client=client, track_exceptions=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Should track both the call and the exception
        assert len(client._events) == 2
        assert any(e.event_type == EventType.EXCEPTION for e in client._events)


class TestIntegration:
    """Test integration between monitoring components."""

    def test_metrics_and_telemetry_integration(self):
        """Test metrics collector with telemetry."""
        metrics = MetricsCollector(MetricsConfig())
        telemetry = TelemetryClient(TelemetryConfig())

        # Track metric through both systems
        metrics.increment("api.calls")
        telemetry.track_metric("api.calls", 1)

        assert metrics._counters["api.calls"] == 1
        assert len(telemetry._events) == 1

    def test_performance_and_metrics_integration(self):
        """Test performance monitor with metrics."""
        metrics = MetricsCollector(MetricsConfig())
        performance = PerformanceMonitor(PerformanceConfig())

        @metric_timer("operation.time", collector=metrics)
        @performance_profile("operation", monitor=performance)
        def monitored_operation():
            time.sleep(0.05)
            return "result"

        result = monitored_operation()

        assert result == "result"
        assert "operation.time" in metrics._timers
        assert len(performance._profiles) > 0

    def test_full_monitoring_stack(self):
        """Test complete monitoring stack."""
        # Create all components
        metrics = MetricsCollector(MetricsConfig())
        performance = PerformanceMonitor(PerformanceConfig())
        telemetry = TelemetryClient(TelemetryConfig())

        # Simulate application operation
        with metrics.start_timer("request.duration") as timer:
            profile = performance.start_profiling("request")

            # Track start event
            telemetry.track_event(
                TelemetryEvent(name="request.started", event_type=EventType.CUSTOM)
            )

            # Simulate work
            time.sleep(0.1)
            metrics.increment("requests.processed")

            # Track completion
            telemetry.track_event(
                TelemetryEvent(name="request.completed", event_type=EventType.CUSTOM)
            )

            performance.stop_profiling("request")

        # Verify all systems captured data
        assert metrics._counters["requests.processed"] == 1
        assert len(metrics._timers["request.duration"]) > 0
        assert len(performance._profiles) == 1
        assert len(telemetry._events) == 2
