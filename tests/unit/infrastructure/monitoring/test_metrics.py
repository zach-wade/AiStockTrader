"""
Comprehensive unit tests for metrics collection module.

Tests business and technical metrics collection, threshold monitoring,
time-series data collection, and alerting functionality.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.domain.services.threshold_policy_service import ThresholdBreachEvent, ThresholdComparison
from src.infrastructure.monitoring.metrics import (
    CustomMetric,
    MetricSnapshot,
    MetricThreshold,
    MetricType,
    SystemMetricsCollector,
    TradingMetrics,
    TradingMetricsCollector,
    get_trading_metrics,
    initialize_trading_metrics,
    track_trading_metric,
)


class TestMetricType:
    """Test MetricType enumeration."""

    def test_metric_types(self):
        """Test all metric type values."""
        assert MetricType.COUNTER == "counter"
        assert MetricType.HISTOGRAM == "histogram"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.UP_DOWN_COUNTER == "up_down_counter"


class TestMetricThreshold:
    """Test MetricThreshold configuration."""

    def test_threshold_creation(self):
        """Test creating metric threshold."""

        def alert_callback(name, value, limit):
            print(f"Alert: {name} = {value} > {limit}")

        threshold = MetricThreshold(
            metric_name="cpu_usage",
            warning_threshold=70.0,
            critical_threshold=90.0,
            comparison="greater_than",
            consecutive_breaches=3,
            alert_callback=alert_callback,
        )

        assert threshold.metric_name == "cpu_usage"
        assert threshold.warning_threshold == 70.0
        assert threshold.critical_threshold == 90.0
        assert threshold.comparison == "greater_than"
        assert threshold.consecutive_breaches == 3
        assert threshold.alert_callback is not None

    def test_threshold_defaults(self):
        """Test threshold default values."""
        threshold = MetricThreshold(metric_name="test_metric")

        assert threshold.warning_threshold is None
        assert threshold.critical_threshold is None
        assert threshold.comparison == "greater_than"
        assert threshold.consecutive_breaches == 1
        assert threshold.alert_callback is None


class TestMetricSnapshot:
    """Test MetricSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating metric snapshot."""
        snapshot = MetricSnapshot(
            name="test_metric",
            value=42.5,
            timestamp=1234567890.0,
            labels={"env": "prod"},
            metric_type=MetricType.GAUGE,
        )

        assert snapshot.name == "test_metric"
        assert snapshot == 42.5
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.labels == {"env": "prod"}
        assert snapshot.metric_type == MetricType.GAUGE

    def test_snapshot_defaults(self):
        """Test snapshot default values."""
        snapshot = MetricSnapshot(name="test", value=10, timestamp=time.time())

        assert snapshot.labels == {}
        assert snapshot.metric_type == MetricType.GAUGE


class TestSystemMetricsCollector:
    """Test SystemMetricsCollector functionality."""

    @patch("psutil.Process")
    def test_initialization(self, mock_process_class):
        """Test system metrics collector initialization."""
        collector = SystemMetricsCollector()

        assert collector.process is not None
        assert collector._last_cpu_times is None
        assert collector._last_io_counters is None
        assert collector._last_net_io is None

    @patch("psutil.cpu_count")
    @patch("psutil.cpu_percent")
    @patch("psutil.Process")
    def test_collect_cpu_metrics(self, mock_process_class, mock_cpu_percent, mock_cpu_count):
        """Test collecting CPU metrics."""
        mock_cpu_percent.return_value = 45.5
        mock_cpu_count.return_value = 8

        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024000
        mock_memory_info.vms = 2048000
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.num_threads.return_value = 10
        mock_process_class.return_value = mock_process

        collector = SystemMetricsCollector()
        metrics = collector.collect_cpu_metrics()

        assert metrics["system_cpu_percent"] == 45.5
        assert metrics["system_cpu_count"] == 8
        assert metrics["process_cpu_percent"] == 25.0
        assert metrics["process_memory_rss"] == 1024000
        assert metrics["process_memory_vms"] == 2048000
        assert metrics["process_threads"] == 10

    @patch("psutil.swap_memory")
    @patch("psutil.virtual_memory")
    @patch("psutil.Process")
    def test_collect_memory_metrics(
        self, mock_process_class, mock_virtual_memory, mock_swap_memory
    ):
        """Test collecting memory metrics."""
        mock_virtual_memory.return_value = Mock(
            total=8192000000, available=4096000000, percent=50.0
        )
        mock_swap_memory.return_value = Mock(total=2048000000, used=512000000, percent=25.0)

        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 512000000
        mock_memory_info.vms = 1024000000
        mock_memory_info.shared = 128000000
        mock_memory_info.text = 64000000
        mock_memory_info.data = 256000000
        mock_process.memory_full_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        collector = SystemMetricsCollector()
        metrics = collector.collect_memory_metrics()

        assert metrics["system_memory_total"] == 8192000000
        assert metrics["system_memory_available"] == 4096000000
        assert metrics["system_memory_percent"] == 50.0
        assert metrics["system_swap_total"] == 2048000000
        assert metrics["system_swap_used"] == 512000000
        assert metrics["system_swap_percent"] == 25.0
        assert metrics["process_memory_rss"] == 512000000
        assert metrics["process_memory_vms"] == 1024000000
        assert metrics["process_memory_shared"] == 128000000

    @patch("psutil.disk_io_counters")
    @patch("psutil.disk_usage")
    @patch("psutil.Process")
    def test_collect_disk_metrics(self, mock_process_class, mock_disk_usage, mock_disk_io):
        """Test collecting disk metrics."""
        mock_disk_usage.return_value = Mock(
            total=1000000000000, used=600000000000, free=400000000000
        )
        mock_disk_io.return_value = Mock(
            read_count=10000, write_count=5000, read_bytes=1024000000, write_bytes=512000000
        )

        mock_process = Mock()
        mock_io_counters = Mock()
        mock_io_counters.read_count = 1000
        mock_io_counters.write_count = 500
        mock_io_counters.read_bytes = 10240000
        mock_io_counters.write_bytes = 5120000
        mock_process.io_counters.return_value = mock_io_counters
        mock_process_class.return_value = mock_process

        collector = SystemMetricsCollector()
        metrics = collector.collect_disk_metrics()

        assert metrics["system_disk_total"] == 1000000000000
        assert metrics["system_disk_used"] == 600000000000
        assert metrics["system_disk_free"] == 400000000000
        assert metrics["system_disk_percent"] == 60.0
        assert metrics["system_disk_read_count"] == 10000
        assert metrics["system_disk_write_count"] == 5000
        assert metrics["process_io_read_count"] == 1000
        assert metrics["process_io_write_count"] == 500

    @patch("psutil.net_io_counters")
    @patch("psutil.Process")
    def test_collect_network_metrics(self, mock_process_class, mock_net_io):
        """Test collecting network metrics."""
        mock_net_io.return_value = Mock(
            bytes_sent=1024000000,
            bytes_recv=2048000000,
            packets_sent=1000000,
            packets_recv=2000000,
            errin=10,
            errout=5,
            dropin=2,
            dropout=1,
        )

        collector = SystemMetricsCollector()
        metrics = collector.collect_network_metrics()

        assert metrics["system_network_bytes_sent"] == 1024000000
        assert metrics["system_network_bytes_recv"] == 2048000000
        assert metrics["system_network_packets_sent"] == 1000000
        assert metrics["system_network_packets_recv"] == 2000000
        assert metrics["system_network_errin"] == 10
        assert metrics["system_network_errout"] == 5

    @patch("logging.Logger.warning")
    @patch("psutil.cpu_percent")
    def test_collect_metrics_error_handling(self, mock_cpu_percent, mock_warning):
        """Test error handling in metrics collection."""
        mock_cpu_percent.side_effect = Exception("CPU error")

        collector = SystemMetricsCollector()
        metrics = collector.collect_cpu_metrics()

        assert metrics == {}
        mock_warning.assert_called_once()

    def test_collect_all_metrics(self):
        """Test collecting all system metrics."""
        collector = SystemMetricsCollector()

        with (
            patch.object(collector, "collect_cpu_metrics") as mock_cpu,
            patch.object(collector, "collect_memory_metrics") as mock_memory,
            patch.object(collector, "collect_disk_metrics") as mock_disk,
            patch.object(collector, "collect_network_metrics") as mock_network,
        ):
            mock_cpu.return_value = {"cpu": 50.0}
            mock_memory.return_value = {"memory": 60.0}
            mock_disk.return_value = {"disk": 70.0}
            mock_network.return_value = {"network": 80.0}

            all_metrics = collector.collect_all_metrics()

            assert all_metrics["cpu"] == 50.0
            assert all_metrics["memory"] == 60.0
            assert all_metrics["disk"] == 70.0
            assert all_metrics["network"] == 80.0


class TestTradingMetricsCollector:
    """Test TradingMetricsCollector functionality."""

    def test_initialization(self):
        """Test trading metrics collector initialization."""
        collector = TradingMetricsCollector()

        assert collector._order_count == 0
        assert collector._filled_orders == 0
        assert collector._rejected_orders == 0
        assert collector._cancelled_orders == 0
        assert len(collector._order_latencies) == 0
        assert len(collector._pnl_history) == 0
        assert collector._position_values == {}
        assert collector._portfolio_values == {}
        assert collector._risk_metrics == {}

    def test_record_order_submitted(self):
        """Test recording order submission."""
        collector = TradingMetricsCollector()

        collector.record_order_submitted("ORD123", "AAPL", 100, 150.50)
        assert collector._order_count == 1

        collector.record_order_submitted("ORD124", "GOOGL", 50)
        assert collector._order_count == 2

    def test_record_order_filled(self):
        """Test recording order fill."""
        collector = TradingMetricsCollector()

        collector.record_order_filled("ORD123", "AAPL", 100, 150.50, 25.5)

        assert collector._filled_orders == 1
        assert len(collector._order_latencies) == 1
        assert collector._order_latencies[0] == 25.5

    def test_record_order_rejected(self):
        """Test recording order rejection."""
        collector = TradingMetricsCollector()

        collector.record_order_rejected("ORD123", "Insufficient funds")
        assert collector._rejected_orders == 1

    def test_record_order_cancelled(self):
        """Test recording order cancellation."""
        collector = TradingMetricsCollector()

        collector.record_order_cancelled("ORD123")
        assert collector._cancelled_orders == 1

    @patch("time.time")
    def test_record_pnl(self, mock_time):
        """Test recording P&L."""
        mock_time.return_value = 1234567890.0

        collector = TradingMetricsCollector()
        collector.record_pnl("PORT123", 1500.50)

        assert len(collector._pnl_history) == 1
        assert collector._pnl_history[0] == (1234567890.0, "PORT123", 1500.50)

    @patch("time.time")
    def test_record_position_value(self, mock_time):
        """Test recording position value."""
        mock_time.return_value = 1234567890.0

        collector = TradingMetricsCollector()
        collector.record_position_value("POS123", "AAPL", 15000.00)

        assert "POS123" in collector._position_values
        assert collector._position_values["POS123"]["symbol"] == "AAPL"
        assert collector._position_values["POS123"]["value"] == 15000.00
        assert collector._position_values["POS123"]["timestamp"] == 1234567890.0

    @patch("time.time")
    def test_record_portfolio_value(self, mock_time):
        """Test recording portfolio value."""
        mock_time.return_value = 1234567890.0

        collector = TradingMetricsCollector()
        collector.record_portfolio_value("PORT123", 100000.00)

        assert "PORT123" in collector._portfolio_values
        assert collector._portfolio_values["PORT123"]["value"] == 100000.00

    @patch("time.time")
    def test_record_risk_metric(self, mock_time):
        """Test recording risk metric."""
        mock_time.return_value = 1234567890.0

        collector = TradingMetricsCollector()

        # Record without portfolio ID
        collector.record_risk_metric("var_95", 5000.00)
        assert "var_95" in collector._risk_metrics

        # Record with portfolio ID
        collector.record_risk_metric("sharpe_ratio", 1.5, "PORT123")
        assert "sharpe_ratio:PORT123" in collector._risk_metrics
        assert collector._risk_metrics["sharpe_ratio:PORT123"]["portfolio_id"] == "PORT123"

    @patch("time.time")
    def test_get_trading_metrics(self, mock_time):
        """Test getting aggregated trading metrics."""
        # Setup collector with data
        collector = TradingMetricsCollector()
        collector._start_time = 1000.0
        mock_time.return_value = 1100.0  # 100 seconds later

        # Add test data
        collector._order_count = 100
        collector._filled_orders = 80
        collector._rejected_orders = 15
        collector._cancelled_orders = 5
        collector._order_latencies.extend([10, 20, 30, 40, 50])
        collector._pnl_history.append((1100.0, "PORT1", 5000.00))
        collector._position_values = {"POS1": {"value": 10000}, "POS2": {"value": 20000}}
        collector._portfolio_values = {"PORT1": {"value": 100000}, "PORT2": {"value": 200000}}

        metrics = collector.get_trading_metrics()

        assert metrics["trading_orders_total"] == 100
        assert metrics["trading_orders_filled"] == 80
        assert metrics["trading_orders_rejected"] == 15
        assert metrics["trading_orders_cancelled"] == 5
        assert metrics["trading_orders_per_second"] == 1.0  # 100 orders / 100 seconds
        assert metrics["trading_fill_rate"] == 0.8  # 80/100
        assert metrics["trading_rejection_rate"] == 0.15  # 15/100
        assert metrics["trading_avg_latency_ms"] == 30.0  # Average of [10,20,30,40,50]
        assert metrics["trading_total_portfolio_value"] == 300000  # 100000 + 200000
        assert metrics["trading_current_pnl"] == 5000.00
        assert metrics["trading_active_positions"] == 2
        assert metrics["trading_active_portfolios"] == 2
        assert metrics["trading_runtime_seconds"] == 100.0


class TestCustomMetric:
    """Test CustomMetric functionality."""

    def test_counter_metric(self):
        """Test counter metric operations."""
        metric = CustomMetric("test_counter", MetricType.COUNTER, "Test counter")

        # Test increment
        metric.increment()
        assert metric.get_value() == 1.0

        metric.increment(5.0)
        assert metric.get_value() == 6.0

        # Test with labels
        metric.increment(2.0, {"env": "prod"})
        assert metric.get_value() == 8.0  # Total
        assert metric.get_value({"env": "prod"}) == 2.0

    def test_gauge_metric(self):
        """Test gauge metric operations."""
        metric = CustomMetric("test_gauge", MetricType.GAUGE, "Test gauge")

        # Test set
        metric.set(42.5)
        assert metric.get_value() == 42.5

        metric.set(100.0)
        assert metric.get_value() == 100.0

        # Test with labels
        metric.set(75.0, {"region": "us-east"})
        assert metric.get_value({"region": "us-east"}) == 75.0

    def test_histogram_metric(self):
        """Test histogram metric operations."""
        metric = CustomMetric("test_histogram", MetricType.HISTOGRAM, "Test histogram")

        # Test observe
        metric.observe(10.5)
        metric.observe(20.0)
        metric.observe(15.5)

        samples = metric.get_samples()
        assert len(samples) == 3
        assert samples[0][1] == 10.5
        assert samples[1][1] == 20.0
        assert samples[2][1] == 15.5

    def test_up_down_counter_metric(self):
        """Test up-down counter metric."""
        metric = CustomMetric("test_updown", MetricType.UP_DOWN_COUNTER)

        metric.increment(10.0)
        assert metric.get_value() == 10.0

        metric.increment(-3.0)
        assert metric.get_value() == 7.0

    def test_invalid_operations(self):
        """Test invalid operations on metrics."""
        counter = CustomMetric("counter", MetricType.COUNTER)
        gauge = CustomMetric("gauge", MetricType.GAUGE)
        histogram = CustomMetric("histogram", MetricType.HISTOGRAM)

        # Counter can't use set
        with pytest.raises(ValueError):
            counter.set(10.0)

        # Gauge can't use increment
        with pytest.raises(ValueError):
            gauge.increment(5.0)

        # Histogram can't use set or increment
        with pytest.raises(ValueError):
            histogram.set(10.0)
        with pytest.raises(ValueError):
            histogram.increment(5.0)

    def test_get_samples_with_limit(self):
        """Test getting samples with limit."""
        metric = CustomMetric("test", MetricType.HISTOGRAM)

        for i in range(10):
            metric.observe(i)

        # Get all samples
        all_samples = metric.get_samples()
        assert len(all_samples) == 10

        # Get limited samples
        limited = metric.get_samples(limit=5)
        assert len(limited) == 5
        assert limited[0][1] == 5  # Should get last 5

    def test_labels_to_key(self):
        """Test label key generation."""
        metric = CustomMetric("test", MetricType.GAUGE)

        key1 = metric._labels_to_key({"env": "prod", "region": "us"})
        assert key1 == "env=prod,region=us"

        key2 = metric._labels_to_key({"region": "us", "env": "prod"})
        assert key2 == "env=prod,region=us"  # Should be sorted


class TestTradingMetrics:
    """Test TradingMetrics main class."""

    def test_initialization(self):
        """Test TradingMetrics initialization."""
        metrics = TradingMetrics()

        assert metrics.meter_provider is None
        assert metrics.meter is None
        assert metrics.system_collector is not None
        assert metrics.trading_collector is not None
        assert metrics._custom_metrics == {}
        assert metrics._thresholds == []
        assert metrics._threshold_policy_service is not None

    @patch("opentelemetry.metrics.get_meter")
    def test_initialization_with_meter_provider(self, mock_get_meter):
        """Test initialization with OpenTelemetry meter provider."""
        mock_meter = Mock()
        mock_get_meter.return_value = mock_meter

        mock_provider = Mock()
        metrics = TradingMetrics(meter_provider=mock_provider)

        assert metrics.meter_provider == mock_provider
        assert metrics.meter is not None
        mock_get_meter.assert_called_once()

    def test_create_counter(self):
        """Test creating counter metric."""
        metrics = TradingMetrics()

        counter = metrics.create_counter("test_counter", "Test description")

        assert isinstance(counter, CustomMetric)
        assert counter.metric_type == MetricType.COUNTER
        assert counter.description == "Test description"
        assert "test_counter" in metrics._custom_metrics

    def test_create_gauge(self):
        """Test creating gauge metric."""
        metrics = TradingMetrics()

        gauge = metrics.create_gauge("test_gauge")

        assert isinstance(gauge, CustomMetric)
        assert gauge.metric_type == MetricType.GAUGE
        assert "test_gauge" in metrics._custom_metrics

    def test_create_histogram(self):
        """Test creating histogram metric."""
        metrics = TradingMetrics()

        histogram = metrics.create_histogram("test_histogram")

        assert isinstance(histogram, CustomMetric)
        assert histogram.metric_type == MetricType.HISTOGRAM

    def test_create_up_down_counter(self):
        """Test creating up-down counter metric."""
        metrics = TradingMetrics()

        counter = metrics.create_up_down_counter("test_updown")

        assert isinstance(counter, CustomMetric)
        assert counter.metric_type == MetricType.UP_DOWN_COUNTER

    def test_create_existing_metric(self):
        """Test creating metric that already exists."""
        metrics = TradingMetrics()

        counter1 = metrics.create_counter("test_metric")
        counter2 = metrics.create_counter("test_metric")

        assert counter1 is counter2  # Should return same instance

    def test_create_existing_metric_different_type(self):
        """Test creating existing metric with different type."""
        metrics = TradingMetrics()

        metrics.create_counter("test_metric")

        with pytest.raises(ValueError, match="already exists with different type"):
            metrics.create_gauge("test_metric")

    def test_get_metric(self):
        """Test getting metric by name."""
        metrics = TradingMetrics()

        counter = metrics.create_counter("test_counter")
        retrieved = metrics.get_metric("test_counter")

        assert retrieved is counter
        assert metrics.get_metric("nonexistent") is None

    def test_record_order_submitted(self):
        """Test recording order submission."""
        metrics = TradingMetrics()

        with patch.object(metrics.trading_collector, "record_order_submitted") as mock_record:
            metrics.record_order_submitted("ORD123", "AAPL", 100, 150.50)
            mock_record.assert_called_once_with("ORD123", "AAPL", 100, 150.50)

    def test_record_order_filled(self):
        """Test recording order fill."""
        metrics = TradingMetrics()

        with patch.object(metrics.trading_collector, "record_order_filled") as mock_record:
            metrics.record_order_filled("ORD123", "AAPL", 100, 150.50, 25.5)
            mock_record.assert_called_once_with("ORD123", "AAPL", 100, 150.50, 25.5)

    def test_record_order_rejected(self):
        """Test recording order rejection."""
        metrics = TradingMetrics()

        with patch.object(metrics.trading_collector, "record_order_rejected") as mock_record:
            metrics.record_order_rejected("ORD123", "Insufficient funds")
            mock_record.assert_called_once_with("ORD123", "Insufficient funds")

    def test_record_pnl(self):
        """Test recording P&L."""
        metrics = TradingMetrics()

        with patch.object(metrics.trading_collector, "record_pnl") as mock_record:
            metrics.record_pnl("PORT123", 5000.00)
            mock_record.assert_called_once_with("PORT123", 5000.00)

    def test_record_portfolio_value(self):
        """Test recording portfolio value."""
        metrics = TradingMetrics()

        with patch.object(metrics.trading_collector, "record_portfolio_value") as mock_record:
            metrics.record_portfolio_value("PORT123", 100000.00)
            mock_record.assert_called_once_with("PORT123", 100000.00)

    def test_add_threshold(self):
        """Test adding metric threshold."""
        metrics = TradingMetrics()

        threshold = MetricThreshold(
            metric_name="cpu_usage",
            warning_threshold=70.0,
            critical_threshold=90.0,
            comparison="greater_than",
        )

        metrics.add_threshold(threshold)

        assert len(metrics._thresholds) == 1
        assert metrics._thresholds[0] == threshold

        # Should also add to domain service
        policies = metrics._threshold_policy_service._policies
        assert len(policies) == 1
        assert policies[0].metric_name == "cpu_usage"

    @patch("time.time")
    def test_check_thresholds(self, mock_time):
        """Test checking thresholds."""
        mock_time.return_value = 1234567890.0

        metrics = TradingMetrics()

        # Add threshold
        alert_called = False

        def alert_callback(name, value, limit):
            nonlocal alert_called
            alert_called = True

        threshold = MetricThreshold(
            metric_name="cpu_usage",
            critical_threshold=80.0,
            comparison="greater_than",
            alert_callback=alert_callback,
        )
        metrics.add_threshold(threshold)

        # Mock metrics collection
        with patch.object(metrics, "collect_all_metrics") as mock_collect:
            mock_collect.return_value = {"cpu_usage": 85.0}

            # Mock domain service evaluation
            breach_event = ThresholdBreachEvent(
                metric_name="cpu_usage",
                current_value=85.0,
                threshold_value=80.0,
                comparison=ThresholdComparison.GREATER_THAN,
                severity="critical",
                timestamp=1234567890.0,
                message="CPU usage critical: 85.0 > 80.0",
            )

            with patch.object(
                metrics._threshold_policy_service, "evaluate_all_thresholds"
            ) as mock_eval:
                mock_eval.return_value = [breach_event]

                metrics.check_thresholds()

                assert alert_called

    def test_handle_breach_event(self):
        """Test handling threshold breach event."""
        metrics = TradingMetrics()

        alert_called = False
        alert_args = None

        def alert_callback(name, value, limit):
            nonlocal alert_called, alert_args
            alert_called = True
            alert_args = (name, value, limit)

        threshold = MetricThreshold(
            metric_name="memory_usage", critical_threshold=90.0, alert_callback=alert_callback
        )
        metrics._thresholds.append(threshold)

        breach_event = ThresholdBreachEvent(
            metric_name="memory_usage",
            current_value=95.0,
            threshold_value=90.0,
            comparison=ThresholdComparison.GREATER_THAN,
            severity="critical",
            timestamp=time.time(),
            message="Memory usage critical",
        )

        with patch("logging.Logger.warning") as mock_warning:
            metrics._handle_breach_event(breach_event)

            mock_warning.assert_called_once()
            assert alert_called
            assert alert_args == ("memory_usage", 95.0, 90.0)

    def test_is_within_warning_range(self):
        """Test checking if value is within warning range."""
        metrics = TradingMetrics()

        # Greater than comparison
        threshold_gt = MetricThreshold(
            metric_name="test",
            warning_threshold=70.0,
            critical_threshold=90.0,
            comparison="greater_than",
        )

        assert metrics._is_within_warning_range(75.0, threshold_gt) is True
        assert metrics._is_within_warning_range(95.0, threshold_gt) is False
        assert metrics._is_within_warning_range(65.0, threshold_gt) is False

        # Less than comparison
        threshold_lt = MetricThreshold(
            metric_name="test",
            warning_threshold=30.0,
            critical_threshold=10.0,
            comparison="less_than",
        )

        assert metrics._is_within_warning_range(25.0, threshold_lt) is True
        assert metrics._is_within_warning_range(5.0, threshold_lt) is False
        assert metrics._is_within_warning_range(35.0, threshold_lt) is False

    def test_collect_all_metrics(self):
        """Test collecting all metrics."""
        metrics = TradingMetrics()

        # Add custom metric
        counter = metrics.create_counter("custom_counter")
        counter.increment(42.0)

        with (
            patch.object(metrics.system_collector, "collect_all_metrics") as mock_system,
            patch.object(metrics.trading_collector, "get_trading_metrics") as mock_trading,
        ):
            mock_system.return_value = {"cpu": 50.0}
            mock_trading.return_value = {"orders": 100}

            all_metrics = metrics.collect_all_metrics()

            assert all_metrics["cpu"] == 50.0
            assert all_metrics["orders"] == 100
            assert all_metrics["custom_counter"] == 42.0

    @patch("time.time")
    def test_get_metric_snapshots(self, mock_time):
        """Test getting metric snapshots."""
        mock_time.return_value = 1234567890.0

        metrics = TradingMetrics()

        with patch.object(metrics, "collect_all_metrics") as mock_collect:
            mock_collect.return_value = {
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "orders_per_second": 1.5,
            }

            snapshots = metrics.get_metric_snapshots()

            assert len(snapshots) == 3
            assert all(s.timestamp == 1234567890.0 for s in snapshots)
            assert any(s.name == "cpu_usage" and s == 50.0 for s in snapshots)
            assert any(s.name == "memory_usage" and s == 60.0 for s in snapshots)
            assert any(s.name == "orders_per_second" and s == 1.5 for s in snapshots)

    @pytest.mark.asyncio
    async def test_background_collection(self):
        """Test background metrics collection."""
        metrics = TradingMetrics()

        # Start collection
        await metrics.start_background_collection(interval=0.05)
        assert metrics._collection_task is not None
        assert not metrics._collection_task.done()

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop collection
        await metrics.stop_background_collection()
        assert metrics._stop_collection is True

    @pytest.mark.asyncio
    async def test_collection_loop(self):
        """Test collection loop functionality."""
        metrics = TradingMetrics()

        collect_called = False
        check_called = False
        update_called = False

        async def mock_collect():
            nonlocal collect_called
            collect_called = True
            return {}

        def mock_check():
            nonlocal check_called
            check_called = True

        def mock_update():
            nonlocal update_called
            update_called = True

        with (
            patch.object(metrics, "collect_all_metrics", side_effect=mock_collect),
            patch.object(metrics, "check_thresholds", side_effect=mock_check),
            patch.object(metrics, "_update_otel_instruments", side_effect=mock_update),
        ):
            # Run one iteration
            metrics._stop_collection = False
            await metrics._collection_loop(0.01)

            # Should have called all functions
            assert collect_called
            assert check_called
            assert update_called

    @pytest.mark.asyncio
    async def test_collection_loop_error_handling(self):
        """Test error handling in collection loop."""
        metrics = TradingMetrics()

        with patch.object(metrics, "collect_all_metrics", side_effect=Exception("Test error")):
            with patch("logging.Logger.error") as mock_error:
                # Run briefly
                await metrics.start_background_collection(interval=0.01)
                await asyncio.sleep(0.05)
                await metrics.stop_background_collection()

                # Should have logged error
                mock_error.assert_called()


class TestGlobalTradingMetrics:
    """Test global trading metrics functions."""

    def test_initialize_trading_metrics(self):
        """Test initializing global metrics."""
        mock_provider = Mock()
        metrics = initialize_trading_metrics(meter_provider=mock_provider)

        assert metrics is not None
        assert metrics.meter_provider == mock_provider

    def test_get_trading_metrics(self):
        """Test getting global metrics."""
        # Initialize first
        initialize_trading_metrics()

        metrics = get_trading_metrics()
        assert metrics is not None

    def test_get_trading_metrics_not_initialized(self):
        """Test getting metrics when not initialized."""
        # Reset global
        import src.infrastructure.monitoring.metrics as metrics_module

        metrics_module._trading_metrics = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_trading_metrics()


class TestTrackTradingMetricDecorator:
    """Test track_trading_metric decorator."""

    @patch("src.infrastructure.monitoring.metrics.get_trading_metrics")
    def test_track_counter_sync(self, mock_get_metrics):
        """Test tracking counter metric on sync function."""
        mock_metrics = Mock()
        mock_counter = Mock()
        mock_metrics.create_counter.return_value = mock_counter
        mock_get_metrics.return_value = mock_metrics

        @track_trading_metric("function_calls", "counter", {"service": "test"})
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        mock_metrics.create_counter.assert_called_once_with("function_calls")
        mock_counter.increment.assert_called_once_with(1.0, {"service": "test"})

    @patch("src.infrastructure.monitoring.metrics.get_trading_metrics")
    def test_track_histogram_sync(self, mock_get_metrics):
        """Test tracking histogram metric on sync function."""
        mock_metrics = Mock()
        mock_histogram = Mock()
        mock_metrics.create_histogram.return_value = mock_histogram
        mock_get_metrics.return_value = mock_metrics

        @track_trading_metric("function_duration", "histogram")
        def test_func():
            return 42

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            result = test_func()

        assert result == 42
        mock_histogram.observe.assert_called_once()
        # Should observe ~100ms
        observed_value = mock_histogram.observe.call_args[0][0]
        assert 99 <= observed_value <= 101

    @patch("src.infrastructure.monitoring.metrics.get_trading_metrics")
    def test_track_counter_error(self, mock_get_metrics):
        """Test tracking counter on error."""
        mock_metrics = Mock()
        mock_counter = Mock()
        mock_metrics.create_counter.return_value = mock_counter
        mock_get_metrics.return_value = mock_metrics

        @track_trading_metric("error_count", "counter")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        mock_counter.increment.assert_called_once()
        call_args = mock_counter.increment.call_args
        assert call_args[0][0] == 1.0
        assert call_args[0][1]["status"] == "error"
        assert call_args[0][1]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.metrics.get_trading_metrics")
    async def test_track_counter_async(self, mock_get_metrics):
        """Test tracking counter metric on async function."""
        mock_metrics = Mock()
        mock_counter = Mock()
        mock_metrics.create_counter.return_value = mock_counter
        mock_get_metrics.return_value = mock_metrics

        @track_trading_metric("async_calls", "counter")
        @pytest.mark.asyncio
        async def test_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await test_func()

        assert result == "async_result"
        mock_counter.increment.assert_called_once_with(1.0, None)

    @pytest.mark.asyncio
    @patch("src.infrastructure.monitoring.metrics.get_trading_metrics")
    async def test_track_histogram_async(self, mock_get_metrics):
        """Test tracking histogram metric on async function."""
        mock_metrics = Mock()
        mock_histogram = Mock()
        mock_metrics.create_histogram.return_value = mock_histogram
        mock_get_metrics.return_value = mock_metrics

        @track_trading_metric("async_duration", "histogram", {"async": "true"})
        @pytest.mark.asyncio
        async def test_func():
            await asyncio.sleep(0.01)
            return "done"

        with patch("time.perf_counter", side_effect=[0.0, 0.05]):
            result = await test_func()

        assert result == "done"
        mock_histogram.observe.assert_called_once()
        observed_value = mock_histogram.observe.call_args[0][0]
        assert observed_value == 50.0  # 0.05 seconds = 50ms
