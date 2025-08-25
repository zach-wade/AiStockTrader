"""
Metrics Collection for AI Trading System

Business and technical metrics collection with:
- Trading metrics (orders per second, P&L, latency)
- System metrics (CPU, memory, database connections)
- Custom metrics for domain services
- Time-series data collection
- Alerting thresholds
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any

import psutil
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider

# Import from domain service
from src.domain.services.threshold_policy_service import (
    ThresholdBreachEvent,
    ThresholdComparison,
    ThresholdPolicy,
    ThresholdPolicyService,
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    UP_DOWN_COUNTER = "up_down_counter"


@dataclass
class MetricThreshold:
    """Alerting threshold configuration."""

    metric_name: str
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    comparison: str = "greater_than"  # greater_than, less_than, equal_to
    consecutive_breaches: int = 1
    alert_callback: Callable[[str, float, float], None] | None = None


@dataclass
class MetricSnapshot:
    """Snapshot of a metric value at a point in time."""

    name: str
    value: int | float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


class SystemMetricsCollector:
    """Collects system-level metrics."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self._last_cpu_times = None
        self._last_io_counters = None
        self._last_net_io = None

    def collect_cpu_metrics(self) -> dict[str, float]:
        """Collect CPU-related metrics."""
        try:
            # System CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # Process CPU
            process_cpu = self.process.cpu_percent()
            process_memory = self.process.memory_info()
            process_threads = self.process.num_threads()

            return {
                "system_cpu_percent": cpu_percent,
                "system_cpu_count": cpu_count,
                "process_cpu_percent": process_cpu,
                "process_memory_rss": process_memory.rss,
                "process_memory_vms": process_memory.vms,
                "process_threads": process_threads,
            }
        except Exception as e:
            logger.warning(f"Failed to collect CPU metrics: {e}")
            return {}

    def collect_memory_metrics(self) -> dict[str, float]:
        """Collect memory-related metrics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Process memory details
            process_memory_info = self.process.memory_full_info()

            return {
                "system_memory_total": memory.total,
                "system_memory_available": memory.available,
                "system_memory_percent": memory.percent,
                "system_swap_total": swap.total,
                "system_swap_used": swap.used,
                "system_swap_percent": swap.percent,
                "process_memory_rss": process_memory_info.rss,
                "process_memory_vms": process_memory_info.vms,
                "process_memory_shared": getattr(process_memory_info, "shared", 0),
                "process_memory_text": getattr(process_memory_info, "text", 0),
                "process_memory_data": getattr(process_memory_info, "data", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to collect memory metrics: {e}")
            return {}

    def collect_disk_metrics(self) -> dict[str, float]:
        """Collect disk I/O metrics."""
        try:
            # System disk usage
            disk_usage = psutil.disk_usage("/")

            # System disk I/O
            disk_io = psutil.disk_io_counters()

            # Process I/O
            process_io = self.process.io_counters()

            return {
                "system_disk_total": disk_usage.total,
                "system_disk_used": disk_usage.used,
                "system_disk_free": disk_usage.free,
                "system_disk_percent": disk_usage.used / disk_usage.total * 100,
                "system_disk_read_count": disk_io.read_count if disk_io else 0,
                "system_disk_write_count": disk_io.write_count if disk_io else 0,
                "system_disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                "system_disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                "process_io_read_count": process_io.read_count,
                "process_io_write_count": process_io.write_count,
                "process_io_read_bytes": process_io.read_bytes,
                "process_io_write_bytes": process_io.write_bytes,
            }
        except Exception as e:
            logger.warning(f"Failed to collect disk metrics: {e}")
            return {}

    def collect_network_metrics(self) -> dict[str, float]:
        """Collect network I/O metrics."""
        try:
            # System network I/O
            net_io = psutil.net_io_counters()

            return {
                "system_network_bytes_sent": net_io.bytes_sent,
                "system_network_bytes_recv": net_io.bytes_recv,
                "system_network_packets_sent": net_io.packets_sent,
                "system_network_packets_recv": net_io.packets_recv,
                "system_network_errin": net_io.errin,
                "system_network_errout": net_io.errout,
                "system_network_dropin": net_io.dropin,
                "system_network_dropout": net_io.dropout,
            }
        except Exception as e:
            logger.warning(f"Failed to collect network metrics: {e}")
            return {}

    def collect_all_metrics(self) -> dict[str, float]:
        """Collect all system metrics."""
        metrics = {}
        metrics.update(self.collect_cpu_metrics())
        metrics.update(self.collect_memory_metrics())
        metrics.update(self.collect_disk_metrics())
        metrics.update(self.collect_network_metrics())
        return metrics


class TradingMetricsCollector:
    """Collects trading-specific business metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._order_count = 0
        self._filled_orders = 0
        self._rejected_orders = 0
        self._cancelled_orders = 0
        self._order_latencies: deque[float] = deque(maxlen=1000)
        self._pnl_history: deque[tuple[float, str, float]] = deque(maxlen=1000)
        self._position_values: dict[str, dict[str, Any]] = {}
        self._portfolio_values: dict[str, dict[str, Any]] = {}
        self._risk_metrics: dict[str, dict[str, Any]] = {}
        self._start_time = time.time()

    def record_order_submitted(
        self, order_id: str, symbol: str, quantity: float, price: float | None = None
    ) -> None:
        """Record an order submission."""
        with self._lock:
            self._order_count += 1

    def record_order_filled(
        self, order_id: str, symbol: str, quantity: float, price: float, execution_latency_ms: float
    ) -> None:
        """Record an order fill."""
        with self._lock:
            self._filled_orders += 1
            self._order_latencies.append(execution_latency_ms)

    def record_order_rejected(self, order_id: str, reason: str) -> None:
        """Record an order rejection."""
        with self._lock:
            self._rejected_orders += 1

    def record_order_cancelled(self, order_id: str) -> None:
        """Record an order cancellation."""
        with self._lock:
            self._cancelled_orders += 1

    def record_pnl(self, portfolio_id: str, pnl: float) -> None:
        """Record profit and loss."""
        with self._lock:
            self._pnl_history.append((time.time(), portfolio_id, pnl))

    def record_position_value(self, position_id: str, symbol: str, value: float) -> None:
        """Record position value."""
        with self._lock:
            self._position_values[position_id] = {
                "symbol": symbol,
                "value": value,
                "timestamp": time.time(),
            }

    def record_portfolio_value(self, portfolio_id: str, value: float) -> None:
        """Record portfolio total value."""
        with self._lock:
            self._portfolio_values[portfolio_id] = {"value": value, "timestamp": time.time()}

    def record_risk_metric(
        self, metric_name: str, value: float, portfolio_id: str | None = None
    ) -> None:
        """Record risk metric."""
        with self._lock:
            key = f"{metric_name}:{portfolio_id}" if portfolio_id else metric_name
            self._risk_metrics[key] = {
                "value": value,
                "timestamp": time.time(),
                "portfolio_id": portfolio_id,
            }

    def get_trading_metrics(self) -> dict[str, float]:
        """Get current trading metrics."""
        with self._lock:
            runtime_seconds = time.time() - self._start_time

            # Calculate rates
            orders_per_second = self._order_count / runtime_seconds if runtime_seconds > 0 else 0
            fill_rate = self._filled_orders / self._order_count if self._order_count > 0 else 0
            rejection_rate = (
                self._rejected_orders / self._order_count if self._order_count > 0 else 0
            )

            # Calculate average latency
            avg_latency = (
                sum(self._order_latencies) / len(self._order_latencies)
                if self._order_latencies
                else 0
            )

            # Calculate total portfolio value
            total_portfolio_value = sum(p["value"] for p in self._portfolio_values.values())

            # Calculate current P&L
            current_pnl = self._pnl_history[-1][2] if self._pnl_history else 0

            return {
                "trading_orders_total": self._order_count,
                "trading_orders_filled": self._filled_orders,
                "trading_orders_rejected": self._rejected_orders,
                "trading_orders_cancelled": self._cancelled_orders,
                "trading_orders_per_second": orders_per_second,
                "trading_fill_rate": fill_rate,
                "trading_rejection_rate": rejection_rate,
                "trading_avg_latency_ms": avg_latency,
                "trading_total_portfolio_value": total_portfolio_value,
                "trading_current_pnl": current_pnl,
                "trading_active_positions": len(self._position_values),
                "trading_active_portfolios": len(self._portfolio_values),
                "trading_runtime_seconds": runtime_seconds,
            }


class CustomMetric:
    """Custom metric implementation."""

    def __init__(self, name: str, metric_type: MetricType, description: str = "") -> None:
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self._lock = Lock()
        self._value = 0.0
        self._samples: deque[tuple[float, float, dict[str, str]]] = deque(maxlen=1000)
        self._labels_values: dict[str, float] = defaultdict(float)

    def increment(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment counter metric."""
        if self.metric_type not in (MetricType.COUNTER, MetricType.UP_DOWN_COUNTER):
            raise ValueError(f"Increment not supported for {self.metric_type}")

        with self._lock:
            self._value += value
            self._samples.append((time.time(), value, labels or {}))

            if labels:
                label_key = self._labels_to_key(labels)
                self._labels_values[label_key] += value

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge metric value."""
        if self.metric_type != MetricType.GAUGE:
            raise ValueError(f"Set not supported for {self.metric_type}")

        with self._lock:
            self._value = value
            self._samples.append((time.time(), value, labels or {}))

            if labels:
                label_key = self._labels_to_key(labels)
                self._labels_values[label_key] = value

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe histogram metric value."""
        if self.metric_type != MetricType.HISTOGRAM:
            raise ValueError(f"Observe not supported for {self.metric_type}")

        with self._lock:
            self._samples.append((time.time(), value, labels or {}))

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get current metric value."""
        with self._lock:
            if labels:
                label_key = self._labels_to_key(labels)
                return self._labels_values.get(label_key, 0.0)
            return self._value

    def get_samples(self, limit: int | None = None) -> list[tuple[float, float, dict[str, str]]]:
        """Get recent samples."""
        with self._lock:
            samples = list(self._samples)
            return samples[-limit:] if limit else samples

    def _labels_to_key(self, labels: dict[str, str]) -> str:
        """Convert labels to string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class TradingMetrics:
    """
    Comprehensive metrics collection system for trading operations.

    Provides:
    - System metrics (CPU, memory, etc.)
    - Trading business metrics (orders, P&L, latency)
    - Custom metrics with labels
    - Threshold monitoring and alerting
    - Time-series data collection
    """

    def __init__(self, meter_provider: MeterProvider | None = None) -> None:
        self.meter_provider = meter_provider
        self.meter = None
        if meter_provider:
            self.meter = metrics.get_meter(__name__, "1.0.0")

        # Metric collectors
        self.system_collector = SystemMetricsCollector()
        self.trading_collector = TradingMetricsCollector()

        # Custom metrics registry
        self._custom_metrics: dict[str, CustomMetric] = {}
        self._lock = Lock()

        # Threshold monitoring with domain service
        self._thresholds: list[MetricThreshold] = []
        self._threshold_states: dict[str, dict[str, Any]] = {}
        self._threshold_policy_service = ThresholdPolicyService()
        self._previous_breach_states: dict[str, bool] = {}

        # Background collection
        self._collection_task: asyncio.Task[None] | None = None
        self._stop_collection = False

        # OpenTelemetry instruments
        self._instruments: dict[str, Any] = {}
        self._setup_otel_instruments()

    def _setup_otel_instruments(self) -> None:
        """Setup OpenTelemetry metric instruments."""
        if not self.meter:
            return

        # Trading instruments
        self._instruments.update(
            {
                "orders_total": self.meter.create_counter(
                    name="trading_orders_total",
                    description="Total number of trading orders",
                    unit="orders",
                ),
                "orders_latency": self.meter.create_histogram(
                    name="trading_orders_latency",
                    description="Trading order execution latency",
                    unit="ms",
                ),
                "portfolio_value": self.meter.create_gauge(
                    name="trading_portfolio_value",
                    description="Current portfolio value",
                    unit="USD",
                ),
                "pnl": self.meter.create_gauge(
                    name="trading_pnl", description="Current profit and loss", unit="USD"
                ),
            }
        )

        # System instruments
        self._instruments.update(
            {
                "cpu_usage": self.meter.create_gauge(
                    name="system_cpu_usage_percent",
                    description="CPU usage percentage",
                    unit="percent",
                ),
                "memory_usage": self.meter.create_gauge(
                    name="system_memory_usage_bytes",
                    description="Memory usage in bytes",
                    unit="bytes",
                ),
            }
        )

    def create_counter(self, name: str, description: str = "") -> CustomMetric:
        """Create a counter metric."""
        return self._create_custom_metric(name, MetricType.COUNTER, description)

    def create_gauge(self, name: str, description: str = "") -> CustomMetric:
        """Create a gauge metric."""
        return self._create_custom_metric(name, MetricType.GAUGE, description)

    def create_histogram(self, name: str, description: str = "") -> CustomMetric:
        """Create a histogram metric."""
        return self._create_custom_metric(name, MetricType.HISTOGRAM, description)

    def create_up_down_counter(self, name: str, description: str = "") -> CustomMetric:
        """Create an up-down counter metric."""
        return self._create_custom_metric(name, MetricType.UP_DOWN_COUNTER, description)

    def _create_custom_metric(
        self, name: str, metric_type: MetricType, description: str
    ) -> CustomMetric:
        """Create custom metric."""
        with self._lock:
            if name in self._custom_metrics:
                existing = self._custom_metrics[name]
                if existing.metric_type != metric_type:
                    raise ValueError(f"Metric {name} already exists with different type")
                return existing

            metric = CustomMetric(name, metric_type, description)
            self._custom_metrics[name] = metric
            return metric

    def get_metric(self, name: str) -> CustomMetric | None:
        """Get custom metric by name."""
        return self._custom_metrics.get(name)

    def record_order_submitted(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        price: float | None = None,
        **labels: Any,
    ) -> None:
        """Record order submission."""
        self.trading_collector.record_order_submitted(order_id, symbol, quantity, price)

        if self._instruments.get("orders_total"):
            self._instruments["orders_total"].add(
                1, {"type": "submitted", "symbol": symbol, **labels}
            )

    def record_order_filled(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        price: float,
        execution_latency_ms: float,
        **labels: Any,
    ) -> None:
        """Record order fill."""
        self.trading_collector.record_order_filled(
            order_id, symbol, quantity, price, execution_latency_ms
        )

        if self._instruments.get("orders_total"):
            self._instruments["orders_total"].add(1, {"type": "filled", "symbol": symbol, **labels})

        if self._instruments.get("orders_latency"):
            self._instruments["orders_latency"].record(
                execution_latency_ms, {"symbol": symbol, **labels}
            )

    def record_order_rejected(self, order_id: str, reason: str, **labels: Any) -> None:
        """Record order rejection."""
        self.trading_collector.record_order_rejected(order_id, reason)

        if self._instruments.get("orders_total"):
            self._instruments["orders_total"].add(
                1, {"type": "rejected", "reason": reason, **labels}
            )

    def record_pnl(self, portfolio_id: str, pnl: float, **labels: Any) -> None:
        """Record profit and loss."""
        self.trading_collector.record_pnl(portfolio_id, pnl)

        if self._instruments.get("pnl"):
            self._instruments["pnl"].set(pnl, {"portfolio_id": portfolio_id, **labels})

    def record_portfolio_value(self, portfolio_id: str, value: float, **labels: Any) -> None:
        """Record portfolio value."""
        self.trading_collector.record_portfolio_value(portfolio_id, value)

        if self._instruments.get("portfolio_value"):
            self._instruments["portfolio_value"].set(
                value, {"portfolio_id": portfolio_id, **labels}
            )

    def add_threshold(self, threshold: MetricThreshold) -> None:
        """
        Add alerting threshold - configures domain service.

        This method converts the infrastructure threshold to a domain policy
        and registers it with the domain service.
        """
        self._thresholds.append(threshold)

        # Convert to domain policy
        comparison_map = {
            "greater_than": ThresholdComparison.GREATER_THAN,
            "less_than": ThresholdComparison.LESS_THAN,
            "equal_to": ThresholdComparison.EQUAL_TO,
        }

        policy = ThresholdPolicy(
            metric_name=threshold.metric_name,
            comparison=comparison_map.get(threshold.comparison, ThresholdComparison.GREATER_THAN),
            warning_threshold=threshold.warning_threshold,
            critical_threshold=threshold.critical_threshold,
            consecutive_breaches_required=threshold.consecutive_breaches,
        )

        self._threshold_policy_service.add_policy(policy)

    def check_thresholds(self) -> None:
        """
        Check all metric thresholds - delegates to domain service.

        This method now delegates threshold evaluation to the domain service.
        The infrastructure layer only handles metric collection and alert triggering.
        """
        current_metrics = self.collect_all_metrics()
        current_time = time.time()

        # Evaluate thresholds using domain service
        breach_events = self._threshold_policy_service.evaluate_all_thresholds(
            current_metrics, current_time
        )

        # Handle breach events
        for breach_event in breach_events:
            self._handle_breach_event(breach_event)

        # Log recoveries
        for metric_name in current_metrics:
            state = self._threshold_policy_service.get_breach_state(metric_name)
            if state and not state["in_breach"] and metric_name in self._previous_breach_states:
                if self._previous_breach_states[metric_name]:
                    logger.info(
                        f"Metric {metric_name} returned to normal: {current_metrics[metric_name]}"
                    )
                    self._previous_breach_states[metric_name] = False

        # Update previous states
        for metric_name in current_metrics:
            state = self._threshold_policy_service.get_breach_state(metric_name)
            if state:
                self._previous_breach_states[metric_name] = state["in_breach"]

    def _handle_breach_event(self, breach_event: ThresholdBreachEvent) -> None:
        """
        Handle a threshold breach event from the domain service.

        Args:
            breach_event: The breach event to handle
        """
        # Find the corresponding threshold configuration
        threshold = None
        for t in self._thresholds:
            if t.metric_name == breach_event.metric_name:
                threshold = t
                break

        if threshold:
            # Log the breach
            logger.warning(breach_event.message)

            # Trigger alert callback if configured
            if threshold.alert_callback:
                threshold.alert_callback(
                    breach_event.metric_name,
                    breach_event.current_value,
                    breach_event.threshold_value,
                )

    def _trigger_threshold_alert(self, threshold: MetricThreshold, current_value: float) -> None:
        """Trigger threshold alert."""
        severity = "CRITICAL"
        limit_value = threshold.critical_threshold

        if threshold.warning_threshold and self._is_within_warning_range(current_value, threshold):
            severity = "WARNING"
            limit_value = threshold.warning_threshold

        logger.warning(
            f"Metric threshold breached: {threshold.metric_name}",
            extra={
                "metric_name": threshold.metric_name,
                "current_value": current_value,
                "threshold_value": limit_value,
                "severity": severity,
                "operation_type": "threshold_breach",
            },
        )

        # Call custom alert callback
        if threshold.alert_callback:
            try:
                threshold.alert_callback(threshold.metric_name, current_value, limit_value or 0.0)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _is_within_warning_range(self, value: float, threshold: MetricThreshold) -> bool:
        """Check if value is within warning range only."""
        if not threshold.warning_threshold or not threshold.critical_threshold:
            return True

        if threshold.comparison == "greater_than":
            return threshold.warning_threshold <= value < threshold.critical_threshold
        elif threshold.comparison == "less_than":
            return threshold.critical_threshold < value <= threshold.warning_threshold

        return False

    def collect_all_metrics(self) -> dict[str, float]:
        """Collect all metrics."""
        all_metrics = {}

        # System metrics
        all_metrics.update(self.system_collector.collect_all_metrics())

        # Trading metrics
        all_metrics.update(self.trading_collector.get_trading_metrics())

        # Custom metrics
        with self._lock:
            for name, metric in self._custom_metrics.items():
                all_metrics[name] = metric.get_value()

        return all_metrics

    def get_metric_snapshots(self) -> list[MetricSnapshot]:
        """Get snapshots of all metrics."""
        snapshots = []
        timestamp = time.time()

        all_metrics = self.collect_all_metrics()

        for name, value in all_metrics.items():
            snapshot = MetricSnapshot(name=name, value=value, timestamp=timestamp)
            snapshots.append(snapshot)

        return snapshots

    async def start_background_collection(self, interval: float = 30.0) -> None:
        """Start background metrics collection."""
        if self._collection_task and not self._collection_task.done():
            logger.warning("Background collection already running")
            return

        self._stop_collection = False
        self._collection_task = asyncio.create_task(self._collection_loop(interval))
        logger.info(f"Started background metrics collection (interval: {interval}s)")

    async def stop_background_collection(self) -> None:
        """Stop background metrics collection."""
        self._stop_collection = True

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background metrics collection")

    async def _collection_loop(self, interval: float) -> None:
        """Background collection loop."""
        while not self._stop_collection:
            try:
                # Collect metrics
                self.collect_all_metrics()

                # Check thresholds
                self.check_thresholds()

                # Update OpenTelemetry instruments
                self._update_otel_instruments()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)

    def _update_otel_instruments(self) -> None:
        """Update OpenTelemetry instruments with current values."""
        if not self.meter:
            return

        try:
            # Update system metrics
            system_metrics = self.system_collector.collect_cpu_metrics()
            if self._instruments.get("cpu_usage") and "process_cpu_percent" in system_metrics:
                self._instruments["cpu_usage"].set(system_metrics["process_cpu_percent"])

            memory_metrics = self.system_collector.collect_memory_metrics()
            if self._instruments.get("memory_usage") and "process_memory_rss" in memory_metrics:
                self._instruments["memory_usage"].set(memory_metrics["process_memory_rss"])

        except Exception as e:
            logger.warning(f"Failed to update OpenTelemetry instruments: {e}")


# Global metrics instance
_trading_metrics: TradingMetrics | None = None


def initialize_trading_metrics(meter_provider: MeterProvider | None = None) -> TradingMetrics:
    """Initialize global trading metrics."""
    global _trading_metrics
    _trading_metrics = TradingMetrics(meter_provider)
    return _trading_metrics


def get_trading_metrics() -> TradingMetrics:
    """Get global trading metrics instance."""
    if not _trading_metrics:
        raise RuntimeError(
            "Trading metrics not initialized. Call initialize_trading_metrics() first."
        )
    return _trading_metrics


# Decorator for automatic metrics collection
def track_trading_metric(
    metric_name: str, metric_type: str = "counter", labels: dict[str, str] | None = None
) -> Any:
    """Decorator to automatically track function calls as metrics."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_trading_metrics()
            metric = getattr(metrics, f"create_{metric_type}")(metric_name)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)

                if metric_type == "counter":
                    metric.increment(1.0, labels)
                elif metric_type == "histogram":
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    metric.observe(duration_ms, labels)

                return result

            except Exception as e:
                error_labels = {**(labels or {}), "status": "error", "error_type": type(e).__name__}

                if metric_type == "counter":
                    metric.increment(1.0, error_labels)

                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_trading_metrics()
            metric = getattr(metrics, f"create_{metric_type}")(metric_name)

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)

                if metric_type == "counter":
                    metric.increment(1.0, labels)
                elif metric_type == "histogram":
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    metric.observe(duration_ms, labels)

                return result

            except Exception as e:
                error_labels = {**(labels or {}), "status": "error", "error_type": type(e).__name__}

                if metric_type == "counter":
                    metric.increment(1.0, error_labels)

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
