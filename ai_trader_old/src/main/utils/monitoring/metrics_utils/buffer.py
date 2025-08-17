"""
Metrics Buffer Module

Provides buffering capabilities for metrics collection to improve performance
and reduce storage overhead.
"""

# Standard library imports
from collections import defaultdict, deque
from collections.abc import Callable
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class MetricsBuffer:
    """
    Buffer for collecting and aggregating metrics before storage/export.

    Features:
    - Configurable buffer size and flush intervals
    - Automatic aggregation of metrics
    - Thread-safe operations
    - Multiple flush strategies
    """

    def __init__(
        self, max_size: int = 10000, flush_interval: float = 60.0, aggregation_window: float = 5.0
    ):
        """
        Initialize metrics buffer.

        Args:
            max_size: Maximum number of metrics to buffer
            flush_interval: Time in seconds between automatic flushes
            aggregation_window: Time window for aggregating metrics
        """
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.aggregation_window = aggregation_window

        # Buffers for different metric types
        self._counters = defaultdict(float)
        self._gauges = {}
        self._histograms = defaultdict(list)
        self._timers = defaultdict(list)
        self._events = deque(maxlen=max_size)

        # Thread safety
        self._lock = threading.RLock()

        # Flush control
        self._last_flush = time.time()
        self._flush_callbacks = []

        # Statistics
        self._stats = {
            "metrics_buffered": 0,
            "metrics_flushed": 0,
            "buffer_overflows": 0,
            "aggregations_performed": 0,
        }

    def add_counter(self, name: str, value: float = 1.0, tags: dict | None = None):
        """Add a counter metric."""
        with self._lock:
            key = self._create_key(name, tags)
            self._counters[key] += value
            self._stats["metrics_buffered"] += 1
            self._check_flush()

    def set_gauge(self, name: str, value: float, tags: dict | None = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._create_key(name, tags)
            self._gauges[key] = {"value": value, "timestamp": time.time(), "tags": tags or {}}
            self._stats["metrics_buffered"] += 1
            self._check_flush()

    def add_histogram(self, name: str, value: float, tags: dict | None = None):
        """Add a value to a histogram."""
        with self._lock:
            key = self._create_key(name, tags)

            # Limit histogram size
            if len(self._histograms[key]) >= self.max_size // 10:
                # Aggregate old values
                self._aggregate_histogram(key)

            self._histograms[key].append({"value": value, "timestamp": time.time()})
            self._stats["metrics_buffered"] += 1
            self._check_flush()

    def add_timer(self, name: str, duration: float, tags: dict | None = None):
        """Add a timer metric."""
        with self._lock:
            key = self._create_key(name, tags)
            self._timers[key].append({"duration": duration, "timestamp": time.time()})
            self._stats["metrics_buffered"] += 1
            self._check_flush()

    def add_event(self, name: str, data: dict, tags: dict | None = None):
        """Add an event."""
        with self._lock:
            event = {"name": name, "data": data, "tags": tags or {}, "timestamp": time.time()}

            if len(self._events) >= self.max_size:
                self._stats["buffer_overflows"] += 1

            self._events.append(event)
            self._stats["metrics_buffered"] += 1
            self._check_flush()

    def flush(self) -> dict[str, Any]:
        """
        Flush all buffered metrics.

        Returns:
            Dictionary containing all buffered metrics
        """
        with self._lock:
            # Collect all metrics
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": self._aggregate_all_histograms(),
                "timers": self._aggregate_all_timers(),
                "events": list(self._events),
            }

            # Clear buffers
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._events.clear()

            # Update stats
            self._stats["metrics_flushed"] += self._stats["metrics_buffered"]
            self._stats["metrics_buffered"] = 0
            self._last_flush = time.time()

            # Call flush callbacks
            for callback in self._flush_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Flush callback error: {e}")

            return metrics

    def add_flush_callback(self, callback: Callable[[dict], None]):
        """Add a callback to be called on flush."""
        with self._lock:
            self._flush_callbacks.append(callback)

    def _create_key(self, name: str, tags: dict | None) -> str:
        """Create a unique key for a metric."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"

    def _check_flush(self):
        """Check if automatic flush is needed."""
        # Check time-based flush
        if time.time() - self._last_flush >= self.flush_interval:
            self.flush()

        # Check size-based flush
        total_size = (
            len(self._counters)
            + len(self._gauges)
            + sum(len(h) for h in self._histograms.values())
            + sum(len(t) for t in self._timers.values())
            + len(self._events)
        )

        if total_size >= self.max_size:
            self.flush()

    def _aggregate_histogram(self, key: str):
        """Aggregate histogram values."""
        values = self._histograms[key]
        if not values:
            return

        # Keep only recent values
        cutoff_time = time.time() - self.aggregation_window
        recent_values = [v for v in values if v["timestamp"] >= cutoff_time]

        # If we have old values, aggregate them
        if len(recent_values) < len(values):
            old_values = [v["value"] for v in values if v["timestamp"] < cutoff_time]

            # Create aggregated entry
            if old_values:
                aggregated = {
                    "value": sum(old_values) / len(old_values),  # Average
                    "timestamp": cutoff_time,
                    "count": len(old_values),
                    "min": min(old_values),
                    "max": max(old_values),
                }
                recent_values.insert(0, aggregated)

            self._histograms[key] = recent_values
            self._stats["aggregations_performed"] += 1

    def _aggregate_all_histograms(self) -> dict[str, dict]:
        """Aggregate all histograms for flushing."""
        aggregated = {}

        for key, values in self._histograms.items():
            if values:
                all_values = [v.get("value", v) for v in values if isinstance(v, dict)]

                if all_values:
                    # Third-party imports
                    import numpy as np

                    aggregated[key] = {
                        "count": len(all_values),
                        "mean": np.mean(all_values),
                        "min": min(all_values),
                        "max": max(all_values),
                        "p50": np.percentile(all_values, 50),
                        "p90": np.percentile(all_values, 90),
                        "p95": np.percentile(all_values, 95),
                        "p99": np.percentile(all_values, 99),
                        "std": np.std(all_values),
                    }

        return aggregated

    def _aggregate_all_timers(self) -> dict[str, dict]:
        """Aggregate all timers for flushing."""
        aggregated = {}

        for key, timers in self._timers.items():
            if timers:
                durations = [t["duration"] for t in timers]

                if durations:
                    # Third-party imports
                    import numpy as np

                    aggregated[key] = {
                        "count": len(durations),
                        "mean": np.mean(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "p50": np.percentile(durations, 50),
                        "p90": np.percentile(durations, 90),
                        "p95": np.percentile(durations, 95),
                        "p99": np.percentile(durations, 99),
                        "total": sum(durations),
                    }

        return aggregated

    def get_stats(self) -> dict[str, int]:
        """Get buffer statistics."""
        with self._lock:
            return self._stats.copy()

    def get_buffer_sizes(self) -> dict[str, int]:
        """Get current buffer sizes."""
        with self._lock:
            return {
                "counters": len(self._counters),
                "gauges": len(self._gauges),
                "histograms": sum(len(h) for h in self._histograms.values()),
                "timers": sum(len(t) for t in self._timers.values()),
                "events": len(self._events),
            }


# Global buffer instance
_global_buffer = None


def get_global_buffer() -> MetricsBuffer:
    """Get the global metrics buffer instance."""
    global _global_buffer
    if _global_buffer is None:
        _global_buffer = MetricsBuffer()
    return _global_buffer


def set_global_buffer(buffer: MetricsBuffer):
    """Set the global metrics buffer instance."""
    global _global_buffer
    _global_buffer = buffer


# Convenience functions
def buffer_counter(name: str, value: float = 1.0, tags: dict | None = None):
    """Buffer a counter metric."""
    get_global_buffer().add_counter(name, value, tags)


def buffer_gauge(name: str, value: float, tags: dict | None = None):
    """Buffer a gauge metric."""
    get_global_buffer().set_gauge(name, value, tags)


def buffer_histogram(name: str, value: float, tags: dict | None = None):
    """Buffer a histogram value."""
    get_global_buffer().add_histogram(name, value, tags)


def buffer_timer(name: str, duration: float, tags: dict | None = None):
    """Buffer a timer metric."""
    get_global_buffer().add_timer(name, duration, tags)


def buffer_event(name: str, data: dict, tags: dict | None = None):
    """Buffer an event."""
    get_global_buffer().add_event(name, data, tags)


def flush_buffer() -> dict[str, Any]:
    """Flush the global buffer."""
    return get_global_buffer().flush()
