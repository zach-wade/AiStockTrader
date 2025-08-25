"""
Memory usage profiler.

Provides memory profiling capabilities including leak detection,
memory snapshots, and growth analysis.
"""

import logging
import time
import tracemalloc
from collections import deque
from typing import Any

import psutil

logger = logging.getLogger(__name__)


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
