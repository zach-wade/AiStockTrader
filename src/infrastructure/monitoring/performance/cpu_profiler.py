"""
CPU usage profiler.

Provides CPU usage tracking and performance analysis capabilities.
"""

import resource
import time
from collections import deque
from typing import Any

import psutil


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
