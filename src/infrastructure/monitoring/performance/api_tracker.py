"""
API endpoint performance tracker.

Tracks API endpoint performance, response times, error rates,
and throughput metrics.
"""

import threading
import time
from collections import defaultdict
from typing import Any


class APIPerformanceTracker:
    """Track API endpoint performance."""

    def __init__(self) -> None:
        self._endpoint_stats: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._slow_request_threshold = 5.0  # 5 second threshold

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
            "is_slow": duration > self._slow_request_threshold,
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

    def get_slow_endpoints(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get slowest API endpoints."""
        endpoint_stats = self.get_endpoint_statistics()
        slow_endpoints = sorted(
            endpoint_stats.items(), key=lambda x: x[1]["p99_duration"], reverse=True
        )[:limit]

        return [
            {
                "endpoint": endpoint,
                "p99_duration": stats["p99_duration"],
                "error_rate": stats["error_rate"],
                "request_count": stats["request_count"],
                "slow_request_count": stats["slow_request_count"],
            }
            for endpoint, stats in slow_endpoints
        ]

    def set_slow_request_threshold(self, threshold: float) -> None:
        """Set the slow request threshold in seconds."""
        self._slow_request_threshold = threshold

    def get_slow_request_threshold(self) -> float:
        """Get the current slow request threshold."""
        return self._slow_request_threshold
