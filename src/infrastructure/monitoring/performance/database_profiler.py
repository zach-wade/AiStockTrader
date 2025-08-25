"""
Database query performance profiler.

Provides database query monitoring, slow query detection,
and query performance statistics.
"""

import re
import threading
import time
from collections import defaultdict
from typing import Any


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

    def set_slow_query_threshold(self, threshold: float) -> None:
        """Set the slow query threshold in seconds."""
        self._slow_query_threshold = threshold

    def get_slow_query_threshold(self) -> float:
        """Get the current slow query threshold."""
        return self._slow_query_threshold
