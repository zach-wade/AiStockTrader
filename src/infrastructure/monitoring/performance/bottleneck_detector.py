"""
Performance bottleneck detection.

Analyzes performance data to identify bottlenecks and generate
recommendations for performance improvements.
"""

from typing import Any

from .api_tracker import APIPerformanceTracker
from .database_profiler import DatabaseQueryProfiler
from .memory_profiler import MemoryProfiler


class BottleneckDetector:
    """Detects performance bottlenecks and generates recommendations."""

    def __init__(
        self,
        memory_profiler: MemoryProfiler | None = None,
        db_profiler: DatabaseQueryProfiler | None = None,
        api_profiler: APIPerformanceTracker | None = None,
    ):
        self.memory_profiler = memory_profiler
        self.db_profiler = db_profiler
        self.api_profiler = api_profiler

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

    def generate_recommendations(self, report: dict[str, Any]) -> list[str]:
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
