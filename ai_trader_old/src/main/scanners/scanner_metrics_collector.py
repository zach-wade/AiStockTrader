"""
Scanner Metrics Collector

This module provides metrics collection for scanner operations.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from datetime import UTC, datetime
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ScannerMetricsCollector:
    """Collects and tracks metrics for scanner operations."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = defaultdict(
            lambda: {
                "total_scans": 0,
                "successful_scans": 0,
                "failed_scans": 0,
                "total_alerts": 0,
                "total_duration_seconds": 0.0,
                "last_scan_time": None,
                "error_count": 0,
                "symbols_processed": 0,
            }
        )
        self._lock = asyncio.Lock()
        logger.info("ScannerMetricsCollector initialized")

    async def record_scan_start(self, scanner_name: str) -> float:
        """
        Record the start of a scan.

        Args:
            scanner_name: Name of the scanner

        Returns:
            Start timestamp
        """
        return time.time()

    async def record_scan_completion(
        self,
        scanner_name: str,
        start_time: float,
        symbols_count: int,
        alerts_count: int,
        success: bool = True,
        error: str | None = None,
    ):
        """
        Record completion of a scan.

        Args:
            scanner_name: Name of the scanner
            start_time: Start timestamp from record_scan_start
            symbols_count: Number of symbols processed
            alerts_count: Number of alerts generated
            success: Whether the scan was successful
            error: Error message if scan failed
        """
        duration = time.time() - start_time

        async with self._lock:
            metrics = self.metrics[scanner_name]
            metrics["total_scans"] += 1

            if success:
                metrics["successful_scans"] += 1
            else:
                metrics["failed_scans"] += 1
                metrics["error_count"] += 1
                if error:
                    logger.error(f"Scanner {scanner_name} failed: {error}")

            metrics["total_alerts"] += alerts_count
            metrics["total_duration_seconds"] += duration
            metrics["last_scan_time"] = datetime.now(UTC)
            metrics["symbols_processed"] += symbols_count

        logger.info(
            f"Scanner {scanner_name} completed in {duration:.2f}s - "
            f"Processed {symbols_count} symbols, generated {alerts_count} alerts"
        )

    async def get_metrics(self, scanner_name: str | None = None) -> dict[str, Any]:
        """
        Get metrics for a specific scanner or all scanners.

        Args:
            scanner_name: Name of scanner, or None for all scanners

        Returns:
            Dictionary of metrics
        """
        async with self._lock:
            if scanner_name:
                return dict(self.metrics.get(scanner_name, {}))
            else:
                return {name: dict(metrics) for name, metrics in self.metrics.items()}

    async def get_summary(self) -> dict[str, Any]:
        """
        Get summary metrics across all scanners.

        Returns:
            Summary dictionary
        """
        async with self._lock:
            total_scans = sum(m["total_scans"] for m in self.metrics.values())
            successful_scans = sum(m["successful_scans"] for m in self.metrics.values())
            failed_scans = sum(m["failed_scans"] for m in self.metrics.values())
            total_alerts = sum(m["total_alerts"] for m in self.metrics.values())
            total_duration = sum(m["total_duration_seconds"] for m in self.metrics.values())
            total_symbols = sum(m["symbols_processed"] for m in self.metrics.values())

            return {
                "total_scanners": len(self.metrics),
                "total_scans": total_scans,
                "successful_scans": successful_scans,
                "failed_scans": failed_scans,
                "success_rate": (successful_scans / total_scans * 100) if total_scans > 0 else 0,
                "total_alerts": total_alerts,
                "total_duration_seconds": total_duration,
                "total_symbols_processed": total_symbols,
                "avg_duration_per_scan": (total_duration / total_scans) if total_scans > 0 else 0,
                "alerts_per_scan": (total_alerts / total_scans) if total_scans > 0 else 0,
            }

    def reset_metrics(self, scanner_name: str | None = None):
        """
        Reset metrics for a specific scanner or all scanners.

        Args:
            scanner_name: Name of scanner to reset, or None for all
        """
        if scanner_name:
            if scanner_name in self.metrics:
                self.metrics[scanner_name] = {
                    "total_scans": 0,
                    "successful_scans": 0,
                    "failed_scans": 0,
                    "total_alerts": 0,
                    "total_duration_seconds": 0.0,
                    "last_scan_time": None,
                    "error_count": 0,
                    "symbols_processed": 0,
                }
        else:
            self.metrics.clear()

        logger.info(f"Metrics reset for: {scanner_name or 'all scanners'}")


# Global instance for easy access
_global_metrics_collector = None


def get_scanner_metrics_collector() -> ScannerMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = ScannerMetricsCollector()
    return _global_metrics_collector
