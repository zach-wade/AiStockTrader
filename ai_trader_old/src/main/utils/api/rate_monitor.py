"""
Real-time API rate monitoring for backfill optimization.

This module tracks API request rates and provides insights for optimization.
"""

# Standard library imports
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class RateStats:
    """Statistics for API rate monitoring."""

    requests_per_second: float = 0.0
    requests_per_minute: float = 0.0
    peak_rps: float = 0.0
    total_requests: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)


class APIRateMonitor:
    """Monitor API request rates in real-time."""

    def __init__(self, window_seconds: int = 60):
        """
        Initialize rate monitor.

        Args:
            window_seconds: Rolling window size for rate calculation
        """
        self.window_seconds = window_seconds
        self.request_times: deque[float] = deque()
        self.stats = RateStats()
        self._lock = asyncio.Lock()
        self._monitoring_task: asyncio.Task | None = None

    async def record_request(self):
        """Record a new API request."""
        async with self._lock:
            current_time = time.time()
            self.request_times.append(current_time)
            self.stats.total_requests += 1

            # Clean old requests outside window
            cutoff_time = current_time - self.window_seconds
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()

            # Update statistics
            self._update_stats_locked()

    def _update_stats_locked(self):
        """Update statistics (must be called with lock held)."""
        if not self.request_times:
            self.stats.requests_per_second = 0.0
            self.stats.requests_per_minute = 0.0
            return

        # Calculate rates
        time_span = self.request_times[-1] - self.request_times[0]
        if time_span > 0:
            request_count = len(self.request_times)
            self.stats.requests_per_second = request_count / time_span
            self.stats.requests_per_minute = self.stats.requests_per_second * 60

            # Update peak
            if self.stats.requests_per_second > self.stats.peak_rps:
                self.stats.peak_rps = self.stats.requests_per_second

        self.stats.last_update = datetime.now()

    async def get_stats(self) -> RateStats:
        """Get current rate statistics."""
        async with self._lock:
            # Clean old requests and update stats
            current_time = time.time()
            cutoff_time = current_time - self.window_seconds
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()

            self._update_stats_locked()
            return RateStats(
                requests_per_second=self.stats.requests_per_second,
                requests_per_minute=self.stats.requests_per_minute,
                peak_rps=self.stats.peak_rps,
                total_requests=self.stats.total_requests,
                window_start=self.stats.window_start,
                last_update=self.stats.last_update,
            )

    def start_monitoring(self, log_interval: int = 30):
        """Start background monitoring task that logs statistics.

        Args:
            log_interval: Seconds between log entries
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already started")
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop(log_interval))

    async def _monitoring_loop(self, log_interval: int):
        """Background task to log statistics periodically."""
        logger.info("API rate monitoring started")

        while True:
            try:
                await asyncio.sleep(log_interval)
                stats = await self.get_stats()

                if stats.requests_per_minute > 0:
                    logger.info(
                        f"API Rate: {stats.requests_per_minute:.1f} req/min "
                        f"({stats.requests_per_second:.1f} req/s), "
                        f"Peak: {stats.peak_rps:.1f} req/s, "
                        f"Total: {stats.total_requests}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def stop_monitoring(self):
        """Stop the background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("API rate monitoring stopped")

    def get_utilization_percentage(self, target_rpm: float) -> float:
        """Calculate utilization percentage against target rate.

        Args:
            target_rpm: Target requests per minute

        Returns:
            Utilization percentage (0-100)
        """
        if target_rpm <= 0:
            return 0.0

        return min(100.0, (self.stats.requests_per_minute / target_rpm) * 100)


class GlobalRateMonitor:
    """Global rate monitor for tracking multiple API sources."""

    def __init__(self):
        self.monitors: dict[str, APIRateMonitor] = {}
        self._lock = asyncio.Lock()

    async def get_monitor(self, source: str) -> APIRateMonitor:
        """Get or create a monitor for a specific source."""
        async with self._lock:
            if source not in self.monitors:
                self.monitors[source] = APIRateMonitor()
                logger.info(f"Created rate monitor for {source}")
            return self.monitors[source]

    async def record_request(self, source: str):
        """Record a request for a specific source."""
        monitor = await self.get_monitor(source)
        await monitor.record_request()

    async def get_all_stats(self) -> dict[str, RateStats]:
        """Get statistics for all monitored sources."""
        stats = {}
        async with self._lock:
            for source, monitor in self.monitors.items():
                stats[source] = await monitor.get_stats()
        return stats

    def start_all_monitoring(self, log_interval: int = 30):
        """Start monitoring for all sources."""
        for source, monitor in self.monitors.items():
            monitor.start_monitoring(log_interval)

    def stop_all_monitoring(self):
        """Stop monitoring for all sources."""
        for monitor in self.monitors.values():
            monitor.stop_monitoring()


# Global instance
_global_rate_monitor = GlobalRateMonitor()


async def record_api_request(source: str):
    """Record an API request for rate monitoring."""
    await _global_rate_monitor.record_request(source)


async def get_rate_stats(source: str | None = None) -> dict[str, RateStats]:
    """Get rate statistics.

    Args:
        source: Specific source to get stats for, or None for all

    Returns:
        Dictionary of source -> RateStats
    """
    if source:
        monitor = await _global_rate_monitor.get_monitor(source)
        stats = await monitor.get_stats()
        return {source: stats}
    else:
        return await _global_rate_monitor.get_all_stats()


def start_rate_monitoring(log_interval: int = 30):
    """Start global rate monitoring."""
    _global_rate_monitor.start_all_monitoring(log_interval)


def stop_rate_monitoring():
    """Stop global rate monitoring."""
    _global_rate_monitor.stop_all_monitoring()
