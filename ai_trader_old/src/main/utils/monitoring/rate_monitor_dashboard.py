"""
Rate monitoring dashboard for backfill operations.

This script provides a simple dashboard to monitor API request rates during backfill.
"""

# Standard library imports
import asyncio
from datetime import datetime
import logging
from typing import Any

# Local imports
from main.utils.api.rate_monitor import get_rate_stats, start_rate_monitoring, stop_rate_monitoring

logger = logging.getLogger(__name__)


class RateMonitorDashboard:
    """Simple dashboard for monitoring API rates during backfill."""

    def __init__(self, update_interval: int = 10):
        """
        Initialize dashboard.

        Args:
            update_interval: Seconds between dashboard updates
        """
        self.update_interval = update_interval
        self.running = False
        self._task = None

        # Rate limits for comparison
        self.rate_limits = {
            "alpaca_market": 10000,  # 10k requests/minute for premium
            "polygon_market": 6000,  # Conservative estimate for premium
            "yahoo_market": 1800,  # 30 requests/second * 60
        }

    async def start(self):
        """Start the dashboard."""
        if self.running:
            logger.warning("Dashboard already running")
            return

        self.running = True
        start_rate_monitoring(log_interval=30)

        self._task = asyncio.create_task(self._update_loop())
        logger.info("Rate monitoring dashboard started")

    async def stop(self):
        """Stop the dashboard."""
        if not self.running:
            return

        self.running = False
        stop_rate_monitoring()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Rate monitoring dashboard stopped")

    async def _update_loop(self):
        """Main dashboard update loop."""
        while self.running:
            try:
                await self._display_stats()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update: {e}")
                await asyncio.sleep(self.update_interval)

    async def _display_stats(self):
        """Display current statistics."""
        stats = await get_rate_stats()

        if not stats:
            return

        # Create summary
        total_rpm = sum(stat.requests_per_minute for stat in stats.values())
        total_requests = sum(stat.total_requests for stat in stats.values())

        print("\\n" + "=" * 60)
        print(f"API Rate Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Overall: {total_rpm:.1f} req/min, {total_requests} total requests\\n")

        for source, stat in stats.items():
            limit = self.rate_limits.get(source, 0)
            utilization = (stat.requests_per_minute / limit * 100) if limit > 0 else 0

            print(
                f"{source:20} | {stat.requests_per_minute:7.1f} req/min | "
                f"{stat.requests_per_second:6.1f} req/s | "
                f"{utilization:5.1f}% util | "
                f"{stat.total_requests:8} total"
            )

        print()

    async def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        stats = await get_rate_stats()

        total_rpm = sum(stat.requests_per_minute for stat in stats.values())
        total_requests = sum(stat.total_requests for stat in stats.values())

        # Calculate efficiency metrics
        alpaca_stats = stats.get("alpaca_market")
        alpaca_util = 0
        if alpaca_stats:
            alpaca_limit = self.rate_limits.get("alpaca_market", 10000)
            alpaca_util = alpaca_stats.requests_per_minute / alpaca_limit * 100

        return {
            "total_requests_per_minute": total_rpm,
            "total_requests": total_requests,
            "alpaca_utilization_percent": alpaca_util,
            "sources": {
                source: {
                    "requests_per_minute": stat.requests_per_minute,
                    "requests_per_second": stat.requests_per_second,
                    "total_requests": stat.total_requests,
                    "peak_rps": stat.peak_rps,
                }
                for source, stat in stats.items()
            },
            "timestamp": datetime.now().isoformat(),
        }


async def run_dashboard():
    """Run the rate monitoring dashboard."""
    dashboard = RateMonitorDashboard()

    try:
        await dashboard.start()

        # Run until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\\nShutting down dashboard...")
    finally:
        await dashboard.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_dashboard())
