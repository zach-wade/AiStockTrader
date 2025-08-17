# File: ai_trader/scanners/volume_scanner.py

# Standard library imports
from datetime import UTC, datetime
import logging

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.events.types import AlertType, ScanAlert
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerRepository
from main.utils.core import timer
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector

from ..catalyst_scanner_base import CatalystScannerBase

logger = logging.getLogger(__name__)


class VolumeScanner(CatalystScannerBase):
    """
    Scans for unusual volume spikes using the new repository pattern
    with hot/cold storage awareness and performance optimizations.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: IEventBus | None = None,
        metrics_collector: ScannerMetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
    ):
        """
        Initializes the VolumeScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "VolumeScanner", config, repository, event_bus, metrics_collector, cache_manager
        )

        # Scanner-specific parameters
        self.params = self.config.get("scanners.volume", {})
        self.volume_spike_threshold = self.params.get("spike_threshold_ratio", 2.5)
        self.lookback_days = self.params.get("lookback_days", 20)
        self.min_volume = self.params.get("min_volume", 100000)

    async def scan(self, symbols: list[str], **kwargs) -> list[ScanAlert]:
        """
        Finds all symbols in the universe with a recent volume spike.

        Uses the new repository pattern with hot/cold storage routing
        for optimal performance.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List[ScanAlert] objects for detected volume spikes
        """
        if not self._initialized:
            await self.initialize()

        with timer() as t:
            logger.info(f"📊 Volume Scanner: Analyzing {len(symbols)} symbols for volume spikes...")

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache first if enabled
                cache_key = self.generate_cache_key(
                    "scan", symbols, lookback_days=self.lookback_days
                )

                if self.cache and self.use_cache:
                    cached_alerts = await self.cache.get(self.name, symbols, {"key": cache_key})
                    if cached_alerts is not None:
                        logger.info("Using cached results for volume scan")
                        return cached_alerts

                # Get latest prices first (with interval info)
                latest_prices = await self.repository.get_latest_prices(symbols)

                # Get volume statistics matched to the data intervals
                volume_stats = await self.repository.get_matched_volume_statistics(
                    symbols, latest_prices, self.lookback_days
                )

                # Check if we got data
                if not volume_stats:
                    logger.warning("No volume statistics returned from repository")
                    volume_stats = {}

                if not latest_prices:
                    logger.warning("No latest prices returned from repository")
                    latest_prices = {}

                # Analyze volume spikes using batch processing
                async def process_symbol(symbol: str) -> ScanAlert | None:
                    stats = volume_stats.get(symbol, {})
                    price_data = latest_prices.get(symbol, {})

                    if not stats or not price_data:
                        return None

                    current_volume = price_data.get("volume", 0)
                    avg_volume = stats.get("avg_volume", 0)

                    # Skip if no average volume or below minimum
                    if avg_volume == 0 or current_volume < self.min_volume:
                        return None

                    # Calculate volume ratio
                    volume_ratio = current_volume / avg_volume

                    # Check for spike
                    if volume_ratio >= self.volume_spike_threshold:
                        # Calculate z-score if std deviation available
                        z_score = 0
                        if stats.get("std_volume", 0) > 0:
                            z_score = (current_volume - avg_volume) / stats["std_volume"]

                        # Normalize score to 0.0-1.0 range
                        normalized_score = min(volume_ratio / 5.0, 1.0)

                        alert = self.create_alert(
                            symbol=symbol,
                            alert_type=AlertType.VOLUME_SPIKE,
                            score=normalized_score,
                            metadata={
                                "volume_ratio": round(volume_ratio, 2),
                                "z_score": round(z_score, 2),
                                "reason": f"Volume Spike: {volume_ratio:.1f}x the {self.lookback_days}-day average",
                                "current_volume": current_volume,
                                "avg_volume": avg_volume,
                                "current_price": price_data.get("close", 0),
                                "timestamp": price_data.get("timestamp", datetime.now(UTC)),
                                "data_interval": price_data.get("data_interval", "unknown"),
                            },
                        )

                        # Record alert metric
                        if self.metrics:
                            self.metrics.record_alert_generated(
                                self.name, AlertType.VOLUME_SPIKE, symbol, normalized_score
                            )

                        return alert

                    return None

                # Process all symbols concurrently
                alerts = await self.process_symbols_individually(
                    symbols,
                    process_symbol,
                    max_concurrent=self.config.get("max_concurrent_symbols", 20),
                )

                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.set(self.name, symbols, {"key": cache_key}, alerts)

                # Publish alerts to event bus
                await self.publish_alerts_batch(alerts)

                logger.info(
                    f"✅ Volume Scanner: Found {len(alerts)} symbols with significant "
                    f"volume spikes in {t.elapsed * 1000:.2f}ms"
                )

                # Record scan metrics
                await self.record_scan_metrics(
                    scan_start if self.metrics else datetime.now(UTC), len(symbols), len(alerts)
                )

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Volume Scanner: {e}", exc_info=True)

                # Record error metric
                await self.record_scan_metrics(
                    scan_start if self.metrics else datetime.now(UTC), len(symbols), 0, error=e
                )

                return []
