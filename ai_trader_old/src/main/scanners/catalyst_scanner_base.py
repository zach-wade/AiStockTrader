"""
Base class for catalyst scanners with shared initialization and utilities.

This class provides common functionality for all catalyst scanners to reduce
code duplication and ensure consistent behavior.
"""

# Standard library imports
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import logging
from typing import Any, TypeVar

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.events.types import ScanAlert
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScanner, IScannerRepository
from main.scanners.base_scanner import BaseScanner
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector, ScannerQueryBuilder

logger = logging.getLogger(__name__)

# Type variable for batch processing
T = TypeVar("T")


class CatalystScannerBase(BaseScanner, IScanner, ABC):
    """
    Base class for all catalyst scanners with shared initialization.

    This class provides:
    - Common initialization pattern
    - Shared cache key generation
    - Common metrics recording
    - Standard error handling
    - Alert publishing utilities
    """

    def __init__(
        self,
        scanner_name: str,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: IEventBus | None = None,
        metrics_collector: ScannerMetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
    ):
        """
        Initialize catalyst scanner with standard dependencies.

        Args:
            scanner_name: Name of the scanner
            config: Scanner configuration
            repository: Scanner data repository
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(scanner_name)
        self.config = config
        self.repository = repository
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.cache = cache_manager

        # Query builder for optimized queries
        self.query_builder = ScannerQueryBuilder()

        # Event publisher for layer qualification events
        self.event_publisher = ScannerEventPublisher(event_bus) if event_bus else None

        # Track initialization state
        self._initialized = False

        # Extract common configuration
        self._extract_config()

        logger.info(f"Initialized {scanner_name} with catalyst base")

    def _extract_config(self):
        """Extract common configuration parameters."""
        # These can be overridden by subclasses
        self.use_cache = self.config.get("use_cache", True)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 300)
        self.batch_size = self.config.get("batch_size", 100)
        self.timeout_seconds = self.config.get("timeout_seconds", 60)

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.name}")

        # Call subclass-specific initialization
        await self._initialize_specific()

        self._initialized = True

    async def _initialize_specific(self) -> None:
        """Subclass-specific initialization. Override if needed."""
        pass

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")

        # Call subclass-specific cleanup
        await self._cleanup_specific()

        self._initialized = False

    async def _cleanup_specific(self) -> None:
        """Subclass-specific cleanup. Override if needed."""
        pass

    def generate_cache_key(self, operation: str, symbols: list[str], **params) -> str:
        """
        Generate a standardized cache key for scanner operations.

        Args:
            operation: Operation name (e.g., 'scan', 'analyze')
            symbols: List of symbols
            **params: Additional parameters to include in key

        Returns:
            Cache key string
        """
        # Use first 10 symbols for key to avoid overly long keys
        symbol_key = ",".join(sorted(symbols[:10]))
        param_key = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{self.name}:{operation}:{symbol_key}:{param_key}"

    async def get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs) -> Any:
        """
        Get data from cache or fetch if not available.

        Args:
            cache_key: Cache key
            fetch_func: Async function to fetch data if not cached
            *args: Arguments for fetch function
            **kwargs: Keyword arguments for fetch function

        Returns:
            Cached or fetched data
        """
        # Check cache if enabled
        if self.cache and self.use_cache:
            cached_result = await self.cache.get(
                self.name, cache_key.split(":")[2].split(","), {"key": cache_key}  # Extract symbols
            )
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result

        # Fetch data
        result = await fetch_func(*args, **kwargs)

        # Cache result if enabled
        if self.cache and self.use_cache and result:
            await self.cache.set(
                self.name,
                cache_key.split(":")[2].split(","),  # Extract symbols
                {"key": cache_key},
                result,
            )

        return result

    async def record_scan_metrics(
        self,
        start_time: datetime,
        symbol_count: int,
        alert_count: int,
        error: Exception | None = None,
    ):
        """
        Record standard scan metrics.

        Args:
            start_time: When the scan started
            symbol_count: Number of symbols scanned
            alert_count: Number of alerts generated
            error: Optional error that occurred
        """
        if not self.metrics:
            return

        duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        # Record duration
        self.metrics.record_scan_duration(self.name, duration_ms, symbol_count)

        # Record error if any
        if error:
            self.metrics.record_scan_error(self.name, type(error).__name__, str(error))

    async def publish_alerts_batch(self, alerts: list[ScanAlert]) -> None:
        """
        Publish alerts to event bus with batching.

        Args:
            alerts: List of alerts to publish
        """
        if not alerts or not self.event_bus:
            return

        # Publish to event bus using parent class method
        await self.publish_alerts_to_event_bus(alerts, self.event_bus)

        # Record metrics for each alert
        if self.metrics:
            for alert in alerts:
                self.metrics.record_alert_generated(
                    self.name, alert.alert_type, alert.symbol, alert.score
                )

    async def publish_symbol_qualified(
        self, symbol: str, layer: DataLayer, reason: str, metrics: dict[str, Any] | None = None
    ):
        """
        Publish symbol qualification event.

        Args:
            symbol: Symbol that qualified
            layer: Layer the symbol qualified for
            reason: Reason for qualification
            metrics: Optional metrics data
        """
        if self.event_publisher:
            await self.event_publisher.publish_symbol_qualified(
                symbol=symbol, layer=layer, qualification_reason=reason, metrics=metrics or {}
            )

    async def publish_symbol_promoted(
        self, symbol: str, from_layer: DataLayer, to_layer: DataLayer, reason: str
    ):
        """
        Publish symbol promotion event.

        Args:
            symbol: Symbol that was promoted
            from_layer: Previous layer
            to_layer: New layer
            reason: Reason for promotion
        """
        if self.event_publisher:
            await self.event_publisher.publish_symbol_promoted(
                symbol=symbol, from_layer=from_layer, to_layer=to_layer, promotion_reason=reason
            )

    async def process_symbols_in_batches(
        self,
        symbols: list[str],
        batch_processor: Callable[[list[str]], Awaitable[list[T]]],
        batch_size: int | None = None,
        max_concurrent: int | None = None,
    ) -> list[T]:
        """
        Process symbols in concurrent batches for improved performance.

        Args:
            symbols: List of symbols to process
            batch_processor: Async function to process a batch of symbols
            batch_size: Size of each batch (default from config)
            max_concurrent: Max concurrent batches (default from config)

        Returns:
            Flattened list of results from all batches
        """
        # Use config defaults if not specified
        if batch_size is None:
            batch_size = self.config.get("batch_size", 50)
        if max_concurrent is None:
            max_concurrent = self.config.get("max_concurrent_batches", 5)

        # Split symbols into batches
        batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch_with_semaphore(batch: list[str]) -> list[T]:
            async with semaphore:
                try:
                    return await batch_processor(batch)
                except Exception as e:
                    logger.error(f"Error processing batch of {len(batch)} symbols: {e}")
                    return []

        # Execute all batches concurrently
        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in batches], return_exceptions=False
        )

        # Flatten results
        all_results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                all_results.extend(batch_result)

        logger.debug(
            f"Processed {len(symbols)} symbols in {len(batches)} batches "
            f"(batch_size={batch_size}, max_concurrent={max_concurrent})"
        )

        return all_results

    async def process_symbols_individually(
        self,
        symbols: list[str],
        processor: Callable[[str], Awaitable[T | None]],
        max_concurrent: int | None = None,
    ) -> list[T]:
        """
        Process symbols individually with concurrency control.

        Args:
            symbols: List of symbols to process
            processor: Async function to process a single symbol
            max_concurrent: Max concurrent operations (default from config)

        Returns:
            List of non-None results
        """
        if max_concurrent is None:
            max_concurrent = self.config.get("max_concurrent_symbols", 10)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(symbol: str) -> T | None:
            async with semaphore:
                try:
                    return await processor(symbol)
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    raise

        # Execute all symbols concurrently
        results = await asyncio.gather(
            *[process_with_semaphore(symbol) for symbol in symbols], return_exceptions=False
        )

        # Filter out None results
        return [r for r in results if r is not None]

    @abstractmethod
    async def scan(self, symbols: list[str], **kwargs) -> list[ScanAlert]:
        """
        Abstract scan method to be implemented by subclasses.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List of scan alerts
        """
        pass

    async def run(self, universe: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Legacy method for backward compatibility.
        Converts ScanAlert objects back to dictionary format.

        Args:
            universe: List of symbols to scan

        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        alerts = await self.scan(universe)

        # Convert alerts to legacy format
        legacy_signals = {}
        for alert in alerts:
            if alert.symbol not in legacy_signals:
                legacy_signals[alert.symbol] = []

            # Extract signal type from alert type
            signal_type = alert.alert_type.value.lower().replace("_", " ")

            legacy_signals[alert.symbol].append(
                {
                    "score": alert.score * 5.0 if alert.score else 0,  # Convert to 0-5 range
                    "reason": alert.metadata.get("reason", ""),
                    "signal_type": signal_type,
                    "metadata": alert.metadata,
                }
            )

        return legacy_signals
