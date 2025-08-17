"""
Parallel scanner execution engine.

This module provides a high-performance engine for running multiple scanners
in parallel, with configurable concurrency limits and result aggregation.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import time

# Local imports
from main.events.types import AlertType, ScanAlert
from main.scanners.base_scanner import BaseScanner
from main.utils.core import async_retry, chunk_list, create_event_tracker, create_task_safely
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Configuration for a scanner in the engine."""

    scanner: BaseScanner
    priority: int = 5  # 1-10, higher is more important
    enabled: bool = True
    timeout: float = 30.0  # Seconds
    retry_attempts: int = 3
    batch_size: int = 50  # Symbols per batch


@dataclass
class ParallelEngineConfig:
    """Configuration for the parallel scanner engine."""

    max_concurrent_scanners: int = 10
    max_concurrent_symbols: int = 100
    scanner_timeout: float = 60.0
    result_deduplication: bool = True
    aggregation_window: float = 5.0  # Seconds to wait for aggregation
    error_threshold: float = 0.5  # Max error rate before disabling scanner


@dataclass
class ScannerMetrics:
    """Metrics for scanner performance."""

    scanner_name: str
    total_scans: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    total_alerts: int = 0
    avg_scan_time: float = 0.0
    last_error: str | None = None
    last_scan_time: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_scans == 0:
            return 1.0
        return self.successful_scans / self.total_scans


@dataclass
class ScanResult:
    """Result from a scan operation."""

    scanner_name: str
    symbols: list[str]
    alerts: list[ScanAlert]
    duration: float
    error: Exception | None = None


class ParallelScannerEngine:
    """
    High-performance parallel scanner execution engine.

    Manages concurrent execution of multiple scanners across symbols,
    with intelligent batching, error handling, and result aggregation.
    """

    def __init__(
        self, config: ParallelEngineConfig, metrics_collector: MetricsCollector | None = None
    ):
        """
        Initialize parallel scanner engine.

        Args:
            config: Engine configuration
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("parallel_scanner_engine")

        # Scanner registry
        self._scanners: dict[str, ScannerConfig] = {}
        self._scanner_metrics: dict[str, ScannerMetrics] = {}

        # Execution state
        self._running = False
        self._scanner_semaphore = asyncio.Semaphore(config.max_concurrent_scanners)
        self._symbol_semaphore = asyncio.Semaphore(config.max_concurrent_symbols)

        # Result management
        self._recent_alerts: dict[str, list[ScanAlert]] = defaultdict(list)
        self._alert_dedup_cache: set[tuple[str, str, AlertType]] = set()

    def register_scanner(self, scanner_config: ScannerConfig) -> None:
        """
        Register a scanner with the engine.

        Args:
            scanner_config: Scanner configuration
        """
        name = scanner_config.scanner.name
        self._scanners[name] = scanner_config
        self._scanner_metrics[name] = ScannerMetrics(scanner_name=name)

        logger.info(f"Registered scanner: {name} (priority={scanner_config.priority})")

    def unregister_scanner(self, scanner_name: str) -> None:
        """Unregister a scanner."""
        if scanner_name in self._scanners:
            del self._scanners[scanner_name]
            logger.info(f"Unregistered scanner: {scanner_name}")

    async def scan_symbols(
        self, symbols: list[str], scanner_filter: Callable[[str], bool] | None = None, **kwargs
    ) -> list[ScanAlert]:
        """
        Scan symbols using all registered scanners in parallel.

        Args:
            symbols: List of symbols to scan
            scanner_filter: Optional filter function for scanners
            **kwargs: Additional parameters passed to scanners

        Returns:
            Aggregated list of scan alerts
        """
        start_time = time.time()

        # Get enabled scanners
        active_scanners = [
            config
            for name, config in self._scanners.items()
            if config.enabled and (not scanner_filter or scanner_filter(name))
        ]

        if not active_scanners:
            logger.warning("No active scanners available")
            return []

        # Sort by priority (descending)
        active_scanners.sort(key=lambda x: x.priority, reverse=True)

        # Create scan tasks
        tasks = []
        for scanner_config in active_scanners:
            task = create_task_safely(self._scan_with_scanner(scanner_config, symbols, **kwargs))
            tasks.append(task)

        # Execute all scans in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_alerts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scanner_name = active_scanners[i].scanner.name
                logger.error(f"Scanner {scanner_name} failed: {result}")
                self._update_metrics(scanner_name, success=False, error=str(result))
            elif isinstance(result, ScanResult):
                if result.error:
                    self._update_metrics(
                        result.scanner_name, success=False, error=str(result.error)
                    )
                else:
                    self._update_metrics(
                        result.scanner_name,
                        success=True,
                        duration=result.duration,
                        alert_count=len(result.alerts),
                    )
                    all_alerts.extend(result.alerts)

        # Deduplicate alerts if configured
        if self.config.result_deduplication:
            all_alerts = self._deduplicate_alerts(all_alerts)

        # Track execution
        duration = time.time() - start_time
        self._track_execution(len(symbols), len(active_scanners), len(all_alerts), duration)

        logger.info(
            f"Parallel scan completed: {len(symbols)} symbols, "
            f"{len(active_scanners)} scanners, {len(all_alerts)} alerts in {duration:.2f}s"
        )

        return all_alerts

    async def _scan_with_scanner(
        self, scanner_config: ScannerConfig, symbols: list[str], **kwargs
    ) -> ScanResult:
        """Execute scan with a single scanner."""
        scanner = scanner_config.scanner
        start_time = time.time()

        try:
            async with self._scanner_semaphore:
                # Batch symbols if needed
                if len(symbols) > scanner_config.batch_size:
                    batches = list(chunk_list(symbols, scanner_config.batch_size))
                    batch_results = []

                    for batch in batches:
                        # Apply symbol concurrency limit
                        async with self._symbol_semaphore:
                            result = await self._scan_batch(
                                scanner,
                                batch,
                                scanner_config.timeout,
                                scanner_config.retry_attempts,
                                **kwargs,
                            )
                            batch_results.extend(result)

                    alerts = batch_results
                else:
                    # Scan all symbols at once
                    async with self._symbol_semaphore:
                        alerts = await self._scan_batch(
                            scanner,
                            symbols,
                            scanner_config.timeout,
                            scanner_config.retry_attempts,
                            **kwargs,
                        )

            duration = time.time() - start_time

            return ScanResult(
                scanner_name=scanner.name, symbols=symbols, alerts=alerts, duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Scanner {scanner.name} failed: {e}")

            return ScanResult(
                scanner_name=scanner.name, symbols=symbols, alerts=[], duration=duration, error=e
            )

    @async_retry(max_attempts=3, delay=1.0)
    async def _scan_batch(
        self,
        scanner: BaseScanner,
        symbols: list[str],
        timeout: float,
        retry_attempts: int,
        **kwargs,
    ) -> list[ScanAlert]:
        """Scan a batch of symbols with timeout and retry."""
        try:
            # Apply timeout
            alerts = await asyncio.wait_for(scanner.scan(symbols, **kwargs), timeout=timeout)
            return alerts

        except TimeoutError:
            logger.warning(f"Scanner {scanner.name} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Scanner {scanner.name} error: {e}")
            raise

    def _deduplicate_alerts(self, alerts: list[ScanAlert]) -> list[ScanAlert]:
        """Deduplicate alerts based on symbol, scanner, and type."""
        unique_alerts = []
        seen = set()

        for alert in alerts:
            # Create unique key
            key = (alert.symbol, alert.scanner_name, alert.alert_type)

            if key not in seen:
                seen.add(key)
                unique_alerts.append(alert)

                # Add to dedup cache for future reference
                self._alert_dedup_cache.add(key)

        return unique_alerts

    def _update_metrics(
        self,
        scanner_name: str,
        success: bool,
        duration: float | None = None,
        alert_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update scanner metrics."""
        metrics = self._scanner_metrics.get(scanner_name)
        if not metrics:
            return

        metrics.total_scans += 1
        metrics.last_scan_time = datetime.utcnow()

        if success:
            metrics.successful_scans += 1
            if alert_count is not None:
                metrics.total_alerts += alert_count
        else:
            metrics.failed_scans += 1
            metrics.last_error = error

        # Update average scan time
        if duration is not None and success:
            if metrics.avg_scan_time == 0:
                metrics.avg_scan_time = duration
            else:
                # Moving average
                metrics.avg_scan_time = (metrics.avg_scan_time * 0.9) + (duration * 0.1)

        # Check if scanner should be disabled due to errors
        if metrics.success_rate < (1 - self.config.error_threshold):
            logger.warning(
                f"Scanner {scanner_name} error rate {1 - metrics.success_rate:.1%} "
                f"exceeds threshold {self.config.error_threshold:.1%}"
            )
            self._scanners[scanner_name].enabled = False

        # Track metrics
        if self.metrics:
            self.metrics.increment(
                "scanner_engine.scans", tags={"scanner": scanner_name, "success": str(success)}
            )

            if duration is not None:
                self.metrics.histogram(
                    "scanner_engine.scan_duration", duration, tags={"scanner": scanner_name}
                )

    def get_scanner_metrics(self, scanner_name: str | None = None) -> dict[str, ScannerMetrics]:
        """Get metrics for scanners."""
        if scanner_name:
            return {scanner_name: self._scanner_metrics.get(scanner_name)}
        return dict(self._scanner_metrics)

    def enable_scanner(self, scanner_name: str) -> None:
        """Enable a scanner."""
        if scanner_name in self._scanners:
            self._scanners[scanner_name].enabled = True
            logger.info(f"Enabled scanner: {scanner_name}")

    def disable_scanner(self, scanner_name: str) -> None:
        """Disable a scanner."""
        if scanner_name in self._scanners:
            self._scanners[scanner_name].enabled = False
            logger.info(f"Disabled scanner: {scanner_name}")

    def set_scanner_priority(self, scanner_name: str, priority: int) -> None:
        """Update scanner priority."""
        if scanner_name in self._scanners:
            self._scanners[scanner_name].priority = max(1, min(10, priority))
            logger.info(f"Updated {scanner_name} priority to {priority}")

    def _track_execution(
        self, symbol_count: int, scanner_count: int, alert_count: int, duration: float
    ) -> None:
        """Track engine execution."""
        self.event_tracker.track(
            "parallel_scan",
            {
                "symbol_count": symbol_count,
                "scanner_count": scanner_count,
                "alert_count": alert_count,
                "duration": duration,
            },
        )

        if self.metrics:
            self.metrics.histogram(
                "scanner_engine.execution_time",
                duration,
                tags={"symbols": str(symbol_count), "scanners": str(scanner_count)},
            )
