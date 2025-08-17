"""
Scanner Orchestrator implementation with dependency injection.

Provides a general-purpose orchestrator for managing multiple scanners,
implementing the IScannerOrchestrator interface with full DI support.

Key Features:
- Implements IScannerOrchestrator interface
- Multiple execution strategies (parallel, sequential, hybrid)
- Automatic result deduplication
- Scanner health monitoring with auto-disable
- Event bus integration for real-time alerts
- Caching support for performance
- Comprehensive metrics collection

Usage:
    # Create with factory
    from main.scanners.scanner_orchestrator_factory import create_scanner_orchestrator

    orchestrator = create_scanner_orchestrator(
        scanner_factory=scanner_factory,
        event_bus=event_bus,
        metrics_collector=metrics,
        cache_manager=cache,
        config=config
    )

    # Scan universe
    alerts = await orchestrator.scan_universe(['AAPL', 'GOOGL', 'MSFT'])

    # Check scanner status
    status = await orchestrator.get_scanner_status()

Execution Strategies:
- parallel: All scanners run concurrently (fastest)
- sequential: Scanners run one by one in priority order (most control)
- hybrid: High priority scanners run first, then others in parallel
"""

# Standard library imports
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from typing import Any

# Local imports
from main.events.types import EventType, ScanAlert, ScannerAlertEvent
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScanner, IScannerOrchestrator, ScannerConfig
from main.scanners.layers.parallel_scanner_engine import ParallelEngineConfig, ParallelScannerEngine
from main.scanners.layers.parallel_scanner_engine import ScannerConfig as EngineConfig
from main.utils.core import timer
from main.utils.monitoring import MetricsCollector
from main.utils.scanners import ScannerCacheManager

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for scanner orchestration."""

    execution_strategy: str = "parallel"  # parallel, sequential, hybrid
    deduplication_window_seconds: int = 300  # 5 minutes
    min_alert_confidence: float = 0.5
    max_alerts_per_symbol: int = 10
    cache_results: bool = True
    cache_ttl_seconds: int = 600  # 10 minutes
    publish_to_event_bus: bool = True
    collect_metrics: bool = True
    error_threshold: float = 0.3  # Disable scanner if error rate > 30%


@dataclass
class ScannerRegistration:
    """Internal registration for a scanner."""

    scanner: IScanner
    config: ScannerConfig
    enabled: bool = True
    error_count: int = 0
    success_count: int = 0
    last_error: Exception | None = None
    last_scan_time: datetime | None = None


@dataclass
class OrchestrationResult:
    """Result of orchestrated scanning."""

    alerts: list[ScanAlert]
    scanner_results: dict[str, list[ScanAlert]]
    total_duration_ms: float
    errors: dict[str, Exception]


class ScannerOrchestrator(IScannerOrchestrator):
    """
    General-purpose scanner orchestrator with dependency injection.

    Manages multiple scanners, coordinates execution, and aggregates results.
    Implements the IScannerOrchestrator interface for standardized orchestration.
    """

    def __init__(
        self,
        engine: ParallelScannerEngine | None = None,
        event_bus: IEventBus | None = None,
        metrics_collector: MetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
        config: OrchestrationConfig | None = None,
    ):
        """
        Initialize scanner orchestrator with dependencies.

        Args:
            engine: Parallel execution engine (created if not provided)
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
            config: Orchestration configuration
        """
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.cache = cache_manager
        self.config = config or OrchestrationConfig()

        # Create engine if not provided
        if engine is None:
            engine_config = ParallelEngineConfig(
                max_concurrent_scanners=10,
                max_concurrent_symbols=100,
                scanner_timeout=60.0,
                result_deduplication=True,
            )
            self.engine = ParallelScannerEngine(engine_config, metrics_collector)
        else:
            self.engine = engine

        # Scanner registry
        self._scanners: dict[str, ScannerRegistration] = {}

        # Deduplication cache
        self._alert_cache: set[tuple[str, str, str]] = set()  # (symbol, alert_type, scanner)
        self._cache_cleanup_time = datetime.now(UTC)

        # Orchestration state
        self._is_scanning = False
        self._last_scan_time: datetime | None = None

        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._shutdown_task: asyncio.Task | None = None

        logger.info(
            f"ScannerOrchestrator initialized with strategy={self.config.execution_strategy}, "
            f"event_bus={'enabled' if event_bus else 'disabled'}"
        )

    async def register_scanner(self, scanner: IScanner, config: ScannerConfig) -> None:
        """
        Register a scanner with the orchestrator.

        Args:
            scanner: Scanner instance
            config: Scanner configuration
        """
        scanner_name = config.name or getattr(scanner, "name", "unknown")

        # Create registration
        registration = ScannerRegistration(scanner=scanner, config=config, enabled=config.enabled)

        # Store in registry
        self._scanners[scanner_name] = registration

        # Register with engine if using parallel execution
        if self.config.execution_strategy in ["parallel", "hybrid"]:
            engine_config = EngineConfig(
                scanner=scanner,
                priority=config.priority,
                enabled=config.enabled,
                timeout=config.timeout_seconds,
                retry_attempts=config.retry_attempts,
                batch_size=config.batch_size,
            )
            self.engine.register_scanner(engine_config)

        logger.info(
            f"Registered scanner '{scanner_name}' with priority={config.priority}, "
            f"enabled={config.enabled}"
        )

        # Record metric
        if self.metrics:
            self.metrics.increment(
                "orchestrator.scanners_registered", tags={"scanner": scanner_name}
            )

    async def unregister_scanner(self, scanner_name: str) -> None:
        """
        Unregister a scanner by name.

        Args:
            scanner_name: Name of scanner to unregister
        """
        if scanner_name in self._scanners:
            del self._scanners[scanner_name]

            # Unregister from engine
            self.engine.unregister_scanner(scanner_name)

            logger.info(f"Unregistered scanner '{scanner_name}'")

            # Record metric
            if self.metrics:
                self.metrics.increment(
                    "orchestrator.scanners_unregistered", tags={"scanner": scanner_name}
                )
        else:
            logger.warning(f"Scanner '{scanner_name}' not found in registry")

    async def scan_universe(
        self, universe: list[str], scanner_names: list[str] | None = None
    ) -> list[ScanAlert]:
        """
        Scan a universe of symbols using registered scanners.

        Args:
            universe: List of symbols to scan
            scanner_names: Specific scanners to use (None for all)

        Returns:
            Aggregated list of scan alerts
        """
        if self._is_scanning:
            logger.warning("Scan already in progress, skipping")
            return []

        self._is_scanning = True
        start_time = datetime.now(UTC)

        try:
            with timer() as t:
                logger.info(
                    f"Starting orchestrated scan of {len(universe)} symbols "
                    f"with {len(self._scanners)} scanners"
                )

                # Clean up deduplication cache if needed
                await self._cleanup_alert_cache()

                # Determine which scanners to use
                scanners_to_use = self._get_scanners_to_use(scanner_names)

                if not scanners_to_use:
                    logger.warning("No enabled scanners available")
                    return []

                # Execute scan based on strategy
                if self.config.execution_strategy == "parallel":
                    result = await self._scan_parallel(universe, scanners_to_use)
                elif self.config.execution_strategy == "sequential":
                    result = await self._scan_sequential(universe, scanners_to_use)
                else:  # hybrid
                    result = await self._scan_hybrid(universe, scanners_to_use)

                # Process results
                aggregated_alerts = await self._process_results(result)

                # Cache results if enabled
                if self.cache and self.config.cache_results:
                    await self._cache_results(aggregated_alerts)

                # Publish to event bus if enabled
                if self.event_bus and self.config.publish_to_event_bus:
                    await self._publish_alerts(aggregated_alerts)

                # Update state
                self._last_scan_time = start_time

                logger.info(
                    f"Orchestrated scan completed: {len(aggregated_alerts)} alerts "
                    f"in {t.elapsed_ms:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record(
                        "orchestrator.scan_duration_ms",
                        t.elapsed_ms,
                        tags={"strategy": self.config.execution_strategy},
                    )
                    self.metrics.record("orchestrator.alerts_generated", len(aggregated_alerts))

                return aggregated_alerts

        except Exception as e:
            logger.error(f"Error in orchestrated scan: {e}", exc_info=True)
            if self.metrics:
                self.metrics.increment(
                    "orchestrator.scan_errors", tags={"error_type": type(e).__name__}
                )
            return []

        finally:
            self._is_scanning = False

    async def get_scanner_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status information for all registered scanners.

        Returns:
            Dictionary with scanner status information
        """
        status = {}

        for name, registration in self._scanners.items():
            error_rate = 0.0
            if registration.success_count + registration.error_count > 0:
                error_rate = registration.error_count / (
                    registration.success_count + registration.error_count
                )

            status[name] = {
                "enabled": registration.enabled,
                "priority": registration.config.priority,
                "success_count": registration.success_count,
                "error_count": registration.error_count,
                "error_rate": error_rate,
                "last_error": str(registration.last_error) if registration.last_error else None,
                "last_scan_time": (
                    registration.last_scan_time.isoformat() if registration.last_scan_time else None
                ),
                "config": {
                    "timeout_seconds": registration.config.timeout_seconds,
                    "retry_attempts": registration.config.retry_attempts,
                    "batch_size": registration.config.batch_size,
                    "min_confidence": registration.config.min_confidence,
                },
            }

        # Add orchestrator status
        status["_orchestrator"] = {
            "total_scanners": len(self._scanners),
            "enabled_scanners": sum(1 for r in self._scanners.values() if r.enabled),
            "is_scanning": self._is_scanning,
            "last_scan_time": (self._last_scan_time.isoformat() if self._last_scan_time else None),
            "execution_strategy": self.config.execution_strategy,
            "dedup_cache_size": len(self._alert_cache),
        }

        return status

    def _get_scanners_to_use(
        self, scanner_names: list[str] | None = None
    ) -> list[ScannerRegistration]:
        """Get list of scanners to use for scanning."""
        if scanner_names:
            # Use specific scanners
            scanners = [
                reg for name, reg in self._scanners.items() if name in scanner_names and reg.enabled
            ]
        else:
            # Use all enabled scanners
            scanners = [reg for reg in self._scanners.values() if reg.enabled]

        # Sort by priority (higher first)
        scanners.sort(key=lambda x: x.config.priority, reverse=True)

        return scanners

    async def _scan_parallel(
        self, universe: list[str], scanners: list[ScannerRegistration]
    ) -> OrchestrationResult:
        """Execute scanners in parallel using the engine."""
        # The engine handles parallel execution
        alerts = await self.engine.scan_symbols(universe)

        # Track results by scanner
        scanner_results = defaultdict(list)
        for alert in alerts:
            scanner_name = getattr(alert, "source_scanner", "unknown")
            scanner_results[scanner_name].append(alert)

        # Update scanner stats
        for name, reg in self._scanners.items():
            if name in scanner_results:
                reg.success_count += 1
                reg.last_scan_time = datetime.now(UTC)

        return OrchestrationResult(
            alerts=alerts,
            scanner_results=dict(scanner_results),
            total_duration_ms=0,  # Engine tracks this
            errors={},
        )

    async def _scan_sequential(
        self, universe: list[str], scanners: list[ScannerRegistration]
    ) -> OrchestrationResult:
        """Execute scanners sequentially in priority order."""
        all_alerts = []
        scanner_results = {}
        errors = {}
        start_time = datetime.now(UTC)

        for registration in scanners:
            scanner_name = registration.config.name

            try:
                # Run scanner
                alerts = await asyncio.wait_for(
                    registration.scanner.scan(universe), timeout=registration.config.timeout_seconds
                )

                # Store results
                scanner_results[scanner_name] = alerts
                all_alerts.extend(alerts)

                # Update stats
                registration.success_count += 1
                registration.last_scan_time = datetime.now(UTC)

                logger.debug(f"Scanner '{scanner_name}' returned {len(alerts)} alerts")

            except TimeoutError:
                error = TimeoutError(f"Scanner '{scanner_name}' timed out")
                errors[scanner_name] = error
                registration.error_count += 1
                registration.last_error = error
                logger.error(f"Scanner '{scanner_name}' timed out")

            except Exception as e:
                errors[scanner_name] = e
                registration.error_count += 1
                registration.last_error = e
                logger.error(f"Error in scanner '{scanner_name}': {e}")

            # Check error threshold
            await self._check_scanner_health(registration)

        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000

        return OrchestrationResult(
            alerts=all_alerts,
            scanner_results=scanner_results,
            total_duration_ms=duration,
            errors=errors,
        )

    async def _scan_hybrid(
        self, universe: list[str], scanners: list[ScannerRegistration]
    ) -> OrchestrationResult:
        """
        Execute scanners in hybrid mode.

        High priority scanners run first sequentially,
        then lower priority run in parallel.
        """
        # Split scanners by priority
        high_priority = [s for s in scanners if s.config.priority >= 7]
        low_priority = [s for s in scanners if s.config.priority < 7]

        all_alerts = []
        scanner_results = {}
        errors = {}

        # Run high priority sequentially
        if high_priority:
            seq_result = await self._scan_sequential(universe, high_priority)
            all_alerts.extend(seq_result.alerts)
            scanner_results.update(seq_result.scanner_results)
            errors.update(seq_result.errors)

        # Run low priority in parallel if any
        if low_priority:
            # Register with engine temporarily
            for reg in low_priority:
                engine_config = EngineConfig(
                    scanner=reg.scanner,
                    priority=reg.config.priority,
                    enabled=True,
                    timeout=reg.config.timeout_seconds,
                )
                self.engine.register_scanner(engine_config)

            try:
                parallel_alerts = await self.engine.scan_symbols(universe)
                all_alerts.extend(parallel_alerts)

                # Track results
                for alert in parallel_alerts:
                    scanner_name = getattr(alert, "source_scanner", "unknown")
                    if scanner_name not in scanner_results:
                        scanner_results[scanner_name] = []
                    scanner_results[scanner_name].append(alert)

            finally:
                # Unregister from engine
                for reg in low_priority:
                    self.engine.unregister_scanner(reg.config.name)

        return OrchestrationResult(
            alerts=all_alerts, scanner_results=scanner_results, total_duration_ms=0, errors=errors
        )

    async def _process_results(self, result: OrchestrationResult) -> list[ScanAlert]:
        """Process and deduplicate scan results."""
        processed_alerts = []

        for alert in result.alerts:
            # Apply confidence threshold
            if alert.confidence < self.config.min_alert_confidence:
                continue

            # Deduplicate
            cache_key = (
                alert.symbol,
                (
                    alert.alert_type.value
                    if hasattr(alert.alert_type, "value")
                    else str(alert.alert_type)
                ),
                getattr(alert, "source_scanner", "unknown"),
            )

            if cache_key not in self._alert_cache:
                self._alert_cache.add(cache_key)
                processed_alerts.append(alert)

        # Limit alerts per symbol
        symbol_counts = defaultdict(int)
        final_alerts = []

        # Sort by score/confidence to keep best alerts
        processed_alerts.sort(key=lambda x: x.confidence, reverse=True)

        for alert in processed_alerts:
            if symbol_counts[alert.symbol] < self.config.max_alerts_per_symbol:
                final_alerts.append(alert)
                symbol_counts[alert.symbol] += 1

        return final_alerts

    async def _cleanup_alert_cache(self) -> None:
        """Clean up old entries from deduplication cache."""
        now = datetime.now(UTC)

        # Clean up every hour
        if (now - self._cache_cleanup_time).seconds > 3600:
            self._alert_cache.clear()
            self._cache_cleanup_time = now
            logger.debug("Cleared alert deduplication cache")

    async def _check_scanner_health(self, registration: ScannerRegistration) -> None:
        """Check scanner health and disable if error rate too high."""
        total_runs = registration.success_count + registration.error_count

        if total_runs >= 10:  # Need minimum runs
            error_rate = registration.error_count / total_runs

            if error_rate > self.config.error_threshold:
                registration.enabled = False
                logger.warning(
                    f"Disabled scanner '{registration.config.name}' due to high error rate: "
                    f"{error_rate:.2%}"
                )

                if self.metrics:
                    self.metrics.increment(
                        "orchestrator.scanner_disabled",
                        tags={"scanner": registration.config.name, "reason": "high_error_rate"},
                    )

    async def _cache_results(self, alerts: list[ScanAlert]) -> None:
        """Cache scan results."""
        if not self.cache:
            return

        try:
            # Cache by symbol
            alerts_by_symbol = defaultdict(list)
            for alert in alerts:
                alerts_by_symbol[alert.symbol].append(
                    {
                        "alert_type": str(alert.alert_type),
                        "confidence": alert.confidence,
                        "severity": alert.severity,
                        "timestamp": alert.timestamp.isoformat(),
                        "metadata": alert.metadata,
                    }
                )

            # Store in cache
            for symbol, symbol_alerts in alerts_by_symbol.items():
                await self.cache.cache_result(
                    "orchestrator",
                    f"alerts:{symbol}",
                    symbol_alerts,
                    ttl_seconds=self.config.cache_ttl_seconds,
                )

            # Cache summary
            await self.cache.cache_result(
                "orchestrator",
                "last_scan_summary",
                {
                    "total_alerts": len(alerts),
                    "symbols_with_alerts": len(alerts_by_symbol),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                ttl_seconds=self.config.cache_ttl_seconds,
            )

        except Exception as e:
            logger.error(f"Error caching results: {e}")

    async def _publish_alerts(self, alerts: list[ScanAlert]) -> None:
        """Publish alerts to event bus."""
        if not self.event_bus or not alerts:
            return

        try:
            # Group by scanner
            alerts_by_scanner = defaultdict(list)
            for alert in alerts:
                scanner_name = getattr(alert, "source_scanner", "orchestrator")
                alerts_by_scanner[scanner_name].append(alert)

            # Publish events
            for scanner_name, scanner_alerts in alerts_by_scanner.items():
                event = ScannerAlertEvent(
                    alerts=scanner_alerts,
                    scanner_name=scanner_name,
                    event_type=EventType.SCANNER_ALERT,
                    timestamp=datetime.now(UTC),
                )

                await self.event_bus.publish(event)

            logger.debug(f"Published {len(alerts)} alerts to event bus")

        except Exception as e:
            logger.error(f"Error publishing alerts: {e}")

    async def enable_scanner(self, scanner_name: str) -> None:
        """Enable a disabled scanner."""
        if scanner_name in self._scanners:
            self._scanners[scanner_name].enabled = True
            logger.info(f"Enabled scanner '{scanner_name}'")

    async def disable_scanner(self, scanner_name: str) -> None:
        """Disable a scanner."""
        if scanner_name in self._scanners:
            self._scanners[scanner_name].enabled = False
            logger.info(f"Disabled scanner '{scanner_name}'")

    def get_orchestration_config(self) -> OrchestrationConfig:
        """Get current orchestration configuration."""
        return self.config

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        return False

    async def shutdown(self) -> None:
        """Graceful shutdown of the orchestrator."""
        logger.info("Initiating graceful shutdown of scanner orchestrator")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for any ongoing scans to complete
        max_wait = 30  # seconds
        start_time = datetime.now(UTC)

        while self._is_scanning:
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > max_wait:
                logger.warning("Forcing shutdown after timeout")
                break
            await asyncio.sleep(0.5)

        # Perform cleanup
        await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        logger.info("Cleaning up scanner orchestrator")

        # Clean up scanner resources
        cleanup_tasks = []
        for registration in self._scanners.values():
            if hasattr(registration.scanner, "cleanup"):
                cleanup_tasks.append(self._cleanup_scanner(registration))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Clean up engine resources
        if hasattr(self.engine, "cleanup"):
            try:
                await self.engine.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up scanner engine: {e}")

        # Clean up cache manager
        if self.cache_manager:
            try:
                await self.cache_manager.cleanup_expired()
            except Exception as e:
                logger.error(f"Error cleaning up cache manager: {e}")

        # Clear internal caches
        self._alert_cache.clear()
        self._scanners.clear()

        logger.info("Scanner orchestrator cleanup complete")

    async def _cleanup_scanner(self, registration: "ScannerRegistration") -> None:
        """Clean up individual scanner."""
        try:
            await registration.scanner.cleanup()
            logger.debug(f"Cleaned up scanner '{registration.config.name}'")
        except Exception as e:
            logger.error(f"Error cleaning up scanner '{registration.config.name}': {e}")
