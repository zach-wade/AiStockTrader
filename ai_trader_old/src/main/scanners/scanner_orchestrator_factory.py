"""
Scanner Orchestrator Factory for dependency injection.

Provides factory methods for creating scanner orchestrator instances
with proper dependency injection and configuration.
"""

# Standard library imports
import logging

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScanner, IScannerOrchestrator, ScannerConfig
from main.scanners.layers.parallel_scanner_engine import ParallelEngineConfig, ParallelScannerEngine
from main.scanners.scanner_factory_v2 import ScannerFactoryV2
from main.scanners.scanner_orchestrator import OrchestrationConfig, ScannerOrchestrator
from main.utils.monitoring import MetricsCollector
from main.utils.scanners import ScannerCacheManager

logger = logging.getLogger(__name__)


class ScannerOrchestratorFactory:
    """
    Factory for creating scanner orchestrator instances with dependency injection.

    This factory ensures orchestrators are created with:
    - Proper scanner registration
    - Configured execution strategies
    - Event bus integration
    - Metrics and caching support
    """

    @staticmethod
    async def create_orchestrator(
        scanner_factory: ScannerFactoryV2,
        event_bus: IEventBus | None = None,
        metrics_collector: MetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
        config: DictConfig | None = None,
        scanner_types: list[str] | None = None,
    ) -> IScannerOrchestrator:
        """
        Create a scanner orchestrator with registered scanners.

        Args:
            scanner_factory: Factory for creating scanner instances
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
            config: System configuration
            scanner_types: Specific scanner types to register (None for all)

        Returns:
            Configured scanner orchestrator
        """
        # Build orchestration config
        orch_config = ScannerOrchestratorFactory._build_orchestration_config(config)

        # Create parallel engine
        engine = ScannerOrchestratorFactory._create_engine(config, metrics_collector)

        # Create orchestrator
        orchestrator = ScannerOrchestrator(
            engine=engine,
            event_bus=event_bus,
            metrics_collector=metrics_collector,
            cache_manager=cache_manager,
            config=orch_config,
        )

        # Register scanners
        scanners_to_register = scanner_types or scanner_factory.get_available_scanners()
        registered_count = 0

        for scanner_type in scanners_to_register:
            try:
                # Create scanner instance
                scanner = await scanner_factory.create_scanner(scanner_type)

                # Get scanner config (use default for now)
                scanner_config = ScannerConfig(
                    name=scanner_type, enabled=True, priority=5, timeout_seconds=60.0
                )

                # Register with orchestrator
                await orchestrator.register_scanner(scanner, scanner_config)

                registered_count += 1

            except Exception as e:
                logger.error(f"Failed to register scanner '{scanner_type}': {e}")

        logger.info(
            f"Created scanner orchestrator with {registered_count} scanners, "
            f"strategy={orch_config.execution_strategy}"
        )

        return orchestrator

    @staticmethod
    async def create_test_orchestrator(
        event_bus: IEventBus | None = None, scanners: dict[str, IScanner] | None = None
    ) -> IScannerOrchestrator:
        """
        Create a scanner orchestrator for testing.

        Args:
            event_bus: Optional event bus
            scanners: Optional dict of scanner name -> instance

        Returns:
            Test orchestrator instance
        """
        # Test configuration
        test_config = OrchestrationConfig(
            execution_strategy="sequential",  # Easier to test
            deduplication_window_seconds=60,
            min_alert_confidence=0.3,
            cache_results=False,
            publish_to_event_bus=event_bus is not None,
            collect_metrics=False,
        )

        # Simple engine for testing
        engine_config = ParallelEngineConfig(
            max_concurrent_scanners=2, max_concurrent_symbols=10, scanner_timeout=10.0
        )
        engine = ParallelScannerEngine(engine_config)

        # Create orchestrator
        orchestrator = ScannerOrchestrator(engine=engine, event_bus=event_bus, config=test_config)

        # Register provided scanners
        if scanners:
            for name, scanner in scanners.items():
                config = ScannerConfig(name=name, enabled=True, priority=5, timeout_seconds=10.0)
                await orchestrator.register_scanner(scanner, config)

        return orchestrator

    @staticmethod
    async def create_realtime_orchestrator(
        scanner_factory: ScannerFactoryV2,
        event_bus: IEventBus,
        metrics_collector: MetricsCollector,
        cache_manager: ScannerCacheManager,
        config: DictConfig | None = None,
    ) -> IScannerOrchestrator:
        """
        Create an orchestrator optimized for real-time scanning.

        Args:
            scanner_factory: Factory for creating scanners
            event_bus: Event bus (required for real-time)
            metrics_collector: Metrics collector
            cache_manager: Cache manager
            config: System configuration

        Returns:
            Real-time optimized orchestrator
        """
        # Real-time configuration
        realtime_config = OrchestrationConfig(
            execution_strategy="hybrid",  # High priority sequential, rest parallel
            deduplication_window_seconds=180,  # 3 minutes
            min_alert_confidence=0.7,  # Higher threshold
            max_alerts_per_symbol=5,  # Limit noise
            cache_results=True,
            cache_ttl_seconds=300,  # 5 minutes
            publish_to_event_bus=True,
            collect_metrics=True,
            error_threshold=0.2,  # More aggressive disabling
        )

        # High-performance engine
        engine_config = ParallelEngineConfig(
            max_concurrent_scanners=20,
            max_concurrent_symbols=200,
            scanner_timeout=30.0,  # Tighter timeout
            result_deduplication=True,
            aggregation_window=2.0,  # Faster aggregation
            error_threshold=0.2,
        )

        engine = ParallelScannerEngine(engine_config, metrics_collector)

        # Create orchestrator
        orchestrator = ScannerOrchestrator(
            engine=engine,
            event_bus=event_bus,
            metrics_collector=metrics_collector,
            cache_manager=cache_manager,
            config=realtime_config,
        )

        # Register high-priority scanners for real-time
        # These are the most important for real-time trading
        priority_scanners = [
            "volume",  # Volume spikes
            "technical",  # Technical breakouts
            "news",  # News catalysts
            "options",  # Options flow
            "social",  # Social sentiment
        ]

        # Register scanners with adjusted priorities
        for i, scanner_type in enumerate(priority_scanners):
            try:
                scanner = await scanner_factory.create_scanner(scanner_type)

                # Create config with boosted priority for real-time scanners
                config = ScannerConfig(
                    name=scanner_type,
                    enabled=True,
                    priority=10 - i,  # 10, 9, 8, etc.
                    timeout_seconds=20.0,  # Tighter timeout
                )

                await orchestrator.register_scanner(scanner, config)

            except Exception as e:
                logger.error(f"Failed to register scanner '{scanner_type}': {e}")

        logger.info("Created real-time scanner orchestrator")

        return orchestrator

    @staticmethod
    def _build_orchestration_config(config: DictConfig | None) -> OrchestrationConfig:
        """Build orchestration configuration from system config."""
        if not config:
            return OrchestrationConfig()

        # Extract orchestrator config section
        orch_cfg = config.get("scanner_orchestrator", {})

        return OrchestrationConfig(
            execution_strategy=orch_cfg.get("execution_strategy", "parallel"),
            deduplication_window_seconds=orch_cfg.get("deduplication_window_seconds", 300),
            min_alert_confidence=orch_cfg.get("min_alert_confidence", 0.5),
            max_alerts_per_symbol=orch_cfg.get("max_alerts_per_symbol", 10),
            cache_results=orch_cfg.get("cache_results", True),
            cache_ttl_seconds=orch_cfg.get("cache_ttl_seconds", 600),
            publish_to_event_bus=orch_cfg.get("publish_to_event_bus", True),
            collect_metrics=orch_cfg.get("collect_metrics", True),
            error_threshold=orch_cfg.get("error_threshold", 0.3),
        )

    @staticmethod
    def _create_engine(
        config: DictConfig | None, metrics_collector: MetricsCollector | None = None
    ) -> ParallelScannerEngine:
        """Create parallel scanner engine."""
        if config:
            engine_cfg = config.get("scanner_engine", {})
            engine_config = ParallelEngineConfig(
                max_concurrent_scanners=engine_cfg.get("max_concurrent_scanners", 10),
                max_concurrent_symbols=engine_cfg.get("max_concurrent_symbols", 100),
                scanner_timeout=engine_cfg.get("scanner_timeout", 60.0),
                result_deduplication=engine_cfg.get("result_deduplication", True),
                aggregation_window=engine_cfg.get("aggregation_window", 5.0),
                error_threshold=engine_cfg.get("error_threshold", 0.5),
            )
        else:
            engine_config = ParallelEngineConfig()

        return ParallelScannerEngine(engine_config, metrics_collector)


# Convenience functions
async def create_scanner_orchestrator(
    scanner_factory: ScannerFactoryV2,
    event_bus: IEventBus | None = None,
    metrics_collector: MetricsCollector | None = None,
    cache_manager: ScannerCacheManager | None = None,
    config: DictConfig | None = None,
) -> IScannerOrchestrator:
    """
    Convenience function to create a scanner orchestrator.

    Args:
        scanner_factory: Scanner factory instance
        event_bus: Optional event bus
        metrics_collector: Optional metrics collector
        cache_manager: Optional cache manager
        config: Optional configuration

    Returns:
        Scanner orchestrator instance
    """
    return await ScannerOrchestratorFactory.create_orchestrator(
        scanner_factory=scanner_factory,
        event_bus=event_bus,
        metrics_collector=metrics_collector,
        cache_manager=cache_manager,
        config=config,
    )
