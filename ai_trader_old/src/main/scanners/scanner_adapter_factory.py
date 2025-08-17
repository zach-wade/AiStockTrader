"""
Scanner Adapter Factory for dependency injection.

Provides factory methods for creating scanner adapter instances
with proper dependency injection and configuration.
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerAdapter
from main.scanners.layers.parallel_scanner_engine import ParallelEngineConfig, ParallelScannerEngine
from main.scanners.scanner_adapter import AlertToSignalConfig, ScannerAdapter, ScannerAdapterConfig
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class ScannerAdapterFactory:
    """
    Factory for creating scanner adapter instances with dependency injection.

    This factory ensures scanner adapters are created with:
    - Proper interface dependencies
    - Configurable engines
    - Event bus integration
    - Metrics collection
    """

    @staticmethod
    def create_adapter(
        config: DictConfig,
        database: IAsyncDatabase,
        event_bus: IEventBus | None = None,
        metrics_collector: MetricsCollector | None = None,
        engine: ParallelScannerEngine | None = None,
    ) -> IScannerAdapter:
        """
        Create a scanner adapter instance with full DI support.

        Args:
            config: System configuration
            database: Database interface
            event_bus: Optional event bus for integration
            metrics_collector: Optional metrics collector
            engine: Optional pre-configured engine

        Returns:
            Scanner adapter instance
        """
        # Extract adapter configuration
        adapter_config = ScannerAdapterFactory._build_adapter_config(config)

        # Create engine if not provided
        if engine is None:
            engine = ScannerAdapterFactory._create_engine(
                adapter_config.engine_config, metrics_collector
            )

        # Create adapter with all dependencies
        adapter = ScannerAdapter(
            config=adapter_config,
            database=database,
            engine=engine,
            event_bus=event_bus,
            metrics_collector=metrics_collector,
        )

        logger.info(
            f"Created scanner adapter with engine_config={adapter_config.engine_config}, "
            f"event_bus={'enabled' if event_bus else 'disabled'}"
        )

        return adapter

    @staticmethod
    def create_test_adapter(
        database: IAsyncDatabase,
        event_bus: IEventBus | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> IScannerAdapter:
        """
        Create a scanner adapter for testing with minimal configuration.

        Args:
            database: Database interface
            event_bus: Optional event bus
            config_overrides: Optional config overrides

        Returns:
            Scanner adapter instance for testing
        """
        # Default test configuration
        test_config = {
            "scanner_adapter": {
                "scan_interval": 5.0,
                "universe_size_limit": 100,
                "enable_continuous_scanning": False,
                "persist_alerts": False,
                "alert_to_signal": {
                    "min_confidence": 0.3,
                    "min_alerts_for_signal": 1,
                    "signal_aggregation_window": 60.0,
                },
                "engine": {
                    "max_concurrent_scanners": 5,
                    "max_concurrent_symbols": 50,
                    "scanner_timeout": 30.0,
                },
            }
        }

        # Apply overrides if provided
        if config_overrides:
            test_config["scanner_adapter"].update(config_overrides)

        # Convert to DictConfig
        # Third-party imports
        from omegaconf import OmegaConf

        config = OmegaConf.create(test_config)

        return ScannerAdapterFactory.create_adapter(
            config=config, database=database, event_bus=event_bus
        )

    @staticmethod
    def create_realtime_adapter(
        config: DictConfig,
        database: IAsyncDatabase,
        event_bus: IEventBus,
        metrics_collector: MetricsCollector,
    ) -> IScannerAdapter:
        """
        Create a scanner adapter optimized for real-time trading.

        Args:
            config: System configuration
            database: Database interface
            event_bus: Event bus (required for real-time)
            metrics_collector: Metrics collector

        Returns:
            Scanner adapter configured for real-time operation
        """
        # Override configuration for real-time
        realtime_overrides = {
            "scan_interval": 30.0,  # 30 second scans
            "enable_continuous_scanning": True,
            "persist_alerts": True,
            "alert_to_signal": {
                "min_confidence": 0.7,  # Higher confidence for real-time
                "min_alerts_for_signal": 2,
                "signal_aggregation_window": 180.0,  # 3 minute window
            },
        }

        # Build config with overrides
        adapter_config = ScannerAdapterFactory._build_adapter_config(
            config, overrides=realtime_overrides
        )

        # Create optimized engine
        engine_config = ParallelEngineConfig(
            max_concurrent_scanners=20,
            max_concurrent_symbols=200,
            scanner_timeout=45.0,
            result_deduplication=True,
            aggregation_window=3.0,
        )

        engine = ParallelScannerEngine(engine_config, metrics_collector)

        return ScannerAdapter(
            config=adapter_config,
            database=database,
            engine=engine,
            event_bus=event_bus,
            metrics_collector=metrics_collector,
        )

    @staticmethod
    def _build_adapter_config(
        config: DictConfig, overrides: dict[str, Any] | None = None
    ) -> ScannerAdapterConfig:
        """Build scanner adapter configuration from system config."""
        # Extract scanner adapter config section
        adapter_cfg = config.get("scanner_adapter", {})

        # Apply overrides if provided
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in adapter_cfg:
                    adapter_cfg[key].update(value)
                else:
                    adapter_cfg[key] = value

        # Build alert to signal config
        alert_cfg = adapter_cfg.get("alert_to_signal", {})
        alert_to_signal_config = AlertToSignalConfig(
            min_confidence=alert_cfg.get("min_confidence", 0.5),
            alert_weights=alert_cfg.get("alert_weights"),  # Uses default if None
            signal_aggregation_window=alert_cfg.get("signal_aggregation_window", 300.0),
            min_alerts_for_signal=alert_cfg.get("min_alerts_for_signal", 2),
            decay_rate=alert_cfg.get("decay_rate", 0.95),
        )

        # Build engine config
        engine_cfg = adapter_cfg.get("engine", {})
        engine_config = ParallelEngineConfig(
            max_concurrent_scanners=engine_cfg.get("max_concurrent_scanners", 10),
            max_concurrent_symbols=engine_cfg.get("max_concurrent_symbols", 100),
            scanner_timeout=engine_cfg.get("scanner_timeout", 60.0),
            result_deduplication=engine_cfg.get("result_deduplication", True),
            aggregation_window=engine_cfg.get("aggregation_window", 5.0),
            error_threshold=engine_cfg.get("error_threshold", 0.5),
        )

        # Build adapter config
        return ScannerAdapterConfig(
            engine_config=engine_config,
            alert_to_signal_config=alert_to_signal_config,
            scan_interval=adapter_cfg.get("scan_interval", 60.0),
            universe_size_limit=adapter_cfg.get("universe_size_limit", 500),
            enable_continuous_scanning=adapter_cfg.get("enable_continuous_scanning", True),
            persist_alerts=adapter_cfg.get("persist_alerts", True),
        )

    @staticmethod
    def _create_engine(
        config: ParallelEngineConfig, metrics_collector: MetricsCollector | None = None
    ) -> ParallelScannerEngine:
        """Create scanner engine instance."""
        return ParallelScannerEngine(config, metrics_collector)


# Convenience functions
def create_scanner_adapter(
    config: DictConfig,
    database: IAsyncDatabase,
    event_bus: IEventBus | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> IScannerAdapter:
    """
    Convenience function to create a scanner adapter.

    Args:
        config: System configuration
        database: Database interface
        event_bus: Optional event bus
        metrics_collector: Optional metrics collector

    Returns:
        Scanner adapter instance
    """
    return ScannerAdapterFactory.create_adapter(
        config=config, database=database, event_bus=event_bus, metrics_collector=metrics_collector
    )
