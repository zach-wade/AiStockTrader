"""
Scanner Factory V2 with Clean Architecture

Factory for creating scanner instances with the new interface-based
storage system that avoids circular dependencies.
"""

# Standard library imports
import logging

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.storage.repositories.repository_provider import RepositoryProvider

# Storage components
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScanner, IScannerRepository
from main.interfaces.storage import IStorageRouter
from main.scanners.catalysts.advanced_sentiment_scanner import AdvancedSentimentScanner
from main.scanners.catalysts.coordinated_activity_scanner import CoordinatedActivityScanner
from main.scanners.catalysts.earnings_scanner import EarningsScanner
from main.scanners.catalysts.insider_scanner import InsiderScanner
from main.scanners.catalysts.intermarket_scanner import IntermarketScanner
from main.scanners.catalysts.market_validation_scanner import MarketValidationScanner
from main.scanners.catalysts.news_scanner import NewsScanner
from main.scanners.catalysts.options_scanner import OptionsScanner
from main.scanners.catalysts.sector_scanner import SectorScanner
from main.scanners.catalysts.social_scanner import SocialScanner
from main.scanners.catalysts.technical_scanner import TechnicalScanner

# Scanner implementations
from main.scanners.catalysts.volume_scanner import VolumeScanner
from main.scanners.scanner_cache_manager import ScannerCacheManager

# Supporting components
from main.scanners.scanner_metrics_collector import ScannerMetricsCollector

logger = logging.getLogger(__name__)


class ScannerFactoryV2:
    """
    Factory for creating scanner instances with clean architecture.

    This version uses the new interface-based storage system to avoid
    circular dependencies while maintaining all functionality.
    """

    # Registry of scanner implementations
    SCANNER_REGISTRY: dict[str, type[IScanner]] = {
        "volume": VolumeScanner,
        "technical": TechnicalScanner,
        "news": NewsScanner,
        "earnings": EarningsScanner,
        "social": SocialScanner,
        "insider": InsiderScanner,
        "options": OptionsScanner,
        "sector": SectorScanner,
        "intermarket": IntermarketScanner,
        "market_validation": MarketValidationScanner,
        "coordinated_activity": CoordinatedActivityScanner,
        "advanced_sentiment": AdvancedSentimentScanner,
    }

    def __init__(
        self,
        config: DictConfig,
        db_adapter: IAsyncDatabase,
        event_bus: IEventBus | None = None,
        metrics_collector: ScannerMetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
    ):
        """
        Initialize scanner factory with clean dependencies.

        Args:
            config: System configuration
            db_adapter: Database adapter
            event_bus: Optional event bus for alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        self.config = config
        self.db_adapter = db_adapter
        self.event_bus = event_bus
        self.metrics_collector = metrics_collector
        self.cache_manager = cache_manager

        # Initialize storage components with clean architecture
        self._storage_router: IStorageRouter | None = None
        # Storage executor removed - file doesn't exist
        self._repository_provider: RepositoryProvider | None = None
        self._scanner_repository: IScannerRepository | None = None

        # Cache created scanners
        self._scanner_instances: dict[str, IScanner] = {}

        logger.info("Initialized ScannerFactoryV2 with clean architecture")

    def _initialize_storage_system(self):
        """Initialize the storage system components."""
        if self._storage_router is None:
            # Create clean storage router (no repository dependencies)
            self._storage_router = StorageRouterV2(self.config)

            # Create repository provider
            self._repository_provider = RepositoryProvider(self.db_adapter)

            # Storage executor removed - file doesn't exist
            # Using repository provider directly instead

            # Create scanner repository with clean interfaces
            # Local imports
            from main.data_pipeline.storage.repositories import get_repository_factory
            from main.data_pipeline.storage.repositories.repository_types import (
                RepositoryConfig,
                ValidationLevel,
            )

            # Create repository config directly
            repo_config = RepositoryConfig(
                enable_caching=self.config.get("storage.enable_caching", True),
                cache_ttl_seconds=self.config.get("storage.cache_ttl_seconds", 300),
                enable_metrics=self.config.get("storage.enable_metrics", True),
                validation_level=ValidationLevel.LENIENT,  # Use lenient validation for scanners
            )

            # Use factory to create scanner repository
            factory = get_repository_factory()
            self._scanner_repository = factory.create_scanner_repository(
                db_adapter=self.db_adapter, config=repo_config
            )

            # Register scanner repository with provider for other components to use
            self._repository_provider.register_repository_instance(
                "scanner", self._scanner_repository
            )

            logger.info("Storage system initialized with clean architecture")

    async def create_scanner(self, scanner_type: str, **kwargs) -> IScanner:
        """
        Create a scanner instance.

        Args:
            scanner_type: Type of scanner to create
            **kwargs: Additional scanner-specific arguments

        Returns:
            Scanner instance

        Raises:
            ValueError: If scanner type is unknown
        """
        if scanner_type not in self.SCANNER_REGISTRY:
            available = ", ".join(sorted(self.SCANNER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown scanner type: '{scanner_type}'. " f"Available types: {available}"
            )

        # Check cache first
        if scanner_type in self._scanner_instances:
            logger.debug(f"Returning cached scanner: {scanner_type}")
            return self._scanner_instances[scanner_type]

        # Ensure storage system is initialized
        self._initialize_storage_system()

        # Get scanner class
        scanner_class = self.SCANNER_REGISTRY[scanner_type]

        # Get scanner-specific configuration
        scanner_config = self.config.get(f"scanners.{scanner_type}", {})

        # Create scanner instance with clean dependencies
        try:
            scanner = scanner_class(
                config=scanner_config,
                repository=self._scanner_repository,
                event_bus=self.event_bus,
                metrics_collector=self.metrics_collector,
                cache_manager=self.cache_manager,
                **kwargs,
            )

            # Initialize scanner if needed
            if hasattr(scanner, "initialize"):
                await scanner.initialize()

            # Cache the instance
            self._scanner_instances[scanner_type] = scanner

            logger.info(f"Created scanner: {scanner_type}")
            return scanner

        except Exception as e:
            logger.error(f"Failed to create scanner '{scanner_type}': {e}")
            raise

    async def create_multiple_scanners(
        self, scanner_types: list[str], **kwargs
    ) -> dict[str, IScanner]:
        """
        Create multiple scanner instances.

        Args:
            scanner_types: List of scanner types to create
            **kwargs: Additional arguments for all scanners

        Returns:
            Dictionary mapping scanner type to instance
        """
        scanners = {}

        for scanner_type in scanner_types:
            try:
                scanner = await self.create_scanner(scanner_type, **kwargs)
                scanners[scanner_type] = scanner
            except Exception as e:
                logger.error(f"Failed to create scanner '{scanner_type}': {e}")
                # Continue with other scanners

        return scanners

    def get_available_scanners(self) -> list[str]:
        """Get list of available scanner types."""
        return list(self.SCANNER_REGISTRY.keys())

    def get_scanner_repository(self) -> IScannerRepository:
        """Get the scanner repository instance."""
        self._initialize_storage_system()
        return self._scanner_repository

    def get_storage_router(self) -> IStorageRouter:
        """Get the storage router instance."""
        self._initialize_storage_system()
        return self._storage_router

    # Storage executor method removed - file doesn't exist
    # Use get_repository_provider() instead for storage operations

    async def cleanup(self) -> None:
        """Clean up scanner instances."""
        for scanner_type, scanner in self._scanner_instances.items():
            if hasattr(scanner, "cleanup"):
                try:
                    await scanner.cleanup()
                    logger.info(f"Cleaned up scanner: {scanner_type}")
                except Exception as e:
                    logger.error(f"Error cleaning up scanner '{scanner_type}': {e}")

        self._scanner_instances.clear()


# Global factory instance
_scanner_factory_v2: ScannerFactoryV2 | None = None


def get_scanner_factory_v2(
    config: DictConfig,
    db_adapter: IAsyncDatabase,
    event_bus: IEventBus | None = None,
    metrics_collector: ScannerMetricsCollector | None = None,
    cache_manager: ScannerCacheManager | None = None,
) -> ScannerFactoryV2:
    """
    Get or create scanner factory instance.

    This uses a singleton pattern to ensure consistent scanner instances
    across the application.

    Args:
        config: System configuration
        db_adapter: Database adapter
        event_bus: Optional event bus
        metrics_collector: Optional metrics collector
        cache_manager: Optional cache manager

    Returns:
        Scanner factory instance
    """
    global _scanner_factory_v2

    # Check if we need a new factory
    needs_new = (
        _scanner_factory_v2 is None
        or _scanner_factory_v2.db_adapter != db_adapter
        or _scanner_factory_v2.event_bus != event_bus
    )

    if needs_new:
        _scanner_factory_v2 = ScannerFactoryV2(
            config=config,
            db_adapter=db_adapter,
            event_bus=event_bus,
            metrics_collector=metrics_collector,
            cache_manager=cache_manager,
        )

    return _scanner_factory_v2
