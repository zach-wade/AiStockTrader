"""
Standardized Application Context Management

This module provides a standardized way to initialize and manage application contexts
across all AI Trader applications, replacing the duplicate AppContext patterns
found in run_backfill.py, run_etl.py, and other app files.
"""

# Standard library imports
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.database import DatabasePool
from main.utils.monitoring import get_global_monitor, record_metric
from main.utils.resilience import get_global_recovery_manager

logger = get_logger(__name__)


class AppContextError(Exception):
    """Exception raised when app context operations fail."""

    pass


class StandardAppContext(ErrorHandlingMixin):
    """
    Standardized application context that replaces the duplicate AppContext classes.

    This class provides a consistent way to initialize and manage:
    - Configuration loading and validation
    - Database connections and pooling
    - Data source clients
    - Monitoring and metrics
    - Error handling and recovery
    - Resource cleanup
    """

    def __init__(self, app_name: str, config: Any | None = None):
        """
        Initialize StandardAppContext.

        Args:
            app_name: Name of the application for logging and monitoring
            config: Optional configuration override
        """
        super().__init__()
        self.app_name = app_name
        self.config = config  # Config should be provided by caller to avoid circular imports
        if not config:
            logger.warning(f"No config provided to {app_name} context - some features may not work")
        self.logger = get_logger(f"{__name__}.{app_name}")

        # Core components
        self.db_pool: DatabasePool | None = None
        self.data_source_manager: Any | None = None
        self.ingestion_orchestrator: Any | None = None
        self.processing_manager: Any | None = None

        # Dual storage components
        self.event_bus: Any | None = None
        self.cold_storage: Any | None = None
        self.dual_storage_consumer_started: bool = False

        # Monitoring and metrics
        self.monitor = get_global_monitor()
        self.recovery_manager = get_global_recovery_manager()

        # State tracking
        self.initialized = False
        self.startup_time = datetime.now()
        self.components_initialized: list[str] = []

        # Register error callback for monitoring
        self.register_error_callback(
            "monitoring",
            lambda error, context: record_metric(
                "app_error", 1, tags={"app": self.app_name, "context": context}
            ),
        )

    async def initialize(self, components: list[str] | None = None) -> "StandardAppContext":
        """
        Initialize the application context with specified components.

        Args:
            components: List of components to initialize. If None, initializes all.
                       Options: ['database', 'data_sources', 'ingestion', 'processing']

        Returns:
            Self for method chaining
        """
        if self.initialized:
            self.logger.warning("Context already initialized, skipping...")
            return self

        start_time = datetime.now()
        components = components or ["database", "dual_storage", "data_sources", "ingestion"]

        try:
            self.logger.info(f"Initializing {self.app_name} context with components: {components}")

            # Initialize database pool
            if "database" in components:
                await self._initialize_database()
                self.components_initialized.append("database")

            # Initialize dual storage (depends on database)
            if "dual_storage" in components and "database" in components:
                await self._initialize_dual_storage()
                self.components_initialized.append("dual_storage")

            # Initialize data source manager
            if "data_sources" in components:
                await self._initialize_data_sources()
                self.components_initialized.append("data_sources")

            # Initialize ingestion orchestrator
            if "ingestion" in components:
                await self._initialize_ingestion()
                self.components_initialized.append("ingestion")

            # Initialize processing manager
            if "processing" in components:
                await self._initialize_processing()
                self.components_initialized.append("processing")

            self.initialized = True
            duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"✅ {self.app_name} context initialized successfully in {duration:.2f}s"
            )

            # Record initialization metrics
            record_metric(
                "app_initialization_duration",
                duration,
                tags={"app": self.app_name, "components": len(components)},
            )

            return self

        except Exception as e:
            self.handle_error(e, f"initializing {self.app_name} context")
            raise AppContextError(f"Failed to initialize {self.app_name} context: {e}")

    async def _initialize_database(self):
        """Initialize database pool with monitoring."""
        try:
            self.logger.info("Initializing database pool...")
            self.db_pool = DatabasePool()
            self.db_pool.initialize(config=self.config)

            # Test connection
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            self.logger.info("✅ Database pool initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def _initialize_data_sources(self):
        """Initialize data source manager with all configured clients."""
        try:
            self.logger.info("Initializing data source manager...")

            # Import here to avoid circular imports
            # Local imports
            from main.data_pipeline.ingestion.data_source_manager import DataSourceManager
            from main.data_pipeline.storage.archive_initializer import initialize_data_archive

            # Initialize DataArchive first to prevent the warning
            self.logger.info("Initializing DataArchive...")
            await initialize_data_archive(self.config)
            self.logger.info("✅ DataArchive initialized successfully")

            self.data_source_manager = DataSourceManager(self.config)

            # Initialize data source clients with resilience
            await self._initialize_data_source_clients()

            self.logger.info("✅ Data source manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize data source manager: {e}")
            raise

    async def _initialize_data_source_clients(self):
        """Initialize and register data source clients with proper error handling."""
        try:
            # Get enabled data sources from config
            enabled_sources = self.config.get("data.sources", ["alpaca", "polygon", "yahoo"])
            self.logger.info(f"Initializing data sources: {enabled_sources}")

            clients = {}

            # Initialize Alpaca clients
            if "alpaca" in enabled_sources:
                clients.update(await self._create_alpaca_clients())

            # Initialize Polygon clients
            if "polygon" in enabled_sources:
                clients.update(await self._create_polygon_clients())

            # Initialize Yahoo clients
            if "yahoo" in enabled_sources:
                clients.update(await self._create_yahoo_clients())

            # Initialize other clients
            if "benzinga" in enabled_sources:
                clients.update(await self._create_benzinga_clients())

            if "reddit" in enabled_sources:
                clients.update(await self._create_reddit_clients())

            # Register all clients
            if clients:
                self.data_source_manager.register_clients(clients)
                self.logger.info(f"Successfully registered {len(clients)} data source clients")
            else:
                self.logger.warning("No data source clients initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize data source clients: {e}")
            raise

    async def _create_alpaca_clients(self) -> dict[str, Any]:
        """Create Alpaca data source clients."""
        clients = {}

        try:
            # Per architecture: Alpaca only for assets discovery, not market data
            # from main.data_pipeline.ingestion.alpaca_market_client import AlpacaMarketClient  # REMOVED
            # from main.data_pipeline.ingestion.alpaca_options_client import AlpacaOptionsClient  # REMOVED
            # Local imports
            from main.data_pipeline.ingestion.alpaca_assets_client import AlpacaAssetsClient

            # Only initialize assets client per architecture
            clients["alpaca_assets"] = AlpacaAssetsClient(self.config)

            self.logger.info("✅ Alpaca clients initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca clients: {e}")

        return clients

    async def _create_polygon_clients(self) -> dict[str, Any]:
        """Create Polygon data source clients."""
        clients = {}

        try:
            # Local imports
            from main.data_pipeline.ingestion.polygon_corporate_actions_client import (
                PolygonCorporateActionsClient,
            )
            from main.data_pipeline.ingestion.polygon_forex_client import PolygonForexClient
            from main.data_pipeline.ingestion.polygon_market_client import PolygonMarketClient
            from main.data_pipeline.ingestion.polygon_news_client import PolygonNewsClient
            from main.data_pipeline.ingestion.polygon_options_client import PolygonOptionsClient

            clients["polygon_market"] = PolygonMarketClient(self.config)
            clients["polygon_news"] = PolygonNewsClient(self.config)
            clients["polygon_options"] = PolygonOptionsClient(self.config)
            clients["polygon_corporate_actions"] = PolygonCorporateActionsClient(self.config)
            clients["polygon_forex"] = PolygonForexClient(self.config)

            self.logger.info("✅ Polygon clients initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Polygon clients: {e}")

        return clients

    async def _create_yahoo_clients(self) -> dict[str, Any]:
        """Create Yahoo data source clients."""
        clients = {}

        try:
            # Local imports
            from main.data_pipeline.ingestion.yahoo_corporate_actions_client import (
                YahooCorporateActionsClient,
            )
            from main.data_pipeline.ingestion.yahoo_financials_client import YahooFinancialsClient
            from main.data_pipeline.ingestion.yahoo_market_client import YahooMarketClient
            from main.data_pipeline.ingestion.yahoo_news_client import YahooNewsClient

            clients["yahoo_market"] = YahooMarketClient(self.config)
            clients["yahoo_financials"] = YahooFinancialsClient(self.config)
            clients["yahoo_news"] = YahooNewsClient(self.config)
            clients["yahoo_corporate_actions"] = YahooCorporateActionsClient(self.config)

            self.logger.info("✅ Yahoo clients initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Yahoo clients: {e}")

        return clients

    async def _create_benzinga_clients(self) -> dict[str, Any]:
        """Create Benzinga data source clients."""
        clients = {}

        try:
            # Local imports
            from main.data_pipeline.ingestion.benzinga_news_client import BenzingaNewsClient

            clients["benzinga_news"] = BenzingaNewsClient(self.config)
            self.logger.info("✅ Benzinga clients initialized")

        except ImportError:
            self.logger.warning("BenzingaNewsClient not implemented, skipping")
        except Exception as e:
            self.logger.error(f"Failed to initialize Benzinga clients: {e}")

        return clients

    async def _create_reddit_clients(self) -> dict[str, Any]:
        """Create Reddit data source clients."""
        clients = {}

        try:
            # Local imports
            from main.data_pipeline.ingestion.reddit_client import RedditClient

            clients["reddit"] = RedditClient(self.config)
            self.logger.info("✅ Reddit clients initialized")

        except Exception as e:
            self.logger.warning(f"Reddit client initialization failed: {e}")

        return clients

    async def _initialize_ingestion(self):
        """Initialize ingestion orchestrator."""
        try:
            self.logger.info("Initializing ingestion orchestrator...")

            # Local imports
            from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator

            if not self.data_source_manager:
                raise AppContextError(
                    "Data source manager must be initialized before ingestion orchestrator"
                )

            self.ingestion_orchestrator = IngestionOrchestrator(self.data_source_manager.clients)

            self.logger.info("✅ Ingestion orchestrator initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ingestion orchestrator: {e}")
            raise

    async def _initialize_dual_storage(self):
        """Initialize dual storage components."""
        try:
            self.logger.info("Initializing dual storage...")

            # Local imports
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.data_pipeline.storage.dual_storage_startup import (
                initialize_dual_storage,
                start_dual_storage_consumer,
            )

            if not self.db_pool:
                raise AppContextError("Database pool must be initialized before dual storage")

            # Create database adapter
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(self.config)

            # Check if dual storage is enabled in config
            enable_dual_storage = (
                self.config.get("data_pipeline", {})
                .get("storage", {})
                .get("enable_dual_storage", True)
            )

            # Initialize dual storage components
            self.event_bus, self.cold_storage = initialize_dual_storage(
                hot_storage=db_adapter, enable_dual_storage=enable_dual_storage
            )

            # Start the cold storage consumer if enabled
            if enable_dual_storage and self.cold_storage:
                await start_dual_storage_consumer()
                self.dual_storage_consumer_started = True
                self.logger.info("✅ Cold storage consumer started")
            else:
                self.dual_storage_consumer_started = False
                self.logger.info("✅ Dual storage initialized (consumer not started)")

            self.logger.info("✅ Dual storage initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize dual storage: {e}")
            # Don't raise - dual storage is optional
            self.event_bus = None
            self.cold_storage = None
            self.dual_storage_consumer_started = False
            self.logger.warning("Continuing without dual storage support")

    async def _initialize_processing(self):
        """Initialize processing manager."""
        try:
            self.logger.info("Initializing processing manager...")

            # Local imports
            from main.data_pipeline.processing.manager import ProcessingManager
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.data_pipeline.storage.repositories import get_repository_factory
            from main.feature_pipeline.calculator_factory import get_calculator_factory

            if not self.db_pool:
                raise AppContextError("Database pool must be initialized before processing manager")

            # Create database adapter using factory
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(self.config)

            # Get factories for dependency injection
            calculator_factory = get_calculator_factory()

            # Pass event bus and cold storage to repository factory if available
            repository_factory = get_repository_factory(
                db_adapter, cold_storage=self.cold_storage, event_bus=self.event_bus
            )

            # Create processing manager with both factories
            self.processing_manager = ProcessingManager(
                self.config, calculator_factory, repository_factory
            )

            self.logger.info("✅ Processing manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize processing manager: {e}")
            raise

    async def safe_shutdown(self):
        """Safely shutdown all components with proper error handling."""
        try:
            shutdown_start = datetime.now()
            self.logger.info(f"Starting safe shutdown of {self.app_name} context...")

            # Close data source sessions
            if hasattr(self.data_source_manager, "close_all_sessions"):
                await self.data_source_manager.close_all_sessions()
                self.logger.info("✅ Data source sessions closed")

            # Shutdown dual storage consumer
            if self.dual_storage_consumer_started:
                try:
                    # Local imports
                    from main.data_pipeline.storage.dual_storage_startup import stop_dual_storage

                    await stop_dual_storage()
                    self.logger.info("✅ Dual storage consumer stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping dual storage: {e}")

            # Shutdown DataArchive
            if "data_sources" in self.components_initialized:
                # Local imports
                from main.data_pipeline.storage.archive_initializer import shutdown_data_archive

                await shutdown_data_archive()
                self.logger.info("✅ DataArchive shut down")

            # Close database pool
            if self.db_pool:
                self.db_pool.dispose()
                self.logger.info("✅ Database pool closed")

            # Close processing manager
            if self.processing_manager and hasattr(self.processing_manager, "close"):
                await self.processing_manager.close()
                self.logger.info("✅ Processing manager closed")

            shutdown_duration = (datetime.now() - shutdown_start).total_seconds()
            uptime = (datetime.now() - self.startup_time).total_seconds()

            self.logger.info(
                f"✅ {self.app_name} context shutdown completed in {shutdown_duration:.2f}s "
                f"(uptime: {uptime:.2f}s)"
            )

            # Record shutdown metrics
            record_metric("app_shutdown_duration", shutdown_duration, tags={"app": self.app_name})

        except Exception as e:
            self.handle_error(e, f"shutting down {self.app_name} context")
            self.logger.error(f"Error during shutdown: {e}")

    def get_component_status(self) -> dict[str, Any]:
        """Get the status of all initialized components."""
        return {
            "app_name": self.app_name,
            "initialized": self.initialized,
            "startup_time": self.startup_time.isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "components_initialized": self.components_initialized,
            "database_pool_active": self.db_pool is not None,
            "data_source_manager_active": self.data_source_manager is not None,
            "ingestion_orchestrator_active": self.ingestion_orchestrator is not None,
            "processing_manager_active": self.processing_manager is not None,
            "dual_storage_active": self.dual_storage_consumer_started,
            "event_bus_active": self.event_bus is not None,
            "error_count": self.get_error_count(),
        }

    async def get_dual_storage_health(self) -> dict[str, Any]:
        """Get detailed health status of dual storage components."""
        health_status = {
            "dual_storage_enabled": self.dual_storage_consumer_started,
            "event_bus_initialized": self.event_bus is not None,
            "cold_storage_initialized": self.cold_storage is not None,
            "consumer_running": False,
            "consumer_metrics": {},
            "cold_storage_metrics": {},
            "overall_health": "unknown",
        }

        try:
            # Check cold storage consumer health
            if self.dual_storage_consumer_started:
                # Local imports
                from main.data_pipeline.storage.dual_storage_startup import get_dual_storage_manager

                manager = get_dual_storage_manager()

                if manager and manager.cold_consumer:
                    consumer_health = await manager.cold_consumer.health_check()
                    health_status["consumer_running"] = consumer_health.get("healthy", False)
                    health_status["consumer_metrics"] = consumer_health.get("metrics", {})

                # Get overall metrics
                metrics = manager.get_metrics() if manager else {}
                health_status["cold_storage_metrics"] = metrics.get("cold_consumer", {})

            # Determine overall health
            if health_status["dual_storage_enabled"]:
                if health_status["consumer_running"] and health_status["event_bus_initialized"]:
                    health_status["overall_health"] = "healthy"
                elif health_status["event_bus_initialized"]:
                    health_status["overall_health"] = "degraded"
                else:
                    health_status["overall_health"] = "unhealthy"
            else:
                health_status["overall_health"] = "disabled"

        except Exception as e:
            self.logger.error(f"Error checking dual storage health: {e}")
            health_status["overall_health"] = "error"
            health_status["error"] = str(e)

        return health_status


async def create_app_context(
    app_name: str, components: list[str] | None = None
) -> StandardAppContext:
    """
    Create and initialize a StandardAppContext.

    Args:
        app_name: Name of the application
        components: Optional list of components to initialize

    Returns:
        Initialized StandardAppContext
    """
    context = StandardAppContext(app_name)
    await context.initialize(components)
    return context


@asynccontextmanager
async def managed_app_context(app_name: str, components: list[str] | None = None):
    """
    Context manager for automatic app context management.

    Args:
        app_name: Name of the application
        components: Optional list of components to initialize

    Yields:
        StandardAppContext instance
    """
    context = None
    try:
        context = await create_app_context(app_name, components)
        yield context
    finally:
        if context:
            await context.safe_shutdown()
