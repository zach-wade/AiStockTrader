"""
Simple Service Container

Lightweight container for registering and resolving existing services.
"""

from typing import Dict, Any, Type, Optional, TypeVar

from main.utils.core import get_logger
from main.interfaces.database import IAsyncDatabase
from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.archive import DataArchive

# Import existing services
from main.data_pipeline.services.storage import (
    QualificationService,
    TableRoutingService,
    PartitionManager
)
from main.data_pipeline.services.ingestion import (
    DeduplicationService,
    MetricExtractionService,
    TextProcessingService
)
from main.data_pipeline.services.processing import CorporateActionsService

T = TypeVar('T')


class SimpleServiceContainer:
    """
    Simple container for managing service instances.
    
    Registers existing services and provides easy access.
    """
    
    def __init__(self):
        """Initialize the container."""
        self._services: Dict[Type, Any] = {}
        self.logger = get_logger(__name__)
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self):
        """Register core infrastructure services."""
        # Config manager
        self._services[type(get_config_manager())] = get_config_manager()
        
        # Database factory
        self._services[DatabaseFactory] = DatabaseFactory()
        
        self.logger.info("Core services registered")
    
    def register(self, service_type: Type[T], instance: T) -> None:
        """
        Register a service instance.
        
        Args:
            service_type: The type of the service
            instance: The service instance
        """
        self._services[service_type] = instance
        self.logger.debug(f"Registered {service_type.__name__}")
    
    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Resolve a service.
        
        Args:
            service_type: The type to resolve
            
        Returns:
            The service instance or None
        """
        return self._services.get(service_type)
    
    async def setup_data_pipeline_services(self, db_adapter: IAsyncDatabase):
        """
        Setup all data pipeline services.
        
        Args:
            db_adapter: Database adapter to use
        """
        # Archive service
        archive_config = {'storage_type': 'local', 'local_path': 'data_lake'}
        self.register(DataArchive, DataArchive(archive_config))
        
        # Qualification service
        self.register(
            QualificationService,
            QualificationService(db_adapter=db_adapter)
        )
        
        # Deduplication service
        self.register(
            DeduplicationService,
            DeduplicationService(db_adapter=db_adapter)
        )
        
        # Table routing service
        self.register(
            TableRoutingService,
            TableRoutingService()
        )
        
        # Partition manager
        self.register(
            PartitionManager,
            PartitionManager(db_adapter=db_adapter)
        )
        
        # Metric extraction service
        self.register(
            MetricExtractionService,
            MetricExtractionService()
        )
        
        # Text processing service
        self.register(
            TextProcessingService,
            TextProcessingService()
        )
        
        # Corporate actions service
        self.register(
            CorporateActionsService,
            CorporateActionsService()
        )
        
        self.logger.info("Data pipeline services registered")
    
    def setup_ingestion_clients(self, api_key: str, layer: DataLayer = DataLayer.BASIC):
        """
        Setup Polygon ingestion clients.
        
        Args:
            api_key: Polygon API key
            layer: Data layer for configuration
        """
        from main.data_pipeline.ingestion.clients import (
            PolygonMarketClient,
            PolygonNewsClient,
            PolygonFundamentalsClient,
            PolygonCorporateActionsClient
        )
        
        # Get existing services for injection
        text_processor = self.resolve(TextProcessingService)
        metric_extractor = self.resolve(MetricExtractionService)
        
        # Create clients with service dependencies
        self.register(
            PolygonMarketClient,
            PolygonMarketClient(api_key, layer)
        )
        
        self.register(
            PolygonNewsClient,
            PolygonNewsClient(api_key, layer, text_processor=text_processor)
        )
        
        self.register(
            PolygonFundamentalsClient,
            PolygonFundamentalsClient(api_key, layer, metric_extractor=metric_extractor)
        )
        
        self.register(
            PolygonCorporateActionsClient,
            PolygonCorporateActionsClient(api_key, layer)
        )
        
        self.logger.info("Ingestion clients registered")
    
    def setup_coordinators(self):
        """Setup orchestration coordinators."""
        from main.data_pipeline.orchestration.coordinators import (
            LayerCoordinator,
            DataFetchCoordinator,
            StorageCoordinator
        )
        
        # Get dependencies
        db_adapter = self.resolve(IAsyncDatabase)
        archive = self.resolve(DataArchive)
        qualification_service = self.resolve(QualificationService)
        routing_service = self.resolve(TableRoutingService)
        partition_manager = self.resolve(PartitionManager)
        
        # Layer coordinator
        self.register(
            LayerCoordinator,
            LayerCoordinator(
                db_adapter=db_adapter,
                qualification_service=qualification_service
            )
        )
        
        # Data fetch coordinator
        self.register(
            DataFetchCoordinator,
            DataFetchCoordinator(
                archive=archive,
                market_client=self.resolve(PolygonMarketClient),
                news_client=self.resolve(PolygonNewsClient),
                fundamentals_client=self.resolve(PolygonFundamentalsClient),
                corporate_actions_client=self.resolve(PolygonCorporateActionsClient)
            )
        )
        
        # Storage coordinator
        self.register(
            StorageCoordinator,
            StorageCoordinator(
                db_adapter=db_adapter,
                routing_service=routing_service,
                partition_manager=partition_manager
            )
        )
        
        self.logger.info("Coordinators registered")
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of service names to instances
        """
        return {
            service_type.__name__: instance
            for service_type, instance in self._services.items()
        }