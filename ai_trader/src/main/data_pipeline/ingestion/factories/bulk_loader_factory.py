"""
Bulk Loader Factory

Factory for creating bulk loaders with layer-based configuration.
Reads layer configurations from storage.yaml and creates loaders
with appropriate buffer sizes and settings.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig
from main.data_pipeline.ingestion.loaders import (
    MarketDataSplitBulkLoader,
    NewsBulkLoader,
    FundamentalsBulkLoader,
    CorporateActionsBulkLoader
)
from main.data_pipeline.services.storage import (
    QualificationService,
    TableRoutingService,
    PartitionManager
)
from main.data_pipeline.storage.archive import DataArchive
from main.utils.core import get_logger
from main.config import get_config_manager

logger = get_logger(__name__)


@dataclass
class LayerConfig:
    """Configuration for a specific layer."""
    market_data_buffer: int = 10000
    news_buffer: int = 500
    fundamentals_buffer: int = 2500
    corporate_actions_buffer: int = 5000
    priority: int = 1
    use_copy_command: bool = True
    parallel_loaders: int = 3
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        """Create LayerConfig from dictionary."""
        return cls(
            market_data_buffer=data.get('market_data_buffer', 10000),
            news_buffer=data.get('news_buffer', 500),
            fundamentals_buffer=data.get('fundamentals_buffer', 2500),
            corporate_actions_buffer=data.get('corporate_actions_buffer', 5000),
            priority=data.get('priority', 1),
            use_copy_command=data.get('use_copy_command', True),
            parallel_loaders=data.get('parallel_loaders', 3)
        )


class BulkLoaderFactory:
    """
    Factory for creating bulk loaders with layer-based configuration.
    
    This factory reads layer configurations from the config system
    and creates bulk loaders with appropriate settings based on the
    layer of the symbols being processed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the bulk loader factory.
        
        Args:
            config: Optional configuration override
        """
        self.logger = get_logger(__name__)
        
        # Load configuration
        if config:
            self.config = config
        else:
            config_manager = get_config_manager()
            self.config = config_manager.load_config('storage')
        
        # Parse layer configurations
        self.layer_configs = self._parse_layer_configs()
        
        self.logger.info(
            f"BulkLoaderFactory initialized with {len(self.layer_configs)} layer configurations"
        )
    
    def _parse_layer_configs(self) -> Dict[int, LayerConfig]:
        """Parse layer configurations from storage config."""
        layer_configs = {}
        
        # Get layer configs from storage optimization section
        optimization = self.config.get('storage', {}).get('optimization', {})
        bulk_ops = optimization.get('bulk_operations', {})
        layer_config_data = bulk_ops.get('layer_configs', {})
        
        # Parse each layer configuration
        for layer_name, layer_data in layer_config_data.items():
            # Extract layer number from name (e.g., 'layer1' -> 1)
            try:
                layer_num = int(layer_name.replace('layer', ''))
                layer_configs[layer_num] = LayerConfig.from_dict(layer_data)
                self.logger.debug(f"Loaded config for layer {layer_num}: {layer_configs[layer_num]}")
            except ValueError:
                self.logger.warning(f"Invalid layer name: {layer_name}")
        
        # Add default configuration if no layers defined
        if not layer_configs:
            self.logger.warning("No layer configurations found, using defaults")
            layer_configs[1] = LayerConfig()
            layer_configs[2] = LayerConfig(
                market_data_buffer=5000,
                news_buffer=250,
                fundamentals_buffer=1000,
                corporate_actions_buffer=2500,
                priority=2,
                parallel_loaders=2
            )
            layer_configs[3] = LayerConfig(
                market_data_buffer=2500,
                news_buffer=100,
                fundamentals_buffer=500,
                corporate_actions_buffer=1000,
                priority=3,
                parallel_loaders=1
            )
        
        return layer_configs
    
    def create_market_data_loader(
        self,
        db_adapter: IAsyncDatabase,
        layer: int = 1,
        archive: Optional[DataArchive] = None,
        qualification_service: Optional[QualificationService] = None,
        routing_service: Optional[TableRoutingService] = None,
        partition_manager: Optional[PartitionManager] = None,
        **kwargs
    ) -> MarketDataSplitBulkLoader:
        """
        Create a market data bulk loader with layer-based configuration.
        
        Args:
            db_adapter: Database adapter
            layer: Layer number (1, 2, or 3)
            archive: Optional archive for cold storage
            qualification_service: Optional qualification service (will create if not provided)
            routing_service: Optional routing service (will create if not provided)
            partition_manager: Optional partition manager (will create if not provided)
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured market data bulk loader
        """
        layer_config = self.layer_configs.get(layer, self.layer_configs.get(1))
        
        config = BulkLoadConfig(
            buffer_size=layer_config.market_data_buffer,
            use_copy_command=layer_config.use_copy_command,
            parallel_workers=layer_config.parallel_loaders,
            **kwargs
        )
        
        # Create services if not provided
        if not qualification_service:
            qualification_service = QualificationService(db_adapter)
        if not routing_service:
            routing_service = TableRoutingService()
        if not partition_manager:
            partition_manager = PartitionManager(db_adapter)
        
        loader = MarketDataSplitBulkLoader(
            db_adapter=db_adapter,
            qualification_service=qualification_service,
            routing_service=routing_service,
            partition_manager=partition_manager,
            archive=archive,
            config=config
        )
        
        self.logger.debug(f"Created MarketDataSplitBulkLoader for layer {layer} with buffer {config.buffer_size}")
        return loader
    
    def create_news_loader(
        self,
        db_adapter: IAsyncDatabase,
        layer: int = 1,
        archive: Optional[DataArchive] = None,
        **kwargs
    ) -> NewsBulkLoader:
        """
        Create a news bulk loader with layer-based configuration.
        
        Args:
            db_adapter: Database adapter
            layer: Layer number (1, 2, or 3)
            archive: Optional archive for cold storage
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured news bulk loader
        """
        layer_config = self.layer_configs.get(layer, self.layer_configs.get(1))
        
        config = BulkLoadConfig(
            buffer_size=layer_config.news_buffer,
            use_copy_command=layer_config.use_copy_command,
            parallel_workers=layer_config.parallel_loaders,
            **kwargs
        )
        
        loader = NewsBulkLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=config
        )
        
        self.logger.debug(f"Created NewsBulkLoader for layer {layer} with buffer {config.buffer_size}")
        return loader
    
    def create_fundamentals_loader(
        self,
        db_adapter: IAsyncDatabase,
        layer: int = 1,
        archive: Optional[DataArchive] = None,
        **kwargs
    ) -> FundamentalsBulkLoader:
        """
        Create a fundamentals bulk loader with layer-based configuration.
        
        Args:
            db_adapter: Database adapter
            layer: Layer number (1, 2, or 3)
            archive: Optional archive for cold storage
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured fundamentals bulk loader
        """
        layer_config = self.layer_configs.get(layer, self.layer_configs.get(1))
        
        config = BulkLoadConfig(
            buffer_size=layer_config.fundamentals_buffer,
            use_copy_command=layer_config.use_copy_command,
            parallel_workers=layer_config.parallel_loaders,
            **kwargs
        )
        
        loader = FundamentalsBulkLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=config
        )
        
        self.logger.debug(f"Created FundamentalsBulkLoader for layer {layer} with buffer {config.buffer_size}")
        return loader
    
    def create_corporate_actions_loader(
        self,
        db_adapter: IAsyncDatabase,
        layer: int = 1,
        archive: Optional[DataArchive] = None,
        **kwargs
    ) -> CorporateActionsBulkLoader:
        """
        Create a corporate actions bulk loader with layer-based configuration.
        
        Args:
            db_adapter: Database adapter
            layer: Layer number (1, 2, or 3)
            archive: Optional archive for cold storage
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured corporate actions bulk loader
        """
        layer_config = self.layer_configs.get(layer, self.layer_configs.get(1))
        
        config = BulkLoadConfig(
            buffer_size=layer_config.corporate_actions_buffer,
            use_copy_command=layer_config.use_copy_command,
            parallel_workers=layer_config.parallel_loaders,
            **kwargs
        )
        
        loader = CorporateActionsBulkLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=config
        )
        
        self.logger.debug(f"Created CorporateActionsBulkLoader for layer {layer} with buffer {config.buffer_size}")
        return loader
    
    def create_loader(
        self,
        data_type: str,
        db_adapter: IAsyncDatabase,
        layer: int = 1,
        archive: Optional[DataArchive] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a bulk loader based on data type.
        
        Args:
            data_type: Type of data ('market_data', 'news', 'fundamentals', 'corporate_actions')
            db_adapter: Database adapter
            layer: Layer number (1, 2, or 3)
            archive: Optional archive for cold storage
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured bulk loader or None if type not recognized
        """
        loader_map = {
            'market_data': self.create_market_data_loader,
            'news': self.create_news_loader,
            'fundamentals': self.create_fundamentals_loader,
            'corporate_actions': self.create_corporate_actions_loader
        }
        
        creator = loader_map.get(data_type)
        if not creator:
            self.logger.error(f"Unknown data type: {data_type}")
            return None
        
        return creator(db_adapter, layer, archive, **kwargs)
    
    def get_layer_priority(self, layer: int) -> int:
        """
        Get the priority for a given layer.
        
        Args:
            layer: Layer number
            
        Returns:
            Priority value (lower is higher priority)
        """
        layer_config = self.layer_configs.get(layer)
        return layer_config.priority if layer_config else 999


# Singleton instance
_factory_instance: Optional[BulkLoaderFactory] = None


def get_bulk_loader_factory(config: Optional[Dict[str, Any]] = None) -> BulkLoaderFactory:
    """
    Get or create the singleton bulk loader factory instance.
    
    Args:
        config: Optional configuration override
        
    Returns:
        BulkLoaderFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = BulkLoaderFactory(config)
    return _factory_instance