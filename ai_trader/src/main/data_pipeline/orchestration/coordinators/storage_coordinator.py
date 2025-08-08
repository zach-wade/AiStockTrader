"""
Storage Coordinator

Coordinates data storage using TableRoutingService and PartitionManager.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from main.utils.core import get_logger
from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.services.storage import (
    TableRoutingService,
    TableRoutingConfig,
    PartitionManager,
    PartitionConfig
)
from main.data_pipeline.ingestion.factories.bulk_loader_factory import BulkLoaderFactory


@dataclass
class StorageRequest:
    """Request for data storage."""
    data_type: str
    symbol: str
    data: List[Dict[str, Any]]
    interval: Optional[str] = None
    flush_immediately: bool = False


@dataclass 
class StorageResult:
    """Result of storage operation."""
    records_stored: int
    table_name: str
    partitions_created: int
    success: bool
    error: Optional[str] = None


class StorageCoordinator:
    """
    Coordinates data storage operations.
    
    Uses TableRoutingService to determine target tables,
    PartitionManager to ensure partitions exist, and
    BulkLoaderFactory to efficiently load data.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        routing_service: Optional[TableRoutingService] = None,
        partition_manager: Optional[PartitionManager] = None,
        bulk_loader_factory: Optional[BulkLoaderFactory] = None
    ):
        """
        Initialize the storage coordinator.
        
        Args:
            db_adapter: Database adapter
            routing_service: Optional table routing service
            partition_manager: Optional partition manager
            bulk_loader_factory: Optional bulk loader factory
        """
        self.db_adapter = db_adapter
        
        self.routing_service = routing_service or TableRoutingService(
            TableRoutingConfig()
        )
        
        self.partition_manager = partition_manager or PartitionManager(
            db_adapter=db_adapter,
            config=PartitionConfig()
        )
        
        self.bulk_loader_factory = bulk_loader_factory or BulkLoaderFactory(
            db_adapter=db_adapter
        )
        
        self.bulk_loaders = {}
        self.logger = get_logger(__name__)
    
    async def store_data(self, request: StorageRequest) -> StorageResult:
        """
        Store data according to the request.
        
        Args:
            request: Storage request
            
        Returns:
            Storage result
        """
        try:
            # Determine target table
            if request.data_type == 'market_data' and request.interval:
                table_name = self.routing_service.get_table_for_interval(request.interval)
            else:
                # Map data type to table
                table_map = {
                    'news': 'news_data',
                    'fundamentals': 'financials_data',
                    'corporate_actions': 'corporate_actions'
                }
                table_name = table_map.get(request.data_type, 'raw_data')
            
            self.logger.debug(f"Routing {request.data_type} to table: {table_name}")
            
            # Ensure partitions exist for market data
            partitions_created = 0
            if 'market_data' in table_name and request.data:
                # Extract date range from data
                timestamps = [d.get('timestamp') for d in request.data if d.get('timestamp')]
                if timestamps:
                    min_date = min(timestamps)
                    max_date = max(timestamps)
                    
                    partitions_created = await self.partition_manager.ensure_partitions_exist(
                        table_name=table_name,
                        start_date=min_date,
                        end_date=max_date
                    )
            
            # Get or create bulk loader
            loader = await self._get_bulk_loader(request.data_type, table_name)
            
            # Add data to loader
            records_added = 0
            for record in request.data:
                # Add symbol if not present
                if 'symbol' not in record:
                    record['symbol'] = request.symbol
                
                await loader.add_record(record)
                records_added += 1
            
            # Flush if requested
            if request.flush_immediately:
                await loader.flush()
            
            return StorageResult(
                records_stored=records_added,
                table_name=table_name,
                partitions_created=partitions_created,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Storage error: {e}")
            return StorageResult(
                records_stored=0,
                table_name="",
                partitions_created=0,
                success=False,
                error=str(e)
            )
    
    async def _get_bulk_loader(self, data_type: str, table_name: str):
        """
        Get or create a bulk loader for the data type.
        
        Args:
            data_type: Type of data
            table_name: Target table name
            
        Returns:
            Bulk loader instance
        """
        loader_key = f"{data_type}_{table_name}"
        
        if loader_key not in self.bulk_loaders:
            self.bulk_loaders[loader_key] = self.bulk_loader_factory.create_loader(
                data_type=data_type,
                table_name=table_name
            )
        
        return self.bulk_loaders[loader_key]
    
    async def flush_all(self):
        """Flush all bulk loaders."""
        self.logger.info("Flushing all bulk loaders")
        
        for loader_key, loader in self.bulk_loaders.items():
            try:
                await loader.flush()
                self.logger.debug(f"Flushed loader: {loader_key}")
            except Exception as e:
                self.logger.error(f"Error flushing {loader_key}: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'active_loaders': len(self.bulk_loaders),
            'loader_stats': {}
        }
        
        for loader_key, loader in self.bulk_loaders.items():
            if hasattr(loader, 'get_stats'):
                stats['loader_stats'][loader_key] = loader.get_stats()
        
        return stats