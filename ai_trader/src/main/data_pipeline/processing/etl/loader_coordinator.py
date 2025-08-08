"""
Loader Coordinator

Manages bulk loader instances and coordinates loading operations.
"""

from typing import Dict, Optional, Any
from datetime import datetime

from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import IBulkLoader, BulkLoadConfig
from main.data_pipeline.core.enums import DataType
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.bulk_loaders import (
    MarketDataSplitBulkLoader,
    NewsBulkLoader,
    FundamentalsBulkLoader,
    CorporateActionsBulkLoader
)
from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import timer, record_metric, MetricType


class LoaderCoordinator(ErrorHandlingMixin):
    """
    Coordinates bulk loader instances for different data types.
    
    Manages loader lifecycle, configuration, and provides
    a unified interface for loading operations.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: DataArchive,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize loader coordinator.
        
        Args:
            db_adapter: Database adapter
            archive: Data archive instance
            config: Optional configuration
        """
        self.db_adapter = db_adapter
        self.archive = archive
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Create bulk load config
        self.bulk_config = self._create_bulk_config()
        
        # Loader instances cache
        self._loaders: Dict[DataType, IBulkLoader] = {}
        
        # Loader statistics
        self._loader_stats = {
            'loaders_created': 0,
            'loads_performed': 0,
            'total_records': 0
        }
    
    def _create_bulk_config(self) -> BulkLoadConfig:
        """Create bulk load configuration from config."""
        return BulkLoadConfig(
            batch_size=self.config.get('batch_size', 1000),
            max_buffer_size=self.config.get('max_buffer_size', 10000),
            flush_interval_seconds=self.config.get('flush_interval', 60),
            max_memory_mb=self.config.get('max_memory_mb', 500),
            compression_enabled=self.config.get('compression', True),
            archive_enabled=self.config.get('archive_enabled', True),
            parallel_workers=self.config.get('parallel_workers', 4)
        )
    
    async def get_loader(self, data_type: DataType) -> Optional[IBulkLoader]:
        """
        Get or create a bulk loader for the specified data type.
        
        Args:
            data_type: Type of data to load
        
        Returns:
            Bulk loader instance or None if not supported
        """
        # Check cache first
        if data_type in self._loaders:
            return self._loaders[data_type]
        
        # Create new loader
        loader = await self._create_loader(data_type)
        
        if loader:
            self._loaders[data_type] = loader
            self._loader_stats['loaders_created'] += 1
            self.logger.info(f"Created {data_type} bulk loader")
        
        return loader
    
    async def _create_loader(self, data_type: DataType) -> Optional[IBulkLoader]:
        """Create a new bulk loader for the specified data type."""
        try:
            if data_type == DataType.MARKET_DATA:
                return MarketDataSplitBulkLoader(
                    db_adapter=self.db_adapter,
                    archive=self.archive,
                    config=self.bulk_config
                )
            
            elif data_type == DataType.NEWS:
                return NewsBulkLoader(
                    db_adapter=self.db_adapter,
                    archive=self.archive,
                    config=self.bulk_config
                )
            
            elif data_type == DataType.FINANCIALS:
                return FundamentalsBulkLoader(
                    db_adapter=self.db_adapter,
                    archive=self.archive,
                    config=self.bulk_config
                )
            
            elif data_type == DataType.CORPORATE_ACTIONS:
                return CorporateActionsBulkLoader(
                    db_adapter=self.db_adapter,
                    archive=self.archive,
                    config=self.bulk_config
                )
            
            else:
                self.logger.warning(f"No loader available for data type: {data_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create loader for {data_type}: {e}")
            return None
    
    async def flush_all(self) -> Dict[str, Any]:
        """
        Flush all active loaders.
        
        Returns:
            Dictionary with flush results for each loader
        """
        results = {}
        
        with timer("loader.flush_all"):
            for data_type, loader in self._loaders.items():
                try:
                    self.logger.info(f"Flushing {data_type} loader")
                    flush_result = await loader.flush_all()
                    
                    results[str(data_type)] = {
                        'success': True,
                        'records_flushed': getattr(flush_result, 'records_loaded', 0),
                        'errors': getattr(flush_result, 'errors', [])
                    }
                    
                    # Update stats
                    self._loader_stats['total_records'] += getattr(flush_result, 'records_loaded', 0)
                    
                    # Record metrics
                    record_metric("loader.flush", 
                                 getattr(flush_result, 'records_loaded', 0),
                                 MetricType.COUNTER,
                                 tags={"data_type": str(data_type)})
                    
                except Exception as e:
                    self.logger.error(f"Failed to flush {data_type} loader: {e}")
                    results[str(data_type)] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    async def reset_loader(self, data_type: DataType) -> bool:
        """
        Reset a specific loader, flushing and recreating it.
        
        Args:
            data_type: Type of loader to reset
        
        Returns:
            True if successful
        """
        try:
            # Flush if exists
            if data_type in self._loaders:
                loader = self._loaders[data_type]
                await loader.flush_all()
                del self._loaders[data_type]
                self.logger.info(f"Reset {data_type} loader")
            
            # Create new loader
            new_loader = await self.get_loader(data_type)
            return new_loader is not None
            
        except Exception as e:
            self.logger.error(f"Failed to reset {data_type} loader: {e}")
            return False
    
    async def get_loader_stats(self, data_type: Optional[DataType] = None) -> Dict[str, Any]:
        """
        Get statistics for loaders.
        
        Args:
            data_type: Optional specific loader type
        
        Returns:
            Dictionary with loader statistics
        """
        if data_type and data_type in self._loaders:
            loader = self._loaders[data_type]
            return {
                'data_type': str(data_type),
                'buffer_size': len(getattr(loader, '_buffer', [])),
                'total_loaded': getattr(loader, '_total_records_loaded', 0),
                'total_failed': getattr(loader, '_total_records_failed', 0),
                'flush_count': getattr(loader, '_flush_count', 0)
            }
        
        # Return overall stats
        stats = self._loader_stats.copy()
        stats['active_loaders'] = list(str(dt) for dt in self._loaders.keys())
        stats['loader_count'] = len(self._loaders)
        
        # Add individual loader stats
        loader_details = {}
        for dt, loader in self._loaders.items():
            loader_details[str(dt)] = {
                'buffer_size': len(getattr(loader, '_buffer', [])),
                'total_loaded': getattr(loader, '_total_records_loaded', 0)
            }
        stats['loader_details'] = loader_details
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration and recreate bulk config.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.bulk_config = self._create_bulk_config()
        self.logger.info("Updated loader coordinator configuration")
    
    async def shutdown(self) -> None:
        """Shutdown coordinator, flushing all loaders."""
        self.logger.info("Shutting down loader coordinator")
        
        # Flush all loaders
        flush_results = await self.flush_all()
        
        # Log results
        for data_type, result in flush_results.items():
            if result['success']:
                self.logger.info(f"Flushed {result.get('records_flushed', 0)} records from {data_type}")
            else:
                self.logger.error(f"Failed to flush {data_type}: {result.get('error')}")
        
        # Clear loader cache
        self._loaders.clear()
        self.logger.info("Loader coordinator shutdown complete")