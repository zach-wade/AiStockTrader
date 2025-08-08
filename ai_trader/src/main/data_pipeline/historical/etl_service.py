"""
ETL Service

Service responsible for coordinating Extract-Transform-Load operations
for historical data processing. Part of the service-oriented architecture.
"""

from typing import Dict, List, Any, Optional, Type
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import asyncio

from main.utils.core import get_logger, timer
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.data_pipeline.types import DataType, TimeInterval, BackfillParams
from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
from main.data_pipeline.historical.data_fetch_service import DataFetchService, FetchRequest
# DataTransformer will be implemented in the new pipeline
# from main.data_pipeline.processing.transformer import DataTransformer
from main.data_pipeline.ingestion.factories.bulk_loader_factory import BulkLoaderFactory
from main.interfaces.ingestion import IBulkLoader, BulkLoadResult

logger = get_logger(__name__)


@dataclass
class ETLRequest:
    """Configuration for an ETL operation."""
    symbol: str
    data_type: DataType
    start_date: datetime
    end_date: datetime
    source: str = "polygon"
    intervals: List[TimeInterval] = field(default_factory=lambda: [TimeInterval.ONE_DAY])
    layer: int = 1
    use_bulk_loader: bool = True
    force_refresh: bool = False
    archive_enabled: bool = True


@dataclass
class ETLResult:
    """Result of an ETL operation."""
    success: bool
    symbol: str
    data_type: DataType
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_archived: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class ETLConfig:
    """Configuration for the ETL service."""
    # Layer-based configuration
    layer_configs: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        1: {
            "batch_size": 10000,
            "parallel_workers": 4,
            "archive_enabled": True,
            "validation_level": "strict"
        },
        2: {
            "batch_size": 5000,
            "parallel_workers": 2,
            "archive_enabled": True,
            "validation_level": "standard"
        },
        3: {
            "batch_size": 2500,
            "parallel_workers": 1,
            "archive_enabled": False,
            "validation_level": "basic"
        }
    })
    
    # Processing configuration
    enable_transformation: bool = True
    enable_validation: bool = True
    enable_deduplication: bool = True
    
    # Archive configuration
    archive_raw_data: bool = True
    archive_transformed_data: bool = False
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: int = 5
    continue_on_error: bool = True


class ETLService:
    """
    Service for coordinating ETL operations.
    
    Responsibilities:
    - Coordinate extraction from data sources
    - Apply transformations and validations
    - Load data using bulk loaders
    - Archive data to cold storage
    - Emit events for monitoring and coordination
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: Optional[DataArchive] = None,
        event_bus: Optional[IEventBus] = None,
        config: Optional[ETLConfig] = None,
        data_fetch_service: Optional[DataFetchService] = None,
        transformer: Optional[Any] = None,  # DataTransformer to be implemented
        bulk_loader_factory: Optional[BulkLoaderFactory] = None
    ):
        """
        Initialize the ETL service.
        
        Args:
            db_adapter: Database adapter for operations
            archive: Archive for cold storage
            event_bus: Event bus for coordination
            config: Service configuration
            data_fetch_service: Service for fetching data
            transformer: Data transformer
            bulk_loader_factory: Factory for creating bulk loaders
        """
        self.logger = get_logger(__name__)
        self.db_adapter = db_adapter
        self.archive = archive
        self.event_bus = event_bus
        self.config = config or ETLConfig()
        
        # Initialize services
        self.data_fetch_service = data_fetch_service or DataFetchService(
            db_adapter=db_adapter
        )
        self.transformer = transformer
        self.bulk_loader_factory = bulk_loader_factory or BulkLoaderFactory()
        
        # Track active operations
        self._active_operations = {}
        self._operation_lock = asyncio.Lock()
        
        self.logger.info("ETLService initialized")
    
    async def initialize(self):
        """Initialize dependencies."""
        await self.data_fetch_service.initialize()
        
        # Transformer will be initialized when implemented in new pipeline
        if not self.transformer:
            self.logger.info("Running without transformer (not yet implemented in new pipeline)")
    
    @timer
    async def process_etl_request(self, request: ETLRequest) -> ETLResult:
        """
        Process a single ETL request.
        
        Args:
            request: ETL request configuration
            
        Returns:
            ETLResult with operation details
        """
        start_time = datetime.now(timezone.utc)
        result = ETLResult(
            success=False,
            symbol=request.symbol,
            data_type=request.data_type
        )
        
        try:
            # Track operation
            operation_id = f"{request.symbol}_{request.data_type.value}_{start_time.timestamp()}"
            async with self._operation_lock:
                self._active_operations[operation_id] = request
            
            # Get layer configuration
            layer_config = self.config.layer_configs.get(
                request.layer,
                self.config.layer_configs.get(3)  # Default to layer 3 config
            )
            
            # Emit start event
            if self.event_bus:
                await self._emit_etl_event("etl_started", request, result)
            
            # Phase 1: Extract
            self.logger.info(f"Extracting {request.data_type.value} for {request.symbol}")
            extracted_data = await self._extract_data(request)
            
            if not extracted_data:
                result.errors.append("No data extracted")
                result.success = True  # No data is not necessarily an error
                return result
            
            result.records_extracted = self._count_records(extracted_data)
            
            # Phase 2: Transform
            transformed_data = extracted_data
            if self.config.enable_transformation and self.transformer:
                self.logger.info(f"Transforming {result.records_extracted} records")
                transformed_data = await self._transform_data(
                    extracted_data, request
                )
                result.records_transformed = self._count_records(transformed_data)
            else:
                result.records_transformed = result.records_extracted
            
            # Phase 3: Load
            if request.use_bulk_loader:
                self.logger.info(f"Loading {result.records_transformed} records")
                load_result = await self._load_data(
                    transformed_data, request, layer_config
                )
                result.records_loaded = load_result.records_loaded
                result.errors.extend(load_result.errors)
            
            # Phase 4: Archive
            if request.archive_enabled and self.archive:
                self.logger.info(f"Archiving {result.records_transformed} records")
                archive_count = await self._archive_data(
                    extracted_data if self.config.archive_raw_data else transformed_data,
                    request
                )
                result.records_archived = archive_count
            
            # Calculate duration
            result.duration_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            
            # Determine success
            result.success = (
                result.records_loaded > 0 or
                result.records_archived > 0 or
                result.records_extracted == 0  # No data is OK
            )
            
            # Emit completion event
            if self.event_bus:
                await self._emit_etl_event("etl_completed", request, result)
            
            self.logger.info(
                f"ETL completed for {request.symbol}/{request.data_type.value}: "
                f"extracted={result.records_extracted}, "
                f"transformed={result.records_transformed}, "
                f"loaded={result.records_loaded}, "
                f"archived={result.records_archived}"
            )
            
        except Exception as e:
            error_msg = f"ETL failed for {request.symbol}/{request.data_type.value}: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
            
            # Emit error event
            if self.event_bus:
                await self._emit_etl_event("etl_failed", request, result)
        
        finally:
            # Remove from active operations
            async with self._operation_lock:
                self._active_operations.pop(operation_id, None)
        
        return result
    
    async def process_batch_etl(
        self,
        requests: List[ETLRequest],
        max_parallel: Optional[int] = None
    ) -> List[ETLResult]:
        """
        Process multiple ETL requests in parallel.
        
        Args:
            requests: List of ETL requests
            max_parallel: Maximum parallel operations
            
        Returns:
            List of ETL results
        """
        if not requests:
            return []
        
        # Determine parallelism based on layer
        if max_parallel is None:
            layer = requests[0].layer if requests else 1
            layer_config = self.config.layer_configs.get(layer, {})
            max_parallel = layer_config.get('parallel_workers', 2)
        
        # Process with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(request: ETLRequest) -> ETLResult:
            async with semaphore:
                return await self.process_etl_request(request)
        
        # Process all requests
        tasks = [process_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ETLResult(
                    success=False,
                    symbol=requests[i].symbol,
                    data_type=requests[i].data_type,
                    errors=[str(result)]
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def _extract_data(self, request: ETLRequest) -> Any:
        """Extract data from source."""
        fetch_request = FetchRequest(
            symbol=request.symbol,
            data_type=request.data_type.value,
            start_date=request.start_date,
            end_date=request.end_date,
            source=request.source,
            intervals=[interval.value for interval in request.intervals],
            layer=request.layer
        )
        
        fetch_result = await self.data_fetch_service.fetch_data(fetch_request)
        
        if not fetch_result.success:
            raise Exception(f"Fetch failed: {fetch_result.errors}")
        
        return fetch_result.data
    
    async def _transform_data(self, data: Any, request: ETLRequest) -> Any:
        """Transform extracted data."""
        if not self.transformer:
            # Transformer not yet implemented in new pipeline
            # For now, pass data through unchanged
            return data
        
        # When transformer is implemented:
        # Handle different data types
        # if request.data_type == DataType.MARKET_DATA:
        #     import pandas as pd
        #     if isinstance(data, pd.DataFrame):
        #         return await self.transformer.transform_market_data(
        #             data, request.source, request.symbol
        #         )
        
        return data
    
    async def _load_data(
        self,
        data: Any,
        request: ETLRequest,
        layer_config: Dict[str, Any]
    ) -> BulkLoadResult:
        """Load data using bulk loader."""
        # Create appropriate bulk loader
        bulk_loader = self._create_bulk_loader(request.data_type, request.layer)
        
        if not bulk_loader:
            return BulkLoadResult(
                success=False,
                data_type=request.data_type.value,
                errors=["No bulk loader available"]
            )
        
        # Load data
        return await bulk_loader.load(
            data=data,
            symbols=[request.symbol],
            source=request.source
        )
    
    async def _archive_data(self, data: Any, request: ETLRequest) -> int:
        """Archive data to cold storage."""
        if not self.archive:
            return 0
        
        try:
            # Create raw data record
            record = RawDataRecord(
                source=request.source,
                data_type=request.data_type.value,
                symbol=request.symbol,
                timestamp=datetime.now(timezone.utc),
                data=data,
                metadata={
                    'start_date': request.start_date.isoformat(),
                    'end_date': request.end_date.isoformat(),
                    'intervals': [i.value for i in request.intervals],
                    'layer': request.layer
                }
            )
            
            # Save to archive
            success = await self.archive.save_raw_record_async(record)
            
            if success:
                return self._count_records(data)
            
        except Exception as e:
            self.logger.error(f"Archive failed: {e}")
        
        return 0
    
    def _create_bulk_loader(
        self,
        data_type: DataType,
        layer: int
    ) -> Optional[IBulkLoader]:
        """Create appropriate bulk loader for data type."""
        if data_type == DataType.MARKET_DATA:
            return self.bulk_loader_factory.create_market_data_loader(
                db_adapter=self.db_adapter,
                layer=layer,
                archive=self.archive
            )
        elif data_type == DataType.NEWS:
            return self.bulk_loader_factory.create_news_loader(
                db_adapter=self.db_adapter,
                archive=self.archive
            )
        elif data_type == DataType.FINANCIALS:
            return self.bulk_loader_factory.create_fundamentals_loader(
                db_adapter=self.db_adapter,
                archive=self.archive
            )
        elif data_type == DataType.CORPORATE_ACTIONS:
            return self.bulk_loader_factory.create_corporate_actions_loader(
                db_adapter=self.db_adapter,
                archive=self.archive
            )
        
        return None
    
    def _count_records(self, data: Any) -> int:
        """Count records in data."""
        if data is None:
            return 0
        elif isinstance(data, list):
            return len(data)
        elif hasattr(data, '__len__'):
            return len(data)
        elif hasattr(data, 'shape'):  # DataFrame
            return data.shape[0]
        else:
            return 1
    
    async def _emit_etl_event(
        self,
        event_type: str,
        request: ETLRequest,
        result: ETLResult
    ):
        """Emit ETL event to event bus."""
        if not self.event_bus:
            return
        
        try:
            event_data = {
                'type': event_type,
                'symbol': request.symbol,
                'data_type': request.data_type.value,
                'layer': request.layer,
                'records_extracted': result.records_extracted,
                'records_transformed': result.records_transformed,
                'records_loaded': result.records_loaded,
                'records_archived': result.records_archived,
                'success': result.success,
                'errors': result.errors
            }
            
            await self.event_bus.publish(event_type, event_data)
            
        except Exception as e:
            self.logger.error(f"Failed to emit event: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            'active_operations': len(self._active_operations),
            'transformation_enabled': self.config.enable_transformation,
            'validation_enabled': self.config.enable_validation,
            'archive_enabled': self.config.archive_raw_data
        }
    
    async def cancel_operation(self, symbol: str, data_type: DataType) -> bool:
        """
        Cancel an active ETL operation.
        
        Args:
            symbol: Symbol to cancel
            data_type: Data type to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        async with self._operation_lock:
            for op_id, request in list(self._active_operations.items()):
                if request.symbol == symbol and request.data_type == data_type:
                    del self._active_operations[op_id]
                    self.logger.info(f"Cancelled ETL for {symbol}/{data_type.value}")
                    return True
        
        return False