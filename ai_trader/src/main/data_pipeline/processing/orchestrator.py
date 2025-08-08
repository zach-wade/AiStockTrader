"""
Processing Orchestrator

Coordinates all processing components (transform, standardize, clean, validate, ETL).
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.storage.archive import DataArchive
from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import timer, record_metric, MetricType
from main.utils.data import get_global_processor, get_global_validator, StreamingConfig

from .transformers import DataTransformer
from .standardizers import DataStandardizer
from .cleaners import DataCleaner
from .validators import PipelineValidator
from .etl import ETLManager


@dataclass
class ProcessingResult:
    """Result of a complete processing pipeline."""
    success: bool
    data: Optional[pd.DataFrame] = None
    records_processed: int = 0
    records_loaded: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class ProcessingOrchestrator(ErrorHandlingMixin):
    """
    Orchestrates the complete data processing pipeline.
    
    Coordinates transformation, standardization, cleaning,
    validation, and ETL operations with layer-aware rules.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: DataArchive,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize processing orchestrator.
        
        Args:
            db_adapter: Database adapter
            archive: Data archive instance
            config: Optional configuration
        """
        self.db_adapter = db_adapter
        self.archive = archive
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Initialize all processing components
        self.transformer = DataTransformer()
        self.standardizer = DataStandardizer()
        self.cleaner = DataCleaner()
        self.validator = PipelineValidator()
        self.etl_manager = ETLManager(db_adapter, archive, config)
        
        # Get global processors for advanced operations
        self.global_processor = get_global_processor()
        self.global_validator = get_global_validator()
        
        # Processing statistics
        self._processing_stats = {
            'total_pipelines': 0,
            'successful': 0,
            'failed': 0,
            'total_records': 0
        }
    
    async def process_data(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer,
        symbol: Optional[str] = None,
        source: str = 'unknown',
        skip_steps: Optional[List[str]] = None
    ) -> ProcessingResult:
        """
        Process data through the complete pipeline.
        
        Args:
            data: Input DataFrame
            data_type: Type of data
            layer: Data layer for processing rules
            symbol: Optional stock symbol
            source: Data source
            skip_steps: Optional list of steps to skip
        
        Returns:
            ProcessingResult with processed data and metrics
        """
        skip_steps = skip_steps or []
        
        with timer("processing.pipeline", tags={
            "data_type": data_type.value if hasattr(data_type, 'value') else str(data_type),
            "layer": layer.name,
            "symbol": symbol or "unknown"
        }):
            try:
                result = ProcessingResult(success=True)
                result.metrics['input_rows'] = len(data)
                
                # Start with input data
                processed_data = data.copy()
                
                # Step 1: Standardize
                if 'standardize' not in skip_steps:
                    self.logger.info("Step 1: Standardizing data")
                    processed_data = await self.standardizer.standardize(
                        processed_data, data_type, source, layer
                    )
                    result.metrics['after_standardize'] = len(processed_data)
                
                # Step 2: Clean
                if 'clean' not in skip_steps:
                    self.logger.info("Step 2: Cleaning data")
                    processed_data = await self.cleaner.clean(
                        processed_data, data_type, layer
                    )
                    result.metrics['after_clean'] = len(processed_data)
                
                # Step 3: Transform
                if 'transform' not in skip_steps:
                    self.logger.info("Step 3: Transforming data")
                    context = {
                        'symbol': symbol,
                        'data_type': data_type,
                        'normalize': layer >= DataLayer.CATALYST
                    }
                    processed_data = await self.transformer.transform(
                        processed_data, 'pandas', 'pandas', layer, context
                    )
                    result.metrics['after_transform'] = len(processed_data)
                
                # Step 4: Validate
                if 'validate' not in skip_steps:
                    self.logger.info("Step 4: Validating data")
                    validation_result = await self.validator.validate_data(
                        processed_data, data_type, layer
                    )
                    
                    if not validation_result.is_valid:
                        result.errors.extend(validation_result.errors)
                        result.warnings.extend(validation_result.warnings)
                        
                        # Decide whether to continue based on layer
                        if layer >= DataLayer.CATALYST:
                            # Strict validation for higher layers
                            result.success = False
                            result.data = None
                            self.logger.error(f"Validation failed: {validation_result.errors}")
                            return result
                    
                    result.warnings.extend(validation_result.warnings)
                    result.metrics.update(validation_result.metrics)
                
                # Store processed data
                result.data = processed_data
                result.records_processed = len(processed_data)
                
                # Update statistics
                self._processing_stats['total_pipelines'] += 1
                self._processing_stats['successful'] += 1
                self._processing_stats['total_records'] += result.records_processed
                
                # Record metrics
                # gauge("processing.output_rows", result.records_processed,
                #       tags={"symbol": symbol or "unknown"})
                record_metric("processing.reduction_rate",
                             1 - (result.records_processed / len(data)) if len(data) > 0 else 0,
                             MetricType.GAUGE,
                             tags={"symbol": symbol or "unknown"})
                
                self.logger.info(f"Processing complete: {result.records_processed} records")
                return result
                
            except Exception as e:
                self.logger.error(f"Processing failed: {e}")
                self._processing_stats['failed'] += 1
                return ProcessingResult(
                    success=False,
                    errors=[str(e)]
                )
    
    async def process_and_load(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        source: str = 'polygon',
        interval: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process data from archive and load to database.
        
        Combines processing pipeline with ETL operations.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data
            start_date: Start date
            end_date: End date
            layer: Data layer
            source: Data source
            interval: Optional interval for market data
        
        Returns:
            ProcessingResult with complete metrics
        """
        with timer("processing.process_and_load", tags={
            "symbol": symbol,
            "data_type": str(data_type),
            "layer": layer.name
        }):
            # Use ETL manager to load data
            etl_result = await self.etl_manager.load_data(
                symbol, data_type, start_date, end_date, layer, source, interval
            )
            
            result = ProcessingResult(
                success=etl_result.success,
                records_loaded=etl_result.records_loaded,
                errors=etl_result.errors,
                warnings=etl_result.warnings,
                metrics=etl_result.metadata
            )
            
            if not etl_result.success:
                self.logger.error(f"ETL failed for {symbol}: {etl_result.errors}")
                self._processing_stats['failed'] += 1
            else:
                self._processing_stats['successful'] += 1
                self._processing_stats['total_records'] += etl_result.records_loaded
            
            return result
    
    async def process_batch(
        self,
        symbols: List[str],
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        max_concurrent: int = 5
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple symbols in batch.
        
        Args:
            symbols: List of symbols
            data_type: Type of data
            start_date: Start date
            end_date: End date
            layer: Data layer
            max_concurrent: Maximum concurrent operations
        
        Returns:
            Dictionary of results by symbol
        """
        import asyncio
        
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_symbol(symbol: str):
            async with semaphore:
                return await self.process_and_load(
                    symbol, data_type, start_date, end_date, layer
                )
        
        # Process all symbols
        tasks = [process_symbol(symbol) for symbol in symbols]
        symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results
        for symbol, result in zip(symbols, symbol_results):
            if isinstance(result, Exception):
                results[symbol] = ProcessingResult(
                    success=False,
                    errors=[str(result)]
                )
            else:
                results[symbol] = result
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        self.logger.info(f"Batch processing complete: {successful}/{len(symbols)} successful")
        
        return results
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._processing_stats.copy()
        
        # Add component stats
        stats['transformer_stats'] = await self.transformer.get_transformation_stats()
        stats['cleaner_stats'] = await self.cleaner.get_cleaning_stats()
        stats['validator_stats'] = await self.validator.get_validation_stats()
        stats['etl_stats'] = await self.etl_manager.get_etl_stats()
        
        # Calculate rates
        if stats['total_pipelines'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_pipelines']
            stats['avg_records'] = stats['total_records'] / stats['total_pipelines']
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown orchestrator and flush all components."""
        self.logger.info("Shutting down processing orchestrator")
        
        # Flush ETL manager
        await self.etl_manager.loader_coordinator.shutdown()
        
        # Log final stats
        stats = await self.get_processing_stats()
        self.logger.info(f"Final processing stats: {stats}")
        
        self.logger.info("Processing orchestrator shutdown complete")