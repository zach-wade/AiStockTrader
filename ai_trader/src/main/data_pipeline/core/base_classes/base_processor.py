"""
Base Processor Class

Provides common functionality for all data processing components
with standardized patterns for logging, error handling, and monitoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import pandas as pd

from main.utils.core import get_logger, ensure_utc, timer
from main.utils.monitoring import MetricsCollector, log_performance
from main.utils.data import ProcessingUtils, ValidationUtils, DataFrameStreamer
from ..exceptions import ProcessingError, convert_exception
from ..enums import ProcessingPriority


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    Provides common functionality including:
    - Standardized logging
    - Error handling with context
    - Performance monitoring
    - Configuration management
    - Processing state tracking
    """
    
    def __init__(
        self,
        processor_name: str,
        config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.processor_name = processor_name
        self.config = config or {}
        self.metrics_collector = metrics_collector
        self.logger = get_logger(f"data_pipeline.{processor_name}")
        
        # Processing state
        self._is_initialized = False
        self._processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'last_processed': None
        }
        
        # Streaming configuration for large datasets
        self.streaming_enabled = self.config.get('streaming_enabled', True)
        self.streaming_chunk_size = self.config.get('streaming_chunk_size', 10000)
        self.streaming_max_memory_mb = self.config.get('streaming_max_memory_mb', 500)
        
        self.logger.debug(f"Initialized {processor_name} processor")
    
    async def initialize(self) -> None:
        """Initialize the processor. Override in subclasses for specific setup."""
        try:
            await self._initialize_processor()
            self._is_initialized = True
            self._processing_stats['start_time'] = ensure_utc(datetime.now(timezone.utc))
            self.logger.info(f"Processor {self.processor_name} initialized successfully")
        except Exception as e:
            error = convert_exception(e, f"Failed to initialize processor {self.processor_name}")
            self.logger.error(f"Initialization failed: {error}")
            raise error
    
    @abstractmethod
    async def _initialize_processor(self) -> None:
        """Processor-specific initialization logic. Override in subclasses."""
        pass
    
    @log_performance
    async def process(
        self,
        data: Any,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Process data with monitoring and error handling.
        
        Args:
            data: Data to process
            priority: Processing priority level
            context: Additional context for processing
            
        Returns:
            Processed data
            
        Raises:
            ProcessingError: If processing fails
        """
        if not self._is_initialized:
            await self.initialize()
        
        context = context or {}
        processing_id = context.get('processing_id', f"{self.processor_name}_{ensure_utc(datetime.now(timezone.utc)).isoformat()}")
        
        self.logger.debug(f"Starting processing: {processing_id} (priority: {priority.name})")
        
        try:
            with timer(f"process_{self.processor_name}"):
                # Pre-processing validation
                await self._validate_input(data, context)
                
                # Check if we should use streaming for large datasets
                if self._should_use_streaming(data):
                    result = await self._process_with_streaming(data, priority, context)
                else:
                    # Core processing logic
                    result = await self._process_data(data, priority, context)
                
                # Post-processing validation
                await self._validate_output(result, context)
            
            # Update statistics
            self._update_stats(success=True)
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_processing_success(
                    processor=self.processor_name,
                    priority=priority.name,
                    context=context
                )
            
            self.logger.debug(f"Processing completed successfully: {processing_id}")
            return result
            
        except Exception as e:
            self._update_stats(success=False)
            
            # Record metrics for failure
            if self.metrics_collector:
                self.metrics_collector.record_processing_failure(
                    processor=self.processor_name,
                    error=str(e),
                    priority=priority.name,
                    context=context
                )
            
            # Convert to ProcessingError with context
            error = ProcessingError(
                message=f"Processing failed in {self.processor_name}",
                processor_type=self.processor_name,
                processing_stage=context.get('stage', 'unknown'),
                original_error=e,
                context={
                    'processing_id': processing_id,
                    'priority': priority.name,
                    **context
                }
            )
            
            self.logger.error(f"Processing failed: {error}")
            raise error
    
    @abstractmethod
    async def _process_data(
        self, 
        data: Any, 
        priority: ProcessingPriority,
        context: Dict[str, Any]
    ) -> Any:
        """Core processing logic. Override in subclasses."""
        pass
    
    async def _validate_input(self, data: Any, context: Dict[str, Any]) -> None:
        """Validate input data. Override in subclasses for specific validation."""
        if data is None:
            raise ProcessingError(
                "Input data cannot be None",
                processor_type=self.processor_name,
                processing_stage="input_validation"
            )
        
        # Use ValidationUtils for DataFrame validation
        if isinstance(data, pd.DataFrame):
            if context.get('data_type') == 'ohlcv':
                is_valid, errors = ValidationUtils.validate_ohlcv_data(data)
                if not is_valid:
                    raise ProcessingError(
                        f"OHLCV validation failed: {'; '.join(errors)}",
                        processor_type=self.processor_name,
                        processing_stage="input_validation",
                        context={'errors': errors}
                    )
            
            # Check data quality
            max_missing = context.get('max_missing_pct', 0.1)
            is_quality, quality_issues = ValidationUtils.check_data_quality(data, max_missing)
            if not is_quality:
                self.logger.warning(f"Data quality issues detected: {quality_issues}")
    
    async def _validate_output(self, result: Any, context: Dict[str, Any]) -> None:
        """Validate output data. Override in subclasses for specific validation."""
        if result is None and not context.get('allow_none_result', False):
            raise ProcessingError(
                "Processing result cannot be None",
                processor_type=self.processor_name,
                processing_stage="output_validation"
            )
    
    def _update_stats(self, success: bool) -> None:
        """Update processing statistics."""
        self._processing_stats['total_processed'] += 1
        if success:
            self._processing_stats['successful'] += 1
        else:
            self._processing_stats['failed'] += 1
        self._processing_stats['last_processed'] = ensure_utc(datetime.now(timezone.utc))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._processing_stats.copy()
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        try:
            await self._shutdown_processor()
            self.logger.info(f"Processor {self.processor_name} shutdown successfully")
        except Exception as e:
            error = convert_exception(e, f"Failed to shutdown processor {self.processor_name}")
            self.logger.error(f"Shutdown failed: {error}")
            raise error
    
    async def _shutdown_processor(self) -> None:
        """Processor-specific shutdown logic. Override in subclasses."""
        pass
    
    def _should_use_streaming(self, data: Any) -> bool:
        """Determine if streaming should be used for processing."""
        if not self.streaming_enabled:
            return False
        
        if isinstance(data, pd.DataFrame):
            # Check DataFrame size
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            return (
                len(data) > self.streaming_chunk_size or 
                memory_mb > self.streaming_max_memory_mb
            )
        
        return False
    
    async def _process_with_streaming(
        self, 
        data: pd.DataFrame, 
        priority: ProcessingPriority,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process large DataFrame using streaming."""
        self.logger.info(
            f"Processing large dataset ({len(data)} rows) using streaming"
        )
        
        # Create streamer with config
        from main.utils.processing.streaming import StreamingConfig
        config = StreamingConfig(
            chunk_size=self.streaming_chunk_size,
            max_memory_mb=self.streaming_max_memory_mb
        )
        streamer = DataFrameStreamer(config)
        
        # Create async processor wrapper
        async def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            return await self._process_data(chunk, priority, context)
        
        # Process the stream and get results
        result = await streamer.process_stream(
            data_source=data,
            processor=process_chunk,
            output_path=None  # Collect results in memory
        )
        
        # Update metrics based on streamer stats
        if self.metrics_collector:
            stats = streamer.stats
            self.metrics_collector.record_streaming_progress(
                processor=self.processor_name,
                progress=100.0,  # Completed
                chunks_processed=stats.chunks_processed
            )
        
        return result if result is not None else pd.DataFrame()
    
    async def preprocess_data(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Apply standard preprocessing using ProcessingUtils."""
        # Handle missing values
        fill_method = context.get('fill_method', 'forward_fill')
        data = ProcessingUtils.handle_missing_values(data, method=fill_method)
        
        # Remove outliers if requested
        if context.get('remove_outliers', False):
            outlier_method = context.get('outlier_method', 'iqr')
            data = ProcessingUtils.remove_outliers(data, method=outlier_method)
        
        # Normalize if requested
        if context.get('normalize', False):
            norm_method = context.get('norm_method', 'zscore')
            data = ProcessingUtils.normalize_data(data, method=norm_method)
        
        return data
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.processor_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.processor_name}, initialized={self._is_initialized})"