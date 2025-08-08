"""
Fundamentals Format Factory

Factory for selecting the appropriate format handler based on data characteristics.
"""

from typing import Optional, Any
from dataclasses import dataclass

from main.utils.core import get_logger
from main.data_pipeline.services.ingestion import (
    MetricExtractionService,
    MetricExtractionConfig
)
from main.data_pipeline.services.format_handlers import (
    BaseFormatHandler,
    FormatHandlerConfig,
    PolygonFormatHandler,
    YahooFormatHandler,
    PreProcessedFormatHandler
)

logger = get_logger(__name__)


@dataclass
class FundamentalsFormatFactoryConfig:
    """Configuration for the format factory."""
    metric_extraction_config: Optional[MetricExtractionConfig] = None
    format_handler_config: Optional[FormatHandlerConfig] = None
    prefer_handler: Optional[str] = None  # Force specific handler if set


class FundamentalsFormatFactory:
    """
    Factory for creating appropriate format handlers for financial data.
    
    Analyzes data characteristics and returns the correct handler
    to process the data into standardized format.
    """
    
    def __init__(self, config: Optional[FundamentalsFormatFactoryConfig] = None):
        """
        Initialize the format factory.
        
        Args:
            config: Factory configuration
        """
        self.config = config or FundamentalsFormatFactoryConfig()
        
        # Create shared metric extractor
        self.metric_extractor = MetricExtractionService(
            self.config.metric_extraction_config
        )
        
        # Initialize handlers
        handler_config = self.config.format_handler_config
        
        self.handlers = {
            'preprocessed': PreProcessedFormatHandler(self.metric_extractor, handler_config),
            'polygon': PolygonFormatHandler(self.metric_extractor, handler_config),
            'yahoo': YahooFormatHandler(self.metric_extractor, handler_config)
        }
        
        logger.info(f"FundamentalsFormatFactory initialized with {len(self.handlers)} handlers")
    
    def get_handler(self, data: Any, source: Optional[str] = None) -> Optional[BaseFormatHandler]:
        """
        Select the appropriate handler for the given data.
        
        Args:
            data: Financial data to analyze
            source: Optional hint about data source
            
        Returns:
            Appropriate format handler or None if no handler can process the data
        """
        # Check if specific handler is preferred
        if self.config.prefer_handler:
            handler = self.handlers.get(self.config.prefer_handler)
            if handler and handler.can_handle(data):
                logger.debug(f"Using preferred handler: {self.config.prefer_handler}")
                return handler
        
        # Use source hint if provided
        if source:
            source_lower = source.lower()
            if source_lower in self.handlers:
                handler = self.handlers[source_lower]
                if handler.can_handle(data):
                    logger.debug(f"Using handler based on source hint: {source_lower}")
                    return handler
        
        # Check handlers in priority order
        # Pre-processed first (most specific format)
        if self.handlers['preprocessed'].can_handle(data):
            logger.debug("Detected pre-processed format")
            return self.handlers['preprocessed']
        
        # Then Polygon (has specific fields)
        if self.handlers['polygon'].can_handle(data):
            logger.debug("Detected Polygon format")
            return self.handlers['polygon']
        
        # Finally Yahoo (most general)
        if self.handlers['yahoo'].can_handle(data):
            logger.debug("Detected Yahoo format")
            return self.handlers['yahoo']
        
        logger.warning(f"No handler found for data type: {type(data)}")
        return None
    
    def reset_all_handlers(self):
        """Reset duplicate tracking for all handlers."""
        for handler in self.handlers.values():
            handler.reset_duplicates()
        logger.debug("All format handlers reset")
    
    def get_metric_extractor(self) -> MetricExtractionService:
        """
        Get the shared metric extraction service.
        
        Returns:
            MetricExtractionService instance
        """
        return self.metric_extractor