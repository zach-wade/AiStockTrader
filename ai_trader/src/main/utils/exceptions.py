"""
Unified Exception Hierarchy for AI Trader System

This module provides a comprehensive exception hierarchy that consolidates
all exception types used throughout the application, eliminating duplicates
and providing consistent error handling.
"""

from typing import Optional, Dict, Any
from main.utils.core import get_logger

logger = get_logger(__name__)


class AITraderException(Exception):
    """
    Base exception for all AI Trader errors.
    
    This is the root of the exception hierarchy and provides rich context
    for error handling and debugging.
    """
    
    def __init__(
        self, 
        message: str, 
        component: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.component = component or self.__class__.__module__.split('.')[-1]
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.original_error = original_error
        
        # Build comprehensive error message
        error_parts = [message]
        if component:
            error_parts.append(f"Component: {component}")
        if error_code:
            error_parts.append(f"Code: {error_code}")
        if original_error:
            error_parts.append(f"Original: {str(original_error)}")
            
        super().__init__(" | ".join(error_parts))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'component': self.component,
            'error_code': self.error_code,
            'context': self.context,
            'original_error': str(self.original_error) if self.original_error else None
        }


# ============================================================================
# Configuration and Setup Errors
# ============================================================================

class ConfigurationError(AITraderException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        self.config_key = config_key
        if config_key:
            kwargs.setdefault('context', {})['config_key'] = config_key
        super().__init__(message, component="configuration", **kwargs)


class EnvironmentError(ConfigurationError):
    """Raised when environment-specific configuration has issues."""
    pass


class MissingDependencyError(ConfigurationError):
    """Raised when a required dependency is not installed."""
    pass


# ============================================================================
# Data Pipeline Errors
# ============================================================================

class DataPipelineError(AITraderException):
    """Base exception for all data pipeline errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="data_pipeline", **kwargs)


class ValidationError(DataPipelineError):
    """Exceptions related to data validation."""
    
    def __init__(
        self,
        message: str,
        validation_rule: Optional[str] = None,
        failed_records: Optional[int] = None,
        **kwargs
    ):
        self.validation_rule = validation_rule
        self.failed_records = failed_records
        
        context = kwargs.get('context', {})
        if validation_rule:
            context['validation_rule'] = validation_rule
        if failed_records is not None:
            context['failed_records'] = failed_records
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class StorageError(DataPipelineError):
    """Exceptions related to data storage operations."""
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.storage_type = storage_type
        self.operation = operation
        
        context = kwargs.get('context', {})
        if storage_type:
            context['storage_type'] = storage_type
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class IngestionError(DataPipelineError):
    """Exceptions related to data ingestion."""
    
    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs
    ):
        self.source = source
        self.data_type = data_type
        
        context = kwargs.get('context', {})
        if source:
            context['source'] = source
        if data_type:
            context['data_type'] = data_type
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ProcessingError(DataPipelineError):
    """Exceptions related to data processing."""
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        records_affected: Optional[int] = None,
        **kwargs
    ):
        self.stage = stage
        self.records_affected = records_affected
        
        context = kwargs.get('context', {})
        if stage:
            context['stage'] = stage
        if records_affected is not None:
            context['records_affected'] = records_affected
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class TransformationError(ProcessingError):
    """Error during data transformation."""
    pass


class AggregationError(ProcessingError):
    """Error during data aggregation."""
    pass


class DataFetchError(DataPipelineError):
    """Raised when data fetching fails."""
    pass


class DataQualityError(DataPipelineError):
    """Raised when data quality checks fail."""
    pass


# ============================================================================
# Database Errors
# ============================================================================

class DatabaseError(AITraderException):
    """Base class for database-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="database", **kwargs)


class ConnectionError(DatabaseError):
    """Database connection errors."""
    pass


class QueryError(DatabaseError):
    """Query execution errors."""
    pass


class TransactionError(DatabaseError):
    """Transaction-related errors."""
    pass


class PoolExhaustedError(DatabaseError):
    """Connection pool exhausted."""
    pass


# ============================================================================
# API and External Service Errors
# ============================================================================

class ExternalServiceError(AITraderException):
    """Base class for external service errors."""
    
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        self.service = service
        if service:
            kwargs.setdefault('context', {})['service'] = service
        super().__init__(message, component="external_service", **kwargs)


class APIError(ExternalServiceError):
    """Base class for API-related errors."""
    pass


class APIConnectionError(APIError):
    """API connection errors."""
    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        if retry_after:
            kwargs.setdefault('context', {})['retry_after'] = retry_after
        super().__init__(message, **kwargs)


class APIAuthenticationError(APIError):
    """API authentication failed."""
    pass


class DataSourceException(ExternalServiceError):
    """Data source specific errors."""
    pass


# ============================================================================
# Trading and Order Management Errors
# ============================================================================

class TradingError(AITraderException):
    """Base class for trading-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="trading", **kwargs)


class OrderExecutionError(TradingError):
    """Order execution failed."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs
    ):
        self.order_id = order_id
        self.symbol = symbol
        
        context = kwargs.get('context', {})
        if order_id:
            context['order_id'] = order_id
        if symbol:
            context['symbol'] = symbol
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class InsufficientFundsError(TradingError):
    """Insufficient funds for trade."""
    pass


class PositionError(TradingError):
    """Position management error."""
    pass


class RiskLimitExceededError(TradingError):
    """Risk limit exceeded."""
    pass


class BrokerError(TradingError):
    """Broker-specific error."""
    pass


# ============================================================================
# Model and Prediction Errors
# ============================================================================

class ModelError(AITraderException):
    """Base class for model-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="model", **kwargs)


class ModelNotFoundError(ModelError):
    """Model not found."""
    pass


class PredictionError(ModelError):
    """Prediction failed."""
    pass


class TrainingError(ModelError):
    """Model training failed."""
    pass


class FeatureError(ModelError):
    """Feature engineering error."""
    pass


# ============================================================================
# Scanner and Alert Errors
# ============================================================================

class ScannerError(AITraderException):
    """Base class for scanner errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="scanner", **kwargs)


class AlertGenerationError(ScannerError):
    """Alert generation failed."""
    pass


class QualificationError(ScannerError):
    """Symbol qualification error."""
    pass


# ============================================================================
# Event System Errors
# ============================================================================

class EventError(AITraderException):
    """Base class for event system errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="events", **kwargs)


class EventPublishError(EventError):
    """Event publishing failed."""
    pass


class EventSubscriptionError(EventError):
    """Event subscription error."""
    pass


class EventHandlerError(EventError):
    """Event handler execution error."""
    pass


# ============================================================================
# Monitoring and System Errors
# ============================================================================

class MonitoringError(AITraderException):
    """Base class for monitoring errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="monitoring", **kwargs)


class MetricsError(MonitoringError):
    """Metrics collection/export error."""
    pass


class AlertError(MonitoringError):
    """Alert system error."""
    pass


class SystemError(AITraderException):
    """System-level errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="system", **kwargs)


class ResourceExhaustedError(SystemError):
    """System resource exhausted."""
    pass


class CircuitBreakerOpenError(SystemError):
    """Circuit breaker is open."""
    pass


# ============================================================================
# Utility function to convert old exception types to new ones
# ============================================================================

def migrate_exception(old_exception: Exception) -> AITraderException:
    """
    Convert old exception types to new unified hierarchy.
    This helps during migration period.
    """
    # Map old exception names to new ones
    exception_map = {
        'DataPipelineException': DataPipelineError,
        'DatabaseException': DatabaseError,
        'TradingException': TradingError,
        # Add more mappings as needed
    }
    
    old_name = old_exception.__class__.__name__
    new_class = exception_map.get(old_name, AITraderException)
    
    return new_class(
        str(old_exception),
        original_error=old_exception
    )


# Export most commonly used exceptions
__all__ = [
    # Base
    'AITraderException',
    
    # Configuration
    'ConfigurationError',
    'EnvironmentError',
    'MissingDependencyError',
    
    # Data Pipeline
    'DataPipelineError',
    'ValidationError',
    'StorageError',
    'IngestionError',
    'ProcessingError',
    'TransformationError',
    'AggregationError',
    'DataFetchError',
    'DataQualityError',
    
    # Database
    'DatabaseError',
    'ConnectionError',
    'QueryError',
    'TransactionError',
    'PoolExhaustedError',
    
    # External Services
    'ExternalServiceError',
    'APIError',
    'APIConnectionError',
    'APIRateLimitError',
    'APIAuthenticationError',
    'DataSourceException',
    
    # Trading
    'TradingError',
    'OrderExecutionError',
    'InsufficientFundsError',
    'PositionError',
    'RiskLimitExceededError',
    'BrokerError',
    
    # Model
    'ModelError',
    'ModelNotFoundError',
    'PredictionError',
    'TrainingError',
    'FeatureError',
    
    # Scanner
    'ScannerError',
    'AlertGenerationError',
    'QualificationError',
    
    # Events
    'EventError',
    'EventPublishError',
    'EventSubscriptionError',
    'EventHandlerError',
    
    # Monitoring
    'MonitoringError',
    'MetricsError',
    'AlertError',
    
    # System
    'SystemError',
    'ResourceExhaustedError',
    'CircuitBreakerOpenError',
    
    # Utility
    'migrate_exception'
]