"""
Custom exception types for the AI Trader system.

This module defines specific exception types to replace generic Exception handling,
providing better error diagnostics and handling.
"""


class AITraderException(Exception):
    """Base exception for all AI Trader errors."""

    pass


# Data Pipeline Exceptions
class DataPipelineException(AITraderException):
    """Base exception for data pipeline errors."""

    pass


class DataSourceException(DataPipelineException):
    """Raised when a data source fails."""

    pass


class APIConnectionError(DataSourceException):
    """Raised when API connection fails."""

    pass


class APIRateLimitError(DataSourceException):
    """Raised when API rate limit is exceeded."""

    pass


class APIAuthenticationError(DataSourceException):
    """Raised when API authentication fails."""

    pass


class DataValidationError(DataPipelineException):
    """Raised when data validation fails."""

    pass


class DataStorageError(DataPipelineException):
    """Raised when data storage operations fail."""

    pass


# Database Exceptions
class DatabaseException(AITraderException):
    """Base exception for database errors."""

    pass


class DatabaseConnectionError(DatabaseException):
    """Raised when database connection fails."""

    pass


class DatabaseQueryError(DatabaseException):
    """Raised when database query fails."""

    pass


class DatabaseIntegrityError(DatabaseException):
    """Raised when database integrity constraints are violated."""

    pass


# Cache Exceptions
class CacheException(AITraderException):
    """Base exception for cache errors."""

    pass


class CacheConnectionError(CacheException):
    """Raised when cache connection fails."""

    pass


class CacheSerializationError(CacheException):
    """Raised when cache serialization/deserialization fails."""

    pass


# Feature Engineering Exceptions
class FeatureEngineeringException(AITraderException):
    """Base exception for feature engineering errors."""

    pass


class FeatureCalculationError(FeatureEngineeringException):
    """Raised when feature calculation fails."""

    pass


class InsufficientDataError(FeatureEngineeringException):
    """Raised when insufficient data for feature calculation."""

    pass


# Model Training Exceptions
class ModelTrainingException(AITraderException):
    """Base exception for model training errors."""

    pass


class ModelConfigError(ModelTrainingException):
    """Raised when model configuration is invalid."""

    pass


class TrainingDataError(ModelTrainingException):
    """Raised when training data is invalid or insufficient."""

    pass


# Trading Exceptions
class TradingException(AITraderException):
    """Base exception for trading errors."""

    pass


class OrderExecutionError(TradingException):
    """Raised when order execution fails."""

    pass


class RiskLimitExceededError(TradingException):
    """Raised when risk limits are exceeded."""

    pass


class BrokerConnectionError(TradingException):
    """Raised when broker connection fails."""

    pass


# Configuration Exceptions
class ConfigurationError(AITraderException):
    """Raised when configuration is invalid."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


# Utility function to convert generic exceptions
def convert_exception(e: Exception, context: str = "") -> AITraderException:
    """
    Convert a generic exception to a specific AI Trader exception based on context.

    Args:
        e: The original exception
        context: Context string to help determine the specific exception type

    Returns:
        A specific AI Trader exception
    """
    error_msg = str(e)

    # API-related errors
    if "rate limit" in error_msg.lower():
        return APIRateLimitError(f"{context}: {error_msg}")
    elif "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
        return APIAuthenticationError(f"{context}: {error_msg}")
    elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        if "database" in context.lower():
            return DatabaseConnectionError(f"{context}: {error_msg}")
        elif "cache" in context.lower():
            return CacheConnectionError(f"{context}: {error_msg}")
        else:
            return APIConnectionError(f"{context}: {error_msg}")

    # Database errors
    elif "duplicate key" in error_msg.lower() or "unique constraint" in error_msg.lower():
        return DatabaseIntegrityError(f"{context}: {error_msg}")
    elif "sql" in error_msg.lower() or "query" in error_msg.lower():
        return DatabaseQueryError(f"{context}: {error_msg}")

    # Data errors
    elif "validation" in error_msg.lower() or "invalid data" in error_msg.lower():
        return DataValidationError(f"{context}: {error_msg}")
    elif "insufficient data" in error_msg.lower() or "not enough" in error_msg.lower():
        return InsufficientDataError(f"{context}: {error_msg}")

    # Default to base exception with context
    return AITraderException(f"{context}: {error_msg}")
