"""
Core utilities for the AI Trader system.

This module provides a unified interface to all core utilities including:
- Async helpers for concurrent processing
- Exception types for proper error handling
- Time helpers for market-aware datetime operations
- File helpers for safe file operations
- Secure random number generation
- Secure serialization
- Error handling and circuit breaker patterns
- Logging configuration

This is the main interface module that imports all core utilities from the 
core/ subdirectory for easy access throughout the system.
"""

# Import and explicitly re-export core utilities
from .core import (
    # Async helpers
    RateLimiter,
    process_in_batches,
    run_in_executor,
    AsyncCircuitBreaker,
    timeout_coro,
    async_retry,
    
    # Exception types
    AITraderException,
    DataPipelineException,
    DataSourceException,
    APIConnectionError,
    APIRateLimitError,
    DatabaseException,
    TradingException,
    OrderExecutionError,
    ConfigurationError,
    
    # Time helpers
    ensure_utc,
    get_last_us_trading_day,
    is_market_open,
    get_market_hours,
    is_trading_day,
    get_next_trading_day,
    
    # File helpers
    load_yaml_config,
    ensure_directory_exists,
    safe_delete_file,
    read_json_file,
    write_json_file,
    get_file_size,
    
    # Security
    SecureRandom,
    secure_serializer,
    secure_dumps,
    secure_loads,
    
    # Error handling
    ErrorHandlingMixin,
    CircuitBreaker,
    CircuitBreakerConfig,
    
    # Logging
    get_logger,
    setup_logging,
    ColoredFormatter,
    
    # Text helpers
    calculate_text_similarity,
    normalize_text_for_comparison,
    tokenize_text,
    calculate_jaccard_similarity,
    calculate_levenshtein_distance,
    normalize_levenshtein_distance
)

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"