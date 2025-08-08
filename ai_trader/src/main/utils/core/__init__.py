"""Core utilities package."""

from .async_helpers import (
    process_in_batches,
    run_in_executor,
    run_cpu_bound_task,
    gather_with_exceptions,
    timeout_coro,
    RateLimiter,
    chunk_async_generator,
    AsyncCircuitBreaker,
    async_lru_cache,
    cleanup_executors,
    async_retry
)

from .exception_types import (
    AITraderException,
    DataPipelineException,
    DataSourceException,
    APIConnectionError,
    APIRateLimitError,
    APIAuthenticationError,
    DataValidationError,
    DataStorageError,
    DatabaseException,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseIntegrityError,
    CacheException,
    CacheConnectionError,
    CacheSerializationError,
    FeatureEngineeringException,
    FeatureCalculationError,
    InsufficientDataError,
    ModelTrainingException,
    ModelConfigError,
    TrainingDataError,
    TradingException,
    OrderExecutionError,
    RiskLimitExceededError,
    BrokerConnectionError,
    ConfigurationError,
    MissingConfigError,
    convert_exception
)

from .time_helpers import (
    ensure_utc,
    get_last_us_trading_day,
    is_market_open,
    get_market_hours,
    get_trading_days_between,
    is_trading_day,
    get_next_trading_day
)

from .file_helpers import (
    load_yaml_config,
    ensure_directory_exists,
    safe_delete_file,
    read_json_file,
    write_json_file,
    safe_json_write,
    get_file_size,
    get_file_size_human,
    clean_old_files,
    copy_with_backup,
    list_files,
    read_file_chunks
)

from .secure_random import (
    SecureRandom,
    secure_uniform,
    secure_normal,
    secure_randint,
    secure_choice,
    secure_sample,
    secure_shuffle,
    secure_numpy_uniform,
    secure_numpy_normal,
    get_secure_random,
    SecureRandomMigrationHelper,
    migration_helper,
    secure_replace_random_uniform,
    secure_replace_numpy_uniform,
    secure_replace_numpy_normal
)

from .secure_serializer import (
    SecurityError,
    SecureSerializer,
    secure_serializer,
    secure_dumps,
    secure_loads,
    migrate_unsafe_pickle
)

from .error_handling import (
    ErrorHandlingMixin,
    CircuitBreaker,
    CircuitBreakerConfig
)

from .logging import (
    ColoredFormatter,
    JsonFormatter,
    get_logger,
    setup_logging
)

from .text_helpers import (
    calculate_text_similarity,
    normalize_text_for_comparison,
    tokenize_text,
    calculate_jaccard_similarity,
    calculate_levenshtein_distance,
    normalize_levenshtein_distance,
    find_common_phrases,
    extract_key_phrases
)

from .json_helpers import (
    EventJSONEncoder,
    to_json,
    from_json,
    parse_iso_datetime,
    dict_to_dataclass
)

# Timer context manager for performance monitoring
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str = "Operation"):
    """Simple timer context manager for measuring execution time."""
    class TimerResult:
        def __init__(self):
            self.start = time.time()
            self.elapsed = 0
            
    result = TimerResult()
    try:
        yield result
    finally:
        result.elapsed = time.time() - result.start
        print(f"{name} took {result.elapsed:.4f} seconds")

# Simple event tracker placeholder
class SimpleEventTracker:
    """Simple event tracker placeholder."""
    def __init__(self, name: str):
        self.name = name
    
    def track_event(self, event_name: str, **kwargs):
        """Track an event."""
        pass
    
    def track_metric(self, metric_name: str, value: float, **kwargs):
        """Track a metric."""
        pass

def create_event_tracker(name: str) -> SimpleEventTracker:
    """Create a simple event tracker."""
    return SimpleEventTracker(name)

# Validation result placeholder
from dataclasses import dataclass
from typing import List, Any

@dataclass
class ValidationResult:
    """Simple validation result."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

# Task management utilities
import asyncio

async def create_task_safely(coro, name: str = None) -> asyncio.Task:
    """Create an asyncio task safely."""
    return asyncio.create_task(coro, name=name)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

__all__ = [
    # Async helpers
    'process_in_batches',
    'run_in_executor',
    'run_cpu_bound_task',
    'gather_with_exceptions',
    'timeout_coro',
    'RateLimiter',
    'chunk_async_generator',
    'AsyncCircuitBreaker',
    'async_lru_cache',
    'cleanup_executors',
    'async_retry',
    
    # Exception types
    'AITraderException',
    'DataPipelineException',
    'DataSourceException',
    'APIConnectionError',
    'APIRateLimitError',
    'APIAuthenticationError',
    'DataValidationError',
    'DataStorageError',
    'DatabaseException',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'DatabaseIntegrityError',
    'CacheException',
    'CacheConnectionError',
    'CacheSerializationError',
    'FeatureEngineeringException',
    'FeatureCalculationError',
    'InsufficientDataError',
    'ModelTrainingException',
    'ModelConfigError',
    'TrainingDataError',
    'TradingException',
    'OrderExecutionError',
    'RiskLimitExceededError',
    'BrokerConnectionError',
    'ConfigurationError',
    'MissingConfigError',
    'convert_exception',
    
    # Time helpers
    'ensure_utc',
    'get_last_us_trading_day',
    'is_market_open',
    'get_market_hours',
    'get_trading_days_between',
    'is_trading_day',
    'get_next_trading_day',
    
    # File helpers
    'load_yaml_config',
    'ensure_directory_exists',
    'safe_delete_file',
    'read_json_file',
    'write_json_file',
    'safe_json_write',
    'get_file_size',
    'get_file_size_human',
    'clean_old_files',
    'copy_with_backup',
    'list_files',
    'read_file_chunks',
    
    # Secure random
    'SecureRandom',
    'secure_uniform',
    'secure_normal',
    'secure_randint',
    'secure_choice',
    'secure_sample',
    'secure_shuffle',
    'secure_numpy_uniform',
    'secure_numpy_normal',
    'get_secure_random',
    'SecureRandomMigrationHelper',
    'migration_helper',
    'secure_replace_random_uniform',
    'secure_replace_numpy_uniform',
    'secure_replace_numpy_normal',
    
    # Secure serializer
    'SecurityError',
    'SecureSerializer',
    'secure_serializer',
    'secure_dumps',
    'secure_loads',
    'migrate_unsafe_pickle',
    
    # Error handling
    'ErrorHandlingMixin',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    
    # Logging
    'ColoredFormatter',
    'JsonFormatter',
    'get_logger',
    'setup_logging',
    
    # Text helpers
    'calculate_text_similarity',
    'normalize_text_for_comparison',
    'tokenize_text',
    'calculate_jaccard_similarity',
    'calculate_levenshtein_distance',
    'normalize_levenshtein_distance',
    'find_common_phrases',
    'extract_key_phrases',
    
    # Timer
    'timer',
    
    # Event tracking
    'SimpleEventTracker',
    'create_event_tracker',
    
    # Validation
    'ValidationResult',
    
    # Task management
    'create_task_safely',
    'chunk_list'
]