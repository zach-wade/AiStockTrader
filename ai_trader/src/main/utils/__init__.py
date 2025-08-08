"""
AI Trader Utilities Package

Comprehensive utilities for the AI trader application, organized into focused subpackages.
"""

# Core utilities
from .core import (
    # Async utilities
    process_in_batches,
    run_in_executor,
    gather_with_exceptions,
    timeout_coro,
    RateLimiter,
    
    # Exception types
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
    convert_exception,
    
    # Time utilities
    ensure_utc,
    get_last_us_trading_day,
    is_market_open,
    get_market_hours,
    get_trading_days_between,
    is_trading_day,
    get_next_trading_day,
    
    # File utilities
    load_yaml_config,
    ensure_directory_exists,
    safe_delete_file,
    read_json_file,
    write_json_file,
    safe_json_write,
    
    # Secure utilities
    SecureRandom,
    secure_uniform,
    secure_normal,
    secure_randint,
    secure_choice,
    get_secure_random,
    secure_dumps,
    secure_loads,
    
    # Error handling
    ErrorHandlingMixin,
    
    # Logging
    ColoredFormatter,
    JsonFormatter,
    get_logger,
    setup_logging
)

# Configuration utilities
from .config import (
    # Configuration optimization
    ConfigOptimizer,
    OptimizationStrategy,
    ParameterType,
    ConfigParameter,
    ParameterConstraint,
    OptimizationTarget,
    OptimizationResult,
    get_global_optimizer,
    create_cache_parameters,
    create_database_parameters,
    create_network_parameters,
    create_performance_parameters,
    create_monitoring_parameters,
    
    # Configuration wrapper
    ConfigurationWrapper,
    ConfigFormat,
    ConfigSchema,
    ConfigValidationError,
    create_config_schema,
    create_basic_schema,
    create_database_schema,
    create_api_schema,
    load_config,
    get_global_config,
    set_global_config,
    init_global_config,
    ensure_global_config,
    reset_global_config,
    is_global_config_initialized,
    get_config_value,
    set_config_value,
    has_config_key
)

# Authentication utilities
from .auth import (
    CredentialValidator,
    CredentialType,
    ValidationResult,
    validate_credential,
    generate_secure_credential,
    get_global_validator
)

# Data utilities
from .data import (
    # Data processing
    DataProcessor,
    DataValidationRule,
    ValidationLevel,
    get_global_processor,
    
    # Data utilities
    chunk_list,
    dataframe_memory_usage,
    safe_divide,
    parse_numeric,
    format_number,
    calculate_percentage_change,
    hash_dataframe,
    compare_dataframes
)

# Event utilities
from .events import (
    # Event management
    CallbackPriority,
    EventStatus,
    Event,
    EventResult,
    
    # Callback system
    CallbackManager,
    CallbackMixin,
    callback,
    event_handler,
    auto_register_callbacks,
    
    # Convenience functions
    get_global_callback_manager,
    on,
    off,
    emit,
    emit_and_wait
)

# Monitoring utilities
from .monitoring import (
    # Performance monitoring
    PerformanceMonitor,
    MetricType,
    AlertLevel,
    PerformanceMetric,
    SystemResources,
    FunctionMetrics,
    Alert,
    
    # Components
    SystemMetricsCollector,
    # AlertManager,  # Removed to avoid circular dependency
    FunctionTracker,
    
    # Global monitoring
    get_global_monitor,
    set_global_monitor,
    reset_global_monitor,
    is_global_monitor_initialized,
    
    # Convenience functions
    record_metric,
    time_function,
    timer,
    start_monitoring,
    stop_monitoring,
    get_system_summary,
    get_function_summary,
    get_alerts_summary,
    set_default_thresholds,
    clear_metrics,
    export_metrics,
    
    # Memory monitoring
    MemoryMonitor,
    MemorySnapshot,
    MemoryThresholds,
    get_memory_monitor,
    memory_profiled
)

# Networking utilities
from .networking import (
    # WebSocket optimization
    OptimizedWebSocketClient,
    WebSocketManager,
    WebSocketConnection,
    MessageBuffer,
    FailoverManager,
    ConnectionPool,
    
    # Types
    ConnectionState,
    MessagePriority,
    LatencyMetrics,
    BufferConfig,
    ConnectionConfig,
    WebSocketMessage,
    ConnectionStats,
    
    # Global WebSocket management
    get_websocket_manager,
    create_optimized_websocket,
    websocket_context
)

# Resilience utilities
from .resilience import (
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerState,
    CircuitBreakerError,
    circuit_breaker,
    get_circuit_breaker,
    circuit_breaker_call,
    get_global_circuit_breaker_manager,
    
    # Error recovery
    ErrorRecoveryManager,
    RetryConfig,
    RetryStrategy,
    RecoveryAction,
    RetryExhaustedError,
    BulkRetryManager,
    retry,
    retry_call,
    get_global_recovery_manager,
    NETWORK_RETRY_CONFIG,
    DATABASE_RETRY_CONFIG,
    API_RETRY_CONFIG
)

# State utilities
from .state import (
    # State management
    StateManager,
    StorageBackend,
    SerializationFormat,
    MemoryBackend,
    FileBackend,
    RedisBackend,
    
    # Persistence
    StatePersistence,
    
    # Context management
    StateContext,
    
    # Global state management
    get_state_manager
)

# Trading utilities
from .trading import (
    # Universe management
    UniverseManager,
    UniverseType,
    FilterCriteria,
    Filter,
    UniverseConfig,
    UniverseSnapshot,
    UniverseAnalyzer,
    UniverseImportExport,
    
    # Filter utilities
    create_market_cap_filter,
    create_volume_filter,
    create_sector_filter,
    create_exchange_filter,
    create_price_range_filter,
    create_liquidity_filter,
    create_volatility_filter,
    create_beta_filter,
    create_dividend_yield_filter,
    create_pe_ratio_filter,
    create_industry_filter,
    create_country_filter,
    create_large_cap_filters,
    create_high_volume_filters,
    create_growth_filters,
    create_dividend_filters,
    create_value_filters,
    
    # Global universe management
    get_global_manager,
    set_global_manager,
    init_global_manager,
    ensure_global_manager,
    reset_global_manager,
    is_global_manager_initialized
)

# Cache utilities
from .cache import (
    # Cache types
    CacheType,
    CacheTier,
    CompressionType,
    
    # Cache models
    CacheEntry,
    CacheStats,
    
    # Cache backends
    CacheBackend,
    MemoryBackend,
    RedisBackend,
    
    # Key generation
    CacheKeyGenerator,
    get_key_generator,
    set_key_generator,
    generate_quotes_key,
    generate_ohlcv_key,
    generate_features_key,
    generate_custom_key,
    
    # Metrics
    CacheMetricsService,
    
    # Compression
    CompressionService,
    
    # Background Tasks
    BackgroundTasksService
)

# Database utilities
from .database import (
    DatabasePool,
    ConnectionPoolMetrics,
    ConnectionHealthStatus,
    PoolHealthMonitor,
    QueryTracker,
    QueryPerformanceTracker,
    QueryType,
    QueryPriority,
    track_query,
    get_global_tracker
)

# Processing utilities
from .processing import (
    StreamingConfig,
    ProcessingStats,
    StreamingAggregator,
    ProcessingUtils
)

# Factory utilities
# from .factories import (
#     make_data_fetcher,
#     UtilityManager
# )

# API utilities
from .api import (
    managed_aiohttp_session,
    managed_httpx_client
)

# Market data utilities
from .market_data import (
    MarketDataCache,
    UniverseLoader
)

# Application utilities
from .app import (
    # Context management
    StandardAppContext,
    AppContextError,
    create_app_context,
    managed_app_context,
    
    # CLI utilities
    create_cli_app,
    CLIAppConfig,
    StandardCLIHandler,
    create_data_pipeline_app,
    create_training_app,
    create_validation_app,
    async_command,
    success_message,
    error_message,
    info_message,
    warning_message,
    
    # Workflow management removed - using HistoricalManager directly
    
    # Validation utilities
    AppConfigValidator,
    ValidationResult,
    ConfigValidationError,
    validate_trading_config,
    validate_data_pipeline_config,
    ensure_critical_config,
    validate_app_startup_config
)

# App factory utilities
from .app_factory import (
    create_event_driven_app,
    run_event_driven_app
)

# Export all main classes and functions
__all__ = [
    # Core utilities
    'process_in_batches', 'run_in_executor', 'gather_with_exceptions', 'timeout_coro',
    'RateLimiter',
    'AITraderException', 'DataPipelineException', 'DataSourceException', 'APIConnectionError',
    'APIRateLimitError', 'APIAuthenticationError', 'DataValidationError', 'DataStorageError',
    'DatabaseException', 'DatabaseConnectionError', 'DatabaseQueryError', 'DatabaseIntegrityError',
    'CacheException', 'CacheConnectionError', 'CacheSerializationError', 'FeatureEngineeringException',
    'FeatureCalculationError', 'InsufficientDataError', 'ModelTrainingException', 'ModelConfigError',
    'TrainingDataError', 'TradingException', 'OrderExecutionError', 'RiskLimitExceededError',
    'BrokerConnectionError', 'ConfigurationError', 'MissingConfigError', 'convert_exception',
    'ensure_utc', 'get_last_us_trading_day', 'is_market_open', 'get_market_hours',
    'get_trading_days_between', 'is_trading_day', 'get_next_trading_day',
    'load_yaml_config', 'ensure_directory_exists', 'safe_delete_file', 'read_json_file',
    'write_json_file', 'safe_json_write',
    'SecureRandom', 'secure_uniform', 'secure_normal', 'secure_randint', 'secure_choice',
    'get_secure_random', 'secure_dumps', 'secure_loads',
    'ErrorHandlingMixin',
    'ColoredFormatter', 'JsonFormatter', 'get_logger', 'setup_logging',
    
    # Configuration utilities
    'ConfigOptimizer', 'OptimizationStrategy', 'ParameterType', 'ConfigParameter',
    'ParameterConstraint', 'OptimizationTarget', 'OptimizationResult', 'get_global_optimizer',
    'create_cache_parameters', 'create_database_parameters', 'create_network_parameters',
    'create_performance_parameters', 'create_monitoring_parameters',
    'ConfigurationWrapper', 'ConfigFormat', 'ConfigSchema', 'ConfigValidationError',
    'create_config_schema', 'create_basic_schema', 'create_database_schema', 'create_api_schema',
    'load_config', 'get_global_config', 'set_global_config', 'init_global_config',
    'ensure_global_config', 'reset_global_config', 'is_global_config_initialized',
    'get_config_value', 'set_config_value', 'has_config_key',
    
    # Authentication utilities
    'CredentialValidator', 'CredentialType', 'ValidationResult',
    'validate_credential', 'generate_secure_credential', 'get_global_validator',
    
    # Data utilities
    'DataProcessor', 'DataValidationRule', 'ValidationLevel', 'get_global_processor',
    'chunk_list', 'dataframe_memory_usage', 'safe_divide', 'parse_numeric', 'format_number',
    'calculate_percentage_change', 'hash_dataframe', 'compare_dataframes',
    
    # Event utilities
    'CallbackPriority', 'EventStatus', 'Event', 'EventResult', 'CallbackManager', 'CallbackMixin', 
    'callback', 'event_handler', 'auto_register_callbacks', 'get_global_callback_manager', 'on', 
    'off', 'emit', 'emit_and_wait',
    
    # Monitoring utilities
    'PerformanceMonitor', 'MetricType', 'AlertLevel', 'PerformanceMetric', 'SystemResources',
    'FunctionMetrics', 'Alert', 'SystemMetricsCollector', 'FunctionTracker',
    'get_global_monitor', 'set_global_monitor', 'reset_global_monitor', 'is_global_monitor_initialized',
    'record_metric', 'time_function', 'timer', 'start_monitoring', 'stop_monitoring',
    'get_system_summary', 'get_function_summary', 'get_alerts_summary', 'set_default_thresholds',
    'clear_metrics', 'export_metrics',
    'MemoryMonitor', 'MemorySnapshot', 'MemoryThresholds', 'get_memory_monitor', 'memory_profiled',
    
    # Networking utilities
    'OptimizedWebSocketClient', 'WebSocketManager', 'WebSocketConnection', 'MessageBuffer',
    'FailoverManager', 'ConnectionPool', 'ConnectionState', 'MessagePriority', 'LatencyMetrics',
    'BufferConfig', 'ConnectionConfig', 'WebSocketMessage', 'ConnectionStats',
    'get_websocket_manager', 'create_optimized_websocket', 'websocket_context',
    
    # Resilience utilities
    'CircuitBreaker', 'CircuitBreakerConfig', 'CircuitBreakerManager', 'CircuitBreakerState',
    'CircuitBreakerError', 'circuit_breaker', 'get_circuit_breaker', 'circuit_breaker_call',
    'get_global_circuit_breaker_manager', 'ErrorRecoveryManager', 'RetryConfig', 'RetryStrategy',
    'RecoveryAction', 'RetryExhaustedError', 'BulkRetryManager', 'retry', 'retry_call',
    'get_global_recovery_manager', 'NETWORK_RETRY_CONFIG', 'DATABASE_RETRY_CONFIG', 'API_RETRY_CONFIG',
    
    # State utilities
    'StateManager', 'StorageBackend', 'SerializationFormat', 'MemoryBackend',
    'FileBackend', 'RedisBackend', 'StatePersistence', 'StateContext', 'get_state_manager',
    
    # Trading utilities
    'UniverseManager', 'UniverseType', 'FilterCriteria', 'Filter', 'UniverseConfig',
    'UniverseSnapshot', 'UniverseAnalyzer', 'UniverseImportExport',
    'create_market_cap_filter', 'create_volume_filter', 'create_sector_filter',
    'create_exchange_filter', 'create_price_range_filter', 'create_liquidity_filter',
    'create_volatility_filter', 'create_beta_filter', 'create_dividend_yield_filter',
    'create_pe_ratio_filter', 'create_industry_filter', 'create_country_filter',
    'create_large_cap_filters', 'create_high_volume_filters', 'create_growth_filters',
    'create_dividend_filters', 'create_value_filters', 'get_global_manager',
    'set_global_manager', 'init_global_manager', 'ensure_global_manager',
    'reset_global_manager', 'is_global_manager_initialized',
    
    # Cache utilities
    'CacheType', 'CacheTier', 'CompressionType', 'CacheEntry', 'CacheStats',
    'CacheBackend', 'MemoryBackend', 'RedisBackend', 'CacheKeyGenerator',
    'get_key_generator', 'set_key_generator', 'generate_quotes_key',
    'generate_ohlcv_key', 'generate_features_key', 'generate_custom_key',
    'CacheMetricsService', 'CompressionService', 'BackgroundTasksService',
    
    # Database utilities
    'DatabasePool', 'ConnectionPoolMetrics', 'ConnectionHealthStatus', 'PoolHealthMonitor',
    'QueryTracker', 'QueryPerformanceTracker', 'QueryType', 'QueryPriority', 'track_query', 'get_global_tracker',
    
    # Processing utilities
    'StreamingConfig', 'ProcessingStats', 'StreamingAggregator', 'ProcessingUtils',
    
    # Factory utilities
    # 'make_data_fetcher', 'UtilityManager',
    
    # API utilities
    'managed_aiohttp_session', 'managed_httpx_client',
    
    # Market data utilities
    'MarketDataCache', 'UniverseLoader',
    
    # Application utilities
    'StandardAppContext', 'AppContextError', 'create_app_context', 'managed_app_context',
    'create_cli_app', 'CLIAppConfig', 'StandardCLIHandler', 'create_data_pipeline_app',
    'create_training_app', 'create_validation_app', 'async_command',
    'success_message', 'error_message', 'info_message', 'warning_message',
    'AppConfigValidator', 'ValidationResult', 'ConfigValidationError',
    'validate_trading_config', 'validate_data_pipeline_config', 'ensure_critical_config',
    'validate_app_startup_config',
    
    # App factory utilities
    'create_event_driven_app', 'run_event_driven_app'
]