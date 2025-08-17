"""Resilience utilities package."""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerManager,
    CircuitBreakerState,
    circuit_breaker,
    circuit_breaker_call,
    get_circuit_breaker,
    get_global_circuit_breaker_manager,
)
from .error_recovery import (
    API_RETRY_CONFIG,
    DATABASE_RETRY_CONFIG,
    NETWORK_RETRY_CONFIG,
    BulkRetryManager,
    ErrorRecoveryManager,
    RecoveryAction,
    RetryConfig,
    RetryExhaustedError,
    RetryStrategy,
    get_global_recovery_manager,
    retry,
    retry_call,
)
from .strategies import (
    ResilienceConfig,
    ResilienceStrategies,
    ResilienceStrategiesFactory,
    create_resilience_strategies,
    get_resilience_strategies_factory,
)

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "CircuitBreakerState",
    "CircuitBreakerError",
    "circuit_breaker",
    "get_circuit_breaker",
    "circuit_breaker_call",
    "get_global_circuit_breaker_manager",
    # Error recovery
    "ErrorRecoveryManager",
    "RetryConfig",
    "RetryStrategy",
    "RecoveryAction",
    "RetryExhaustedError",
    "BulkRetryManager",
    "retry",
    "retry_call",
    "get_global_recovery_manager",
    "NETWORK_RETRY_CONFIG",
    "DATABASE_RETRY_CONFIG",
    "API_RETRY_CONFIG",
    # Resilience strategies
    "ResilienceStrategies",
    "ResilienceConfig",
    "ResilienceStrategiesFactory",
    "get_resilience_strategies_factory",
    "create_resilience_strategies",
]
