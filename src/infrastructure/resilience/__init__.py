"""
Resilience Infrastructure Package

Provides production-grade resilience patterns including:
- Circuit breakers for external service calls
- Retry mechanisms with exponential backoff
- Fallback strategies and graceful degradation
- Health monitoring and diagnostics
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .fallback import CacheFirstStrategy, FallbackStrategy, TimeoutStrategy
from .health import HealthChecker, HealthStatus, ServiceHealth
from .retry import ExponentialBackoff, RetryConfig, retry_with_backoff

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "RetryConfig",
    "ExponentialBackoff",
    "retry_with_backoff",
    "FallbackStrategy",
    "CacheFirstStrategy",
    "TimeoutStrategy",
    "HealthChecker",
    "HealthStatus",
    "ServiceHealth",
]
