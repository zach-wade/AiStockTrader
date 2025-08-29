"""
Rate limiting system for AI Trading System.

Provides comprehensive rate limiting capabilities including:
- Multiple algorithms (Token Bucket, Sliding Window, Fixed Window)
- Multi-tier rate limiting (per user, API key, IP, global)
- Redis-backed distributed rate limiting
- Trading-specific rate limits
- Middleware integration
"""

from .algorithms import FixedWindowRateLimit, SlidingWindowRateLimit, TokenBucketRateLimit
from .config import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitRule,
    RateLimitTier,
    TimeWindow,
    TradingRateLimits,
)
from .decorators import (
    api_rate_limit,
    initialize_rate_limiting,
    ip_rate_limit,
    rate_limit,
    trading_rate_limit,
)
from .endpoint_limiting import (
    EndpointConfig,
    EndpointPriority,
    EndpointRateLimiter,
    create_development_endpoint_configs,
    create_trading_endpoint_configs,
)
from .enhanced_algorithms import (
    AdaptiveConfig,
    BackoffConfig,
    BackoffStrategy,
    EnhancedRateLimitAlgorithm,
    EnhancedSlidingWindow,
    EnhancedTokenBucket,
    create_enhanced_rate_limiter,
    create_trading_adaptive_config,
    create_trading_backoff_config,
)
from .exceptions import (
    APIRateLimitExceeded,
    IPRateLimitExceeded,
    RateLimitConfigError,
    RateLimitExceeded,
    RateLimitStorageError,
    TradingRateLimitExceeded,
)
from .manager import RateLimitContext, RateLimitManager, RateLimitStatus
from .middleware import RateLimitMiddleware
from .storage import MemoryRateLimitStorage, RateLimitStorage, RedisRateLimitStorage

__all__ = [
    # Core algorithms
    "TokenBucketRateLimit",
    "SlidingWindowRateLimit",
    "FixedWindowRateLimit",
    # Enhanced algorithms
    "EnhancedRateLimitAlgorithm",
    "EnhancedTokenBucket",
    "EnhancedSlidingWindow",
    "BackoffConfig",
    "BackoffStrategy",
    "AdaptiveConfig",
    "create_enhanced_rate_limiter",
    "create_trading_backoff_config",
    "create_trading_adaptive_config",
    # Endpoint-specific limiting
    "EndpointConfig",
    "EndpointPriority",
    "EndpointRateLimiter",
    "create_trading_endpoint_configs",
    "create_development_endpoint_configs",
    # Configuration
    "RateLimitConfig",
    "RateLimitTier",
    "RateLimitRule",
    "TimeWindow",
    "RateLimitAlgorithm",
    "TradingRateLimits",
    # Exceptions
    "RateLimitExceeded",
    "RateLimitConfigError",
    "RateLimitStorageError",
    "TradingRateLimitExceeded",
    "APIRateLimitExceeded",
    "IPRateLimitExceeded",
    # Management
    "RateLimitManager",
    "RateLimitContext",
    "RateLimitStatus",
    # Integration
    "rate_limit",
    "trading_rate_limit",
    "api_rate_limit",
    "ip_rate_limit",
    "initialize_rate_limiting",
    "RateLimitMiddleware",
    # Storage
    "RateLimitStorage",
    "RedisRateLimitStorage",
    "MemoryRateLimitStorage",
]
