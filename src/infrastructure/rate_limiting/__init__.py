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
