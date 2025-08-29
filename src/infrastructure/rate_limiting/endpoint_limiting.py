"""
Per-endpoint rate limiting system for the AI Trading System.

Provides granular rate limiting controls for different API endpoints
with trading-specific configurations and priority handling.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

from .config import RateLimitRule, RateLimitTier, TimeWindow
from .enhanced_algorithms import (
    AdaptiveConfig,
    BackoffConfig,
    EnhancedRateLimitAlgorithm,
    create_enhanced_rate_limiter,
    create_trading_adaptive_config,
    create_trading_backoff_config,
)
from .exceptions import RateLimitConfigError

logger = logging.getLogger(__name__)


class EndpointPriority(Enum):
    """Priority levels for endpoints."""

    CRITICAL = "critical"  # Critical trading operations
    HIGH = "high"  # Important operations
    MEDIUM = "medium"  # Standard operations
    LOW = "low"  # Background operations
    MONITORING = "monitoring"  # Health/metrics endpoints


@dataclass
class EndpointConfig:
    """Configuration for a specific endpoint."""

    pattern: str | Pattern[str]  # URL pattern (regex)
    methods: list[str] = field(default_factory=lambda: ["*"])  # HTTP methods (* = all)
    priority: EndpointPriority = EndpointPriority.MEDIUM
    rules: dict[RateLimitTier, RateLimitRule] = field(default_factory=dict)
    backoff_config: BackoffConfig | None = None
    adaptive_config: AdaptiveConfig | None = None
    bypass_global_limits: bool = False
    require_authentication: bool = True
    custom_headers: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        """Compile regex pattern if string provided."""
        if isinstance(self.pattern, str):
            try:
                self.pattern = re.compile(self.pattern)
            except re.error as e:
                raise RateLimitConfigError(f"Invalid regex pattern: {self.pattern} - {e}")


class EndpointRateLimiter:
    """
    Per-endpoint rate limiter with priority handling and adaptive features.
    """

    def __init__(self) -> None:
        self.endpoint_configs: list[EndpointConfig] = []
        self.limiters: dict[str, dict[str, EnhancedRateLimitAlgorithm]] = {}
        self._compiled_patterns: list[tuple[Pattern[str], EndpointConfig]] = []

    def add_endpoint_config(self, config: EndpointConfig) -> None:
        """Add endpoint configuration."""
        self.endpoint_configs.append(config)

        # Compile pattern if needed
        pattern = (
            config.pattern if isinstance(config.pattern, Pattern) else re.compile(config.pattern)
        )
        self._compiled_patterns.append((pattern, config))

        # Initialize limiters for this endpoint
        self._initialize_endpoint_limiters(config)

        logger.info(f"Added rate limiting for endpoint: {config.pattern} ({config.priority.value})")

    def _initialize_endpoint_limiters(self, config: EndpointConfig) -> None:
        """Initialize rate limiters for an endpoint configuration."""
        endpoint_key = str(
            config.pattern.pattern if hasattr(config.pattern, "pattern") else config.pattern
        )

        if endpoint_key not in self.limiters:
            self.limiters[endpoint_key] = {}

        # Create limiters for each tier
        for tier, rule in config.rules.items():
            limiter_key = f"{tier.value}"
            self.limiters[endpoint_key][limiter_key] = create_enhanced_rate_limiter(
                rule=rule,
                backoff_config=config.backoff_config,
                adaptive_config=config.adaptive_config,
            )

    def find_endpoint_config(self, path: str, method: str = "GET") -> EndpointConfig | None:
        """Find matching endpoint configuration for path and method."""
        for pattern, config in self._compiled_patterns:
            if pattern.match(path):
                # Check method match
                if "*" in config.methods or method.upper() in [m.upper() for m in config.methods]:
                    return config

        return None

    def check_endpoint_rate_limit(
        self,
        path: str,
        method: str,
        user_tier: RateLimitTier,
        identifier: str,
        tokens: int = 1,
    ) -> dict[str, Any]:
        """
        Check rate limit for specific endpoint.

        Returns:
            Dict with rate limit result and metadata
        """
        # Find matching endpoint config
        config = self.find_endpoint_config(path, method)

        if not config:
            return {
                "allowed": True,
                "endpoint_matched": False,
                "reason": "No rate limiting configured for endpoint",
            }

        # Check if user tier has specific rule for this endpoint
        if user_tier not in config.rules:
            # Try to fall back to lower tiers
            fallback_tiers = [
                RateLimitTier.BASIC,
                RateLimitTier.PREMIUM,
                RateLimitTier.ENTERPRISE,
            ]

            user_tier_found = None
            for tier in fallback_tiers:
                if tier in config.rules:
                    user_tier_found = tier
                    break

            if not user_tier_found:
                return {
                    "allowed": True,
                    "endpoint_matched": True,
                    "reason": "No rate limit rule for user tier",
                    "config": config,
                }

            user_tier = user_tier_found

        # Get the appropriate limiter
        endpoint_key = str(
            config.pattern.pattern if hasattr(config.pattern, "pattern") else config.pattern
        )
        limiter_key = f"{user_tier.value}"

        if endpoint_key not in self.limiters or limiter_key not in self.limiters[endpoint_key]:
            return {
                "allowed": True,
                "endpoint_matched": True,
                "reason": "Rate limiter not initialized",
                "config": config,
            }

        limiter = self.limiters[endpoint_key][limiter_key]

        try:
            # Perform rate limit check
            result = limiter.check_rate_limit(identifier, tokens)

            return {
                "allowed": result.allowed,
                "endpoint_matched": True,
                "config": config,
                "limit": result.limit,
                "remaining": result.remaining,
                "current_count": result.current_count,
                "reset_time": result.reset_time,
                "retry_after": result.retry_after,
                "user_tier": user_tier.value,
                "priority": config.priority.value,
                "custom_headers": config.custom_headers,
            }

        except Exception as e:
            logger.error(f"Error checking endpoint rate limit: {e}")
            return {
                "allowed": True,  # Fail open for availability
                "endpoint_matched": True,
                "error": str(e),
                "config": config,
            }

    def get_endpoint_status(self, path: str, method: str, identifier: str) -> dict[str, Any]:
        """Get current status for endpoint and identifier."""
        config = self.find_endpoint_config(path, method)

        if not config:
            return {"error": "Endpoint not found"}

        endpoint_key = str(
            config.pattern.pattern if hasattr(config.pattern, "pattern") else config.pattern
        )
        status = {}

        for tier_str, limiter in self.limiters.get(endpoint_key, {}).items():
            try:
                current_count, limit = limiter.get_current_usage(identifier)
                status[tier_str] = {
                    "current_count": current_count,
                    "limit": limit,
                    "remaining": limit - current_count,
                }
            except Exception as e:
                status[tier_str] = {
                    "current_count": 0,
                    "limit": 0,
                    "remaining": 0,
                    "error": str(e),  # type: ignore[dict-item]
                }

        return {
            "endpoint": endpoint_key,
            "priority": config.priority.value,
            "tiers": status,
        }

    def cleanup_expired(self) -> dict[str, int]:
        """Clean up expired rate limit entries for all endpoints."""
        cleanup_results = {}

        for endpoint_key, tier_limiters in self.limiters.items():
            endpoint_cleaned = 0
            for limiter_key, limiter in tier_limiters.items():
                try:
                    cleaned = limiter.cleanup_expired()
                    endpoint_cleaned += cleaned
                except Exception as e:
                    logger.error(f"Error cleaning up {endpoint_key}:{limiter_key}: {e}")

            if endpoint_cleaned > 0:
                cleanup_results[endpoint_key] = endpoint_cleaned

        return cleanup_results


def create_trading_endpoint_configs() -> list[EndpointConfig]:
    """Create standard endpoint configurations for trading system."""
    configs = []

    # Critical trading endpoints
    configs.extend(
        [
            EndpointConfig(
                pattern=r"/api/v1/orders/?$",
                methods=["POST"],
                priority=EndpointPriority.CRITICAL,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=10,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=50,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=200,
                        window=TimeWindow("1m"),
                    ),
                },
                backoff_config=create_trading_backoff_config(),
                adaptive_config=create_trading_adaptive_config(),
                description="Order submission",
            ),
            EndpointConfig(
                pattern=r"/api/v1/orders/[^/]+/cancel/?$",
                methods=["POST", "DELETE"],
                priority=EndpointPriority.CRITICAL,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=20,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=100,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=500,
                        window=TimeWindow("1m"),
                    ),
                },
                backoff_config=create_trading_backoff_config(),
                description="Order cancellation",
            ),
        ]
    )

    # High priority endpoints
    configs.extend(
        [
            EndpointConfig(
                pattern=r"/api/v1/portfolio/?$",
                methods=["GET"],
                priority=EndpointPriority.HIGH,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=60,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=300,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=1000,
                        window=TimeWindow("1m"),
                    ),
                },
                adaptive_config=create_trading_adaptive_config(),
                description="Portfolio queries",
            ),
            EndpointConfig(
                pattern=r"/api/v1/market-data/.*",
                methods=["GET"],
                priority=EndpointPriority.HIGH,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=100,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=500,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=2000,
                        window=TimeWindow("1m"),
                    ),
                },
                description="Market data requests",
            ),
        ]
    )

    # Medium priority endpoints
    configs.extend(
        [
            EndpointConfig(
                pattern=r"/api/v1/positions/?.*",
                methods=["GET"],
                priority=EndpointPriority.MEDIUM,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=120,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=600,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=2000,
                        window=TimeWindow("1m"),
                    ),
                },
                description="Position queries",
            ),
            EndpointConfig(
                pattern=r"/api/v1/risk/.*",
                methods=["GET", "POST"],
                priority=EndpointPriority.MEDIUM,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=30,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=150,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=500,
                        window=TimeWindow("1m"),
                    ),
                },
                description="Risk calculations",
            ),
        ]
    )

    # Low priority endpoints
    configs.extend(
        [
            EndpointConfig(
                pattern=r"/api/v1/analytics/.*",
                methods=["GET"],
                priority=EndpointPriority.LOW,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=20,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=100,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=300,
                        window=TimeWindow("1m"),
                    ),
                },
                description="Analytics and reporting",
            ),
            EndpointConfig(
                pattern=r"/api/v1/history/.*",
                methods=["GET"],
                priority=EndpointPriority.LOW,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=10,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.PREMIUM: RateLimitRule(
                        limit=50,
                        window=TimeWindow("1m"),
                    ),
                    RateLimitTier.ENTERPRISE: RateLimitRule(
                        limit=200,
                        window=TimeWindow("1m"),
                    ),
                },
                description="Historical data",
            ),
        ]
    )

    # Monitoring endpoints (higher limits, no backoff)
    configs.extend(
        [
            EndpointConfig(
                pattern=r"/health/?$",
                methods=["GET"],
                priority=EndpointPriority.MONITORING,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=60,
                        window=TimeWindow("1m"),
                    ),
                },
                bypass_global_limits=True,
                require_authentication=False,
                description="Health check",
            ),
            EndpointConfig(
                pattern=r"/metrics/?$",
                methods=["GET"],
                priority=EndpointPriority.MONITORING,
                rules={
                    RateLimitTier.BASIC: RateLimitRule(
                        limit=30,
                        window=TimeWindow("1m"),
                    ),
                },
                bypass_global_limits=True,
                require_authentication=False,
                description="Metrics endpoint",
            ),
        ]
    )

    return configs


def create_development_endpoint_configs() -> list[EndpointConfig]:
    """Create more permissive endpoint configurations for development."""
    configs = create_trading_endpoint_configs()

    # Increase all limits by 10x for development
    for config in configs:
        for tier, rule in config.rules.items():
            config.rules[tier] = RateLimitRule(
                limit=rule.limit * 10,
                window=rule.window,
                algorithm=rule.algorithm,
                burst_allowance=rule.burst_allowance,
                action=rule.action,
            )

    return configs
