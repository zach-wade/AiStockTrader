"""
Configuration management for rate limiting system.

Provides comprehensive configuration options for different rate limiting scenarios.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any


class RateLimitAlgorithm(Enum):
    """Available rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


class RateLimitTier(Enum):
    """Rate limit tiers for different priority levels."""

    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"
    SYSTEM = "system"


class RateLimitAction(Enum):
    """Actions to take when rate limit is exceeded."""

    BLOCK = "block"  # Reject the request
    QUEUE = "queue"  # Queue the request
    THROTTLE = "throttle"  # Slow down the request
    LOG_ONLY = "log_only"  # Log but allow the request


class TimeWindow:
    """Time window specification for rate limits."""

    def __init__(self, value: str | int | timedelta) -> None:
        if isinstance(value, str):
            self._parse_string(value)
        elif isinstance(value, int):
            self.seconds = value
        elif isinstance(value, timedelta):
            self.seconds = int(value.total_seconds())
        else:
            raise ValueError(f"Invalid time window value: {value}")

    def _parse_string(self, value: str) -> None:
        """Parse string time window (e.g., '1min', '5s', '1h')."""
        value = value.lower().strip()

        # Extract number and unit
        import re

        match = re.match(r"^(\d+)([smhd]|min|sec|hour|day)$", value)
        if not match:
            raise ValueError(f"Invalid time window format: {value}")

        number, unit = match.groups()
        number = int(number)

        # Convert to seconds
        unit_map = {
            "s": 1,
            "sec": 1,
            "m": 60,
            "min": 60,
            "h": 3600,
            "hour": 3600,
            "d": 86400,
            "day": 86400,
        }

        if unit not in unit_map:
            raise ValueError(f"Unknown time unit: {unit}")

        self.seconds = number * unit_map[unit]

    def __str__(self) -> str:
        if self.seconds < 60:
            return f"{self.seconds}s"
        elif self.seconds < 3600:
            return f"{self.seconds // 60}min"
        elif self.seconds < 86400:
            return f"{self.seconds // 3600}h"
        else:
            return f"{self.seconds // 86400}d"

    def __repr__(self) -> str:
        return f"TimeWindow({self.seconds}s)"


@dataclass
class RateLimitRule:
    """Configuration for a single rate limit rule."""

    # Core rate limit parameters
    limit: int  # Number of requests allowed
    window: TimeWindow  # Time window
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET

    # Behavior configuration
    action: RateLimitAction = RateLimitAction.BLOCK
    burst_allowance: int | None = None  # Extra requests allowed in burst

    # Identification
    identifier: str | None = None  # Rule identifier
    description: str | None = None  # Human-readable description

    # Advanced options
    enable_cooldown: bool = False  # Enable cooldown after limit exceeded
    cooldown_period: TimeWindow | None = None  # Cooldown duration

    # Whitelisting/Blacklisting
    whitelist: list[str] = field(default_factory=list)  # Always allow these identifiers
    blacklist: list[str] = field(default_factory=list)  # Always block these identifiers

    # Metadata
    priority: int = 100  # Rule priority (lower = higher priority)
    enabled: bool = True  # Whether this rule is active

    def __post_init__(self) -> None:
        if isinstance(self.window, (str, int, timedelta)):
            self.window = TimeWindow(self.window)

        if self.burst_allowance is None:
            # Default burst allowance is 50% of limit
            self.burst_allowance = max(1, int(self.limit * 0.5))

        if self.enable_cooldown and self.cooldown_period is None:
            # Default cooldown is 2x the window
            self.cooldown_period = TimeWindow(self.window.seconds * 2)
        elif self.cooldown_period is not None and isinstance(
            self.cooldown_period, (str, int, timedelta)
        ):
            self.cooldown_period = TimeWindow(self.cooldown_period)


@dataclass
class TradingRateLimits:
    """Trading-specific rate limit configurations."""

    # Order management limits
    order_submission: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=100,
            window=TimeWindow("1min"),
            identifier="order_submission",
            description="Orders per minute per user",
        )
    )

    order_cancellation: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=200,
            window=TimeWindow("1min"),
            identifier="order_cancellation",
            description="Order cancellations per minute per user",
        )
    )

    # Market data limits
    market_data_requests: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=1000,
            window=TimeWindow("1min"),
            identifier="market_data",
            description="Market data requests per minute per API key",
        )
    )

    # Portfolio operations
    portfolio_queries: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=50,
            window=TimeWindow("1min"),
            identifier="portfolio_queries",
            description="Portfolio queries per minute per user",
        )
    )

    # Risk calculations
    risk_calculations: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=200,
            window=TimeWindow("1min"),
            identifier="risk_calculations",
            description="Risk calculations per minute per portfolio",
        )
    )

    # WebSocket connections
    websocket_connections: RateLimitRule = field(
        default_factory=lambda: RateLimitRule(
            limit=10,
            window=TimeWindow("1min"),
            identifier="websocket_connections",
            description="Concurrent WebSocket connections per user",
        )
    )


@dataclass
class RateLimitConfig:
    """Main configuration for rate limiting system."""

    # Storage configuration
    storage_backend: str = "redis"  # redis, memory
    redis_url: str | None = None
    redis_key_prefix: str = "rate_limit:"

    # Default rate limits
    default_limits: dict[RateLimitTier, dict[str, RateLimitRule]] = field(default_factory=dict)

    # Trading-specific limits
    trading_limits: TradingRateLimits = field(default_factory=TradingRateLimits)

    # Global configuration
    global_rate_limit: RateLimitRule | None = None
    enable_ip_limiting: bool = True
    enable_user_limiting: bool = True
    enable_api_key_limiting: bool = True

    # Rate limit headers
    include_rate_limit_headers: bool = True
    rate_limit_header_prefix: str = "X-RateLimit"

    # Monitoring and alerting
    enable_monitoring: bool = True
    alert_threshold: float = 0.8  # Alert when usage > 80% of limit

    # Performance options
    cache_ttl: int = 300  # Cache TTL in seconds
    cleanup_interval: int = 3600  # Cleanup old entries every hour

    # Admin bypass
    admin_bypass_enabled: bool = True
    admin_api_keys: list[str] = field(default_factory=list)
    admin_user_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.default_limits:
            self._setup_default_limits()

        if self.redis_url is None:
            self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    def _setup_default_limits(self) -> None:
        """Setup default rate limits for different tiers."""

        # Basic tier limits
        basic_limits = {
            "api_requests": RateLimitRule(
                limit=100, window=TimeWindow("1min"), identifier="api_requests_basic"
            ),
            "data_requests": RateLimitRule(
                limit=500, window=TimeWindow("1hour"), identifier="data_requests_basic"
            ),
        }

        # Premium tier limits
        premium_limits = {
            "api_requests": RateLimitRule(
                limit=500, window=TimeWindow("1min"), identifier="api_requests_premium"
            ),
            "data_requests": RateLimitRule(
                limit=2000, window=TimeWindow("1hour"), identifier="data_requests_premium"
            ),
        }

        # Enterprise tier limits
        enterprise_limits = {
            "api_requests": RateLimitRule(
                limit=2000, window=TimeWindow("1min"), identifier="api_requests_enterprise"
            ),
            "data_requests": RateLimitRule(
                limit=10000, window=TimeWindow("1hour"), identifier="data_requests_enterprise"
            ),
        }

        # Admin tier (very high limits)
        admin_limits = {
            "api_requests": RateLimitRule(
                limit=10000, window=TimeWindow("1min"), identifier="api_requests_admin"
            ),
            "data_requests": RateLimitRule(
                limit=100000, window=TimeWindow("1hour"), identifier="data_requests_admin"
            ),
        }

        self.default_limits = {
            RateLimitTier.BASIC: basic_limits,
            RateLimitTier.PREMIUM: premium_limits,
            RateLimitTier.ENTERPRISE: enterprise_limits,
            RateLimitTier.ADMIN: admin_limits,
            RateLimitTier.SYSTEM: admin_limits,  # System uses admin limits
        }

    def get_rule(self, tier: RateLimitTier, rule_type: str) -> RateLimitRule | None:
        """Get a specific rate limit rule."""
        return self.default_limits.get(tier, {}).get(rule_type)

    def is_admin_bypass(self, api_key: str | None = None, user_id: str | None = None) -> bool:
        """Check if request should bypass rate limits due to admin status."""
        if not self.admin_bypass_enabled:
            return False

        if api_key and api_key in self.admin_api_keys:
            return True

        if user_id and user_id in self.admin_user_ids:
            return True

        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RateLimitConfig":
        """Create configuration from dictionary."""
        # Convert nested dictionaries to appropriate objects
        if "default_limits" in data:
            default_limits = {}
            for tier_name, limits in data["default_limits"].items():
                tier = RateLimitTier(tier_name)
                tier_limits = {}
                for rule_name, rule_data in limits.items():
                    tier_limits[rule_name] = RateLimitRule(**rule_data)
                default_limits[tier] = tier_limits
            data["default_limits"] = default_limits

        if "trading_limits" in data:
            data["trading_limits"] = TradingRateLimits(**data["trading_limits"])

        if data.get("global_rate_limit"):
            data["global_rate_limit"] = RateLimitRule(**data["global_rate_limit"])

        return cls(**data)

    @classmethod
    def from_file(cls, file_path: str) -> "RateLimitConfig":
        """Load configuration from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables
        storage_backend = os.getenv("RATE_LIMIT_STORAGE_BACKEND")
        if storage_backend:
            config.storage_backend = storage_backend

        redis_url = os.getenv("RATE_LIMIT_REDIS_URL")
        if redis_url:
            config.redis_url = redis_url

        enable_monitoring = os.getenv("RATE_LIMIT_ENABLE_MONITORING")
        if enable_monitoring:
            config.enable_monitoring = enable_monitoring.lower() == "true"

        admin_bypass = os.getenv("RATE_LIMIT_ADMIN_BYPASS")
        if admin_bypass:
            config.admin_bypass_enabled = admin_bypass.lower() == "true"

        # Parse admin API keys
        admin_keys = os.getenv("RATE_LIMIT_ADMIN_API_KEYS")
        if admin_keys:
            config.admin_api_keys = admin_keys.split(",")

        # Parse admin user IDs
        admin_users = os.getenv("RATE_LIMIT_ADMIN_USER_IDS")
        if admin_users:
            config.admin_user_ids = admin_users.split(",")

        return config
