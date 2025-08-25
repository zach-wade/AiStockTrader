"""
High-level rate limit management for the AI Trading System.

Provides a unified interface for managing multiple rate limiters,
multi-tier rate limiting, and integration with storage backends.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .algorithms import RateLimitAlgorithm, RateLimitResult, create_rate_limiter
from .config import RateLimitConfig, RateLimitTier
from .exceptions import (
    APIRateLimitExceeded,
    IPRateLimitExceeded,
    RateLimitConfigError,
    RateLimitExceeded,
    TradingRateLimitExceeded,
)
from .storage import create_storage

logger = logging.getLogger(__name__)


@dataclass
class RateLimitContext:
    """Context information for rate limit checks."""

    user_id: str | None = None
    api_key: str | None = None
    ip_address: str | None = None
    endpoint: str | None = None
    method: str | None = None
    user_tier: RateLimitTier = RateLimitTier.BASIC
    trading_action: str | None = None
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitStatus:
    """Status information for rate limits."""

    identifier: str
    rule_id: str
    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_time: datetime | None
    retry_after: int | None
    tier: RateLimitTier
    action_taken: str | None = None


class RateLimitManager:
    """
    High-level rate limit manager.

    Manages multiple rate limiters, provides multi-tier rate limiting,
    and integrates with storage backends for distributed operation.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self.storage = create_storage(config)

        # Rate limiter instances
        self._limiters: dict[str, RateLimitAlgorithm] = {}
        self._limiter_lock = threading.RLock()

        # Monitoring and metrics
        self._metrics: dict[str, int] = defaultdict(int)
        self._last_cleanup = time.time()

        # Initialize default limiters
        self._initialize_limiters()

        logger.info(f"RateLimitManager initialized with {config.storage_backend} storage")

    def _initialize_limiters(self) -> None:
        """Initialize rate limiters from configuration."""
        with self._limiter_lock:
            # Initialize default tier limiters
            for tier, rules in self.config.default_limits.items():
                for rule_type, rule in rules.items():
                    limiter_id = f"{tier.value}:{rule_type}"
                    self._limiters[limiter_id] = create_rate_limiter(rule)

            # Initialize trading-specific limiters
            trading_rules = {
                "order_submission": self.config.trading_limits.order_submission,
                "order_cancellation": self.config.trading_limits.order_cancellation,
                "market_data_requests": self.config.trading_limits.market_data_requests,
                "portfolio_queries": self.config.trading_limits.portfolio_queries,
                "risk_calculations": self.config.trading_limits.risk_calculations,
                "websocket_connections": self.config.trading_limits.websocket_connections,
            }

            for rule_type, rule in trading_rules.items():
                limiter_id = f"trading:{rule_type}"
                self._limiters[limiter_id] = create_rate_limiter(rule)

            # Initialize global rate limiter if configured
            if self.config.global_rate_limit:
                self._limiters["global"] = create_rate_limiter(self.config.global_rate_limit)

    def check_rate_limit(
        self, context: RateLimitContext, tokens: int = 1, rule_types: list[str] | None = None
    ) -> list[RateLimitStatus]:
        """
        Check rate limits for the given context.

        Returns list of rate limit statuses for all applicable rules.
        """
        statuses = []

        # Check admin bypass
        if self.config.is_admin_bypass(context.api_key, context.user_id):
            logger.debug(f"Admin bypass for user {context.user_id} or API key {context.api_key}")
            return []

        # Determine which limiters to check
        limiters_to_check = self._get_applicable_limiters(context, rule_types)

        # Check each applicable limiter
        for limiter_id, rule_type, tier in limiters_to_check:
            try:
                identifier = self._build_identifier(context, rule_type)
                result = self._check_limiter(limiter_id, identifier, tokens)

                status = RateLimitStatus(
                    identifier=identifier,
                    rule_id=limiter_id,
                    allowed=result.allowed,
                    current_count=result.current_count,
                    limit=result.limit,
                    remaining=result.remaining,
                    reset_time=result.reset_time,
                    retry_after=result.retry_after,
                    tier=tier,
                )

                statuses.append(status)

                # Record metrics
                self._record_metrics(limiter_id, result.allowed)

                # If any limiter denies the request, handle the denial
                if not result.allowed:
                    self._handle_rate_limit_exceeded(context, status, result)

            except Exception as e:
                logger.error(f"Error checking rate limit {limiter_id}: {e}")
                # In case of error, we might want to allow or deny based on policy
                # For safety, we'll allow the request but log the error
                continue

        return statuses

    def _get_applicable_limiters(
        self, context: RateLimitContext, rule_types: list[str] | None = None
    ) -> list[tuple[str, str, RateLimitTier]]:
        """Get list of limiters applicable to the context."""
        limiters = []

        # Global rate limit (always applies)
        if "global" in self._limiters:
            limiters.append(("global", "global", RateLimitTier.SYSTEM))

        # IP-based rate limiting
        if self.config.enable_ip_limiting and context.ip_address:
            ip_limiters = self._get_ip_limiters(context)
            limiters.extend(ip_limiters)

        # User-based rate limiting
        if self.config.enable_user_limiting and context.user_id:
            user_limiters = self._get_user_limiters(context, rule_types)
            limiters.extend(user_limiters)

        # API key-based rate limiting
        if self.config.enable_api_key_limiting and context.api_key:
            api_limiters = self._get_api_key_limiters(context, rule_types)
            limiters.extend(api_limiters)

        # Trading-specific rate limiting
        if context.trading_action:
            trading_limiters = self._get_trading_limiters(context)
            limiters.extend(trading_limiters)

        return limiters

    def _get_ip_limiters(self, context: RateLimitContext) -> list[tuple[str, str, RateLimitTier]]:
        """Get IP-based rate limiters."""
        # Use basic tier for IP limiting
        limiters = []
        for rule_type in ["api_requests", "data_requests"]:
            limiter_id = f"{RateLimitTier.BASIC.value}:{rule_type}"
            if limiter_id in self._limiters:
                limiters.append((limiter_id, f"ip:{rule_type}", RateLimitTier.BASIC))
        return limiters

    def _get_user_limiters(
        self, context: RateLimitContext, rule_types: list[str] | None = None
    ) -> list[tuple[str, str, RateLimitTier]]:
        """Get user-based rate limiters."""
        limiters = []
        tier = context.user_tier

        if rule_types:
            # Check specific rule types
            for rule_type in rule_types:
                limiter_id = f"{tier.value}:{rule_type}"
                if limiter_id in self._limiters:
                    limiters.append((limiter_id, f"user:{rule_type}", tier))
        else:
            # Check all available rule types for the tier
            for rule_type in ["api_requests", "data_requests"]:
                limiter_id = f"{tier.value}:{rule_type}"
                if limiter_id in self._limiters:
                    limiters.append((limiter_id, f"user:{rule_type}", tier))

        return limiters

    def _get_api_key_limiters(
        self, context: RateLimitContext, rule_types: list[str] | None = None
    ) -> list[tuple[str, str, RateLimitTier]]:
        """Get API key-based rate limiters."""
        # Similar to user limiters but for API keys
        return self._get_user_limiters(context, rule_types)

    def _get_trading_limiters(
        self, context: RateLimitContext
    ) -> list[tuple[str, str, RateLimitTier]]:
        """Get trading-specific rate limiters."""
        limiters = []

        # Map trading actions to limiter types
        action_mapping = {
            "submit_order": "order_submission",
            "cancel_order": "order_cancellation",
            "get_market_data": "market_data_requests",
            "get_portfolio": "portfolio_queries",
            "calculate_risk": "risk_calculations",
            "websocket_connect": "websocket_connections",
        }

        if context.trading_action in action_mapping:
            rule_type = action_mapping[context.trading_action]
            limiter_id = f"trading:{rule_type}"
            if limiter_id in self._limiters:
                limiters.append((limiter_id, f"trading:{rule_type}", context.user_tier))

        return limiters

    def _build_identifier(self, context: RateLimitContext, rule_type: str) -> str:
        """Build unique identifier for rate limiting."""
        parts = [rule_type]

        if rule_type.startswith("ip:") and context.ip_address:
            parts.append(context.ip_address)
        elif rule_type.startswith("user:") and context.user_id:
            parts.append(context.user_id)
        elif rule_type.startswith("trading:"):
            if context.user_id:
                parts.append(context.user_id)
            if context.symbol:
                parts.append(context.symbol)
        elif context.api_key:
            parts.append(context.api_key)
        elif context.user_id:
            parts.append(context.user_id)
        elif context.ip_address:
            parts.append(context.ip_address)

        return ":".join(parts)

    def _check_limiter(self, limiter_id: str, identifier: str, tokens: int) -> RateLimitResult:
        """Check a specific rate limiter."""
        with self._limiter_lock:
            if limiter_id not in self._limiters:
                raise RateLimitConfigError(f"Rate limiter {limiter_id} not found")

            limiter = self._limiters[limiter_id]
            return limiter.check_rate_limit(identifier, tokens)

    def _handle_rate_limit_exceeded(
        self, context: RateLimitContext, status: RateLimitStatus, result: RateLimitResult
    ) -> None:
        """Handle rate limit exceeded scenarios."""
        # Determine appropriate exception type
        if context.trading_action:
            raise TradingRateLimitExceeded(
                f"Trading rate limit exceeded for {context.trading_action}",
                trading_action=context.trading_action,
                user_id=context.user_id,
                symbol=context.symbol,
                limit=result.limit,
                window_size=str(self._get_rule_window(status.rule_id)),
                current_count=result.current_count,
                retry_after=result.retry_after,
            )
        elif context.endpoint:
            raise APIRateLimitExceeded(
                f"API rate limit exceeded for {context.endpoint}",
                endpoint=context.endpoint,
                method=context.method,
                api_key=context.api_key,
                limit=result.limit,
                window_size=str(self._get_rule_window(status.rule_id)),
                current_count=result.current_count,
                retry_after=result.retry_after,
            )
        elif context.ip_address:
            raise IPRateLimitExceeded(
                f"IP rate limit exceeded for {context.ip_address}",
                ip_address=context.ip_address,
                limit=result.limit,
                window_size=str(self._get_rule_window(status.rule_id)),
                current_count=result.current_count,
                retry_after=result.retry_after,
            )
        else:
            raise RateLimitExceeded(
                "Rate limit exceeded",
                limit=result.limit,
                window_size=str(self._get_rule_window(status.rule_id)),
                current_count=result.current_count,
                retry_after=result.retry_after,
            )

    def _get_rule_window(self, limiter_id: str) -> str:
        """Get the window size for a rate limiter."""
        with self._limiter_lock:
            if limiter_id in self._limiters:
                return str(self._limiters[limiter_id].rule.window)
            return "unknown"

    def _record_metrics(self, limiter_id: str, allowed: bool) -> None:
        """Record rate limiting metrics."""
        self._metrics[f"{limiter_id}:total"] += 1
        if allowed:
            self._metrics[f"{limiter_id}:allowed"] += 1
        else:
            self._metrics[f"{limiter_id}:denied"] += 1

    def get_status(self, context: RateLimitContext) -> dict[str, Any]:
        """Get current rate limit status for context."""
        status = {}

        limiters_to_check = self._get_applicable_limiters(context)

        for limiter_id, rule_type, tier in limiters_to_check:
            identifier = self._build_identifier(context, rule_type)

            try:
                with self._limiter_lock:
                    limiter = self._limiters[limiter_id]
                    current_count, limit = limiter.get_current_usage(identifier)

                    status[rule_type] = {
                        "current_count": current_count,
                        "limit": limit,
                        "remaining": limit - current_count,
                        "tier": tier.value,
                        "window": str(limiter.rule.window),
                    }
            except Exception as e:
                logger.error(f"Error getting status for {limiter_id}: {e}")
                status[rule_type] = {"error": str(e)}

        return status

    def reset_limits(self, context: RateLimitContext, rule_types: list[str] | None = None) -> None:
        """Reset rate limits for context."""
        limiters_to_reset = self._get_applicable_limiters(context, rule_types)

        for limiter_id, rule_type, tier in limiters_to_reset:
            identifier = self._build_identifier(context, rule_type)

            try:
                with self._limiter_lock:
                    limiter = self._limiters[limiter_id]
                    limiter.reset_limit(identifier)

                logger.info(f"Reset rate limit {rule_type} for {identifier}")
            except Exception as e:
                logger.error(f"Error resetting rate limit {limiter_id}: {e}")

    def cleanup_expired(self) -> dict[str, int]:
        """Clean up expired rate limit entries."""
        current_time = time.time()

        # Only cleanup if enough time has passed
        if current_time - self._last_cleanup < self.config.cleanup_interval:
            return {}

        cleanup_results = {}

        # Cleanup storage
        try:
            storage_cleaned = self.storage.cleanup_expired()
            cleanup_results["storage"] = storage_cleaned
        except Exception as e:
            logger.error(f"Error cleaning up storage: {e}")
            cleanup_results["storage"] = 0

        # Cleanup individual limiters
        with self._limiter_lock:
            for limiter_id, limiter in self._limiters.items():
                try:
                    cleaned = limiter.cleanup_expired()
                    cleanup_results[limiter_id] = cleaned
                except Exception as e:
                    logger.error(f"Error cleaning up limiter {limiter_id}: {e}")
                    cleanup_results[limiter_id] = 0

        self._last_cleanup = current_time

        total_cleaned = sum(cleanup_results.values())
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} expired rate limit entries")

        return cleanup_results

    def get_metrics(self) -> dict[str, Any]:
        """Get rate limiting metrics."""
        return {
            "limiters": list(self._limiters.keys()),
            "metrics": dict(self._metrics),
            "storage_health": self.storage.health_check(),
            "last_cleanup": self._last_cleanup,
            "config": {
                "storage_backend": self.config.storage_backend,
                "enable_monitoring": self.config.enable_monitoring,
                "admin_bypass_enabled": self.config.admin_bypass_enabled,
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Perform health check on rate limiting system."""
        health = {
            "healthy": True,
            "storage": self.storage.health_check(),
            "limiters_count": len(self._limiters),
            "errors": [],
        }

        # Test storage operations
        try:
            test_key = f"health_check_{int(time.time())}"
            self.storage.set(test_key, "test", 10)
            test_value = self.storage.get(test_key)
            self.storage.delete(test_key)

            if test_value != "test":
                errors_list = health.get("errors", [])
                if isinstance(errors_list, list):
                    errors_list.append("Storage read/write test failed")
                health["healthy"] = False
        except Exception as e:
            errors_list = health.get("errors", [])
            if isinstance(errors_list, list):
                errors_list.append(f"Storage test error: {e}")
            health["healthy"] = False

        return health
