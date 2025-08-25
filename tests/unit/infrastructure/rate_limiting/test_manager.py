"""
Comprehensive unit tests for RateLimitManager.

Tests the high-level rate limit management, multi-tier limiting,
and integration with different storage backends.
"""

import threading
from unittest.mock import Mock

import pytest

from src.infrastructure.rate_limiting.config import (
    RateLimitConfig,
    RateLimitRule,
    RateLimitTier,
    TradingRateLimits,
)
from src.infrastructure.rate_limiting.exceptions import (
    APIRateLimitExceeded,
    IPRateLimitExceeded,
    RateLimitExceeded,
    TradingRateLimitExceeded,
)
from src.infrastructure.rate_limiting.manager import RateLimitContext, RateLimitManager
from src.infrastructure.rate_limiting.storage import MemoryRateLimitStorage


class TestRateLimitContext:
    """Test RateLimitContext functionality."""

    def test_basic_context_creation(self):
        """Test basic context creation."""
        context = RateLimitContext(user_id="user123", api_key="key456", ip_address="192.168.1.1")

        assert context.user_id == "user123"
        assert context.api_key == "key456"
        assert context.ip_address == "192.168.1.1"
        assert context.user_tier == RateLimitTier.BASIC

    def test_trading_context(self):
        """Test trading-specific context."""
        context = RateLimitContext(
            user_id="trader1",
            trading_action="submit_order",
            symbol="AAPL",
            user_tier=RateLimitTier.PREMIUM,
        )

        assert context.trading_action == "submit_order"
        assert context.symbol == "AAPL"
        assert context.user_tier == RateLimitTier.PREMIUM

    def test_api_context(self):
        """Test API-specific context."""
        context = RateLimitContext(
            api_key="api123", endpoint="/api/v1/orders", method="POST", ip_address="10.0.0.1"
        )

        assert context.endpoint == "/api/v1/orders"
        assert context.method == "POST"


class TestRateLimitManager:
    """Test RateLimitManager functionality."""

    def test_manager_initialization(self):
        """Test manager initialization with default config."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        assert manager.config == config
        assert isinstance(manager.storage, MemoryRateLimitStorage)
        assert len(manager._limiters) > 0  # Should have default limiters

    def test_admin_bypass(self):
        """Test admin bypass functionality."""
        config = RateLimitConfig(
            storage_backend="memory",
            admin_bypass_enabled=True,
            admin_api_keys=["admin_key"],
            admin_user_ids=["admin_user"],
        )
        manager = RateLimitManager(config)

        # Admin API key should bypass
        context = RateLimitContext(api_key="admin_key")
        statuses = manager.check_rate_limit(context)
        assert len(statuses) == 0  # No rate limits applied

        # Admin user should bypass
        context = RateLimitContext(user_id="admin_user")
        statuses = manager.check_rate_limit(context)
        assert len(statuses) == 0

        # Regular user should not bypass
        context = RateLimitContext(user_id="regular_user")
        statuses = manager.check_rate_limit(context)
        assert len(statuses) > 0  # Rate limits applied

    def test_user_tier_rate_limiting(self):
        """Test rate limiting based on user tier."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Basic tier user
        basic_context = RateLimitContext(user_id="basic_user", user_tier=RateLimitTier.BASIC)

        # Premium tier user
        premium_context = RateLimitContext(user_id="premium_user", user_tier=RateLimitTier.PREMIUM)

        # Check rate limits for both tiers
        basic_statuses = manager.check_rate_limit(basic_context)
        premium_statuses = manager.check_rate_limit(premium_context)

        # Both should have rate limits applied
        assert len(basic_statuses) > 0
        assert len(premium_statuses) > 0

        # Premium should have higher limits
        basic_api_limit = next(s.limit for s in basic_statuses if "api_requests" in s.rule_id)
        premium_api_limit = next(s.limit for s in premium_statuses if "api_requests" in s.rule_id)
        assert premium_api_limit > basic_api_limit

    def test_trading_rate_limiting(self):
        """Test trading-specific rate limiting."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="trader1", trading_action="submit_order", symbol="AAPL")

        statuses = manager.check_rate_limit(context)

        # Should have trading-specific rate limits
        trading_statuses = [s for s in statuses if "trading:" in s.rule_id]
        assert len(trading_statuses) > 0

        # Should have order submission limit
        order_status = next((s for s in trading_statuses if "order_submission" in s.rule_id), None)
        assert order_status is not None

    def test_ip_rate_limiting(self):
        """Test IP-based rate limiting."""
        config = RateLimitConfig(storage_backend="memory", enable_ip_limiting=True)
        manager = RateLimitManager(config)

        context = RateLimitContext(ip_address="192.168.1.1")
        statuses = manager.check_rate_limit(context)

        # Should have IP-based rate limits
        assert len(statuses) > 0
        ip_statuses = [s for s in statuses if "ip:" in s.identifier]
        assert len(ip_statuses) > 0

    def test_api_key_rate_limiting(self):
        """Test API key-based rate limiting."""
        config = RateLimitConfig(storage_backend="memory", enable_api_key_limiting=True)
        manager = RateLimitManager(config)

        context = RateLimitContext(api_key="test_key")
        statuses = manager.check_rate_limit(context)

        # Should have API key-based rate limits
        assert len(statuses) > 0

    def test_multiple_limiters_applied(self):
        """Test that multiple limiters are applied when applicable."""
        config = RateLimitConfig(
            storage_backend="memory",
            enable_ip_limiting=True,
            enable_user_limiting=True,
            enable_api_key_limiting=True,
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(
            user_id="user1", api_key="key1", ip_address="192.168.1.1", trading_action="submit_order"
        )

        statuses = manager.check_rate_limit(context)

        # Should have multiple types of rate limits
        assert len(statuses) >= 3  # IP, user, trading at minimum

    def test_rate_limit_exceeded_exception(self):
        """Test rate limit exceeded exception handling."""
        config = RateLimitConfig(storage_backend="memory")
        # Create a very restrictive rule
        config.trading_limits.order_submission = RateLimitRule(
            limit=1, window="1min", identifier="order_submission"
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="trader1", trading_action="submit_order")

        # First request should pass
        manager.check_rate_limit(context)

        # Second request should raise exception
        with pytest.raises(TradingRateLimitExceeded) as exc_info:
            manager.check_rate_limit(context)

        assert exc_info.trading_action == "submit_order"
        assert exc_info.user_id == "trader1"

    def test_api_rate_limit_exceeded(self):
        """Test API rate limit exceeded exception."""
        config = RateLimitConfig(storage_backend="memory")
        # Make basic tier very restrictive
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=1, window="1min"
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(api_key="test_key", endpoint="/api/orders", method="POST")

        # First request should pass
        manager.check_rate_limit(context)

        # Second request should raise API-specific exception
        with pytest.raises(APIRateLimitExceeded) as exc_info:
            manager.check_rate_limit(context)

        assert exc_info.endpoint == "/api/orders"
        assert exc_info.api_key == "test_key"

    def test_ip_rate_limit_exceeded(self):
        """Test IP rate limit exceeded exception."""
        config = RateLimitConfig(storage_backend="memory", enable_ip_limiting=True)
        # Make IP limits very restrictive
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=1, window="1min"
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(ip_address="192.168.1.1")

        # First request should pass
        manager.check_rate_limit(context)

        # Second request should raise IP-specific exception
        with pytest.raises(IPRateLimitExceeded) as exc_info:
            manager.check_rate_limit(context)

        assert exc_info.ip_address == "192.168.1.1"

    def test_get_status(self):
        """Test getting rate limit status."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")

        # Make some requests
        manager.check_rate_limit(context)
        manager.check_rate_limit(context)

        # Get status
        status = manager.get_status(context)

        assert isinstance(status, dict)
        assert len(status) > 0

        # Check that status contains expected fields
        for rule_type, rule_status in status.items():
            if "error" not in rule_status:
                assert "current_count" in rule_status
                assert "limit" in rule_status
                assert "remaining" in rule_status
                assert "tier" in rule_status

    def test_reset_limits(self):
        """Test resetting rate limits."""
        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=2, window="1min"
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")

        # Use up the limit
        manager.check_rate_limit(context)
        manager.check_rate_limit(context)

        # Should be denied
        with pytest.raises(RateLimitExceeded):
            manager.check_rate_limit(context)

        # Reset limits
        manager.reset_limits(context)

        # Should work again
        statuses = manager.check_rate_limit(context)
        assert all(s.allowed for s in statuses)

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Create some entries
        context1 = RateLimitContext(user_id="user1")
        context2 = RateLimitContext(user_id="user2")

        manager.check_rate_limit(context1)
        manager.check_rate_limit(context2)

        # Force cleanup
        cleanup_results = manager.cleanup_expired()

        assert isinstance(cleanup_results, dict)
        assert "storage" in cleanup_results

    def test_get_metrics(self):
        """Test getting rate limiting metrics."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")
        manager.check_rate_limit(context)

        metrics = manager.get_metrics()

        assert isinstance(metrics, dict)
        assert "limiters" in metrics
        assert "metrics" in metrics
        assert "storage_health" in metrics
        assert "config" in metrics

        assert len(metrics["limiters"]) > 0
        assert metrics["storage_health"] is True

    def test_health_check(self):
        """Test health check functionality."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        health = manager.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health
        assert "storage" in health
        assert "limiters_count" in health
        assert "errors" in health

        assert health["healthy"] is True
        assert health["storage"] is True
        assert health["limiters_count"] > 0
        assert isinstance(health["errors"], list)

    def test_build_identifier(self):
        """Test identifier building for different contexts."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # User identifier
        context = RateLimitContext(user_id="user123")
        identifier = manager._build_identifier(context, "user:api_requests")
        assert "user123" in identifier

        # IP identifier
        context = RateLimitContext(ip_address="192.168.1.1")
        identifier = manager._build_identifier(context, "ip:api_requests")
        assert "192.168.1.1" in identifier

        # Trading identifier with symbol
        context = RateLimitContext(user_id="trader1", symbol="AAPL")
        identifier = manager._build_identifier(context, "trading:order_submission")
        assert "trader1" in identifier
        assert "AAPL" in identifier

    def test_concurrent_access(self):
        """Test manager under concurrent access."""
        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=100, window="1min"
        )
        manager = RateLimitManager(config)

        results = []

        def worker(user_id):
            context = RateLimitContext(user_id=f"user{user_id}")
            try:
                manager.check_rate_limit(context)
                results.append(True)
            except RateLimitExceeded:
                results.append(False)

        # Start multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All requests should succeed (different users)
        assert all(results)
        assert len(results) == 50

    def test_storage_error_handling(self):
        """Test handling of storage errors."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Mock storage to raise error
        manager.storage.get = Mock(side_effect=Exception("Storage error"))

        context = RateLimitContext(user_id="user1")

        # Should handle storage errors gracefully
        # In this implementation, storage errors in limiters are caught and logged
        # The request is typically allowed to proceed for safety
        try:
            statuses = manager.check_rate_limit(context)
            # Should either succeed or raise a known rate limit exception
        except Exception as e:
            # Should not be a storage exception
            assert "Storage error" not in str(e)

    def test_custom_rule_types(self):
        """Test checking specific rule types."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")

        # Check only specific rule types
        statuses = manager.check_rate_limit(context, rule_types=["api_requests"])

        # Should only have API request limits
        assert len(statuses) > 0
        assert all("api_requests" in s.rule_id for s in statuses)

    def test_tokens_parameter(self):
        """Test rate limiting with custom token count."""
        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=10, window="1min"
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")

        # Use 5 tokens at once
        statuses = manager.check_rate_limit(context, tokens=5)
        assert all(s.allowed for s in statuses)

        # Check remaining tokens
        status = manager.get_status(context)
        api_status = status.get("user:api_requests", {})
        if "remaining" in api_status:
            assert api_status["remaining"] == 5


class TestRateLimitManagerConfiguration:
    """Test manager configuration and initialization."""

    def test_custom_trading_limits(self):
        """Test manager with custom trading limits."""
        trading_limits = TradingRateLimits()
        trading_limits.order_submission = RateLimitRule(
            limit=50, window="30s", identifier="custom_order_submission"
        )

        config = RateLimitConfig(storage_backend="memory", trading_limits=trading_limits)
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="trader1", trading_action="submit_order")

        statuses = manager.check_rate_limit(context)

        # Should use custom trading limits
        order_status = next((s for s in statuses if "order_submission" in s.rule_id), None)
        assert order_status is not None
        assert order_status.limit == 50

    def test_global_rate_limit(self):
        """Test global rate limiting."""
        config = RateLimitConfig(
            storage_backend="memory",
            global_rate_limit=RateLimitRule(limit=1000, window="1min", identifier="global"),
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1")
        statuses = manager.check_rate_limit(context)

        # Should have global rate limit
        global_status = next((s for s in statuses if s.rule_id == "global"), None)
        assert global_status is not None

    def test_disabled_limiting_types(self):
        """Test disabling specific limiting types."""
        config = RateLimitConfig(
            storage_backend="memory",
            enable_ip_limiting=False,
            enable_user_limiting=False,
            enable_api_key_limiting=True,
        )
        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="user1", api_key="key1", ip_address="192.168.1.1")

        statuses = manager.check_rate_limit(context)

        # Should not have IP or user limits
        ip_statuses = [s for s in statuses if "ip:" in s.identifier]
        user_statuses = [s for s in statuses if "user:" in s.identifier]

        assert len(ip_statuses) == 0
        assert len(user_statuses) == 0

    def test_rate_limit_headers_config(self):
        """Test rate limit headers configuration."""
        config = RateLimitConfig(
            storage_backend="memory",
            include_rate_limit_headers=True,
            rate_limit_header_prefix="X-Custom-RateLimit",
        )
        manager = RateLimitManager(config)

        assert manager.config.include_rate_limit_headers is True
        assert manager.config.rate_limit_header_prefix == "X-Custom-RateLimit"
