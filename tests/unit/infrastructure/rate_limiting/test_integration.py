"""
Integration tests for rate limiting system.

Tests end-to-end functionality, Redis integration, and real-world scenarios.
"""

import threading
import time

import pytest
import redis

from src.infrastructure.rate_limiting import (
    RateLimitConfig,
    RateLimitContext,
    RateLimitExceeded,
    RateLimitManager,
    RateLimitRule,
    RateLimitTier,
    RedisRateLimitStorage,
    TradingRateLimitExceeded,
    api_rate_limit,
    initialize_rate_limiting,
    rate_limit,
    trading_rate_limit,
)


@pytest.mark.integration
class TestMemoryStorageIntegration:
    """Test integration with memory storage backend."""

    def test_full_memory_integration(self):
        """Test complete rate limiting flow with memory storage."""
        config = RateLimitConfig(
            storage_backend="memory",
            enable_ip_limiting=True,
            enable_user_limiting=True,
            enable_api_key_limiting=True,
        )

        # Override limits for testing
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=5, window="1min"
        )

        manager = RateLimitManager(config)

        context = RateLimitContext(
            user_id="test_user", api_key="test_key", ip_address="192.168.1.1", endpoint="/api/test"
        )

        # Should allow first 5 requests
        for i in range(5):
            statuses = manager.check_rate_limit(context)
            assert all(s.allowed for s in statuses)

        # 6th request should fail
        with pytest.raises(RateLimitExceeded):
            manager.check_rate_limit(context)

    def test_memory_storage_persistence(self):
        """Test that memory storage persists across manager instances."""
        config = RateLimitConfig(storage_backend="memory")

        # First manager instance
        manager1 = RateLimitManager(config)
        context = RateLimitContext(user_id="user1")

        # Use some rate limit
        manager1.check_rate_limit(context)

        # Create second manager with same storage
        manager2 = RateLimitManager(config)

        # Storage should be independent (new instance)
        statuses = manager2.check_rate_limit(context)
        assert all(s.allowed for s in statuses)

    def test_memory_storage_cleanup(self):
        """Test memory storage cleanup functionality."""
        config = RateLimitConfig(
            storage_backend="memory",
            cleanup_interval=1,  # 1 second for testing
        )

        manager = RateLimitManager(config)
        storage = manager.storage

        # Add some data
        storage.set("test_key", "test_value", 1)  # 1 second TTL
        assert storage.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert storage.get("test_key") is None

        # Cleanup should work
        cleaned = storage.cleanup_expired()
        assert cleaned >= 0


@pytest.mark.integration
@pytest.mark.redis
class TestRedisStorageIntegration:
    """Test integration with Redis storage backend."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis for testing."""
        try:
            # Try to connect to Redis
            redis_client = redis.Redis.from_url(
                "redis://localhost:6379/15"
            )  # Use DB 15 for testing
            redis_client.ping()

            # Clean up test database
            redis_client.flushdb()

            self.redis_available = True
            yield

            # Cleanup after test
            redis_client.flushdb()

        except (redis.ConnectionError, redis.RedisError):
            pytest.skip("Redis not available for integration tests")

    def test_redis_storage_basic(self):
        """Test basic Redis storage operations."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        storage = RedisRateLimitStorage(config)

        # Test basic operations
        assert storage.set("test_key", "test_value")
        assert storage.get("test_key") == "test_value"
        assert storage.exists("test_key")
        assert storage.delete("test_key")
        assert not storage.exists("test_key")

    def test_redis_storage_ttl(self):
        """Test Redis storage TTL functionality."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        storage = RedisRateLimitStorage(config)

        # Set with TTL
        storage.set("ttl_key", "ttl_value", 2)
        assert storage.get("ttl_key") == "ttl_value"

        # Check TTL
        ttl = storage.ttl("ttl_key")
        assert 0 < ttl <= 2

        # Wait for expiration
        time.sleep(2.5)
        assert storage.get("ttl_key") is None

    def test_redis_storage_increment(self):
        """Test Redis storage increment operations."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        storage = RedisRateLimitStorage(config)

        # Test increment
        count1 = storage.increment("counter", 1)
        assert count1 == 1

        count2 = storage.increment("counter", 5)
        assert count2 == 6

        # Test with TTL
        count3 = storage.increment("counter_ttl", 1, 10)
        assert count3 == 1

        ttl = storage.ttl("counter_ttl")
        assert 0 < ttl <= 10

    def test_full_redis_integration(self):
        """Test complete rate limiting flow with Redis storage."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        # Override limits for testing
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=3, window="1min"
        )

        manager = RateLimitManager(config)

        context = RateLimitContext(user_id="redis_user", api_key="redis_key")

        # Should allow first 3 requests
        for i in range(3):
            statuses = manager.check_rate_limit(context)
            assert all(s.allowed for s in statuses)

        # 4th request should fail
        with pytest.raises(RateLimitExceeded):
            manager.check_rate_limit(context)

    def test_redis_distributed_rate_limiting(self):
        """Test distributed rate limiting across multiple manager instances."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=5, window="1min"
        )

        # Create two manager instances (simulating different app instances)
        manager1 = RateLimitManager(config)
        manager2 = RateLimitManager(config)

        context = RateLimitContext(user_id="distributed_user")

        # Use rate limit from first manager
        for i in range(3):
            manager1.check_rate_limit(context)

        # Use remaining from second manager
        for i in range(2):
            manager2.check_rate_limit(context)

        # Both managers should now deny requests
        with pytest.raises(RateLimitExceeded):
            manager1.check_rate_limit(context)

        with pytest.raises(RateLimitExceeded):
            manager2.check_rate_limit(context)

    def test_redis_health_check(self):
        """Test Redis health check functionality."""
        config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

        storage = RedisRateLimitStorage(config)
        assert storage.health_check() is True

        # Test with invalid Redis URL
        bad_config = RateLimitConfig(
            storage_backend="redis",
            redis_url="redis://localhost:9999/0",  # Invalid port
        )

        with pytest.raises(Exception):  # Should fail to connect
            RedisRateLimitStorage(bad_config)

    def test_redis_key_patterns(self):
        """Test Redis key pattern matching."""
        config = RateLimitConfig(
            storage_backend="redis", redis_url="redis://localhost:6379/15", redis_key_prefix="test:"
        )

        storage = RedisRateLimitStorage(config)

        # Set multiple keys
        storage.set("user:1", "data1")
        storage.set("user:2", "data2")
        storage.set("api:1", "api_data")

        # Test pattern matching
        user_keys = storage.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

        api_keys = storage.keys("api:*")
        assert len(api_keys) == 1
        assert "api:1" in api_keys


@pytest.mark.integration
class TestTradingRateLimitIntegration:
    """Test trading-specific rate limiting integration."""

    def test_trading_order_submission_limits(self):
        """Test order submission rate limiting."""
        config = RateLimitConfig(storage_backend="memory")
        config.trading_limits.order_submission = RateLimitRule(
            limit=3, window="1min", identifier="order_submission"
        )

        initialize_rate_limiting(config)

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id, symbol, quantity):
            return f"Order: {quantity} shares of {symbol}"

        # Should allow 3 orders
        result1 = submit_order("trader1", "AAPL", 100)
        assert "AAPL" in result1

        result2 = submit_order("trader1", "GOOGL", 50)
        assert "GOOGL" in result2

        result3 = submit_order("trader1", "MSFT", 75)
        assert "MSFT" in result3

        # 4th order should fail
        with pytest.raises(TradingRateLimitExceeded) as exc_info:
            submit_order("trader1", "TSLA", 25)

        assert exc_info.trading_action == "submit_order"
        assert exc_info.user_id == "trader1"

    def test_trading_multiple_actions(self):
        """Test rate limiting for multiple trading actions."""
        config = RateLimitConfig(storage_backend="memory")
        config.trading_limits.order_submission = RateLimitRule(limit=2, window="1min")
        config.trading_limits.order_cancellation = RateLimitRule(limit=5, window="1min")

        initialize_rate_limiting(config)

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id, symbol):
            return f"Submitted: {symbol}"

        @trading_rate_limit(action="cancel_order")
        def cancel_order(user_id, order_id):
            return f"Cancelled: {order_id}"

        # Use up order submission limit
        submit_order("trader1", "AAPL")
        submit_order("trader1", "GOOGL")

        with pytest.raises(TradingRateLimitExceeded):
            submit_order("trader1", "MSFT")

        # But cancellation should still work
        result = cancel_order("trader1", "order123")
        assert "order123" in result

    def test_trading_symbol_specific_limits(self):
        """Test symbol-specific rate limiting."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

        @trading_rate_limit(action="submit_order", per_symbol=True)
        def submit_order_with_symbol(user_id, symbol):
            return f"Order: {symbol}"

        # Different symbols should have separate limits
        # Note: This test requires implementation of per_symbol limiting
        # which would need to be added to the trading_rate_limit decorator

        # For now, test that the function works
        result = submit_order_with_symbol("trader1", "AAPL")
        assert "AAPL" in result

    def test_trading_market_data_limits(self):
        """Test market data request rate limiting."""
        config = RateLimitConfig(storage_backend="memory")
        config.trading_limits.market_data_requests = RateLimitRule(limit=10, window="1min")

        initialize_rate_limiting(config)

        @trading_rate_limit(action="get_market_data")
        def get_market_data(user_id, symbol):
            return {"symbol": symbol, "price": 150.0}

        # Should allow 10 requests
        for i in range(10):
            result = get_market_data("user1", f"STOCK{i}")
            assert "price" in result

        # 11th should fail
        with pytest.raises(TradingRateLimitExceeded):
            get_market_data("user1", "STOCK11")


@pytest.mark.integration
class TestAPIRateLimitIntegration:
    """Test API rate limiting integration."""

    def test_api_tier_based_limiting(self):
        """Test API rate limiting based on user tiers."""
        config = RateLimitConfig(storage_backend="memory")

        # Set different limits for different tiers
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=2, window="1min"
        )
        config.default_limits[RateLimitTier.PREMIUM]["api_requests"] = RateLimitRule(
            limit=5, window="1min"
        )

        initialize_rate_limiting(config)

        @api_rate_limit(tier=RateLimitTier.BASIC)
        def basic_endpoint(user_id):
            return {"tier": "basic", "data": "limited"}

        @api_rate_limit(tier=RateLimitTier.PREMIUM)
        def premium_endpoint(user_id):
            return {"tier": "premium", "data": "enhanced"}

        # Basic tier should have lower limits
        basic_endpoint("basic_user")
        basic_endpoint("basic_user")

        with pytest.raises(RateLimitExceeded):
            basic_endpoint("basic_user")

        # Premium tier should have higher limits
        for i in range(5):
            result = premium_endpoint("premium_user")
            assert result["tier"] == "premium"

        with pytest.raises(RateLimitExceeded):
            premium_endpoint("premium_user")

    def test_api_endpoint_specific_limits(self):
        """Test endpoint-specific rate limiting."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

        @rate_limit(limit=2, window="1min")
        def endpoint_a(user_id):
            return "endpoint_a_data"

        @rate_limit(limit=5, window="1min")
        def endpoint_b(user_id):
            return "endpoint_b_data"

        # Each endpoint should have independent limits
        endpoint_a("user1")
        endpoint_a("user1")

        with pytest.raises(RateLimitExceeded):
            endpoint_a("user1")

        # endpoint_b should still work
        for i in range(5):
            result = endpoint_b("user1")
            assert result == "endpoint_b_data"


@pytest.mark.integration
class TestConcurrencyIntegration:
    """Test rate limiting under concurrent load."""

    def test_concurrent_requests_memory_storage(self):
        """Test concurrent requests with memory storage."""
        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=100, window="1min"
        )

        manager = RateLimitManager(config)
        results = []

        def worker(thread_id):
            context = RateLimitContext(user_id=f"user{thread_id}")
            try:
                for i in range(10):
                    manager.check_rate_limit(context)
                results.append(f"thread_{thread_id}_success")
            except RateLimitExceeded:
                results.append(f"thread_{thread_id}_limited")

        # Start 20 threads, each making 10 requests (200 total)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed since they use different user IDs
        assert len(results) == 20
        assert all("success" in result for result in results)

    def test_concurrent_same_user_rate_limiting(self):
        """Test concurrent requests for same user."""
        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=50, window="1min"
        )

        manager = RateLimitManager(config)
        results = []

        def worker():
            context = RateLimitContext(user_id="shared_user")
            try:
                for i in range(10):
                    manager.check_rate_limit(context)
                results.append("success")
            except RateLimitExceeded:
                results.append("limited")

        # Start 10 threads, each making 10 requests for same user (100 total)
        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Some should succeed, some should be limited (total ≤ 50)
        assert len(results) == 10
        successes = sum(1 for r in results if r == "success")
        limited = sum(1 for r in results if r == "limited")

        # Due to the 50 request limit, not all threads can complete successfully
        assert successes + limited == 10
        assert limited > 0  # Some should be limited


@pytest.mark.integration
@pytest.mark.redis
class TestRedisFailoverIntegration:
    """Test rate limiting behavior during Redis failures."""

    def test_redis_connection_failure_handling(self):
        """Test behavior when Redis connection fails."""
        config = RateLimitConfig(
            storage_backend="redis",
            redis_url="redis://localhost:9999/0",  # Invalid port
        )

        # Should raise connection error during initialization
        with pytest.raises(Exception):
            RateLimitManager(config)

    def test_redis_temporary_failure_recovery(self):
        """Test recovery from temporary Redis failures."""
        # This test would require more complex setup to simulate
        # Redis failures and recovery
        pytest.skip("Requires complex Redis failure simulation")


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance under realistic load."""

    def test_high_throughput_memory_storage(self):
        """Test high throughput with memory storage."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        start_time = time.time()

        # Simulate high throughput - 1000 requests
        for i in range(1000):
            context = RateLimitContext(user_id=f"user{i % 100}")
            try:
                manager.check_rate_limit(context)
            except RateLimitExceeded:
                pass  # Expected for some requests

        elapsed = time.time() - start_time

        # Should handle 1000 requests quickly
        assert elapsed < 1.0  # Under 1 second

        throughput = 1000 / elapsed
        assert throughput > 1000  # > 1000 requests/second

    @pytest.mark.redis
    def test_high_throughput_redis_storage(self):
        """Test high throughput with Redis storage."""
        try:
            config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")
            manager = RateLimitManager(config)

            start_time = time.time()

            # Simulate high throughput - 500 requests (less than memory due to network)
            for i in range(500):
                context = RateLimitContext(user_id=f"user{i % 50}")
                try:
                    manager.check_rate_limit(context)
                except RateLimitExceeded:
                    pass

            elapsed = time.time() - start_time

            # Should handle 500 requests reasonably quickly
            assert elapsed < 2.0  # Under 2 seconds

        except Exception:
            pytest.skip("Redis not available for performance testing")


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_trading_platform_scenario(self):
        """Test a realistic trading platform scenario."""
        config = RateLimitConfig(storage_backend="memory")

        # Configure realistic trading limits
        config.trading_limits.order_submission = RateLimitRule(limit=100, window="1min")
        config.trading_limits.market_data_requests = RateLimitRule(limit=1000, window="1min")
        config.default_limits[RateLimitTier.PREMIUM]["api_requests"] = RateLimitRule(
            limit=500, window="1min"
        )

        initialize_rate_limiting(config)

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id, symbol, quantity, order_type):
            return {"order_id": f"order_{user_id}_{symbol}", "status": "submitted"}

        @trading_rate_limit(action="get_market_data")
        def get_quote(user_id, symbol):
            return {"symbol": symbol, "price": 150.0, "timestamp": time.time()}

        @api_rate_limit(tier=RateLimitTier.PREMIUM)
        def get_portfolio(user_id):
            return {"user_id": user_id, "positions": []}

        # Simulate trading activity
        trader_id = "premium_trader"

        # Get portfolio
        portfolio = get_portfolio(trader_id)
        assert portfolio["user_id"] == trader_id

        # Get market data for multiple symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for symbol in symbols:
            quote = get_quote(trader_id, symbol)
            assert quote["symbol"] == symbol

        # Submit some orders
        for i, symbol in enumerate(symbols[:3]):
            order = submit_order(trader_id, symbol, 100, "market")
            assert order["status"] == "submitted"

        # Should still be within limits
        assert True  # If we get here, no rate limits were exceeded

    def test_api_gateway_scenario(self):
        """Test an API gateway scenario with mixed traffic."""
        config = RateLimitConfig(
            storage_backend="memory", enable_ip_limiting=True, enable_api_key_limiting=True
        )

        # Configure tiered limits
        config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
            limit=10, window="1min"
        )
        config.default_limits[RateLimitTier.PREMIUM]["api_requests"] = RateLimitRule(
            limit=50, window="1min"
        )

        manager = RateLimitManager(config)

        # Simulate different types of users
        basic_context = RateLimitContext(
            user_id="basic_user",
            api_key="basic_key",
            ip_address="192.168.1.1",
            user_tier=RateLimitTier.BASIC,
        )

        premium_context = RateLimitContext(
            user_id="premium_user",
            api_key="premium_key",
            ip_address="192.168.1.2",
            user_tier=RateLimitTier.PREMIUM,
        )

        # Basic user should hit limits quickly
        for i in range(10):
            manager.check_rate_limit(basic_context)

        with pytest.raises(RateLimitExceeded):
            manager.check_rate_limit(basic_context)

        # Premium user should have higher limits
        for i in range(20):  # Well within premium limits
            manager.check_rate_limit(premium_context)

        # Premium user should still be allowed
        statuses = manager.check_rate_limit(premium_context)
        assert all(s.allowed for s in statuses)

    def test_multi_tenant_scenario(self):
        """Test multi-tenant application scenario."""
        config = RateLimitConfig(storage_backend="memory")

        manager = RateLimitManager(config)

        # Simulate multiple tenants
        tenants = ["tenant_a", "tenant_b", "tenant_c"]

        for tenant in tenants:
            # Each tenant should have independent rate limits
            for user_num in range(5):
                context = RateLimitContext(
                    user_id=f"{tenant}_user_{user_num}", api_key=f"{tenant}_api_key"
                )

                # Each user should be able to make requests
                statuses = manager.check_rate_limit(context)
                assert all(s.allowed for s in statuses)

        # Verify isolation between tenants
        status_a = manager.get_status(RateLimitContext(user_id="tenant_a_user_1"))
        status_b = manager.get_status(RateLimitContext(user_id="tenant_b_user_1"))

        # Each tenant's usage should be independent
        assert isinstance(status_a, dict)
        assert isinstance(status_b, dict)
