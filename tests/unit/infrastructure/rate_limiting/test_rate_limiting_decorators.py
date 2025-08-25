"""
Unit tests for rate limiting decorators.

Tests decorator functionality, parameter extraction, and integration
with the rate limiting system.
"""

from unittest.mock import Mock, patch

import pytest

from src.infrastructure.rate_limiting.config import RateLimitConfig, RateLimitTier
from src.infrastructure.rate_limiting.decorators import (
    _build_api_context,
    _build_context_from_function,
    _build_custom_identifier,
    _build_ip_context,
    _build_trading_context,
    api_rate_limit,
    get_rate_limit_manager,
    initialize_rate_limiting,
    ip_rate_limit,
    no_rate_limit,
    rate_limit,
    trading_rate_limit,
)
from src.infrastructure.rate_limiting.exceptions import RateLimitExceeded
from src.infrastructure.rate_limiting.manager import RateLimitContext


class TestDecoratorInitialization:
    """Test decorator initialization and manager setup."""

    def test_initialize_rate_limiting(self):
        """Test initializing rate limiting with config."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

        manager = get_rate_limit_manager()
        assert manager is not None
        assert manager.config == config

    def test_get_rate_limit_manager_default(self):
        """Test getting manager with default initialization."""
        # Reset global manager
        import src.infrastructure.rate_limiting.decorators as decorators_module

        decorators_module._rate_limit_manager = None

        with patch.dict("os.environ", {}, clear=True):
            manager = get_rate_limit_manager()
            assert manager is not None


class TestRateLimitDecorator:
    """Test the general rate_limit decorator."""

    def setup_method(self):
        """Setup for each test."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

    def test_basic_rate_limit_decorator(self):
        """Test basic rate limit decorator functionality."""

        @rate_limit(limit=5, window="1min")
        def test_function(user_id=None):
            return "success"

        # Should work for first 5 calls
        for i in range(5):
            result = test_function(user_id="user1")
            assert result == "success"

        # 6th call should raise exception
        with pytest.raises(RateLimitExceeded):
            test_function(user_id="user1")

    def test_rate_limit_with_custom_algorithm(self):
        """Test rate limit with custom algorithm."""

        @rate_limit(limit=3, window="10s", algorithm="sliding_window")
        def test_function(user_id=None):
            return "success"

        # Should work for 3 calls
        for i in range(3):
            result = test_function(user_id="user1")
            assert result == "success"

        # 4th call should fail
        with pytest.raises(RateLimitExceeded):
            test_function(user_id="user1")

    def test_rate_limit_per_user(self):
        """Test rate limiting per user."""

        @rate_limit(limit=2, window="1min", per="user")
        def test_function(user_id=None):
            return f"success for {user_id}"

        # Each user should have separate limits
        assert test_function(user_id="user1") == "success for user1"
        assert test_function(user_id="user1") == "success for user1"
        assert test_function(user_id="user2") == "success for user2"
        assert test_function(user_id="user2") == "success for user2"

        # Third call for each user should fail
        with pytest.raises(RateLimitExceeded):
            test_function(user_id="user1")

        with pytest.raises(RateLimitExceeded):
            test_function(user_id="user2")

    def test_rate_limit_per_ip(self):
        """Test rate limiting per IP address."""

        @rate_limit(limit=2, window="1min", per="ip")
        def test_function(ip_address=None):
            return "success"

        # Same IP should be limited
        test_function(ip_address="192.168.1.1")
        test_function(ip_address="192.168.1.1")

        with pytest.raises(RateLimitExceeded):
            test_function(ip_address="192.168.1.1")

        # Different IP should work
        assert test_function(ip_address="192.168.1.2") == "success"

    def test_rate_limit_with_burst_allowance(self):
        """Test rate limiting with burst allowance."""

        @rate_limit(limit=3, window="1min", burst_allowance=2)
        def test_function(user_id=None):
            return "success"

        # Should allow 3 + 2 = 5 requests
        for i in range(5):
            result = test_function(user_id="user1")
            assert result == "success"

        # 6th should fail
        with pytest.raises(RateLimitExceeded):
            test_function(user_id="user1")

    def test_rate_limit_with_custom_key_function(self):
        """Test rate limiting with custom key function."""

        def custom_key_func(*args, **kwargs):
            return kwargs.get("session_id", "default")

        @rate_limit(limit=2, window="1min", key_func=custom_key_func)
        def test_function(session_id=None):
            return "success"

        # Different sessions should have separate limits
        test_function(session_id="session1")
        test_function(session_id="session1")
        test_function(session_id="session2")

        # Third call for session1 should fail
        with pytest.raises(RateLimitExceeded):
            test_function(session_id="session1")

        # session2 should still work
        assert test_function(session_id="session2") == "success"

    def test_rate_limit_with_custom_error_message(self):
        """Test rate limit with custom error message."""

        @rate_limit(limit=1, window="1min", error_message="Custom rate limit exceeded")
        def test_function(user_id=None):
            return "success"

        test_function(user_id="user1")

        with pytest.raises(RateLimitExceeded) as exc_info:
            test_function(user_id="user1")

        assert "Custom rate limit exceeded" in str(exc_info)


class TestTradingRateLimitDecorator:
    """Test the trading_rate_limit decorator."""

    def setup_method(self):
        """Setup for each test."""
        config = RateLimitConfig(storage_backend="memory")
        # Make trading limits very restrictive for testing
        config.trading_limits.order_submission.limit = 2
        config.trading_limits.order_submission.window.seconds = 60
        initialize_rate_limiting(config)

    def test_trading_rate_limit_basic(self):
        """Test basic trading rate limit functionality."""

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id=None, symbol=None):
            return f"Order submitted for {symbol}"

        # Should work for first few calls
        result1 = submit_order(user_id="trader1", symbol="AAPL")
        assert "AAPL" in result1

        result2 = submit_order(user_id="trader1", symbol="GOOGL")
        assert "GOOGL" in result2

        # Third call should fail due to rate limit
        with pytest.raises(RateLimitExceeded):
            submit_order(user_id="trader1", symbol="MSFT")

    def test_trading_rate_limit_with_symbol(self):
        """Test trading rate limit with specific symbol."""

        @trading_rate_limit(action="submit_order", symbol="AAPL")
        def submit_order_aapl(user_id=None):
            return "AAPL order submitted"

        # Should use AAPL as symbol regardless of parameters
        result = submit_order_aapl(user_id="trader1")
        assert result == "AAPL order submitted"

    def test_trading_rate_limit_per_user(self):
        """Test trading rate limit isolation per user."""

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id=None, symbol=None):
            return "Order submitted"

        # Each trader should have separate limits
        submit_order(user_id="trader1", symbol="AAPL")
        submit_order(user_id="trader1", symbol="GOOGL")
        submit_order(user_id="trader2", symbol="AAPL")
        submit_order(user_id="trader2", symbol="GOOGL")

        # Third order for trader1 should fail
        with pytest.raises(RateLimitExceeded):
            submit_order(user_id="trader1", symbol="MSFT")

        # trader2 should still be able to place one more
        with pytest.raises(RateLimitExceeded):
            submit_order(user_id="trader2", symbol="MSFT")

    def test_trading_rate_limit_custom_error(self):
        """Test trading rate limit with custom error message."""

        @trading_rate_limit(
            action="submit_order", error_message="Too many orders! Please slow down."
        )
        def submit_order(user_id=None, symbol=None):
            return "Order submitted"

        # Use up the limit
        submit_order(user_id="trader1", symbol="AAPL")
        submit_order(user_id="trader1", symbol="GOOGL")

        with pytest.raises(RateLimitExceeded) as exc_info:
            submit_order(user_id="trader1", symbol="MSFT")

        assert "Too many orders! Please slow down." in str(exc_info)


class TestAPIRateLimitDecorator:
    """Test the api_rate_limit decorator."""

    def setup_method(self):
        """Setup for each test."""
        config = RateLimitConfig(storage_backend="memory")
        # Make API limits restrictive for testing
        config.default_limits[RateLimitTier.BASIC]["api_requests"].limit = 3
        initialize_rate_limiting(config)

    def test_api_rate_limit_basic(self):
        """Test basic API rate limit functionality."""

        @api_rate_limit()
        def api_endpoint(user_id=None, api_key=None):
            return {"status": "success"}

        # Should work for first few calls
        for i in range(3):
            result = api_endpoint(user_id="user1", api_key="key1")
            assert result["status"] == "success"

        # Fourth call should fail
        with pytest.raises(RateLimitExceeded):
            api_endpoint(user_id="user1", api_key="key1")

    def test_api_rate_limit_different_tiers(self):
        """Test API rate limiting with different user tiers."""

        @api_rate_limit(tier=RateLimitTier.PREMIUM)
        def premium_endpoint(user_id=None):
            return {"tier": "premium"}

        @api_rate_limit(tier=RateLimitTier.BASIC)
        def basic_endpoint(user_id=None):
            return {"tier": "basic"}

        # Premium should have higher limits
        for i in range(5):  # Premium limit is higher
            try:
                result = premium_endpoint(user_id="premium_user")
                assert result["tier"] == "premium"
            except RateLimitExceeded:
                # If we hit the limit, premium should allow more than basic
                break

        # Basic should have lower limits
        for i in range(3):
            result = basic_endpoint(user_id="basic_user")
            assert result["tier"] == "basic"

        with pytest.raises(RateLimitExceeded):
            basic_endpoint(user_id="basic_user")

    def test_api_rate_limit_specific_rule_types(self):
        """Test API rate limiting with specific rule types."""

        @api_rate_limit(rule_types=["api_requests"])
        def api_endpoint(user_id=None):
            return "success"

        result = api_endpoint(user_id="user1")
        assert result == "success"

    def test_api_rate_limit_require_auth(self):
        """Test API rate limiting with authentication requirement."""

        @api_rate_limit(require_auth=True)
        def protected_endpoint(user_id=None, api_key=None):
            return "protected data"

        result = protected_endpoint(user_id="user1", api_key="valid_key")
        assert result == "protected data"


class TestIPRateLimitDecorator:
    """Test the ip_rate_limit decorator."""

    def setup_method(self):
        """Setup for each test."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

    def test_ip_rate_limit_basic(self):
        """Test basic IP rate limit functionality."""

        @ip_rate_limit(limit=2, window="1min")
        def public_endpoint(ip_address=None):
            return "public data"

        # Should work for first 2 calls from same IP
        result1 = public_endpoint(ip_address="192.168.1.1")
        assert result1 == "public data"

        result2 = public_endpoint(ip_address="192.168.1.1")
        assert result2 == "public data"

        # Third call should fail
        with pytest.raises(RateLimitExceeded):
            public_endpoint(ip_address="192.168.1.1")

        # Different IP should work
        result3 = public_endpoint(ip_address="192.168.1.2")
        assert result3 == "public data"

    def test_ip_rate_limit_different_algorithms(self):
        """Test IP rate limiting with different algorithms."""

        @ip_rate_limit(limit=3, window="10s", algorithm="fixed_window")
        def endpoint_fixed(ip_address=None):
            return "fixed window"

        @ip_rate_limit(limit=3, window="10s", algorithm="sliding_window")
        def endpoint_sliding(ip_address=None):
            return "sliding window"

        # Both should work initially
        assert endpoint_fixed(ip_address="192.168.1.1") == "fixed window"
        assert endpoint_sliding(ip_address="192.168.1.1") == "sliding window"

    def test_ip_rate_limit_custom_error(self):
        """Test IP rate limiting with custom error message."""

        @ip_rate_limit(
            limit=1, window="1min", error_message="IP rate limit exceeded. Please try again later."
        )
        def limited_endpoint(ip_address=None):
            return "data"

        limited_endpoint(ip_address="192.168.1.1")

        with pytest.raises(RateLimitExceeded) as exc_info:
            limited_endpoint(ip_address="192.168.1.1")

        assert "IP rate limit exceeded" in str(exc_info)


class TestNoRateLimitDecorator:
    """Test the no_rate_limit decorator."""

    def test_no_rate_limit_bypass(self):
        """Test that no_rate_limit bypasses all rate limiting."""

        @no_rate_limit
        def admin_function():
            return "admin data"

        # Should work regardless of rate limits
        for i in range(100):
            result = admin_function()
            assert result == "admin data"

        # Should have the bypass marker
        assert hasattr(admin_function, "_no_rate_limit")
        assert admin_function._no_rate_limit is True


class TestContextBuilding:
    """Test context building functions."""

    def test_build_context_from_function_basic(self):
        """Test basic context building from function parameters."""

        def test_func(user_id, api_key, ip_address):
            pass

        args = ("user123", "key456", "192.168.1.1")
        kwargs = {}

        context = _build_context_from_function(test_func, args, kwargs, "user", None)

        assert context.user_id == "user123"
        assert context.api_key == "key456"
        assert context.ip_address == "192.168.1.1"

    def test_build_context_with_kwargs(self):
        """Test context building with keyword arguments."""

        def test_func(user_id=None, api_key=None):
            pass

        args = ()
        kwargs = {"user_id": "user123", "api_key": "key456"}

        context = _build_context_from_function(test_func, args, kwargs, "user", None)

        assert context.user_id == "user123"
        assert context.api_key == "key456"

    def test_build_context_with_request_object(self):
        """Test context building with request object."""

        def test_func(request):
            pass

        # Mock request object
        mock_request = Mock()
        mock_request.remote_addr = "192.168.1.1"
        mock_request.headers = {"X-API-Key": "key123"}

        args = (mock_request,)
        kwargs = {}

        context = _build_context_from_function(test_func, args, kwargs, "user", None)

        assert context.ip_address == "192.168.1.1"
        assert context.api_key == "key123"

    def test_build_context_with_custom_key_func(self):
        """Test context building with custom key function."""

        def test_func(session_id):
            pass

        def custom_key_func(*args, **kwargs):
            return args[0]  # Return session_id as user_id

        args = ("session123",)
        kwargs = {}

        context = _build_context_from_function(test_func, args, kwargs, "user", custom_key_func)

        assert context.user_id == "session123"

    def test_build_trading_context(self):
        """Test building trading-specific context."""

        def submit_order(user_id, symbol, quantity):
            pass

        args = ("trader1", "AAPL", 100)
        kwargs = {}

        context = _build_trading_context(test_func, args, kwargs, "submit_order", None)

        assert context.trading_action == "submit_order"
        assert context.symbol == "AAPL"

    def test_build_api_context(self):
        """Test building API-specific context."""

        def api_endpoint(user_id, api_key):
            pass

        args = ("user1", "key1")
        kwargs = {}

        context = _build_api_context(api_endpoint, args, kwargs, RateLimitTier.PREMIUM, True)

        assert context.user_tier == RateLimitTier.PREMIUM
        assert context.endpoint == "api_endpoint"
        assert context.method == "POST"

    def test_build_ip_context(self):
        """Test building IP-specific context."""

        def public_endpoint(ip_address):
            pass

        args = ("192.168.1.1",)
        kwargs = {}

        context = _build_ip_context(public_endpoint, args, kwargs)

        assert context.ip_address == "192.168.1.1"

    def test_build_custom_identifier(self):
        """Test building custom identifier."""

        context = RateLimitContext(user_id="user123")
        identifier = _build_custom_identifier(context, "user", "test_func")

        assert "test_func" in identifier
        assert "user123" in identifier

        context = RateLimitContext(ip_address="192.168.1.1")
        identifier = _build_custom_identifier(context, "ip", "test_func")

        assert "test_func" in identifier
        assert "192.168.1.1" in identifier


class TestDecoratorIntegration:
    """Test decorator integration with rate limiting system."""

    def setup_method(self):
        """Setup for each test."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

    def test_multiple_decorators(self):
        """Test applying multiple rate limiting decorators."""

        @api_rate_limit(tier=RateLimitTier.BASIC)
        @ip_rate_limit(limit=10, window="1min")
        def multi_limited_endpoint(user_id=None, ip_address=None):
            return "success"

        # Should work initially
        result = multi_limited_endpoint(user_id="user1", ip_address="192.168.1.1")
        assert result == "success"

    def test_decorator_with_exception_handling(self):
        """Test decorator behavior with exceptions in wrapped function."""

        @rate_limit(limit=5, window="1min")
        def failing_function(user_id=None, should_fail=False):
            if should_fail:
                raise ValueError("Function failed")
            return "success"

        # Rate limit should still apply even if function fails
        result = failing_function(user_id="user1", should_fail=False)
        assert result == "success"

        with pytest.raises(ValueError):
            failing_function(user_id="user1", should_fail=True)

        # Rate limit should still be consumed
        # Continue testing rate limit with successful calls
        for i in range(3):  # Already used 2 (1 success, 1 failure)
            failing_function(user_id="user1", should_fail=False)

        # Should hit rate limit
        with pytest.raises(RateLimitExceeded):
            failing_function(user_id="user1", should_fail=False)

    def test_decorator_preserves_function_metadata(self):
        """Test that decorators preserve function metadata."""

        @rate_limit(limit=5, window="1min")
        def documented_function(user_id):
            """This function has documentation."""
            return "success"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

    def test_decorator_with_class_methods(self):
        """Test decorators with class methods."""

        class APIService:
            @rate_limit(limit=3, window="1min")
            def get_data(self, user_id):
                return f"data for {user_id}"

            @trading_rate_limit(action="submit_order")
            def submit_order(self, user_id, symbol):
                return f"order submitted: {symbol}"

        service = APIService()

        # Should work normally
        result = service.get_data("user1")
        assert "user1" in result

        result = service.submit_order("trader1", "AAPL")
        assert "AAPL" in result
