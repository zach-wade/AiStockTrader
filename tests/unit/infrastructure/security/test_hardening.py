"""
Comprehensive unit tests for security hardening module.

Tests rate limiting, request throttling, security headers, authentication,
request signing, and all security control mechanisms.
"""

import threading
import time
from unittest.mock import patch

import pytest

from src.infrastructure.security.hardening import (
    InvalidTokenError,
    RateLimiter,
    RateLimitExceeded,
    RateLimitRule,
    RequestSigner,
    RequestThrottler,
    SecurityConfig,
    SecurityError,
    SecurityHardening,
    SecurityHeaders,
    ThrottlingError,
    create_trading_security_config,
    get_trading_rate_limits,
    secure_endpoint,
)


class TestSecurityErrors:
    """Test security error classes."""

    def test_security_error(self):
        """Test base SecurityError."""
        error = SecurityError("Security violation detected")
        assert str(error) == "Security violation detected"

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded error."""
        error = RateLimitExceeded("Rate limit hit", retry_after=60)
        assert str(error) == "Rate limit hit"
        assert error.retry_after == 60

    def test_rate_limit_exceeded_without_retry(self):
        """Test RateLimitExceeded without retry_after."""
        error = RateLimitExceeded("Rate limit hit")
        assert error.retry_after is None

    def test_throttling_error(self):
        """Test ThrottlingError."""
        error = ThrottlingError("Request throttled")
        assert str(error) == "Request throttled"
        assert isinstance(error, SecurityError)

    def test_invalid_token_error(self):
        """Test InvalidTokenError."""
        error = InvalidTokenError("Invalid auth token")
        assert str(error) == "Invalid auth token"
        assert isinstance(error, SecurityError)


class TestRateLimitRule:
    """Test RateLimitRule configuration."""

    def test_default_rule(self):
        """Test default rate limit rule."""
        rule = RateLimitRule(max_requests=100, window_seconds=60)
        assert rule.max_requests == 100
        assert rule.window_seconds == 60
        assert rule.burst_allowance == 0
        assert rule.cooldown_seconds == 60

    def test_custom_rule(self):
        """Test custom rate limit rule."""
        rule = RateLimitRule(
            max_requests=50, window_seconds=30, burst_allowance=10, cooldown_seconds=120
        )
        assert rule.max_requests == 50
        assert rule.window_seconds == 30
        assert rule.burst_allowance == 10
        assert rule.cooldown_seconds == 120


class TestSecurityConfig:
    """Test SecurityConfig class."""

    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.enable_rate_limiting is True
        assert config.enable_request_throttling is True
        assert config.enable_security_headers is True
        assert config.enable_request_signing is False
        assert config.enable_ip_whitelist is False
        assert config.max_request_size == 1024 * 1024
        assert config.hmac_secret_key is None

    def test_custom_config(self):
        """Test custom security configuration."""
        rule = RateLimitRule(200, 120)
        config = SecurityConfig(
            enable_rate_limiting=False,
            enable_request_signing=True,
            hmac_secret_key="secret123",
            default_rate_limit=rule,
            whitelisted_ips={"192.168.1.1", "10.0.0.1"},
            max_request_size=2048,
        )
        assert config.enable_rate_limiting is False
        assert config.enable_request_signing is True
        assert config.hmac_secret_key == "secret123"
        assert config.default_rate_limit == rule
        assert "192.168.1.1" in config.whitelisted_ips
        assert config.max_request_size == 2048


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limit_allows_requests_under_limit(self):
        """Test rate limiter allows requests under limit."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=1)

        # First 3 requests should be allowed
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True

        # 4th request should be denied
        assert limiter.is_allowed("user1", rule) is False

    def test_rate_limit_window_expiry(self):
        """Test rate limit window expiry."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=1)

        # Use up the limit
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed("user1", rule) is True

    def test_rate_limit_different_identifiers(self):
        """Test rate limiting for different identifiers."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=1, window_seconds=60)

        # Different identifiers have separate limits
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user2", rule) is True
        assert limiter.is_allowed("user3", rule) is True

        # Each identifier exhausted individually
        assert limiter.is_allowed("user1", rule) is False
        assert limiter.is_allowed("user2", rule) is False
        assert limiter.is_allowed("user3", rule) is False

    def test_get_remaining_requests(self):
        """Test getting remaining requests."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=5, window_seconds=60)

        # Initially should have all requests
        assert limiter.get_remaining_requests("user1", rule) == 5

        # Use some requests
        limiter.is_allowed("user1", rule)
        assert limiter.get_remaining_requests("user1", rule) == 4

        limiter.is_allowed("user1", rule)
        limiter.is_allowed("user1", rule)
        assert limiter.get_remaining_requests("user1", rule) == 2

    def test_reset_limit(self):
        """Test resetting rate limit."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=60)

        # Use up limit
        limiter.is_allowed("user1", rule)
        limiter.is_allowed("user1", rule)
        assert limiter.is_allowed("user1", rule) is False

        # Reset
        limiter.reset_limit("user1")

        # Should be allowed again
        assert limiter.is_allowed("user1", rule) is True

    def test_rate_limit_thread_safety(self):
        """Test rate limiter thread safety."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=100, window_seconds=1)
        results = []

        def make_requests():
            for _ in range(50):
                results.append(limiter.is_allowed("shared", rule))

        # Create multiple threads
        threads = [threading.Thread(target=make_requests) for _ in range(4)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have exactly 100 True values (the limit)
        assert sum(results) == 100


class TestRequestThrottler:
    """Test RequestThrottler class."""

    def test_throttler_allows_concurrent_requests(self):
        """Test throttler allows requests under limit."""
        throttler = RequestThrottler(max_concurrent=3)

        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")

        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")

        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")

        # Should deny 4th concurrent request
        assert throttler.can_process_request("user1") is False

    def test_throttler_releases_on_end_request(self):
        """Test throttler releases slots on end_request."""
        throttler = RequestThrottler(max_concurrent=2)

        # Start 2 requests
        throttler.start_request("user1")
        throttler.start_request("user1")
        assert throttler.can_process_request("user1") is False

        # End one request
        throttler.end_request("user1")
        assert throttler.can_process_request("user1") is True

        # End another
        throttler.end_request("user1")
        assert throttler.can_process_request("user1") is True

    def test_throttler_different_identifiers(self):
        """Test throttling for different identifiers."""
        throttler = RequestThrottler(max_concurrent=1)

        throttler.start_request("user1")
        throttler.start_request("user2")

        assert throttler.can_process_request("user1") is False
        assert throttler.can_process_request("user2") is False

        throttler.end_request("user1")
        assert throttler.can_process_request("user1") is True
        assert throttler.can_process_request("user2") is False

    def test_throttler_thread_safety(self):
        """Test throttler thread safety."""
        throttler = RequestThrottler(max_concurrent=10)

        def process_requests():
            for _ in range(20):
                if throttler.can_process_request("shared"):
                    throttler.start_request("shared")
                    time.sleep(0.01)  # Simulate processing
                    throttler.end_request("shared")

        threads = [threading.Thread(target=process_requests) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should end with no active requests
        assert throttler._active_requests["shared"] == 0


class TestSecurityHeaders:
    """Test SecurityHeaders class."""

    def test_default_headers(self):
        """Test default security headers."""
        headers = SecurityHeaders.get_security_headers(include_trading_headers=False)

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers

    def test_trading_headers(self):
        """Test trading-specific headers."""
        headers = SecurityHeaders.get_security_headers(include_trading_headers=True)

        # Should include default headers
        assert "X-Content-Type-Options" in headers

        # Should include trading headers
        assert headers["X-Trading-System"] == "AI-Trader-v1.0"
        assert headers["X-API-Version"] == "1.0"
        assert headers["Cache-Control"] == "no-store, no-cache, must-revalidate, private"
        assert headers["Pragma"] == "no-cache"

    @patch(
        "src.domain.services.request_validation_service.RequestValidationService.validate_request_headers"
    )
    def test_check_request_headers(self, mock_validate):
        """Test request header validation."""
        mock_validate.return_value = []  # No errors

        headers = {"Authorization": "Bearer token123"}
        assert SecurityHeaders.check_request_headers(headers) is True
        mock_validate.assert_called_once_with(headers)

        mock_validate.return_value = ["Missing required header"]
        assert SecurityHeaders.check_request_headers(headers) is False


class TestRequestSigner:
    """Test RequestSigner class."""

    def test_signer_initialization(self):
        """Test RequestSigner initialization."""
        signer = RequestSigner("my_secret_key")
        assert signer.secret_key == b"my_secret_key"

        # Test with bytes
        signer = RequestSigner(b"byte_secret")
        assert signer.secret_key == b"byte_secret"

    def test_signer_initialization_empty_key(self):
        """Test RequestSigner with empty key."""
        with pytest.raises(ValueError) as exc_info:
            RequestSigner("")
        assert "Secret key cannot be empty" in str(exc_info)

    def test_sign_request(self):
        """Test request signing."""
        signer = RequestSigner("test_secret")

        headers = signer.sign_request(
            "POST", "/api/orders", '{"symbol":"AAPL"}', timestamp=1234567890
        )

        assert "X-Signature" in headers
        assert headers["X-Timestamp"] == "1234567890"
        assert len(headers["X-Signature"]) == 64  # SHA256 hex digest length

    def test_sign_request_auto_timestamp(self):
        """Test request signing with automatic timestamp."""
        signer = RequestSigner("test_secret")

        with patch("time.time", return_value=1234567890.5):
            headers = signer.sign_request("GET", "/api/portfolio")
            assert headers["X-Timestamp"] == "1234567890"

    def test_check_request_valid(self):
        """Test valid request signature verification."""
        signer = RequestSigner("test_secret")

        # Sign a request
        headers = signer.sign_request("POST", "/api/orders", "data", timestamp=1234567890)

        # Verify with same parameters
        with patch("time.time", return_value=1234567890):
            is_valid = signer.check_request(
                "POST", "/api/orders", "data", headers["X-Signature"], headers["X-Timestamp"]
            )
            assert is_valid is True

    def test_check_request_invalid_signature(self):
        """Test invalid signature verification."""
        signer = RequestSigner("test_secret")

        is_valid = signer.check_request(
            "POST", "/api/orders", "data", "invalid_signature", "1234567890"
        )
        assert is_valid is False

    def test_check_request_expired_timestamp(self):
        """Test expired timestamp verification."""
        signer = RequestSigner("test_secret")

        # Sign with old timestamp
        headers = signer.sign_request("GET", "/api", timestamp=1000000000)

        # Verify with current time much later
        with patch("time.time", return_value=2000000000):
            is_valid = signer.check_request(
                "GET", "/api", "", headers["X-Signature"], headers["X-Timestamp"], max_age=300
            )
            assert is_valid is False

    def test_check_request_invalid_timestamp(self):
        """Test invalid timestamp format."""
        signer = RequestSigner("test_secret")

        is_valid = signer.check_request("GET", "/api", "", "some_signature", "not_a_number")
        assert is_valid is False


class TestSecurityHardening:
    """Test SecurityHardening main class."""

    def test_initialization_default(self):
        """Test default initialization."""
        hardening = SecurityHardening()
        assert hardening.config.enable_rate_limiting is True
        assert hardening.rate_limiter is not None
        assert hardening.throttler is not None
        assert hardening.request_signer is None

    def test_initialization_with_signing(self):
        """Test initialization with request signing."""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret123")
        hardening = SecurityHardening(config)
        assert hardening.request_signer is not None

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_check_rate_limit_success(self, mock_service):
        """Test successful rate limit check."""
        mock_service.get_rate_limit_for_endpoint.return_value = {
            "max_requests": 10,
            "window_seconds": 60,
            "burst_allowance": 0,
        }
        mock_service.get_cooldown_period.return_value = 60

        hardening = SecurityHardening()

        # Should pass for first request
        assert hardening.check_rate_limit("user1", "/api/orders") is True

    def test_check_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        config = SecurityConfig(default_rate_limit=RateLimitRule(max_requests=1, window_seconds=60))
        hardening = SecurityHardening(config)

        # First request OK
        hardening.check_rate_limit("user1")

        # Second request should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            hardening.check_rate_limit("user1")
        assert "Rate limit exceeded" in str(exc_info)
        assert exc_info.retry_after == 60

    def test_check_rate_limit_disabled(self):
        """Test rate limiting when disabled."""
        config = SecurityConfig(enable_rate_limiting=False)
        hardening = SecurityHardening(config)

        # Should always pass
        for _ in range(100):
            assert hardening.check_rate_limit("user1") is True

    def test_check_throttling_success(self):
        """Test successful throttling check."""
        hardening = SecurityHardening()
        assert hardening.check_throttling("user1") is True

    def test_check_throttling_exceeded(self):
        """Test throttling exceeded."""
        hardening = SecurityHardening()

        # Simulate max concurrent requests
        hardening.throttler._active_requests["user1"] = 100

        with pytest.raises(ThrottlingError) as exc_info:
            hardening.check_throttling("user1")
        assert "Request throttled" in str(exc_info)

    def test_check_throttling_disabled(self):
        """Test throttling when disabled."""
        config = SecurityConfig(enable_request_throttling=False)
        hardening = SecurityHardening(config)

        # Should always pass
        hardening.throttler._active_requests["user1"] = 1000
        assert hardening.check_throttling("user1") is True

    def test_request_processing_lifecycle(self):
        """Test request processing lifecycle."""
        hardening = SecurityHardening()

        # Start processing
        hardening.start_request_processing("user1")
        assert hardening.throttler._active_requests["user1"] == 1

        # End processing
        hardening.end_request_processing("user1")
        assert hardening.throttler._active_requests["user1"] == 0

    @patch(
        "src.domain.services.request_validation_service.RequestValidationService.validate_request_headers"
    )
    def test_check_request_headers(self, mock_validate):
        """Test request header validation."""
        mock_validate.return_value = []

        hardening = SecurityHardening()
        headers = {"Content-Type": "application/json"}

        assert hardening.check_request_headers(headers) is True
        mock_validate.assert_called_once_with(headers)

    def test_get_security_headers(self):
        """Test getting security headers."""
        hardening = SecurityHardening()
        headers = hardening.get_security_headers()

        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers

    def test_get_security_headers_disabled(self):
        """Test getting headers when disabled."""
        config = SecurityConfig(enable_security_headers=False)
        hardening = SecurityHardening(config)

        headers = hardening.get_security_headers()
        assert headers == {}

    def test_check_request_signature_valid(self):
        """Test valid request signature check."""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret123")
        hardening = SecurityHardening(config)

        # Create valid signature
        headers = hardening.request_signer.sign_request("POST", "/api", "body")

        with patch("time.time", return_value=int(headers["X-Timestamp"])):
            is_valid = hardening.check_request_signature("POST", "/api", "body", headers)
            assert is_valid is True

    def test_check_request_signature_disabled(self):
        """Test signature check when disabled."""
        hardening = SecurityHardening()

        # Should always pass when disabled
        assert hardening.check_request_signature("GET", "/", "", {}) is True

    def test_check_ip_whitelist_allowed(self):
        """Test IP whitelist with allowed IP."""
        config = SecurityConfig(
            enable_ip_whitelist=True, whitelisted_ips={"192.168.1.1", "10.0.0.1"}
        )
        hardening = SecurityHardening(config)

        assert hardening.check_ip_whitelist("192.168.1.1") is True

    def test_check_ip_whitelist_blocked(self):
        """Test IP whitelist with blocked IP."""
        config = SecurityConfig(
            enable_ip_whitelist=True, whitelisted_ips={"192.168.1.1"}, blacklisted_ips={"10.0.0.1"}
        )
        hardening = SecurityHardening(config)

        # Blacklisted IP
        with pytest.raises(SecurityError) as exc_info:
            hardening.check_ip_whitelist("10.0.0.1")
        assert "blacklisted" in str(exc_info)

        # Not whitelisted
        with pytest.raises(SecurityError) as exc_info:
            hardening.check_ip_whitelist("192.168.1.2")
        assert "not whitelisted" in str(exc_info)

    def test_check_ip_whitelist_disabled(self):
        """Test IP whitelist when disabled."""
        config = SecurityConfig(enable_ip_whitelist=False)
        hardening = SecurityHardening(config)

        # Should always pass
        assert hardening.check_ip_whitelist("any.ip.address") is True


class TestSecureEndpointDecorator:
    """Test secure_endpoint decorator."""

    def test_secure_endpoint_basic(self):
        """Test basic endpoint security."""

        @secure_endpoint()
        def api_endpoint(**kwargs):
            return "success"

        # Without hardening configured
        result = api_endpoint()
        assert result == "success"

        # With hardening
        hardening = SecurityHardening()
        result = api_endpoint(_security_hardening=hardening)
        assert result == "success"

    def test_secure_endpoint_with_rate_limit(self):
        """Test endpoint with rate limiting."""
        config = SecurityConfig(default_rate_limit=RateLimitRule(max_requests=1, window_seconds=60))
        hardening = SecurityHardening(config)

        @secure_endpoint(identifier_func=lambda **k: "test_user")
        def api_endpoint(**kwargs):
            return "success"

        # First call should succeed
        result = api_endpoint(_security_hardening=hardening)
        assert result == "success"

        # Second call should fail
        with pytest.raises(RateLimitExceeded):
            api_endpoint(_security_hardening=hardening)

    def test_secure_endpoint_with_custom_identifier(self):
        """Test endpoint with custom identifier function."""
        calls = []

        def get_identifier(**kwargs):
            user_id = kwargs.get("user_id", "anonymous")
            calls.append(user_id)
            return user_id

        @secure_endpoint(identifier_func=get_identifier)
        def api_endpoint(**kwargs):
            return f"User: {kwargs.get('user_id')}"

        hardening = SecurityHardening()
        result = api_endpoint(user_id="john", _security_hardening=hardening)

        assert result == "User: john"
        assert calls == ["john"]

    def test_secure_endpoint_cleanup_on_error(self):
        """Test endpoint cleanup on error."""
        hardening = SecurityHardening()

        @secure_endpoint()
        def failing_endpoint(**kwargs):
            raise ValueError("Intentional error")

        with pytest.raises(ValueError):
            failing_endpoint(client_ip="192.168.1.1", _security_hardening=hardening)

        # Should have cleaned up
        assert hardening.throttler._active_requests["192.168.1.1"] == 0


class TestTradingFunctions:
    """Test trading-specific functions."""

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_get_trading_rate_limits(self, mock_service):
        """Test getting trading rate limits."""
        mock_service.TRADING_RATE_LIMITS = {
            "/api/orders": {"max_requests": 100, "window_seconds": 60, "burst_allowance": 10},
            "/api/positions": {"max_requests": 200, "window_seconds": 60, "burst_allowance": 20},
        }

        limits = get_trading_rate_limits()

        assert "/api/orders" in limits
        assert limits["/api/orders"].max_requests == 100
        assert limits["/api/orders"].burst_allowance == 10

        assert "/api/positions" in limits
        assert limits["/api/positions"].max_requests == 200

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_create_trading_security_config(self, mock_service):
        """Test creating trading security configuration."""
        mock_service.get_max_request_size.return_value = 2048
        mock_service.TRADING_RATE_LIMITS = {}

        config = create_trading_security_config(
            hmac_secret="trading_secret", whitelisted_ips={"10.0.0.1"}, enable_signing=True
        )

        assert config.enable_rate_limiting is True
        assert config.enable_request_signing is True
        assert config.hmac_secret_key == "trading_secret"
        assert "10.0.0.1" in config.whitelisted_ips
        assert config.max_request_size == 2048
        mock_service.get_max_request_size.assert_called_once_with("/api/trading/")


class TestConcurrencyAndThreading:
    """Test concurrency and threading scenarios."""

    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent access."""
        config = SecurityConfig(default_rate_limit=RateLimitRule(max_requests=50, window_seconds=1))
        hardening = SecurityHardening(config)
        successes = []
        failures = []

        def make_request(request_id):
            try:
                hardening.check_rate_limit("shared_user")
                successes.append(request_id)
            except RateLimitExceeded:
                failures.append(request_id)

        threads = []
        for i in range(100):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have exactly 50 successes
        assert len(successes) == 50
        assert len(failures) == 50

    def test_concurrent_throttling(self):
        """Test throttling under concurrent access."""
        hardening = SecurityHardening()
        results = []

        def process_request():
            if hardening.check_throttling("user1"):
                hardening.start_request_processing("user1")
                time.sleep(0.01)
                hardening.end_request_processing("user1")
                results.append("success")

        threads = [threading.Thread(target=process_request) for _ in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed as they're processed sequentially
        assert len(results) == 50
