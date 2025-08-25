"""
Comprehensive unit tests for Security Hardening.

Tests the security hardening system including rate limiting, throttling,
security headers, request signing, and related security mechanisms.
"""

import time

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
)


@pytest.mark.unit
class TestRateLimitRule:
    """Test RateLimitRule dataclass."""

    def test_rate_limit_rule_defaults(self):
        """Test RateLimitRule default values."""
        rule = RateLimitRule(max_requests=100, window_seconds=60)
        assert rule.max_requests == 100
        assert rule.window_seconds == 60
        assert rule.burst_allowance == 0
        assert rule.cooldown_seconds == 60

    def test_rate_limit_rule_custom(self):
        """Test RateLimitRule with custom values."""
        rule = RateLimitRule(
            max_requests=50, window_seconds=120, burst_allowance=5, cooldown_seconds=300
        )
        assert rule.max_requests == 50
        assert rule.window_seconds == 120
        assert rule.burst_allowance == 5
        assert rule.cooldown_seconds == 300


@pytest.mark.unit
class TestSecurityConfig:
    """Test SecurityConfig dataclass."""

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()
        assert config.enable_rate_limiting is True
        assert config.enable_request_throttling is True
        assert config.enable_security_headers is True
        assert config.enable_request_signing is False
        assert config.enable_ip_whitelist is False
        assert config.max_request_size == 1024 * 1024
        assert config.signature_header == "X-Signature"
        assert config.timestamp_header == "X-Timestamp"


@pytest.mark.unit
class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limiter_basic(self):
        """Test basic rate limiter functionality."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=60)

        # First 3 requests should be allowed
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True

        # 4th request should be denied
        assert limiter.is_allowed("user1", rule) is False

    def test_rate_limiter_different_users(self):
        """Test rate limiter with different users."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=60)

        # Each user gets their own limit
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user2", rule) is True
        assert limiter.is_allowed("user2", rule) is True

        # Both users hit limit
        assert limiter.is_allowed("user1", rule) is False
        assert limiter.is_allowed("user2", rule) is False

    def test_rate_limiter_window_expiry(self):
        """Test rate limiter window expiry."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=1, cooldown_seconds=1)  # Short cooldown

        # Use up limit
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is False

        # Wait for both window and cooldown to expire
        time.sleep(1.5)

        # Should be allowed again
        assert limiter.is_allowed("user1", rule) is True

    def test_rate_limiter_burst_allowance(self):
        """Test rate limiter with burst allowance."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=60, burst_allowance=1)

        # Regular requests
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True

        # Burst request should be allowed
        assert limiter.is_allowed("user1", rule) is True

        # No more burst tokens
        assert limiter.is_allowed("user1", rule) is False

    def test_rate_limiter_get_remaining(self):
        """Test getting remaining requests."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=5, window_seconds=60)

        # Initial state
        assert limiter.get_remaining_requests("user1", rule) == 5

        # After some requests
        limiter.is_allowed("user1", rule)
        limiter.is_allowed("user1", rule)
        assert limiter.get_remaining_requests("user1", rule) == 3

    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=1, window_seconds=60)

        # Use up limit
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is False

        # Reset
        limiter.reset_limit("user1")

        # Should be allowed again
        assert limiter.is_allowed("user1", rule) is True


@pytest.mark.unit
class TestRequestThrottler:
    """Test RequestThrottler class."""

    def test_throttler_basic(self):
        """Test basic throttler functionality."""
        throttler = RequestThrottler(max_concurrent=2)

        # Should allow requests under limit
        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")
        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")

        # Should deny when at limit
        assert throttler.can_process_request("user1") is False

        # Should allow after ending request
        throttler.end_request("user1")
        assert throttler.can_process_request("user1") is True

    def test_throttler_suspicious_pattern(self):
        """Test throttler detection of suspicious patterns."""
        throttler = RequestThrottler(max_concurrent=100)  # High limit

        # Simulate many requests quickly
        for _ in range(25):  # More than the 20 per minute threshold
            throttler.start_request("user1")
            throttler.end_request("user1")

        # Should detect suspicious pattern
        assert throttler.can_process_request("user1") is False


@pytest.mark.unit
class TestSecurityHeaders:
    """Test SecurityHeaders class."""

    def test_get_security_headers(self):
        """Test getting security headers."""
        headers = SecurityHeaders.get_security_headers()

        # Check required headers are present
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "X-Trading-System" in headers
        assert "Cache-Control" in headers

        # Check values
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "no-store" in headers["Cache-Control"]

    def test_get_security_headers_no_trading(self):
        """Test getting security headers without trading-specific ones."""
        headers = SecurityHeaders.get_security_headers(include_trading_headers=False)

        # Should have default headers but not trading ones
        assert "X-Content-Type-Options" in headers
        assert "X-Trading-System" not in headers

    def test_validate_request_headers_valid(self):
        """Test validating valid request headers."""
        headers = {
            "User-Agent": "Mozilla/5.0 (legitimate browser)",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        issues = SecurityHeaders.validate_request_headers(headers)
        assert issues == []

    def test_validate_request_headers_suspicious_ua(self):
        """Test validating headers with suspicious User-Agent."""
        headers = {"User-Agent": "python-requests/2.25.1", "Accept": "application/json"}  # Bot-like
        issues = SecurityHeaders.validate_request_headers(headers)
        assert len(issues) > 0
        assert any("Suspicious User-Agent" in issue for issue in issues)

    def test_validate_request_headers_no_ua(self):
        """Test validating headers with missing User-Agent."""
        headers = {"Accept": "application/json"}
        issues = SecurityHeaders.validate_request_headers(headers)
        assert len(issues) > 0
        assert any("Suspicious User-Agent" in issue for issue in issues)

    def test_validate_request_headers_invalid_ip(self):
        """Test validating headers with invalid IP."""
        headers = {"X-Forwarded-For": "invalid.ip.address", "User-Agent": "Mozilla/5.0"}
        issues = SecurityHeaders.validate_request_headers(headers)
        assert len(issues) > 0
        assert any("Suspicious X-Forwarded-For" in issue for issue in issues)


@pytest.mark.unit
class TestRequestSigner:
    """Test RequestSigner class."""

    def test_request_signer_init(self):
        """Test RequestSigner initialization."""
        signer = RequestSigner("secret_key")
        assert signer.secret_key == b"secret_key"

        # Test with bytes
        signer = RequestSigner(b"secret_key")
        assert signer.secret_key == b"secret_key"

        # Test empty key
        with pytest.raises(ValueError):
            RequestSigner("")

    def test_sign_request(self):
        """Test request signing."""
        signer = RequestSigner("secret_key")
        headers = signer.sign_request("GET", "/api/orders", "")

        assert "X-Signature" in headers
        assert "X-Timestamp" in headers
        assert len(headers["X-Signature"]) == 64  # SHA256 hex digest
        assert headers["X-Timestamp"].isdigit()

    def test_sign_request_with_body(self):
        """Test request signing with body."""
        signer = RequestSigner("secret_key")
        body = '{"symbol": "AAPL", "quantity": 100}'
        headers = signer.sign_request("POST", "/api/orders", body)

        assert "X-Signature" in headers
        assert "X-Timestamp" in headers

    def test_verify_request_valid(self):
        """Test request verification with valid signature."""
        signer = RequestSigner("secret_key")
        method = "GET"
        path = "/api/orders"
        body = ""
        timestamp = str(int(time.time()))

        # Create signature
        headers = signer.sign_request(method, path, body, int(timestamp))
        signature = headers["X-Signature"]

        # Verify
        assert signer.verify_request(method, path, body, signature, timestamp) is True

    def test_verify_request_invalid_signature(self):
        """Test request verification with invalid signature."""
        signer = RequestSigner("secret_key")
        method = "GET"
        path = "/api/orders"
        body = ""
        timestamp = str(int(time.time()))

        # Use wrong signature
        assert signer.verify_request(method, path, body, "wrong_signature", timestamp) is False

    def test_verify_request_expired(self):
        """Test request verification with expired timestamp."""
        signer = RequestSigner("secret_key")
        method = "GET"
        path = "/api/orders"
        body = ""
        old_timestamp = str(int(time.time()) - 400)  # 400 seconds ago

        # Create signature with old timestamp
        headers = signer.sign_request(method, path, body, int(old_timestamp))
        signature = headers["X-Signature"]

        # Should fail due to age
        assert (
            signer.verify_request(method, path, body, signature, old_timestamp, max_age=300)
            is False
        )

    def test_verify_request_invalid_timestamp(self):
        """Test request verification with invalid timestamp."""
        signer = RequestSigner("secret_key")

        assert signer.verify_request("GET", "/api/orders", "", "signature", "invalid") is False


@pytest.mark.unit
class TestSecurityHardening:
    """Test SecurityHardening main class."""

    def test_security_hardening_init(self):
        """Test SecurityHardening initialization."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        assert hardening.config is config
        assert hardening.rate_limiter is not None
        assert hardening.throttler is not None
        assert hardening.request_signer is None  # Not enabled by default

    def test_security_hardening_with_signing(self):
        """Test SecurityHardening with request signing enabled."""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret")
        hardening = SecurityHardening(config)

        assert hardening.request_signer is not None

    def test_check_rate_limit_allowed(self):
        """Test rate limit check when allowed."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        # Should not raise exception
        result = hardening.check_rate_limit("user1")
        assert result is True

    def test_check_rate_limit_disabled(self):
        """Test rate limit check when disabled."""
        config = SecurityConfig(enable_rate_limiting=False)
        hardening = SecurityHardening(config)

        # Should always allow when disabled
        result = hardening.check_rate_limit("user1")
        assert result is True

    def test_check_throttling_allowed(self):
        """Test throttling check when allowed."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        result = hardening.check_throttling("user1")
        assert result is True

    def test_check_throttling_disabled(self):
        """Test throttling check when disabled."""
        config = SecurityConfig(enable_request_throttling=False)
        hardening = SecurityHardening(config)

        result = hardening.check_throttling("user1")
        assert result is True

    def test_start_end_request_processing(self):
        """Test request processing tracking."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        # Should not raise exceptions
        hardening.start_request_processing("user1")
        hardening.end_request_processing("user1")

    def test_validate_request_headers(self):
        """Test request header validation."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        headers = {"User-Agent": "Mozilla/5.0"}
        issues = hardening.validate_request_headers(headers)
        assert isinstance(issues, list)

    def test_get_security_headers_enabled(self):
        """Test getting security headers when enabled."""
        config = SecurityConfig(enable_security_headers=True)
        hardening = SecurityHardening(config)

        headers = hardening.get_security_headers()
        assert len(headers) > 0
        assert "X-Content-Type-Options" in headers

    def test_get_security_headers_disabled(self):
        """Test getting security headers when disabled."""
        config = SecurityConfig(enable_security_headers=False)
        hardening = SecurityHardening(config)

        headers = hardening.get_security_headers()
        assert headers == {}

    def test_verify_request_signature_disabled(self):
        """Test request signature verification when disabled."""
        config = SecurityConfig(enable_request_signing=False)
        hardening = SecurityHardening(config)

        # Should always pass when disabled
        result = hardening.verify_request_signature("GET", "/api", "", {})
        assert result is True

    def test_verify_request_signature_missing_headers(self):
        """Test request signature verification with missing headers."""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret")
        hardening = SecurityHardening(config)

        with pytest.raises(InvalidTokenError):
            hardening.verify_request_signature("GET", "/api", "", {})

    def test_check_ip_whitelist_disabled(self):
        """Test IP whitelist check when disabled."""
        config = SecurityConfig(enable_ip_whitelist=False)
        hardening = SecurityHardening(config)

        result = hardening.check_ip_whitelist("192.168.1.1")
        assert result is True

    def test_check_ip_whitelist_blacklisted(self):
        """Test IP whitelist check with blacklisted IP."""
        config = SecurityConfig(enable_ip_whitelist=True, blacklisted_ips={"192.168.1.100"})
        hardening = SecurityHardening(config)

        with pytest.raises(SecurityError):
            hardening.check_ip_whitelist("192.168.1.100")

    def test_check_ip_whitelist_not_whitelisted(self):
        """Test IP whitelist check with non-whitelisted IP."""
        config = SecurityConfig(enable_ip_whitelist=True, whitelisted_ips={"192.168.1.1"})
        hardening = SecurityHardening(config)

        with pytest.raises(SecurityError):
            hardening.check_ip_whitelist("192.168.1.100")


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_create_trading_security_config(self):
        """Test creating trading security configuration."""
        config = create_trading_security_config()

        assert isinstance(config, SecurityConfig)
        assert config.enable_rate_limiting is True
        assert config.enable_request_signing is False
        assert len(config.api_rate_limits) > 0
        assert "place_order" in config.api_rate_limits
        assert "get_market_data" in config.api_rate_limits

    def test_create_trading_security_config_with_signing(self):
        """Test creating trading security config with signing enabled."""
        config = create_trading_security_config(
            hmac_secret="secret123",
            enable_signing=True,
            whitelisted_ips={"192.168.1.1", "10.0.0.1"},
        )

        assert config.enable_request_signing is True
        assert config.hmac_secret_key == "secret123"
        assert config.enable_ip_whitelist is True
        assert "192.168.1.1" in config.whitelisted_ips


@pytest.mark.unit
class TestDecorators:
    """Test security decorators."""

    def test_secure_endpoint_decorator_basic(self):
        """Test secure_endpoint decorator basic functionality."""

        @secure_endpoint()
        def test_endpoint():
            return "success"

        # Without security hardening, should just call function
        result = test_endpoint()
        assert result == "success"

    def test_secure_endpoint_decorator_with_hardening(self):
        """Test secure_endpoint decorator with security hardening."""
        config = SecurityConfig()
        hardening = SecurityHardening(config)

        @secure_endpoint()
        def test_endpoint():
            return "success"

        # With security hardening
        result = test_endpoint(_security_hardening=hardening)
        assert result == "success"

    def test_secure_endpoint_decorator_rate_limit_exceeded(self):
        """Test secure_endpoint decorator with rate limit exceeded."""
        # Create restrictive config
        config = SecurityConfig()
        config.default_rate_limit = RateLimitRule(max_requests=1, window_seconds=60)
        hardening = SecurityHardening(config)

        @secure_endpoint()
        def test_endpoint():
            return "success"

        # First call should succeed
        result = test_endpoint(_security_hardening=hardening, client_ip="user1")
        assert result == "success"

        # Second call should fail
        with pytest.raises(RateLimitExceeded):
            test_endpoint(_security_hardening=hardening, client_ip="user1")


@pytest.mark.unit
class TestExceptions:
    """Test custom exception classes."""

    def test_security_error(self):
        """Test SecurityError exception."""
        error = SecurityError("Test security error")
        assert str(error) == "Test security error"
        assert isinstance(error, Exception)

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded exception."""
        error = RateLimitExceeded("Rate limit hit", retry_after=60)
        assert str(error) == "Rate limit hit"
        assert error.retry_after == 60
        assert isinstance(error, SecurityError)

    def test_throttling_error(self):
        """Test ThrottlingError exception."""
        error = ThrottlingError("Request throttled")
        assert str(error) == "Request throttled"
        assert isinstance(error, SecurityError)

    def test_invalid_token_error(self):
        """Test InvalidTokenError exception."""
        error = InvalidTokenError("Invalid signature")
        assert str(error) == "Invalid signature"
        assert isinstance(error, SecurityError)
