"""
Extended unit tests for security hardening module to achieve 80%+ coverage.

Focuses on rate limiting, throttling, security headers, request signing,
and comprehensive security controls for trading systems.
"""

import threading
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


class TestExceptionTypes:
    """Test custom exception types"""

    def test_security_error(self):
        """Test SecurityError base exception"""
        error = SecurityError("Security violation")
        assert str(error) == "Security violation"
        assert isinstance(error, Exception)

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded exception"""
        error = RateLimitExceeded("Rate limit hit", retry_after=60)
        assert str(error) == "Rate limit hit"
        assert error.retry_after == 60
        assert isinstance(error, SecurityError)

    def test_rate_limit_exceeded_no_retry(self):
        """Test RateLimitExceeded without retry_after"""
        error = RateLimitExceeded("Rate limit hit")
        assert error.retry_after is None

    def test_throttling_error(self):
        """Test ThrottlingError exception"""
        error = ThrottlingError("Request throttled")
        assert str(error) == "Request throttled"
        assert isinstance(error, SecurityError)

    def test_invalid_token_error(self):
        """Test InvalidTokenError exception"""
        error = InvalidTokenError("Invalid signature")
        assert str(error) == "Invalid signature"
        assert isinstance(error, SecurityError)


class TestRateLimitRule:
    """Test RateLimitRule configuration"""

    def test_default_values(self):
        """Test default RateLimitRule values"""
        rule = RateLimitRule(max_requests=100, window_seconds=60)
        assert rule.max_requests == 100
        assert rule.window_seconds == 60
        assert rule.burst_allowance == 0
        assert rule.cooldown_seconds == 60

    def test_custom_values(self):
        """Test custom RateLimitRule values"""
        rule = RateLimitRule(
            max_requests=50, window_seconds=30, burst_allowance=10, cooldown_seconds=120
        )
        assert rule.max_requests == 50
        assert rule.window_seconds == 30
        assert rule.burst_allowance == 10
        assert rule.cooldown_seconds == 120


class TestSecurityConfig:
    """Test SecurityConfig configuration"""

    def test_default_config(self):
        """Test default SecurityConfig values"""
        config = SecurityConfig()
        assert config.enable_rate_limiting is True
        assert config.enable_request_throttling is True
        assert config.enable_security_headers is True
        assert config.enable_request_signing is False
        assert config.enable_ip_whitelist is False
        assert config.default_rate_limit.max_requests == 100
        assert config.default_rate_limit.window_seconds == 60
        assert config.api_rate_limits == {}
        assert config.whitelisted_ips == set()
        assert config.blacklisted_ips == set()
        assert config.max_request_size == 1024 * 1024
        assert config.hmac_secret_key is None
        assert config.signature_header == "X-Signature"
        assert config.timestamp_header == "X-Timestamp"
        assert config.max_timestamp_skew == 300

    def test_custom_config(self):
        """Test custom SecurityConfig values"""
        custom_rule = RateLimitRule(200, 120)
        config = SecurityConfig(
            enable_rate_limiting=False,
            enable_request_signing=True,
            default_rate_limit=custom_rule,
            whitelisted_ips={"192.168.1.1", "192.168.1.2"},
            blacklisted_ips={"10.0.0.1"},
            hmac_secret_key="secret123",
            max_request_size=2048,
        )
        assert config.enable_rate_limiting is False
        assert config.enable_request_signing is True
        assert config.default_rate_limit == custom_rule
        assert "192.168.1.1" in config.whitelisted_ips
        assert "10.0.0.1" in config.blacklisted_ips
        assert config.hmac_secret_key == "secret123"
        assert config.max_request_size == 2048


class TestRateLimiter:
    """Test RateLimiter functionality"""

    def test_init(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter()
        assert isinstance(limiter._windows, dict)
        assert isinstance(limiter._burst_tokens, dict)
        assert isinstance(limiter._cooldowns, dict)
        assert limiter._lock is not None

    def test_is_allowed_under_limit(self):
        """Test requests allowed under limit"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=60)

        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True

    def test_is_allowed_over_limit(self):
        """Test requests blocked over limit"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is False

    @patch("time.time")
    def test_window_cleanup(self, mock_time):
        """Test old requests are cleaned up"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=10)

        # Make requests at time 100
        mock_time.return_value = 100
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user1", rule) is False

        # Move time forward past window
        mock_time.return_value = 111  # 11 seconds later
        assert limiter.is_allowed("user1", rule) is True

    def test_multiple_identifiers(self):
        """Test rate limiting for multiple identifiers"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=1, window_seconds=60)

        assert limiter.is_allowed("user1", rule) is True
        assert limiter.is_allowed("user2", rule) is True
        assert limiter.is_allowed("user1", rule) is False
        assert limiter.is_allowed("user2", rule) is False

    def test_get_remaining_requests(self):
        """Test getting remaining request count"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=5, window_seconds=60)

        assert limiter.get_remaining_requests("user1", rule) == 5

        limiter.is_allowed("user1", rule)
        assert limiter.get_remaining_requests("user1", rule) == 4

        limiter.is_allowed("user1", rule)
        limiter.is_allowed("user1", rule)
        assert limiter.get_remaining_requests("user1", rule) == 2

    def test_get_remaining_with_burst(self):
        """Test remaining requests with burst tokens"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=60, burst_allowance=2)

        limiter._burst_tokens["user1"] = 2
        assert limiter.get_remaining_requests("user1", rule) == 5  # 3 + 2 burst

    @patch("time.time")
    def test_get_remaining_with_expired(self, mock_time):
        """Test remaining requests with expired entries"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=10)

        mock_time.return_value = 100
        limiter.is_allowed("user1", rule)
        limiter.is_allowed("user1", rule)

        # Some requests still in window
        assert limiter.get_remaining_requests("user1", rule) == 1

        # All requests expired
        mock_time.return_value = 111
        assert limiter.get_remaining_requests("user1", rule) == 3

    def test_reset_limit(self):
        """Test resetting limits for an identifier"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=1, window_seconds=60)

        # Use up limit and add burst/cooldown
        limiter.is_allowed("user1", rule)
        limiter._burst_tokens["user1"] = 5
        limiter._cooldowns["user1"] = 1000.0

        assert limiter.is_allowed("user1", rule) is False

        # Reset the limit
        limiter.reset_limit("user1")

        assert limiter.is_allowed("user1", rule) is True
        assert "user1" not in limiter._burst_tokens
        assert "user1" not in limiter._cooldowns

    def test_reset_nonexistent(self):
        """Test resetting non-existent identifier"""
        limiter = RateLimiter()
        # Should not raise error
        limiter.reset_limit("nonexistent")

    def test_thread_safety(self):
        """Test thread safety of rate limiter"""
        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=100, window_seconds=1)
        results = []

        def make_requests(identifier):
            for _ in range(50):
                result = limiter.is_allowed(identifier, rule)
                results.append(result)

        threads = [threading.Thread(target=make_requests, args=("user1",)) for _ in range(3)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have exactly 100 True values
        assert sum(results) == 100


class TestRequestThrottler:
    """Test RequestThrottler functionality"""

    def test_init(self):
        """Test RequestThrottler initialization"""
        throttler = RequestThrottler(max_concurrent=50)
        assert throttler.max_concurrent == 50
        assert isinstance(throttler._active_requests, dict)
        assert isinstance(throttler._request_times, dict)

    def test_can_process_under_limit(self):
        """Test request allowed under concurrent limit"""
        throttler = RequestThrottler(max_concurrent=3)

        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")
        assert throttler.can_process_request("user1") is True
        throttler.start_request("user1")
        assert throttler.can_process_request("user1") is True

    def test_can_process_at_limit(self):
        """Test request blocked at concurrent limit"""
        throttler = RequestThrottler(max_concurrent=2)

        throttler.start_request("user1")
        throttler.start_request("user1")
        assert throttler.can_process_request("user1") is False

    def test_start_end_request(self):
        """Test starting and ending requests"""
        throttler = RequestThrottler(max_concurrent=2)

        throttler.start_request("user1")
        assert throttler._active_requests["user1"] == 1

        throttler.start_request("user1")
        assert throttler._active_requests["user1"] == 2

        throttler.end_request("user1")
        assert throttler._active_requests["user1"] == 1

        throttler.end_request("user1")
        assert throttler._active_requests["user1"] == 0

    def test_end_request_no_active(self):
        """Test ending request with no active requests"""
        throttler = RequestThrottler()
        throttler.end_request("user1")  # Should not raise error
        assert throttler._active_requests["user1"] == 0

    def test_request_times_tracking(self):
        """Test request times are tracked"""
        throttler = RequestThrottler()

        with patch("time.time", return_value=1000):
            throttler.start_request("user1")
            throttler.start_request("user1")

        assert len(throttler._request_times["user1"]) == 2
        assert all(t == 1000 for t in throttler._request_times["user1"])

    def test_multiple_identifiers(self):
        """Test throttling for multiple identifiers"""
        throttler = RequestThrottler(max_concurrent=1)

        throttler.start_request("user1")
        throttler.start_request("user2")

        assert throttler.can_process_request("user1") is False
        assert throttler.can_process_request("user2") is False

        throttler.end_request("user1")
        assert throttler.can_process_request("user1") is True
        assert throttler.can_process_request("user2") is False


class TestSecurityHeaders:
    """Test SecurityHeaders functionality"""

    def test_default_headers(self):
        """Test default security headers"""
        headers = SecurityHeaders.get_security_headers(include_trading_headers=False)

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "Referrer-Policy" in headers
        assert "Permissions-Policy" in headers
        assert "X-Trading-System" not in headers

    def test_with_trading_headers(self):
        """Test headers with trading-specific additions"""
        headers = SecurityHeaders.get_security_headers(include_trading_headers=True)

        # Should have all default headers
        assert "X-Content-Type-Options" in headers

        # Plus trading headers
        assert headers["X-Trading-System"] == "AI-Trader-v1.0"
        assert headers["X-API-Version"] == "1.0"
        assert headers["Cache-Control"] == "no-store, no-cache, must-revalidate, private"
        assert headers["Pragma"] == "no-cache"

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_check_request_headers(self, mock_service):
        """Test request header validation"""
        mock_service.validate_request_headers.return_value = []

        headers = {"Content-Type": "application/json"}
        result = SecurityHeaders.check_request_headers(headers)

        assert result is True
        mock_service.validate_request_headers.assert_called_once_with(headers)

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_check_request_headers_invalid(self, mock_service):
        """Test invalid request headers"""
        mock_service.validate_request_headers.return_value = ["Invalid header X"]

        headers = {"X": "invalid"}
        result = SecurityHeaders.check_request_headers(headers)

        assert result is False


class TestRequestSigner:
    """Test RequestSigner functionality"""

    def test_init(self):
        """Test RequestSigner initialization"""
        signer = RequestSigner("secret123")
        assert signer.secret_key == b"secret123"

    def test_init_with_bytes(self):
        """Test initialization with bytes key"""
        signer = RequestSigner(b"secret123")
        assert signer.secret_key == b"secret123"

    def test_init_empty_key(self):
        """Test initialization with empty key raises error"""
        with pytest.raises(ValueError) as exc_info:
            RequestSigner("")
        assert "Secret key cannot be empty" in str(exc_info)

    def test_sign_request_basic(self):
        """Test basic request signing"""
        signer = RequestSigner("secret123")

        headers = signer.sign_request("GET", "/api/data", "")

        assert "X-Signature" in headers
        assert "X-Timestamp" in headers
        assert len(headers["X-Signature"]) == 64  # SHA256 hex digest

    def test_sign_request_with_body(self):
        """Test signing request with body"""
        signer = RequestSigner("secret123")

        body = '{"key": "value"}'
        headers = signer.sign_request("POST", "/api/create", body)

        assert "X-Signature" in headers
        assert "X-Timestamp" in headers

    def test_sign_request_with_timestamp(self):
        """Test signing with specific timestamp"""
        signer = RequestSigner("secret123")

        headers = signer.sign_request("GET", "/api/data", "", timestamp=1234567890)

        assert headers["X-Timestamp"] == "1234567890"

    def test_check_request_valid(self):
        """Test validating a correctly signed request"""
        signer = RequestSigner("secret123")

        # Sign a request
        headers = signer.sign_request("GET", "/api/data", "body", timestamp=1000)

        # Verify it
        with patch("time.time", return_value=1100):  # Within 300s window
            result = signer.check_request(
                "GET", "/api/data", "body", headers["X-Signature"], headers["X-Timestamp"]
            )
            assert result is True

    def test_check_request_invalid_signature(self):
        """Test invalid signature detection"""
        signer = RequestSigner("secret123")

        result = signer.check_request("GET", "/api/data", "body", "invalid_signature", "1000")
        assert result is False

    def test_check_request_expired_timestamp(self):
        """Test expired timestamp detection"""
        signer = RequestSigner("secret123")

        headers = signer.sign_request("GET", "/api/data", "", timestamp=1000)

        with patch("time.time", return_value=2000):  # 1000s later, beyond 300s window
            result = signer.check_request(
                "GET", "/api/data", "", headers["X-Signature"], headers["X-Timestamp"], max_age=300
            )
            assert result is False

    def test_check_request_invalid_timestamp(self):
        """Test invalid timestamp format"""
        signer = RequestSigner("secret123")

        result = signer.check_request("GET", "/api/data", "", "signature", "not_a_number")
        assert result is False

    def test_check_request_case_sensitive(self):
        """Test method case normalization"""
        signer = RequestSigner("secret123")

        # Sign with lowercase
        headers = signer.sign_request("get", "/api/data", "", timestamp=1000)

        # Verify with different case - should still work
        with patch("time.time", return_value=1100):
            result = signer.check_request(
                "GET", "/api/data", "", headers["X-Signature"], headers["X-Timestamp"]
            )
            assert result is True


class TestSecurityHardening:
    """Test main SecurityHardening coordinator"""

    def test_init_default(self):
        """Test initialization with default config"""
        hardening = SecurityHardening()
        assert hardening.config is not None
        assert hardening.rate_limiter is not None
        assert hardening.throttler is not None
        assert hardening.request_signer is None

    def test_init_with_signing(self):
        """Test initialization with request signing"""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret123")
        hardening = SecurityHardening(config)
        assert hardening.request_signer is not None

    def test_check_rate_limit_disabled(self):
        """Test rate limiting when disabled"""
        config = SecurityConfig(enable_rate_limiting=False)
        hardening = SecurityHardening(config)

        result = hardening.check_rate_limit("user1")
        assert result is True

    def test_check_rate_limit_default(self):
        """Test rate limiting with default rule"""
        hardening = SecurityHardening()

        # Should pass initially
        assert hardening.check_rate_limit("user1") is True

    def test_check_rate_limit_exceeded(self):
        """Test rate limit exceeded"""
        config = SecurityConfig(default_rate_limit=RateLimitRule(1, 60, cooldown_seconds=120))
        hardening = SecurityHardening(config)

        hardening.check_rate_limit("user1")

        with pytest.raises(RateLimitExceeded) as exc_info:
            hardening.check_rate_limit("user1")

        assert "Rate limit exceeded" in str(exc_info)
        assert exc_info.retry_after == 120

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_check_rate_limit_with_endpoint(self, mock_service):
        """Test rate limiting with specific endpoint"""
        mock_service.get_rate_limit_for_endpoint.return_value = {
            "max_requests": 10,
            "window_seconds": 30,
            "burst_allowance": 2,
        }
        mock_service.get_cooldown_period.return_value = 60

        hardening = SecurityHardening()
        result = hardening.check_rate_limit("user1", "/api/orders")

        assert result is True
        mock_service.get_rate_limit_for_endpoint.assert_called_once_with("/api/orders")

    def test_check_throttling_disabled(self):
        """Test throttling when disabled"""
        config = SecurityConfig(enable_request_throttling=False)
        hardening = SecurityHardening(config)

        result = hardening.check_throttling("user1")
        assert result is True

    def test_check_throttling_allowed(self):
        """Test throttling allows request"""
        hardening = SecurityHardening()

        result = hardening.check_throttling("user1")
        assert result is True

    def test_check_throttling_blocked(self):
        """Test throttling blocks request"""
        hardening = SecurityHardening()
        hardening.throttler._active_requests["user1"] = 100  # Max concurrent

        with pytest.raises(ThrottlingError) as exc_info:
            hardening.check_throttling("user1")

        assert "Request throttled" in str(exc_info)

    def test_start_end_request_processing(self):
        """Test request processing lifecycle"""
        hardening = SecurityHardening()

        hardening.start_request_processing("user1")
        assert hardening.throttler._active_requests["user1"] == 1

        hardening.end_request_processing("user1")
        assert hardening.throttler._active_requests["user1"] == 0

    def test_start_end_processing_disabled(self):
        """Test processing when throttling disabled"""
        config = SecurityConfig(enable_request_throttling=False)
        hardening = SecurityHardening(config)

        # Should not raise errors
        hardening.start_request_processing("user1")
        hardening.end_request_processing("user1")

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_check_request_headers(self, mock_service):
        """Test request header validation"""
        mock_service.validate_request_headers.return_value = []

        hardening = SecurityHardening()
        headers = {"Content-Type": "application/json"}
        result = hardening.check_request_headers(headers)

        assert result is True
        mock_service.validate_request_headers.assert_called_once_with(headers)

    def test_get_security_headers_enabled(self):
        """Test getting security headers when enabled"""
        hardening = SecurityHardening()
        headers = hardening.get_security_headers()

        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers

    def test_get_security_headers_disabled(self):
        """Test getting headers when disabled"""
        config = SecurityConfig(enable_security_headers=False)
        hardening = SecurityHardening(config)

        headers = hardening.get_security_headers()
        assert headers == {}

    def test_check_request_signature_disabled(self):
        """Test signature check when disabled"""
        hardening = SecurityHardening()

        result = hardening.check_request_signature("GET", "/api", "", {})
        assert result is True

    def test_check_request_signature_missing_headers(self):
        """Test signature check with missing headers"""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret")
        hardening = SecurityHardening(config)

        result = hardening.check_request_signature("GET", "/api", "", {})
        assert result is False

    def test_check_request_signature_valid(self):
        """Test valid signature verification"""
        config = SecurityConfig(enable_request_signing=True, hmac_secret_key="secret")
        hardening = SecurityHardening(config)

        # Create valid signature
        headers = hardening.request_signer.sign_request("GET", "/api", "body")

        with patch.object(hardening.request_signer, "check_request", return_value=True):
            result = hardening.check_request_signature("GET", "/api", "body", headers)
            assert result is True

    def test_check_ip_whitelist_disabled(self):
        """Test IP whitelist when disabled"""
        hardening = SecurityHardening()

        result = hardening.check_ip_whitelist("192.168.1.1")
        assert result is True

    def test_check_ip_blacklisted(self):
        """Test blacklisted IP"""
        config = SecurityConfig(enable_ip_whitelist=True, blacklisted_ips={"10.0.0.1"})
        hardening = SecurityHardening(config)

        with pytest.raises(SecurityError) as exc_info:
            hardening.check_ip_whitelist("10.0.0.1")

        assert "IP 10.0.0.1 is blacklisted" in str(exc_info)

    def test_check_ip_not_whitelisted(self):
        """Test IP not in whitelist"""
        config = SecurityConfig(enable_ip_whitelist=True, whitelisted_ips={"192.168.1.1"})
        hardening = SecurityHardening(config)

        with pytest.raises(SecurityError) as exc_info:
            hardening.check_ip_whitelist("192.168.1.2")

        assert "IP 192.168.1.2 is not whitelisted" in str(exc_info)

    def test_check_ip_whitelisted(self):
        """Test whitelisted IP"""
        config = SecurityConfig(enable_ip_whitelist=True, whitelisted_ips={"192.168.1.1"})
        hardening = SecurityHardening(config)

        result = hardening.check_ip_whitelist("192.168.1.1")
        assert result is True


class TestSecureEndpointDecorator:
    """Test secure_endpoint decorator"""

    def test_secure_endpoint_no_hardening(self):
        """Test decorator without hardening configured"""

        @secure_endpoint()
        def api_endpoint():
            return "success"

        with patch("src.infrastructure.security.hardening.logger") as mock_logger:
            result = api_endpoint()
            assert result == "success"
            mock_logger.warning.assert_called_once()

    def test_secure_endpoint_with_hardening(self):
        """Test decorator with hardening"""
        hardening = SecurityHardening()

        @secure_endpoint()
        def api_endpoint(**kwargs):
            return "success"

        result = api_endpoint(_security_hardening=hardening, client_ip="192.168.1.1")
        assert result == "success"

    def test_secure_endpoint_rate_limit(self):
        """Test decorator with rate limiting"""
        config = SecurityConfig(default_rate_limit=RateLimitRule(1, 60))
        hardening = SecurityHardening(config)

        @secure_endpoint(endpoint_name="/api/test")
        def api_endpoint(**kwargs):
            return "success"

        # First call succeeds
        result = api_endpoint(_security_hardening=hardening, client_ip="192.168.1.1")
        assert result == "success"

        # Second call rate limited
        with patch("src.infrastructure.security.hardening.logger") as mock_logger:
            with pytest.raises(RateLimitExceeded):
                api_endpoint(_security_hardening=hardening, client_ip="192.168.1.1")
            mock_logger.warning.assert_called()

    def test_secure_endpoint_with_identifier_func(self):
        """Test decorator with custom identifier function"""
        hardening = SecurityHardening()

        def get_user_id(*args, **kwargs):
            return kwargs.get("user_id", "unknown")

        @secure_endpoint(identifier_func=get_user_id)
        def api_endpoint(**kwargs):
            return f"User: {kwargs.get('user_id')}"

        result = api_endpoint(_security_hardening=hardening, user_id="user123")
        assert result == "User: user123"

    def test_secure_endpoint_lifecycle(self):
        """Test full request lifecycle"""
        hardening = SecurityHardening()
        call_count = [0]

        @secure_endpoint()
        def api_endpoint(**kwargs):
            call_count[0] += 1
            # Verify processing started
            assert hardening.throttler._active_requests.get("unknown", 0) > 0
            return "success"

        result = api_endpoint(_security_hardening=hardening)
        assert result == "success"
        assert call_count[0] == 1
        # Verify processing ended
        assert hardening.throttler._active_requests.get("unknown", 0) == 0


class TestTradingSpecificFunctions:
    """Test trading-specific helper functions"""

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_get_trading_rate_limits(self, mock_service):
        """Test getting trading rate limits"""
        mock_service.TRADING_RATE_LIMITS = {
            "/api/orders": {"max_requests": 50, "window_seconds": 60, "burst_allowance": 5},
            "/api/positions": {"max_requests": 100, "window_seconds": 60, "burst_allowance": 10},
        }

        limits = get_trading_rate_limits()

        assert "/api/orders" in limits
        assert limits["/api/orders"].max_requests == 50
        assert limits["/api/positions"].burst_allowance == 10

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_create_trading_security_config(self, mock_service):
        """Test creating trading security config"""
        mock_service.get_max_request_size.return_value = 2048
        mock_service.TRADING_RATE_LIMITS = {}

        config = create_trading_security_config(
            hmac_secret="secret123", whitelisted_ips={"192.168.1.1"}, enable_signing=True
        )

        assert config.enable_rate_limiting is True
        assert config.enable_request_signing is True
        assert config.hmac_secret_key == "secret123"
        assert "192.168.1.1" in config.whitelisted_ips
        assert config.max_request_size == 2048
        mock_service.get_max_request_size.assert_called_once_with("/api/trading/")

    @patch("src.domain.services.request_validation_service.RequestValidationService")
    def test_create_trading_config_defaults(self, mock_service):
        """Test creating config with defaults"""
        mock_service.get_max_request_size.return_value = 1024
        mock_service.TRADING_RATE_LIMITS = {}

        config = create_trading_security_config()

        assert config.enable_request_signing is False
        assert config.enable_ip_whitelist is False
        assert config.whitelisted_ips == set()
        assert config.hmac_secret_key is None
