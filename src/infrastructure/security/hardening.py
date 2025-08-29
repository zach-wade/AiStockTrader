"""
Security Hardening - Infrastructure layer for security controls.

This module provides security hardening features including rate limiting,
request throttling, security headers, and defense mechanisms for trading systems.
"""

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class RateLimitExceeded(SecurityError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ThrottlingError(SecurityError):
    """Raised when request is throttled."""

    pass


class InvalidTokenError(SecurityError):
    """Raised when security token is invalid."""

    pass


@dataclass
class RateLimitRule:
    """Configuration for rate limiting rules."""

    max_requests: int
    window_seconds: int
    burst_allowance: int = 0  # Allow burst requests
    cooldown_seconds: int = 60  # Cooldown after limit exceeded


@dataclass
class SecurityConfig:
    """Security hardening configuration."""

    enable_rate_limiting: bool = True
    enable_request_throttling: bool = True
    enable_security_headers: bool = True
    enable_request_signing: bool = False
    enable_ip_whitelist: bool = False

    # Rate limiting configuration
    default_rate_limit: RateLimitRule = field(default_factory=lambda: RateLimitRule(100, 60))
    api_rate_limits: dict[str, RateLimitRule] = field(default_factory=dict)

    # IP and security configuration
    whitelisted_ips: set[str] = field(default_factory=set)
    blacklisted_ips: set[str] = field(default_factory=set)
    max_request_size: int = 1024 * 1024  # 1MB

    # Request signing
    hmac_secret_key: str | None = None
    signature_header: str = "X-Signature"
    timestamp_header: str = "X-Timestamp"
    max_timestamp_skew: int = 300  # 5 minutes


class RateLimiter:
    """Advanced rate limiter with multiple algorithms."""

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._burst_tokens: dict[str, int] = defaultdict(int)
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.RLock()

    def is_allowed(self, identifier: str, rule: RateLimitRule) -> bool:
        """Simple rate limit check - technical implementation only."""
        with self._lock:
            now = time.time()
            window = self._windows[identifier]

            # Clean up old requests
            cutoff = now - rule.window_seconds
            while window and window[0] < cutoff:
                window.popleft()

            # Simple check against limit
            if len(window) >= rule.max_requests:
                return False

            window.append(now)
            return True

    def get_remaining_requests(self, identifier: str, rule: RateLimitRule) -> int:
        """Get remaining requests for identifier."""
        with self._lock:
            window = self._windows[identifier]
            now = time.time()
            cutoff = now - rule.window_seconds

            # Count valid requests
            valid_requests = sum(1 for req_time in window if req_time >= cutoff)
            remaining = max(0, rule.max_requests - valid_requests)

            # Add burst tokens
            remaining += self._burst_tokens[identifier]

            return remaining

    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        with self._lock:
            if identifier in self._windows:
                del self._windows[identifier]
            if identifier in self._burst_tokens:
                del self._burst_tokens[identifier]
            if identifier in self._cooldowns:
                del self._cooldowns[identifier]


class RequestThrottler:
    """Request throttling with adaptive algorithms."""

    def __init__(self, max_concurrent: int = 100) -> None:
        self.max_concurrent = max_concurrent
        self._active_requests: dict[str, int] = defaultdict(int)
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def can_process_request(self, identifier: str) -> bool:
        """Simple concurrency check - no business logic."""
        with self._lock:
            # Simple check against max concurrent
            return self._active_requests[identifier] < self.max_concurrent

    def start_request(self, identifier: str) -> None:
        """Mark request as started."""
        with self._lock:
            self._active_requests[identifier] += 1
            self._request_times[identifier].append(time.time())

    def end_request(self, identifier: str) -> None:
        """Mark request as ended."""
        with self._lock:
            if self._active_requests[identifier] > 0:
                self._active_requests[identifier] -= 1


class SecurityHeaders:
    """Security headers management for web responses."""

    DEFAULT_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    TRADING_SPECIFIC_HEADERS = {
        "X-Trading-System": "AI-Trader-v1.0",
        "X-API-Version": "1.0",
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
    }

    @classmethod
    def get_security_headers(cls, include_trading_headers: bool = True) -> dict[str, str]:
        """Get all security headers."""
        headers = cls.DEFAULT_HEADERS.copy()
        if include_trading_headers:
            headers.update(cls.TRADING_SPECIFIC_HEADERS)
        return headers

    @classmethod
    def check_request_headers(cls, headers: dict[str, str]) -> bool:
        """Delegate header validation to domain service."""
        from src.domain.services.request_validation_service import RequestValidationService

        errors = RequestValidationService.validate_request_headers(headers)
        return len(errors) == 0


class RequestSigner:
    """HMAC-based request signing for API security."""

    def __init__(self, secret_key: str) -> None:
        if not secret_key:
            raise ValueError("Secret key cannot be empty")
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key

    def sign_request(
        self, method: str, path: str, body: str = "", timestamp: int | None = None
    ) -> dict[str, str]:
        """Sign a request and return headers to include."""
        if timestamp is None:
            timestamp = int(time.time())

        # Create signature string: METHOD|PATH|BODY|TIMESTAMP
        signature_string = f"{method.upper()}|{path}|{body}|{timestamp}"

        # Create HMAC signature
        signature = hmac.new(
            self.secret_key, signature_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return {"X-Signature": signature, "X-Timestamp": str(timestamp)}

    def check_request(
        self,
        method: str,
        path: str,
        body: str,
        signature: str,
        timestamp_str: str,
        max_age: int = 300,
    ) -> bool:
        """Simple signature verification - technical implementation only."""
        try:
            timestamp = int(timestamp_str)
            now = int(time.time())

            # Simple time check
            if abs(now - timestamp) > max_age:
                return False

            # Simple signature comparison
            signature_string = f"{method.upper()}|{path}|{body}|{timestamp}"
            expected = hmac.new(
                self.secret_key, signature_string.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False


class SecurityHardening:
    """Main security hardening coordinator."""

    def __init__(self, config: SecurityConfig | None = None) -> None:
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter()
        self.throttler = RequestThrottler()
        self.request_signer = None

        if self.config.enable_request_signing and self.config.hmac_secret_key:
            self.request_signer = RequestSigner(self.config.hmac_secret_key)

    def check_rate_limit(self, identifier: str, api_endpoint: str | None = None) -> bool:
        """Check if request passes rate limiting."""
        if not self.config.enable_rate_limiting:
            return True

        # Get rate limit from domain service if endpoint provided
        if api_endpoint:
            from src.domain.services.request_validation_service import RequestValidationService

            limit_config = RequestValidationService.get_rate_limit_for_endpoint(api_endpoint)
            rule = RateLimitRule(
                max_requests=limit_config["max_requests"],
                window_seconds=limit_config["window_seconds"],
                burst_allowance=limit_config["burst_allowance"],
                cooldown_seconds=RequestValidationService.get_cooldown_period(api_endpoint),
            )
        else:
            rule = self.config.default_rate_limit

        if not self.rate_limiter.is_allowed(identifier, rule):
            raise RateLimitExceeded(
                f"Rate limit exceeded for {identifier}", retry_after=rule.cooldown_seconds
            )

        return True

    def check_throttling(self, identifier: str) -> bool:
        """Check if request passes throttling."""
        if not self.config.enable_request_throttling:
            return True

        if not self.throttler.can_process_request(identifier):
            raise ThrottlingError(f"Request throttled for {identifier}")

        return True

    def start_request_processing(self, identifier: str) -> None:
        """Mark start of request processing."""
        if self.config.enable_request_throttling:
            self.throttler.start_request(identifier)

    def end_request_processing(self, identifier: str) -> None:
        """Mark end of request processing."""
        if self.config.enable_request_throttling:
            self.throttler.end_request(identifier)

    def check_request_headers(self, headers: dict[str, str]) -> bool:
        """Delegate header validation to domain service."""
        from src.domain.services.request_validation_service import RequestValidationService

        errors = RequestValidationService.validate_request_headers(headers)
        return len(errors) == 0

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for response."""
        if not self.config.enable_security_headers:
            return {}
        return SecurityHeaders.get_security_headers()

    def check_request_signature(
        self, method: str, path: str, body: str, headers: dict[str, str]
    ) -> bool:
        """Simple signature verification wrapper."""
        if not self.config.enable_request_signing or not self.request_signer:
            return True

        signature = headers.get(self.config.signature_header)
        timestamp = headers.get(self.config.timestamp_header)

        if not signature or not timestamp:
            return False

        return self.request_signer.check_request(method, path, body, signature, timestamp)

    def check_ip_whitelist(self, client_ip: str) -> bool:
        """Check if IP is whitelisted."""
        if not self.config.enable_ip_whitelist:
            return True

        if client_ip in self.config.blacklisted_ips:
            raise SecurityError(f"IP {client_ip} is blacklisted")

        if self.config.whitelisted_ips and client_ip not in self.config.whitelisted_ips:
            raise SecurityError(f"IP {client_ip} is not whitelisted")

        return True


def secure_endpoint(
    identifier_func: Callable[..., str] | None = None,
    endpoint_name: str | None = None,
    require_signature: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for securing endpoints with all hardening features."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get security hardening instance (should be injected or configured)
            hardening = kwargs.pop("_security_hardening", None)
            if not hardening:
                logger.warning("No security hardening configured for endpoint")
                return func(*args, **kwargs)

            # Extract identifier (IP, user ID, API key, etc.)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = kwargs.get("client_ip", "unknown")

            try:
                # Security checks
                hardening.check_rate_limit(identifier, endpoint_name)
                hardening.check_throttling(identifier)

                if require_signature:
                    method = kwargs.get("method", "GET")
                    path = kwargs.get("path", "/")
                    body = kwargs.get("body", "")
                    headers = kwargs.get("headers", {})
                    hardening.verify_request_signature(method, path, body, headers)

                # Start processing
                hardening.start_request_processing(identifier)

                try:
                    # Execute original function
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # End processing
                    hardening.end_request_processing(identifier)

            except (RateLimitExceeded, ThrottlingError, SecurityError) as e:
                logger.warning(f"Security check failed for {identifier}: {e}")
                raise

        return wrapper

    return decorator


# Trading-specific rate limits are now in the domain service
def get_trading_rate_limits() -> dict[str, RateLimitRule]:
    """Get trading rate limits from domain service."""

    limits = {}
    from src.domain.services.rate_limiting_service import RateLimitingService

    for endpoint, config in RateLimitingService.TRADING_RATE_LIMITS.items():
        limits[endpoint] = RateLimitRule(
            max_requests=config["max_requests"],
            window_seconds=config["window_seconds"],
            burst_allowance=config["burst_allowance"],
        )
    return limits


def create_trading_security_config(
    hmac_secret: str | None = None,
    whitelisted_ips: set[str] | None = None,
    enable_signing: bool = False,
) -> SecurityConfig:
    """Create security configuration optimized for trading systems."""
    from src.domain.services.request_validation_service import RequestValidationService

    # Get trading-specific configuration from domain service
    max_request_size = RequestValidationService.get_max_request_size("/api/trading/")

    return SecurityConfig(
        enable_rate_limiting=True,
        enable_request_throttling=True,
        enable_security_headers=True,
        enable_request_signing=enable_signing,
        enable_ip_whitelist=bool(whitelisted_ips),
        api_rate_limits=get_trading_rate_limits(),
        whitelisted_ips=whitelisted_ips or set(),
        hmac_secret_key=hmac_secret,
        max_request_size=max_request_size,
    )
