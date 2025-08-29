"""
Security hardening middleware for the AI Trading System.

Provides additional security hardening beyond HTTPS enforcement.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Request, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityViolation(Exception):
    """Raised when a security violation is detected."""

    pass


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers."""

    # Content Security Policy
    csp_policy: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' wss: https:; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )

    # Feature Policy / Permissions Policy
    permissions_policy: str = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=(), "
        "usb=(), "
        "magnetometer=(), "
        "gyroscope=(), "
        "speaker=(), "
        "sync-xhr=()"
    )

    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Cache Control
    cache_control: str = "no-cache, no-store, must-revalidate, private"

    # Custom headers
    custom_headers: dict[str, str] | None = None


class SecurityHardeningMiddleware(BaseHTTPMiddleware):
    """
    Security hardening middleware for production trading systems.

    Features:
    - Request validation and sanitization
    - Security header enforcement
    - Request rate limiting per IP
    - Suspicious request detection
    - Security event logging
    - Input validation
    """

    def __init__(
        self,
        app: ASGIApp,
        headers_config: SecurityHeadersConfig | None = None,
        max_request_size: int = 1024 * 1024,  # 1MB
        max_header_size: int = 8192,  # 8KB
        blocked_user_agents: list[str] | None = None,
        blocked_ips: set[str] | None = None,
        suspicious_patterns: list[str] | None = None,
        enable_request_signing: bool = False,
        audit_all_requests: bool = False,
        rate_limit_per_ip: int = 1000,  # requests per hour
        rate_limit_window: int = 3600,  # 1 hour
    ) -> None:
        """
        Initialize security hardening middleware.

        Args:
            app: ASGI application
            headers_config: Security headers configuration
            max_request_size: Maximum request body size in bytes
            max_header_size: Maximum total header size in bytes
            blocked_user_agents: List of blocked user agent patterns
            blocked_ips: Set of blocked IP addresses
            suspicious_patterns: List of suspicious URL patterns
            enable_request_signing: Whether to require request signing
            audit_all_requests: Whether to audit all requests
            rate_limit_per_ip: Rate limit per IP address
            rate_limit_window: Rate limit time window in seconds
        """
        super().__init__(app)
        self.headers_config = headers_config or SecurityHeadersConfig()
        self.max_request_size = max_request_size
        self.max_header_size = max_header_size
        self.blocked_user_agents = blocked_user_agents or [
            "curl",
            "wget",
            "python-requests",
            "bot",
            "crawler",
            "spider",
            "scraper",
        ]
        self.blocked_ips = blocked_ips or set()
        self.suspicious_patterns = suspicious_patterns or [
            "/../",
            "/etc/",
            "/proc/",
            "/var/",
            "SELECT",
            "UNION",
            "DROP",
            "INSERT",
            "DELETE",
            "<script",
            "javascript:",
            "eval(",
            "exec(",
        ]
        self.enable_request_signing = enable_request_signing
        self.audit_all_requests = audit_all_requests
        self.rate_limit_per_ip = rate_limit_per_ip
        self.rate_limit_window = rate_limit_window

        # In-memory rate limiting (use Redis in production)
        self.rate_limit_store: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """
        Process request through security hardening pipeline.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler

        Returns:
            Response with security measures applied
        """
        start_time = time.time()

        try:
            # Validate request security
            await self._validate_request_security(request)

            # Process request
            response: Response = await call_next(request)

            # Apply security headers
            response = self._apply_security_headers(response)

            # Audit if enabled
            if self.audit_all_requests:
                await self._audit_request(request, response, time.time() - start_time)

            return response

        except SecurityViolation as e:
            logger.warning(f"Security violation detected: {e}")
            await self._audit_security_violation(request, str(e))
            return Response(
                content="Request blocked for security reasons",
                status_code=status.HTTP_403_FORBIDDEN,
            )
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return Response(
                content="Internal security error", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def _validate_request_security(self, request: Request) -> None:
        """
        Validate request against security policies.

        Args:
            request: FastAPI request object

        Raises:
            SecurityViolation: If security validation fails
        """
        # Check blocked IPs
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            raise SecurityViolation(f"Blocked IP address: {client_ip}")

        # Check rate limiting
        if not self._check_rate_limit(client_ip):
            raise SecurityViolation(f"Rate limit exceeded for IP: {client_ip}")

        # Check user agent
        user_agent = request.headers.get("user-agent", "").lower()
        for blocked_pattern in self.blocked_user_agents:
            if blocked_pattern.lower() in user_agent:
                raise SecurityViolation(f"Blocked user agent: {user_agent}")

        # Check request size
        content_length = int(request.headers.get("content-length", "0"))
        if content_length > self.max_request_size:
            raise SecurityViolation(f"Request too large: {content_length} bytes")

        # Check header size
        headers_size = sum(len(k) + len(v) for k, v in request.headers.items())
        if headers_size > self.max_header_size:
            raise SecurityViolation(f"Headers too large: {headers_size} bytes")

        # Check for suspicious patterns in URL
        url_path = str(request.url.path).lower()
        query_string = str(request.url.query).lower()
        full_url = f"{url_path}?{query_string}"

        for pattern in self.suspicious_patterns:
            if pattern.lower() in full_url:
                raise SecurityViolation(f"Suspicious pattern detected: {pattern}")

        # Check for suspicious headers
        await self._check_suspicious_headers(request)

        # Validate request signing if enabled
        if self.enable_request_signing:
            await self._validate_request_signature(request)

    async def _check_suspicious_headers(self, request: Request) -> None:
        """
        Check for suspicious request headers.

        Args:
            request: FastAPI request object

        Raises:
            SecurityViolation: If suspicious headers detected
        """
        suspicious_headers = {
            "x-forwarded-host",
            "x-originating-ip",
            "x-cluster-client-ip",
        }

        for header_name in suspicious_headers:
            if header_name in request.headers:
                value = request.headers[header_name]
                # Check for header injection attempts
                if any(char in value for char in ["\r", "\n", "\0"]):
                    raise SecurityViolation(f"Header injection attempt: {header_name}")

    async def _validate_request_signature(self, request: Request) -> None:
        """
        Validate request signature for critical operations.

        Args:
            request: FastAPI request object

        Raises:
            SecurityViolation: If signature validation fails
        """
        signature = request.headers.get("x-request-signature")
        if not signature:
            raise SecurityViolation("Missing request signature")

        # In a real implementation, you would validate the signature
        # using a shared secret or public key cryptography
        # This is a simplified example

        timestamp = request.headers.get("x-timestamp")
        if not timestamp:
            raise SecurityViolation("Missing timestamp in signed request")

        try:
            request_time = float(timestamp)
            current_time = time.time()

            # Check timestamp skew (allow 5 minutes)
            if abs(current_time - request_time) > 300:
                raise SecurityViolation("Request timestamp too old or in future")

        except ValueError:
            raise SecurityViolation("Invalid timestamp format")

    def _check_rate_limit(self, ip: str) -> bool:
        """
        Check rate limit for IP address.

        Args:
            ip: Client IP address

        Returns:
            True if within rate limit, False otherwise
        """
        current_time = time.time()
        window_start = current_time - self.rate_limit_window

        # Get or create request history for IP
        if ip not in self.rate_limit_store:
            self.rate_limit_store[ip] = []

        request_times = self.rate_limit_store[ip]

        # Remove old requests outside the window
        request_times[:] = [t for t in request_times if t > window_start]

        # Check if limit exceeded
        if len(request_times) >= self.rate_limit_per_ip:
            return False

        # Add current request
        request_times.append(current_time)

        # Clean up old entries periodically
        if len(self.rate_limit_store) > 10000:  # Prevent memory growth
            cutoff_time = current_time - (self.rate_limit_window * 2)
            to_remove = []
            for stored_ip, times in self.rate_limit_store.items():
                if not times or max(times) < cutoff_time:
                    to_remove.append(stored_ip)

            for stored_ip in to_remove:
                del self.rate_limit_store[stored_ip]

        return True

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, considering proxies.

        Args:
            request: FastAPI request object

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _apply_security_headers(self, response: Response) -> Response:
        """
        Apply security headers to response.

        Args:
            response: Response object

        Returns:
            Response with security headers
        """
        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = self.headers_config.referrer_policy

        # CSP header
        response.headers["Content-Security-Policy"] = self.headers_config.csp_policy

        # Permissions policy
        response.headers["Permissions-Policy"] = self.headers_config.permissions_policy

        # Cache control
        response.headers["Cache-Control"] = self.headers_config.cache_control
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Custom headers
        if self.headers_config.custom_headers:
            for key, value in self.headers_config.custom_headers.items():
                response.headers[key] = value

        return response

    async def _audit_request(self, request: Request, response: Response, duration: float) -> None:
        """
        Audit request for security monitoring.

        Args:
            request: FastAPI request object
            response: Response object
            duration: Request processing time
        """
        audit_data = {
            "timestamp": time.time(),
            "client_ip": self._get_client_ip(request),
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query) if request.url.query else None,
            "user_agent": request.headers.get("user-agent"),
            "referer": request.headers.get("referer"),
            "status_code": response.status_code,
            "duration": duration,
            "request_size": int(request.headers.get("content-length", "0")),
            "response_size": len(response.body) if hasattr(response, "body") else None,
        }

        # Log audit data (in production, send to security monitoring system)
        logger.info(f"Request audit: {json.dumps(audit_data)}")

    async def _audit_security_violation(self, request: Request, violation: str) -> None:
        """
        Audit security violation.

        Args:
            request: FastAPI request object
            violation: Violation description
        """
        audit_data = {
            "timestamp": time.time(),
            "event_type": "security_violation",
            "client_ip": self._get_client_ip(request),
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query) if request.url.query else None,
            "user_agent": request.headers.get("user-agent"),
            "violation": violation,
            "headers": dict(request.headers),
        }

        # Log security violation (in production, send to SIEM)
        logger.warning(f"Security violation: {json.dumps(audit_data)}")


def create_production_security_middleware(
    app: ASGIApp,
    blocked_ips: set[str] | None = None,
    rate_limit_per_ip: int = 500,
    enable_request_signing: bool = True,
) -> SecurityHardeningMiddleware:
    """
    Create production-ready security hardening middleware.

    Args:
        app: ASGI application
        blocked_ips: Set of blocked IP addresses
        rate_limit_per_ip: Rate limit per IP address
        enable_request_signing: Whether to require request signing

    Returns:
        Configured security hardening middleware
    """
    headers_config = SecurityHeadersConfig(
        csp_policy=(
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'"
        ),
        custom_headers={
            "X-Trading-System": "AI-Trader-v1.0",
            "X-Security-Policy": "strict",
        },
    )

    return SecurityHardeningMiddleware(
        app=app,
        headers_config=headers_config,
        max_request_size=512 * 1024,  # 512KB for production
        blocked_ips=blocked_ips,
        enable_request_signing=enable_request_signing,
        audit_all_requests=True,
        rate_limit_per_ip=rate_limit_per_ip,
    )
