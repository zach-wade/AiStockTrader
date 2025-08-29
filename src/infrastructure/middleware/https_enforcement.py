"""
HTTPS/TLS enforcement middleware for the AI Trading System.

Ensures all communications are encrypted and properly secured for financial operations.
"""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import Request, status
from fastapi.responses import RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class TLSConfigurationError(Exception):
    """Raised when TLS configuration is invalid."""

    pass


class TLSValidationError(Exception):
    """Raised when TLS validation fails."""

    pass


class HTTPSRedirectResponse(RedirectResponse):
    """Custom redirect response for HTTPS enforcement."""

    def __init__(self, url: str, status_code: int = status.HTTP_301_MOVED_PERMANENTLY) -> None:
        super().__init__(url=url, status_code=status_code)
        # Add security headers for redirect
        self.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        self.headers["X-Content-Type-Options"] = "nosniff"


class HTTPSEnforcementMiddleware(BaseHTTPMiddleware):
    """
    HTTPS enforcement middleware for production financial trading system.

    Features:
    - Forces HTTPS connections
    - Validates TLS certificates
    - Adds strict security headers
    - Blocks insecure protocols
    - Audit logging for security events
    """

    def __init__(
        self,
        app: ASGIApp,
        enforce_https: bool = True,
        redirect_http: bool = True,
        hsts_max_age: int = 31536000,
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = True,
        allowed_hosts: list[str] | None = None,
        trusted_proxies: list[str] | None = None,
        validate_tls: bool = True,
        min_tls_version: str = "TLSv1.2",
        blocked_ciphers: list[str] | None = None,
    ) -> None:
        """
        Initialize HTTPS enforcement middleware.

        Args:
            app: ASGI application
            enforce_https: Whether to enforce HTTPS
            redirect_http: Whether to redirect HTTP to HTTPS
            hsts_max_age: HSTS max-age in seconds
            hsts_include_subdomains: Whether to include subdomains in HSTS
            hsts_preload: Whether to include preload in HSTS
            allowed_hosts: List of allowed host headers
            trusted_proxies: List of trusted proxy IPs for X-Forwarded-Proto
            validate_tls: Whether to validate TLS configuration
            min_tls_version: Minimum TLS version required
            blocked_ciphers: List of blocked cipher suites
        """
        super().__init__(app)
        self.enforce_https = enforce_https
        self.redirect_http = redirect_http
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.allowed_hosts = allowed_hosts or ["*"]
        self.trusted_proxies = set(trusted_proxies or [])
        self.validate_tls = validate_tls
        self.min_tls_version = min_tls_version
        self.blocked_ciphers = set(blocked_ciphers or ["RC4", "3DES", "MD5", "SHA1", "DES", "NULL"])

        # Validate configuration
        if self.validate_tls:
            self._validate_tls_config()

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """
        Process request through HTTPS enforcement pipeline.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler

        Returns:
            Response with security headers

        Raises:
            HTTPException: If security validation fails
        """
        # Check if HTTPS enforcement is disabled (development only)
        if not self.enforce_https:
            response = await call_next(request)
            return self._add_security_headers(response, is_secure=False)

        # Determine if connection is secure
        is_secure = self._is_secure_connection(request)

        # Validate host header
        if not self._is_allowed_host(request):
            logger.warning(
                f"Blocked request with invalid host header: {request.headers.get('host')}"
            )
            return Response(content="Invalid host header", status_code=status.HTTP_400_BAD_REQUEST)

        # Handle non-HTTPS requests
        if not is_secure:
            if self.redirect_http:
                # Redirect to HTTPS
                https_url = self._build_https_url(request)
                logger.info(f"Redirecting HTTP request to HTTPS: {https_url}")
                return HTTPSRedirectResponse(url=https_url)
            else:
                # Block non-HTTPS requests
                logger.warning(f"Blocked non-HTTPS request: {request.url}")
                return Response(
                    content="HTTPS required for all connections",
                    status_code=status.HTTP_426_UPGRADE_REQUIRED,
                    headers={"Upgrade": "TLS/1.2, HTTP/1.1"},
                )

        # Validate TLS properties for secure connections
        if is_secure and self.validate_tls:
            try:
                self._validate_tls_connection(request)
            except TLSValidationError as e:
                logger.error(f"TLS validation failed: {e}")
                return Response(
                    content="TLS validation failed", status_code=status.HTTP_400_BAD_REQUEST
                )

        # Process request
        response = await call_next(request)

        # Add security headers
        return self._add_security_headers(response, is_secure=is_secure)

    def _is_secure_connection(self, request: Request) -> bool:
        """
        Determine if the connection is secure.

        Args:
            request: FastAPI request object

        Returns:
            True if connection is secure
        """
        # Check direct HTTPS
        if request.url.scheme == "https":
            return True

        # Check X-Forwarded-Proto from trusted proxies
        forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
        if forwarded_proto == "https":
            # Verify request is from trusted proxy
            client_ip = self._get_client_ip(request)
            if client_ip in self.trusted_proxies or "*" in self.trusted_proxies:
                return True
            else:
                logger.warning(f"X-Forwarded-Proto header from untrusted proxy: {client_ip}")

        # Check other proxy headers
        if request.headers.get("x-forwarded-ssl") == "on":
            client_ip = self._get_client_ip(request)
            if client_ip in self.trusted_proxies or "*" in self.trusted_proxies:
                return True

        return False

    def _is_allowed_host(self, request: Request) -> bool:
        """
        Validate host header against allowed hosts.

        Args:
            request: FastAPI request object

        Returns:
            True if host is allowed
        """
        if "*" in self.allowed_hosts:
            return True

        host = request.headers.get("host", "").lower()
        return host in [h.lower() for h in self.allowed_hosts]

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
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _build_https_url(self, request: Request) -> str:
        """
        Build HTTPS URL from HTTP request.

        Args:
            request: FastAPI request object

        Returns:
            HTTPS URL string
        """
        url = request.url
        return str(url.replace(scheme="https"))

    def _validate_tls_config(self) -> None:
        """
        Validate TLS configuration.

        Raises:
            TLSConfigurationError: If configuration is invalid
        """
        # Validate minimum TLS version
        valid_versions = ["TLSv1.2", "TLSv1.3"]
        if self.min_tls_version not in valid_versions:
            raise TLSConfigurationError(f"Invalid TLS version: {self.min_tls_version}")

        # Validate HSTS settings
        if self.hsts_max_age < 0:
            raise TLSConfigurationError("HSTS max-age must be non-negative")

        logger.info(f"TLS configuration validated: min_version={self.min_tls_version}")

    def _validate_tls_connection(self, request: Request) -> None:
        """
        Validate TLS connection properties.

        Args:
            request: FastAPI request object

        Raises:
            TLSValidationError: If TLS validation fails
        """
        # In a real deployment, you would access the actual SSL context
        # This is a simplified validation for demonstration

        # Check for SSL/TLS headers that might indicate weak encryption
        ssl_cipher = request.headers.get("ssl-cipher")
        if ssl_cipher:
            cipher_upper = ssl_cipher.upper()
            for blocked_cipher in self.blocked_ciphers:
                if blocked_cipher.upper() in cipher_upper:
                    raise TLSValidationError(f"Blocked cipher suite: {ssl_cipher}")

        # Check TLS version header (if provided by proxy)
        tls_version = request.headers.get("ssl-protocol", request.headers.get("tls-version"))
        if tls_version and not self._is_acceptable_tls_version(tls_version):
            raise TLSValidationError(f"TLS version too old: {tls_version}")

    def _is_acceptable_tls_version(self, version: str) -> bool:
        """
        Check if TLS version meets minimum requirements.

        Args:
            version: TLS version string

        Returns:
            True if version is acceptable
        """
        version_upper = version.upper().replace("V", "v")

        # Map versions to numeric values for comparison
        version_values = {
            "TLSv1.0": 1.0,
            "TLSv1.1": 1.1,
            "TLSv1.2": 1.2,
            "TLSv1.3": 1.3,
        }

        current_val = version_values.get(version_upper)
        min_val = version_values.get(self.min_tls_version)

        if current_val is None or min_val is None:
            logger.warning(f"Unknown TLS version: {version}")
            return False

        return current_val >= min_val

    def _add_security_headers(self, response: Response, is_secure: bool) -> Response:
        """
        Add security headers to response.

        Args:
            response: Response object
            is_secure: Whether connection is secure

        Returns:
            Response with security headers
        """
        # Always add basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add HSTS only for HTTPS connections
        if is_secure:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Add CSP for financial applications
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "form-action 'self'"
        )

        # Permissions policy for sensitive APIs
        response.headers["Permissions-Policy"] = (
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

        # Additional security headers for financial systems
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response


def create_production_https_middleware(
    app: ASGIApp,
    allowed_hosts: list[str],
    trusted_proxies: list[str] | None = None,
) -> HTTPSEnforcementMiddleware:
    """
    Create production-ready HTTPS enforcement middleware.

    Args:
        app: ASGI application
        allowed_hosts: List of allowed host headers
        trusted_proxies: List of trusted proxy IPs

    Returns:
        Configured HTTPS enforcement middleware
    """
    return HTTPSEnforcementMiddleware(
        app=app,
        enforce_https=True,
        redirect_http=False,  # Block rather than redirect in production
        hsts_max_age=31536000,  # 1 year
        hsts_include_subdomains=True,
        hsts_preload=True,
        allowed_hosts=allowed_hosts,
        trusted_proxies=trusted_proxies,
        validate_tls=True,
        min_tls_version="TLSv1.2",
        blocked_ciphers=[
            "RC4",
            "3DES",
            "MD5",
            "SHA1",
            "DES",
            "NULL",
            "EXPORT",
            "aDSS",
            "aNULL",
            "eNULL",
            "MEDIUM",
            "LOW",
        ],
    )


def create_development_https_middleware(
    app: ASGIApp,
    redirect_http: bool = True,
) -> HTTPSEnforcementMiddleware:
    """
    Create development-friendly HTTPS enforcement middleware.

    Args:
        app: ASGI application
        redirect_http: Whether to redirect HTTP to HTTPS

    Returns:
        Configured HTTPS enforcement middleware for development
    """
    return HTTPSEnforcementMiddleware(
        app=app,
        enforce_https=False,  # Allow HTTP in development
        redirect_http=redirect_http,
        hsts_max_age=0,  # No HSTS in development
        allowed_hosts=["*"],
        trusted_proxies=["*"],
        validate_tls=False,
    )
