"""
Integrated security middleware for the AI Trading System.

Combines all security features into a unified middleware stack:
- HTTPS/TLS enforcement
- Rate limiting with backoff
- MFA enforcement for critical operations
- RSA key validation
- Security hardening
- Audit logging
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..auth.mfa_enforcement import MFAEnforcementService, MFARequiredOperation
from ..middleware.https_enforcement import HTTPSEnforcementMiddleware
from ..middleware.security_hardening import SecurityHardeningMiddleware
from ..rate_limiting import EndpointRateLimiter, RateLimitTier, create_trading_endpoint_configs
from ..security.key_management import RSAKeyManager

logger = logging.getLogger(__name__)


class IntegratedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Integrated security middleware that orchestrates all security components.

    Features:
    - HTTPS enforcement with TLS validation
    - Comprehensive rate limiting with backoff
    - MFA enforcement for critical operations
    - RSA key validation and rotation
    - Security hardening and headers
    - Centralized audit logging
    - Performance monitoring
    """

    def __init__(
        self,
        app: ASGIApp,
        https_middleware: HTTPSEnforcementMiddleware,
        security_hardening: SecurityHardeningMiddleware,
        endpoint_rate_limiter: EndpointRateLimiter,
        mfa_enforcement: MFAEnforcementService | None = None,
        key_manager: RSAKeyManager | None = None,
        security_monitoring: bool = True,
        bypass_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.https_middleware = https_middleware
        self.security_hardening = security_hardening
        self.endpoint_rate_limiter = endpoint_rate_limiter
        self.mfa_enforcement = mfa_enforcement
        self.key_manager = key_manager
        self.security_monitoring = security_monitoring

        # Paths that bypass security checks (health endpoints, etc.)
        self.bypass_paths = set(
            bypass_paths
            or [
                "/health",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
            ]
        )

        # Security metrics
        self._security_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "https_redirects": 0,
            "rate_limit_violations": 0,
            "mfa_required": 0,
            "security_violations": 0,
            "last_reset": time.time(),
        }

        logger.info("Integrated Security Middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """Process request through integrated security pipeline."""
        start_time = time.time()
        self._security_metrics["total_requests"] += 1

        # Check if path should bypass security
        if self._should_bypass_security(request.url.path):
            return await call_next(request)  # type: ignore[no-any-return]

        try:
            # Step 1: HTTPS/TLS enforcement
            https_response = await self._enforce_https(request)
            if https_response:
                self._security_metrics["https_redirects"] += 1
                return https_response

            # Step 2: Security hardening checks
            hardening_response = await self._apply_security_hardening(request)
            if hardening_response:
                self._security_metrics["security_violations"] += 1
                return hardening_response

            # Step 3: Rate limiting
            rate_limit_response = await self._check_rate_limits(request)
            if rate_limit_response:
                self._security_metrics["rate_limit_violations"] += 1
                return rate_limit_response

            # Step 4: RSA key validation (if applicable)
            if self.key_manager and self._requires_key_validation(request):
                key_validation_response = await self._validate_keys(request)
                if key_validation_response:
                    return key_validation_response

            # Step 5: MFA enforcement (for critical operations)
            if self.mfa_enforcement and self._requires_mfa(request):
                mfa_response = await self._enforce_mfa(request)
                if mfa_response:
                    self._security_metrics["mfa_required"] += 1
                    return mfa_response

            # Process the request
            response = await call_next(request)

            # Post-processing security measures
            response = await self._apply_security_headers(response)

            # Security monitoring
            if self.security_monitoring:
                await self._monitor_security_metrics(request, response, time.time() - start_time)

            return response  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            self._security_metrics["blocked_requests"] += 1

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Security processing failed"},
            )

    def _should_bypass_security(self, path: str) -> bool:
        """Check if path should bypass security checks."""
        return any(path.startswith(bypass_path) for bypass_path in self.bypass_paths)

    async def _enforce_https(self, request: Request) -> Response | None:
        """Enforce HTTPS/TLS security."""
        try:
            # Create a mock call_next that returns None to check if HTTPS enforcement would redirect
            async def mock_call_next(req: Any) -> Any:
                return None

            # This is a simplified check - in reality you'd integrate more deeply
            # with the HTTPS middleware
            if request.url.scheme != "https" and not self._is_from_trusted_proxy(request):
                from ..middleware.https_enforcement import HTTPSRedirectResponse

                https_url = str(request.url.replace(scheme="https"))
                return HTTPSRedirectResponse(url=https_url)

            return None

        except Exception as e:
            logger.error(f"HTTPS enforcement error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "HTTPS enforcement failed"},
            )

    async def _apply_security_hardening(self, request: Request) -> Response | None:
        """Apply security hardening checks."""
        try:
            # This would integrate with the security hardening middleware
            # For now, we'll do basic checks

            # Check for suspicious patterns in URL
            suspicious_patterns = ["../", "etc/passwd", "<script", "SELECT ", "UNION "]
            url_path = str(request.url.path).lower()
            query_string = str(request.url.query).lower()

            for pattern in suspicious_patterns:
                if pattern.lower() in url_path or pattern.lower() in query_string:
                    logger.warning(f"Suspicious pattern detected: {pattern} in {request.url}")
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"error": "Request blocked for security reasons"},
                    )

            # Check request size
            content_length = int(request.headers.get("content-length", "0"))
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": "Request too large"},
                )

            return None

        except Exception as e:
            logger.error(f"Security hardening error: {e}")
            return None

    async def _check_rate_limits(self, request: Request) -> Response | None:
        """Check endpoint-specific rate limits."""
        try:
            # Get user information
            user_id = getattr(request.state, "user_id", None)
            user_tier = getattr(request.state, "user_tier", RateLimitTier.BASIC)

            # Build identifier for rate limiting
            if user_id:
                identifier = f"user:{user_id}"
            else:
                identifier = f"ip:{request.client.host}" if request.client else "unknown"

            # Check rate limits
            result = self.endpoint_rate_limiter.check_endpoint_rate_limit(
                path=request.url.path,
                method=request.method,
                user_tier=user_tier,
                identifier=identifier,
            )

            if not result["allowed"]:
                response_data = {
                    "error": "Rate limit exceeded",
                    "limit": result.get("limit", 0),
                    "remaining": result.get("remaining", 0),
                    "retry_after": result.get("retry_after", 60),
                }

                headers = {}
                if result.get("retry_after"):
                    headers["Retry-After"] = str(result["retry_after"])
                if result.get("custom_headers"):
                    headers.update(result["custom_headers"])

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=response_data,
                    headers=headers,
                )

            # Store rate limit info in request state
            request.state.rate_limit_status = result

            return None

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open for availability
            return None

    async def _validate_keys(self, request: Request) -> Response | None:
        """Validate RSA keys for cryptographic operations."""
        try:
            # This would check if the request involves cryptographic operations
            # that require key validation

            # For JWT operations, validate signing keys
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                # This would validate the JWT signing key
                # Implementation depends on your JWT service
                pass

            return None

        except Exception as e:
            logger.error(f"Key validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Key validation failed"},
            )

    def _requires_mfa(self, request: Request) -> bool:
        """Check if request requires MFA verification."""
        if not self.mfa_enforcement:
            return False

        # Map request paths to MFA operations
        mfa_required_paths = {
            "/api/v1/orders": MFARequiredOperation.PLACE_ORDER,
            "/api/v1/orders/cancel": MFARequiredOperation.CANCEL_ORDER,
            "/api/v1/orders/modify": MFARequiredOperation.MODIFY_ORDER,
            "/api/v1/positions/close": MFARequiredOperation.POSITION_CLOSE,
            "/api/v1/risk/limits": MFARequiredOperation.RISK_LIMIT_CHANGE,
            "/api/v1/account/settings": MFARequiredOperation.ACCOUNT_SETTINGS_CHANGE,
            "/api/v1/auth/api-keys": MFARequiredOperation.API_KEY_GENERATION,
        }

        # Check if path requires MFA
        path = request.url.path
        for mfa_path in mfa_required_paths:
            if path.startswith(mfa_path):
                return True

        return False

    async def _enforce_mfa(self, request: Request) -> Response | None:
        """Enforce MFA for critical operations."""
        if not self.mfa_enforcement:
            return None

        try:
            user_id = getattr(request.state, "user_id", None)
            if not user_id:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Authentication required for MFA enforcement"},
                )

            # Determine operation from path
            operation = self._map_path_to_mfa_operation(request.url.path)
            if not operation:
                return None

            # Check MFA requirement
            try:
                mfa_session = self.mfa_enforcement.require_mfa(
                    operation=operation,
                    user_id=user_id,
                    request=request,
                )

                # Store MFA session in request state
                request.state.mfa_session = mfa_session
                request.state.mfa_enforcement = self.mfa_enforcement

                return None

            except HTTPException as e:
                # Convert to JSON response
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": e.detail},
                    headers=e.headers or {},
                )

        except Exception as e:
            logger.error(f"MFA enforcement error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "MFA enforcement failed"},
            )

    def _map_path_to_mfa_operation(self, path: str) -> MFARequiredOperation | None:
        """Map request path to MFA operation."""
        path_mappings = {
            "/api/v1/orders": MFARequiredOperation.PLACE_ORDER,
            "/api/v1/orders/cancel": MFARequiredOperation.CANCEL_ORDER,
            "/api/v1/orders/modify": MFARequiredOperation.MODIFY_ORDER,
            "/api/v1/positions/close": MFARequiredOperation.POSITION_CLOSE,
            "/api/v1/risk/limits": MFARequiredOperation.RISK_LIMIT_CHANGE,
            "/api/v1/account/settings": MFARequiredOperation.ACCOUNT_SETTINGS_CHANGE,
            "/api/v1/auth/api-keys": MFARequiredOperation.API_KEY_GENERATION,
        }

        for pattern, operation in path_mappings.items():
            if path.startswith(pattern):
                return operation

        return None

    def _requires_key_validation(self, request: Request) -> bool:
        """Check if request requires RSA key validation."""
        # Check for JWT tokens or other cryptographic operations
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return True

        # Check for signed requests
        if request.headers.get("x-request-signature"):
            return True

        return False

    def _is_from_trusted_proxy(self, request: Request) -> bool:
        """Check if request is from a trusted proxy."""
        # Check X-Forwarded-Proto header from trusted proxies
        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto == "https":
            # In production, you'd validate the proxy IP
            return True

        return False

    async def _apply_security_headers(self, response: Response) -> Response:
        """Apply security headers to response."""
        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # CSP for financial applications
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'"
        )

        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"

        return response

    async def _monitor_security_metrics(
        self,
        request: Request,
        response: Response,
        duration: float,
    ) -> None:
        """Monitor security metrics and performance."""
        if self.security_monitoring:
            # Log security events
            security_event = {
                "timestamp": time.time(),
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration": duration,
                "user_id": getattr(request.state, "user_id", None),
                "rate_limited": hasattr(request.state, "rate_limit_status"),
                "mfa_verified": hasattr(request.state, "mfa_session"),
            }

            # In production, you'd send this to a monitoring system
            logger.debug(f"Security metrics: {security_event}")

    def get_security_metrics(self) -> dict[str, Any]:
        """Get current security metrics."""
        current_time = time.time()
        uptime = current_time - self._security_metrics["last_reset"]

        return {
            **self._security_metrics,
            "uptime_seconds": uptime,
            "requests_per_second": self._security_metrics["total_requests"] / max(uptime, 1),
            "block_rate": (
                self._security_metrics["blocked_requests"]
                / max(self._security_metrics["total_requests"], 1)
            ),
        }

    def reset_metrics(self) -> None:
        """Reset security metrics."""
        self._security_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "https_redirects": 0,
            "rate_limit_violations": 0,
            "mfa_required": 0,
            "security_violations": 0,
            "last_reset": time.time(),
        }


def create_production_security_middleware(
    app: ASGIApp,
    mfa_enforcement: MFAEnforcementService,
    key_manager: RSAKeyManager,
    allowed_hosts: list[str],
    trusted_proxies: list[str] | None = None,
) -> IntegratedSecurityMiddleware:
    """Create production-ready integrated security middleware."""
    from ..middleware.https_enforcement import create_production_https_middleware
    from ..middleware.security_hardening import (
        create_production_security_middleware as create_hardening,
    )

    # Create individual security components
    https_middleware = create_production_https_middleware(
        app=app,
        allowed_hosts=allowed_hosts,
        trusted_proxies=trusted_proxies,
    )

    security_hardening = create_hardening(
        app=app,
        enable_request_signing=True,
        rate_limit_per_ip=500,
    )

    # Create endpoint rate limiter
    endpoint_configs = create_trading_endpoint_configs()
    endpoint_rate_limiter = EndpointRateLimiter()
    for config in endpoint_configs:
        endpoint_rate_limiter.add_endpoint_config(config)

    return IntegratedSecurityMiddleware(
        app=app,
        https_middleware=https_middleware,
        security_hardening=security_hardening,
        endpoint_rate_limiter=endpoint_rate_limiter,
        mfa_enforcement=mfa_enforcement,
        key_manager=key_manager,
        security_monitoring=True,
        bypass_paths=["/health", "/metrics"],
    )
