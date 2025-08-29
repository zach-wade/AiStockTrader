"""
Authentication and authorization middleware for FastAPI.

This module provides middleware components for JWT validation,
rate limiting, session management, and security headers.
"""

import logging
import secrets
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import redis
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .jwt_service import (
    InvalidTokenException,
    JWTService,
    TokenExpiredException,
    TokenRevokedException,
)
from .models import AuthAuditLog, UserSession
from .permissions import PermissionMatrix
from .rbac_service import RBACService

logger = logging.getLogger(__name__)


class JWTBearer(HTTPBearer):
    """
    JWT Bearer token authentication.

    Validates JWT tokens in Authorization header.
    """

    def __init__(self, jwt_service: JWTService, db_session: Session, auto_error: bool = True):
        """
        Initialize JWT Bearer authentication.

        Args:
            jwt_service: JWT service instance
            db_session: Database session
            auto_error: Automatically raise HTTPException on error
        """
        super().__init__(auto_error=auto_error)
        self.jwt_service = jwt_service
        self.db = db_session

    async def __call__(self, request: Request) -> dict[str, Any] | None:  # type: ignore[override]
        """
        Validate JWT token from Authorization header.

        Args:
            request: FastAPI request object

        Returns:
            Token payload if valid

        Raises:
            HTTPException: If token is invalid
        """
        credentials: HTTPAuthorizationCredentials | None = await super().__call__(request)

        if not credentials:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization required"
                )
            return None

        if credentials.scheme != "Bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication scheme"
                )
            return None

        # Verify JWT token
        try:
            payload = self.jwt_service.verify_access_token(credentials.credentials)

            # Validate session
            session_id = payload.get("sid")
            if session_id:
                session = (
                    self.db.query(UserSession).filter_by(id=session_id, is_active=True).first()
                )

                if not session or not session.is_valid():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session expired or invalid",
                    )

                # Update session activity
                session.update_activity()
                self.db.commit()

            # Store user context in request state
            request.state.user_id = payload["sub"]
            request.state.email = payload.get("email")
            request.state.username = payload.get("username")
            request.state.roles = payload.get("roles", [])
            request.state.permissions = payload.get("permissions", [])
            request.state.session_id = session_id
            request.state.device_id = payload.get("device_id")

            # IP validation (optional, can be strict or just log)
            if "ip" in payload:
                client_ip = request.client.host if request.client else None
                if client_ip and payload["ip"] != client_ip:
                    logger.warning(
                        f"IP mismatch for user {payload['sub']}: "
                        f"token IP {payload['ip']} != request IP {client_ip}"
                    )

            return dict(payload)

        except TokenExpiredException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except TokenRevokedException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except InvalidTokenException as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e!s}")
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
                )
            return None


class APIKeyAuth:
    """
    API Key authentication.

    Validates API keys in X-API-Key header or api_key query parameter.
    """

    def __init__(
        self,
        rbac_service: RBACService,
        redis_client: redis.Redis | None = None,
        auto_error: bool = True,
    ):
        """
        Initialize API Key authentication.

        Args:
            rbac_service: RBAC service instance
            redis_client: Redis client for rate limiting
            auto_error: Automatically raise HTTPException on error
        """
        self.rbac_service = rbac_service
        self.redis = redis_client
        self.auto_error = auto_error
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def __call__(self, request: Request) -> dict[str, Any] | None:
        """
        Validate API key from header or query parameter.

        Args:
            request: FastAPI request object

        Returns:
            API key information if valid

        Raises:
            HTTPException: If API key is invalid
        """
        # Try header first
        api_key = await self.api_key_header(request)

        # Fall back to query parameter
        if not api_key:
            api_key = request.query_params.get("api_key")

        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
                )
            return None

        # Validate API key
        key_info = await self.rbac_service.validate_api_key(api_key)

        if not key_info:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
                )
            return None

        # Check rate limit
        if not await self._check_rate_limit(key_info.id, key_info.rate_limit):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
            )

        # Store context in request
        request.state.user_id = key_info.user_id
        request.state.api_key_id = key_info.id
        request.state.permissions = key_info.permissions
        request.state.auth_type = "api_key"

        # Update last used
        client_ip = request.client.host if request.client else None
        await self.rbac_service.update_api_key_last_used(key_info.id, client_ip)

        return {
            "user_id": key_info.user_id,
            "api_key_id": key_info.id,
            "permissions": key_info.permissions,
        }

    async def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check rate limit for API key."""
        if not self.redis:
            return True

        # Use sliding window rate limiting
        now = time.time()
        window_start = now - 3600  # 1 hour window
        key = f"rate_limit:api_key:{key_id}"

        # Remove old entries
        if self.redis:
            await self.redis.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            request_count = await self.redis.zcard(key)

            if request_count >= limit:
                return False

            # Add current request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, 3600)

        return True


class RequirePermission:
    """
    Permission requirement dependency.

    Checks if authenticated user has required permission.
    """

    def __init__(self, resource: str, action: str) -> None:
        """
        Initialize permission requirement.

        Args:
            resource: Resource name
            action: Action name
        """
        self.resource = resource
        self.action = action
        self.permission = f"{resource}:{action}"

    async def __call__(self, request: Request) -> bool:
        """
        Check if user has required permission.

        Args:
            request: FastAPI request object

        Returns:
            True if permission granted

        Raises:
            HTTPException: If permission denied
        """
        if not hasattr(request.state, "permissions"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )

        # Check for admin bypass
        if hasattr(request.state, "roles") and "admin" in request.state.roles:
            return True

        if self.permission not in request.state.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{self.permission}' required",
            )

        return True


class RequireRole:
    """
    Role requirement dependency.

    Checks if authenticated user has required role.
    """

    def __init__(self, *roles: str) -> None:
        """
        Initialize role requirement.

        Args:
            roles: Required roles (user must have at least one)
        """
        self.roles = roles

    async def __call__(self, request: Request) -> bool:
        """
        Check if user has required role.

        Args:
            request: FastAPI request object

        Returns:
            True if role granted

        Raises:
            HTTPException: If role not found
        """
        if not hasattr(request.state, "roles"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )

        if not any(role in request.state.roles for role in self.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {', '.join(self.roles)}",
            )

        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware.

    Adds security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # CSP header (adjust based on your needs)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )

        return response  # type: ignore[no-any-return]


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Request ID middleware.

    Adds unique request ID for tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Add request ID to request and response."""
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = f"req_{secrets.token_urlsafe(16)}"

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response  # type: ignore[no-any-return]


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware.

    Logs all authenticated requests for security audit.
    """

    def __init__(self, app: ASGIApp, db_session: Session) -> None:
        """
        Initialize audit logging middleware.

        Args:
            app: ASGI application
            db_session: Database session
        """
        super().__init__(app)
        self.db = db_session

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Log request for audit."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Log if authenticated
        if hasattr(request.state, "user_id"):
            process_time = time.time() - start_time

            audit_log = AuthAuditLog(
                user_id=request.state.user_id,
                event_type="api_request",
                event_data={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "request_id": getattr(request.state, "request_id", None),
                    "auth_type": getattr(request.state, "auth_type", "jwt"),
                },
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                success=response.status_code < 400,
            )

            self.db.add(audit_log)
            self.db.commit()

        return response  # type: ignore[no-any-return]


class RateLimitMiddleware:
    """
    Rate limiting middleware using Redis.

    Implements sliding window rate limiting per user/IP.
    """

    def __init__(
        self, redis_client: redis.Redis, default_limit: int = 100, window_seconds: int = 60
    ):
        """
        Initialize rate limit middleware.

        Args:
            redis_client: Redis client
            default_limit: Default rate limit
            window_seconds: Time window in seconds
        """
        self.redis = redis_client
        self.default_limit = default_limit
        self.window = window_seconds

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Any], limit: int | None = None
    ) -> Response:
        """
        Check rate limit for request.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            limit: Override default limit

        Returns:
            Response or rate limit error
        """
        # Get identifier (user ID or IP)
        if hasattr(request.state, "user_id"):
            identifier = f"user:{request.state.user_id}"
        elif request.client:
            identifier = f"ip:{request.client.host}"
        else:
            # Can't rate limit without identifier
            return await call_next(request)  # type: ignore[no-any-return]

        # Check rate limit
        limit = limit or self.default_limit
        key = f"rate_limit:{identifier}:{request.url.path}"

        now = time.time()
        window_start = now - self.window

        # Remove old entries
        if self.redis:
            await self.redis.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            request_count = await self.redis.zcard(key)

            if request_count >= limit:
                # Get oldest request time to calculate retry-after
                oldest = await self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(self.window - (now - oldest[0][1]))
                else:
                    retry_after = self.window

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + retry_after)),
                    },
                )

            # Add current request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, self.window)
        else:
            request_count = 0

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - request_count - 1)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window))

        return response  # type: ignore[no-any-return]


class SessionManagementMiddleware(BaseHTTPMiddleware):
    """
    Session management middleware.

    Handles session validation and refresh.
    """

    def __init__(
        self,
        app: ASGIApp,
        db_session: Session,
        session_timeout_minutes: int = 30,
        refresh_threshold_minutes: int = 5,
    ):
        """
        Initialize session management middleware.

        Args:
            app: ASGI application
            db_session: Database session
            session_timeout_minutes: Session timeout
            refresh_threshold_minutes: Auto-refresh if expiring soon
        """
        super().__init__(app)
        self.db = db_session
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.refresh_threshold = timedelta(minutes=refresh_threshold_minutes)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Manage user session."""
        response = await call_next(request)

        # Check if authenticated with session
        if hasattr(request.state, "session_id"):
            session = (
                self.db.query(UserSession)
                .filter_by(id=request.state.session_id, is_active=True)
                .first()
            )

            if session:
                # Check if session needs refresh
                time_until_expiry = session.expires_at - datetime.utcnow()

                if time_until_expiry < self.refresh_threshold:
                    # Extend session
                    session.extend(self.session_timeout)
                    self.db.commit()

                    # Add header to indicate session was refreshed
                    response.headers["X-Session-Refreshed"] = "true"
                    response.headers["X-Session-Expires"] = session.expires_at.isoformat()

        return response  # type: ignore[no-any-return]


# Convenience functions for dependency injection


def get_current_user(request: Request, jwt_bearer: dict[str, Any] = Depends(JWTBearer)) -> str:
    """Get current authenticated user ID."""
    return request.state.user_id  # type: ignore[no-any-return]


def get_current_user_permissions(
    request: Request, jwt_bearer: dict[str, Any] = Depends(JWTBearer)
) -> list[str]:
    """Get current user permissions."""
    return request.state.permissions  # type: ignore[no-any-return]


def get_current_user_roles(
    request: Request, jwt_bearer: dict[str, Any] = Depends(JWTBearer)
) -> list[str]:
    """Get current user roles."""
    return request.state.roles  # type: ignore[no-any-return]


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive authentication middleware for the trading system.

    Handles:
    - JWT token validation
    - API key authentication
    - Session management
    - Rate limiting
    - Audit logging
    - Security headers
    """

    def __init__(
        self,
        app: ASGIApp,
        jwt_service: JWTService,
        rbac_service: RBACService,
        db_session: Session,
        redis_client: redis.Redis | None = None,
        excluded_paths: list[str] | None = None,
        require_auth: bool = True,
    ):
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            jwt_service: JWT service instance
            rbac_service: RBAC service instance
            db_session: Database session
            redis_client: Redis client for caching
            excluded_paths: Paths that don't require authentication
            require_auth: Whether to require authentication by default
        """
        super().__init__(app)
        self.jwt_service = jwt_service
        self.rbac_service = rbac_service
        self.db = db_session
        self.redis = redis_client
        self.excluded_paths = excluded_paths or [
            "/auth/register",
            "/auth/login",
            "/auth/password/reset",
            "/auth/email/verify",
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Process request through authentication pipeline."""
        start_time = time.time()

        # Check if path is excluded from authentication
        if self._is_excluded_path(request.url.path):
            return await call_next(request)  # type: ignore[no-any-return]

        # Try to authenticate request
        auth_result = await self._authenticate_request(request)

        if self.require_auth and not auth_result:
            return Response(
                content="Authentication required",
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Process request
        response = await call_next(request)

        # Log audit event if authenticated
        if auth_result:
            self._log_request(request, response, time.time() - start_time)

        return response  # type: ignore[no-any-return]

    async def _authenticate_request(self, request: Request) -> bool:
        """
        Authenticate request using JWT or API key.

        Returns:
            True if authenticated, False otherwise
        """
        # Check for Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return await self._validate_jwt_token(request, token)

        # Check for API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            return await self._validate_api_key(request, api_key)

        return False

    async def _validate_jwt_token(self, request: Request, token: str) -> bool:
        """Validate JWT token and set request context."""
        try:
            payload = self.jwt_service.verify_access_token(token)

            # Validate session if present
            session_id = payload.get("sid")
            if session_id:
                session = (
                    self.db.query(UserSession).filter_by(id=session_id, is_active=True).first()
                )

                if not session or not session.is_valid():
                    return False

                # Update session activity
                session.update_activity()
                self.db.commit()

            # Set request context
            request.state.authenticated = True
            request.state.auth_type = "jwt"
            request.state.user_id = payload["sub"]
            request.state.email = payload.get("email")
            request.state.username = payload.get("username")
            request.state.roles = payload.get("roles", [])
            request.state.permissions = payload.get("permissions", [])
            request.state.session_id = session_id
            request.state.device_id = payload.get("device_id")
            request.state.mfa_verified = payload.get("mfa_verified", False)

            return True

        except (TokenExpiredException, TokenRevokedException, InvalidTokenException) as e:
            logger.debug(f"Token validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during token validation: {e}")
            return False

    async def _validate_api_key(self, request: Request, api_key: str) -> bool:
        """Validate API key and set request context."""
        try:
            key_info = await self.rbac_service.validate_api_key(api_key)

            if not key_info:
                return False

            # Check rate limit
            if self.redis and not await self._check_api_key_rate_limit(
                key_info.id, key_info.rate_limit
            ):
                return False

            # Set request context
            request.state.authenticated = True
            request.state.auth_type = "api_key"
            request.state.user_id = key_info.user_id
            request.state.api_key_id = key_info.id
            request.state.permissions = key_info.permissions
            request.state.roles = []  # API keys don't have roles

            # Update last used
            client_ip = request.client.host if request.client else None
            await self.rbac_service.update_api_key_last_used(key_info.id, client_ip)

            return True

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    async def _check_api_key_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check API key rate limit."""
        now = time.time()
        window_start = now - 3600  # 1 hour window
        key = f"rate_limit:api_key:{key_id}"

        # Remove old entries
        if self.redis:
            await self.redis.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            request_count = await self.redis.zcard(key)

            if request_count >= limit:
                return False

            # Add current request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, 3600)

        return True

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False

    def _log_request(self, request: Request, response: Response, duration: float) -> None:
        """Log authenticated request for audit."""
        if not hasattr(request.state, "user_id"):
            return

        # Check if operation should be audited
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2:
            resource = path_parts[0]
            action = request.method.lower()
            permission = f"{resource}:{action}"

            if PermissionMatrix.is_auditable_operation(permission):
                audit_log = AuthAuditLog(
                    user_id=request.state.user_id,
                    event_type="api_request",
                    event_data={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration": duration,
                        "auth_type": getattr(request.state, "auth_type", "unknown"),
                        "permission": permission,
                    },
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("User-Agent"),
                    success=response.status_code < 400,
                )

                self.db.add(audit_log)
                self.db.commit()


class CORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware for handling cross-origin requests.
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list[str] | None = None,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        allow_credentials: bool = True,
        max_age: int = 3600,
    ):
        """Initialize CORS middleware."""
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Handle CORS headers."""
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)

        # Process request
        response = await call_next(request)

        # Add CORS headers
        origin = request.headers.get("Origin")
        if origin and self._is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = str(
                self.allow_credentials
            ).lower()
            response.headers["Vary"] = "Origin"

        return response  # type: ignore[no-any-return]

    def _create_preflight_response(self, request: Request) -> Response:
        """Create preflight response for OPTIONS requests."""
        headers = {
            "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
            "Access-Control-Max-Age": str(self.max_age),
        }

        origin = request.headers.get("Origin")
        if origin and self._is_allowed_origin(origin):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()

        return Response(status_code=200, headers=headers)

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins
