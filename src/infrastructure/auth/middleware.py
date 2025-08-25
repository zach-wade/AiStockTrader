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

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:  # type: ignore[override]
        """
        Validate JWT token from Authorization header.

        Args:
            request: FastAPI request object

        Returns:
            Token payload if valid

        Raises:
            HTTPException: If token is invalid
        """
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

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
                await self.db.commit()

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

            return payload

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

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:  # type: ignore[override]
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
        self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = self.redis.zcard(key)

        if request_count >= limit:
            return False

        # Add current request
        self.redis.zadd(key, {str(now): now})
        self.redis.expire(key, 3600)

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

    async def dispatch(self, request: Request, call_next) -> Response:
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

    async def dispatch(self, request: Request, call_next) -> Any:
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

    async def dispatch(self, request: Request, call_next) -> Response:
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
            await self.db.commit()

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
        self, request: Request, call_next: Callable[..., Any], limit: int | None = None
    ):
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
            return await call_next(request)

        # Check rate limit
        limit = limit or self.default_limit
        key = f"rate_limit:{identifier}:{request.url.path}"

        now = time.time()
        window_start = now - self.window

        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = self.redis.zcard(key)

        if request_count >= limit:
            # Get oldest request time to calculate retry-after
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
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
        self.redis.zadd(key, {str(now): now})
        self.redis.expire(key, self.window)

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

    async def dispatch(self, request: Request, call_next) -> Response:
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
                    await self.db.commit()

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
