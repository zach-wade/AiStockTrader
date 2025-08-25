"""
Unit tests for authentication middleware.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
import redis
from fastapi import HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from src.infrastructure.auth.jwt_service import JWTService, TokenExpiredException
from src.infrastructure.auth.middleware import (
    APIKeyAuth,
    AuthenticationMiddleware,
    CORSMiddleware,
    JWTBearer,
    RateLimitMiddleware,
    RequirePermission,
    RequireRole,
)
from src.infrastructure.auth.models import UserSession
from src.infrastructure.auth.rbac_service import APIKeyInfo, RBACService


class TestJWTBearer:
    """Test JWT Bearer authentication."""

    @pytest.fixture
    def jwt_service(self):
        """Create mock JWT service."""
        return Mock(spec=JWTService)

    @pytest.fixture
    def db_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit = AsyncMock()
        return session

    @pytest.fixture
    def jwt_bearer(self, jwt_service, db_session):
        """Create JWT Bearer instance."""
        return JWTBearer(jwt_service, db_session)

    @pytest.mark.asyncio
    async def test_valid_jwt_token(self, jwt_bearer, jwt_service, db_session):
        """Test valid JWT token authentication."""
        # Setup
        request = Mock(spec=Request)
        request.client = Mock(host="127.0.0.1")
        request.state = Mock()

        credentials = Mock()
        credentials.credentials = "valid_token"
        credentials.scheme = "Bearer"

        payload = {
            "sub": "user123",
            "email": "user@example.com",
            "username": "testuser",
            "roles": ["trader"],
            "permissions": ["trades:execute"],
            "sid": "session123",
        }

        jwt_service.verify_access_token.return_value = payload

        session = Mock(spec=UserSession)
        session.is_valid.return_value = True
        session.update_activity = Mock()
        db_session.query.return_value.filter_by.return_value.first.return_value = session

        # Mock parent __call__
        with patch.object(jwt_bearer.__class__.__bases__[0], "__call__", return_value=credentials):
            # Execute
            result = await jwt_bearer(request)

        # Assert
        assert result == payload
        jwt_service.verify_access_token.assert_called_once_with("valid_token")
        assert request.state.user_id == "user123"
        assert request.state.email == "user@example.com"
        assert request.state.roles == ["trader"]
        session.update_activity.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_token(self, jwt_bearer, jwt_service):
        """Test expired JWT token."""
        # Setup
        request = Mock(spec=Request)
        credentials = Mock()
        credentials.credentials = "expired_token"
        credentials.scheme = "Bearer"

        jwt_service.verify_access_token.side_effect = TokenExpiredException("Token expired")

        # Mock parent __call__
        with patch.object(jwt_bearer.__class__.__bases__[0], "__call__", return_value=credentials):
            # Execute and assert
            with pytest.raises(HTTPException) as exc_info:
                await jwt_bearer(request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_session(self, jwt_bearer, jwt_service, db_session):
        """Test JWT with invalid session."""
        # Setup
        request = Mock(spec=Request)
        credentials = Mock()
        credentials.credentials = "valid_token"
        credentials.scheme = "Bearer"

        payload = {"sub": "user123", "sid": "invalid_session"}
        jwt_service.verify_access_token.return_value = payload

        # No session found
        db_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock parent __call__
        with patch.object(jwt_bearer.__class__.__bases__[0], "__call__", return_value=credentials):
            # Execute and assert
            with pytest.raises(HTTPException) as exc_info:
                await jwt_bearer(request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Session" in exc_info.value.detail


class TestAPIKeyAuth:
    """Test API Key authentication."""

    @pytest.fixture
    def rbac_service(self):
        """Create mock RBAC service."""
        service = Mock(spec=RBACService)
        service.validate_api_key = AsyncMock()
        service.update_api_key_last_used = AsyncMock()
        return service

    @pytest.fixture
    def redis_client(self):
        """Create mock Redis client."""
        return Mock(spec=redis.Redis)

    @pytest.fixture
    def api_key_auth(self, rbac_service, redis_client):
        """Create API Key auth instance."""
        return APIKeyAuth(rbac_service, redis_client)

    @pytest.mark.asyncio
    async def test_valid_api_key_header(self, api_key_auth, rbac_service):
        """Test valid API key in header."""
        # Setup
        request = Mock(spec=Request)
        request.client = Mock(host="127.0.0.1")
        request.state = Mock()
        request.query_params = {}

        key_info = APIKeyInfo(
            id="key123",
            user_id="user123",
            name="Test Key",
            permissions=["trades:read"],
            rate_limit=1000,
            expires_at=None,
            is_active=True,
        )

        rbac_service.validate_api_key.return_value = key_info

        # Mock header extraction
        with patch.object(api_key_auth, "api_key_header", return_value="test_api_key"):
            with patch.object(
                api_key_auth, "_check_rate_limit", return_value=True
            ) as mock_rate_limit:
                # Execute
                result = await api_key_auth(request)

        # Assert
        assert result["user_id"] == "user123"
        assert request.state.user_id == "user123"
        assert request.state.api_key_id == "key123"
        assert request.state.auth_type == "api_key"
        rbac_service.update_api_key_last_used.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, api_key_auth, rbac_service, redis_client):
        """Test API key rate limit exceeded."""
        # Setup
        request = Mock(spec=Request)
        request.query_params = {"api_key": "test_key"}

        key_info = APIKeyInfo(
            id="key123",
            user_id="user123",
            name="Test Key",
            permissions=[],
            rate_limit=10,
            expires_at=None,
            is_active=True,
        )

        rbac_service.validate_api_key.return_value = key_info

        # Simulate rate limit exceeded
        redis_client.zcard.return_value = 11  # Over limit

        # Mock header extraction returning None
        with patch.object(api_key_auth, "api_key_header", return_value=None):
            # Execute and assert
            with pytest.raises(HTTPException) as exc_info:
                await api_key_auth(request)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestRequirePermission:
    """Test permission requirement decorator."""

    @pytest.mark.asyncio
    async def test_has_permission(self):
        """Test user with required permission."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.permissions = ["trades:execute", "portfolio:read"]
        request.state.roles = []

        require_perm = RequirePermission("trades", "execute")

        # Execute
        result = await require_perm(request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_admin_bypass(self):
        """Test admin role bypasses permission check."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.permissions = []
        request.state.roles = ["admin"]

        require_perm = RequirePermission("any", "action")

        # Execute
        result = await require_perm(request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_missing_permission(self):
        """Test user without required permission."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.permissions = ["portfolio:read"]
        request.state.roles = ["viewer"]

        require_perm = RequirePermission("trades", "execute")

        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await require_perm(request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "trades:execute" in exc_info.value.detail


class TestRequireRole:
    """Test role requirement decorator."""

    @pytest.mark.asyncio
    async def test_has_role(self):
        """Test user with required role."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.roles = ["trader", "viewer"]

        require_role = RequireRole("trader")

        # Execute
        result = await require_role(request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_roles(self):
        """Test user with one of multiple required roles."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.roles = ["viewer"]

        require_role = RequireRole("admin", "trader", "viewer")

        # Execute
        result = await require_role(request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_missing_role(self):
        """Test user without required role."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.roles = ["viewer"]

        require_role = RequireRole("admin", "trader")

        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await require_role(request)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    @pytest.fixture
    def redis_client(self):
        """Create mock Redis client."""
        client = Mock(spec=redis.Redis)
        client.zremrangebyscore = Mock()
        client.zcard = Mock(return_value=0)
        client.zadd = Mock()
        client.expire = Mock()
        return client

    @pytest.fixture
    def rate_limiter(self, redis_client):
        """Create rate limiter instance."""
        return RateLimitMiddleware(redis_client, default_limit=10, window_seconds=60)

    @pytest.mark.asyncio
    async def test_within_limit(self, rate_limiter, redis_client):
        """Test request within rate limit."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock(user_id="user123")
        request.client = Mock(host="127.0.0.1")
        request.url = Mock(path="/api/trades")

        redis_client.zcard.return_value = 5  # Under limit

        call_next = AsyncMock(return_value=Response())

        # Execute
        response = await rate_limiter(request, call_next)

        # Assert
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "10"
        assert response.headers["X-RateLimit-Remaining"] == "4"
        redis_client.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter, redis_client):
        """Test rate limit exceeded."""
        # Setup
        request = Mock(spec=Request)
        request.state = Mock(user_id="user123")
        request.client = Mock(host="127.0.0.1")
        request.url = Mock(path="/api/trades")

        redis_client.zcard.return_value = 10  # At limit
        redis_client.zrange.return_value = [(b"timestamp", time.time() - 30)]

        call_next = AsyncMock()

        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await rate_limiter(request, call_next)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in exc_info.value.headers


class TestAuthenticationMiddleware:
    """Test comprehensive authentication middleware."""

    @pytest.fixture
    def jwt_service(self):
        """Create mock JWT service."""
        return Mock(spec=JWTService)

    @pytest.fixture
    def rbac_service(self):
        """Create mock RBAC service."""
        service = Mock(spec=RBACService)
        service.validate_api_key = AsyncMock()
        service.update_api_key_last_used = AsyncMock()
        return service

    @pytest.fixture
    def db_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture
    def auth_middleware(self, jwt_service, rbac_service, db_session):
        """Create authentication middleware."""
        app = Mock()
        return AuthenticationMiddleware(
            app=app, jwt_service=jwt_service, rbac_service=rbac_service, db_session=db_session
        )

    @pytest.mark.asyncio
    async def test_excluded_path(self, auth_middleware):
        """Test excluded path bypasses authentication."""
        # Setup
        request = Mock(spec=Request)
        request.url = Mock(path="/health")

        call_next = AsyncMock(return_value=Response())

        # Execute
        response = await auth_middleware.dispatch(request, call_next)

        # Assert
        call_next.assert_called_once_with(request)
        assert response is not None

    @pytest.mark.asyncio
    async def test_jwt_authentication(self, auth_middleware, jwt_service):
        """Test successful JWT authentication."""
        # Setup
        request = Mock(spec=Request)
        request.url = Mock(path="/api/trades")
        request.headers = {"Authorization": "Bearer valid_token"}
        request.state = Mock()
        request.client = Mock(host="127.0.0.1")
        request.method = "GET"

        payload = {
            "sub": "user123",
            "email": "user@example.com",
            "roles": ["trader"],
            "permissions": ["trades:read"],
        }
        jwt_service.verify_access_token.return_value = payload

        call_next = AsyncMock(return_value=Response(status_code=200))

        # Execute
        response = await auth_middleware.dispatch(request, call_next)

        # Assert
        assert request.state.authenticated is True
        assert request.state.user_id == "user123"
        assert request.state.auth_type == "jwt"

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, auth_middleware, rbac_service):
        """Test successful API key authentication."""
        # Setup
        request = Mock(spec=Request)
        request.url = Mock(path="/api/trades")
        request.headers = {"X-API-Key": "test_key"}
        request.query_params = {}
        request.state = Mock()
        request.client = Mock(host="127.0.0.1")
        request.method = "GET"

        key_info = APIKeyInfo(
            id="key123",
            user_id="user123",
            name="Test Key",
            permissions=["trades:read"],
            rate_limit=1000,
            expires_at=None,
            is_active=True,
        )
        rbac_service.validate_api_key.return_value = key_info

        call_next = AsyncMock(return_value=Response(status_code=200))

        # Execute
        response = await auth_middleware.dispatch(request, call_next)

        # Assert
        assert request.state.authenticated is True
        assert request.state.user_id == "user123"
        assert request.state.auth_type == "api_key"

    @pytest.mark.asyncio
    async def test_authentication_required(self, auth_middleware):
        """Test authentication required for protected endpoints."""
        # Setup
        request = Mock(spec=Request)
        request.url = Mock(path="/api/trades")
        request.headers = {}
        request.query_params = {}

        call_next = AsyncMock()

        # Execute
        response = await auth_middleware.dispatch(request, call_next)

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        call_next.assert_not_called()


class TestCORSMiddleware:
    """Test CORS middleware."""

    @pytest.fixture
    def cors_middleware(self):
        """Create CORS middleware."""
        app = Mock()
        return CORSMiddleware(
            app=app, allow_origins=["http://localhost:3000", "https://app.example.com"]
        )

    @pytest.mark.asyncio
    async def test_preflight_request(self, cors_middleware):
        """Test OPTIONS preflight request."""
        # Setup
        request = Mock(spec=Request)
        request.method = "OPTIONS"
        request.headers = {"Origin": "http://localhost:3000"}

        call_next = AsyncMock()

        # Execute
        response = await cors_middleware.dispatch(request, call_next)

        # Assert
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_cors_headers_added(self, cors_middleware):
        """Test CORS headers added to response."""
        # Setup
        request = Mock(spec=Request)
        request.method = "GET"
        request.headers = {"Origin": "https://app.example.com"}

        response = Response(status_code=200)
        call_next = AsyncMock(return_value=response)

        # Execute
        result = await cors_middleware.dispatch(request, call_next)

        # Assert
        assert result.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert result.headers["Access-Control-Allow-Credentials"] == "true"

    @pytest.mark.asyncio
    async def test_disallowed_origin(self, cors_middleware):
        """Test disallowed origin doesn't get CORS headers."""
        # Setup
        request = Mock(spec=Request)
        request.method = "GET"
        request.headers = {"Origin": "http://evil.com"}

        response = Response(status_code=200)
        call_next = AsyncMock(return_value=response)

        # Execute
        result = await cors_middleware.dispatch(request, call_next)

        # Assert
        assert "Access-Control-Allow-Origin" not in result.headers
