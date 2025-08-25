"""
Integration tests for complete authentication and authorization flows.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
import redis
from fastapi import Depends, FastAPI, status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.infrastructure.auth.endpoints import (
    get_current_user,
    get_db,
    get_jwt_service,
    get_rbac_service,
    get_user_service,
)
from src.infrastructure.auth.endpoints import router as auth_router
from src.infrastructure.auth.jwt_service import JWTService
from src.infrastructure.auth.middleware import (
    RequirePermission,
    RequireRole,
    SecurityHeadersMiddleware,
)
from src.infrastructure.auth.models import Base, User
from src.infrastructure.auth.rbac_service import RBACService
from src.infrastructure.auth.user_service import UserService


@pytest.fixture
def test_db():
    """Create test database."""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    yield db

    db.close()


@pytest.fixture
def redis_client():
    """Create mock Redis client."""
    client = Mock(spec=redis.Redis)
    client.setex = Mock()
    client.get = Mock(return_value=None)
    client.delete = Mock()
    client.sadd = Mock()
    client.expire = Mock()
    client.zremrangebyscore = Mock()
    client.zcard = Mock(return_value=0)
    client.zadd = Mock()
    return client


@pytest.fixture
def jwt_service(redis_client):
    """Create JWT service."""
    # Use test keys for CI/CD
    return JWTService(redis_client=redis_client)


@pytest.fixture
def rbac_service(test_db):
    """Create RBAC service."""
    return RBACService(test_db)


@pytest.fixture
def user_service(test_db, jwt_service):
    """Create user service."""
    return UserService(test_db, jwt_service, require_email_verification=False)


@pytest.fixture
def test_app(test_db, jwt_service, rbac_service, user_service, redis_client):
    """Create test FastAPI app with auth."""
    app = FastAPI()

    # Add auth router
    app.include_router(auth_router)

    # Add middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Add test endpoints
    @app.get("/protected")
    async def protected_endpoint(user_id: str = Depends(get_current_user)):
        return {"user_id": user_id}

    @app.get("/admin-only")
    async def admin_endpoint(_: bool = Depends(RequireRole("admin"))):
        return {"message": "Admin access granted"}

    @app.post("/trades/execute")
    async def execute_trade(_: bool = Depends(RequirePermission("trades", "execute"))):
        return {"message": "Trade executed"}

    # Override dependency injection
    def get_test_db():
        yield test_db

    def get_test_jwt_service():
        return jwt_service

    def get_test_user_service():
        return user_service

    def get_test_rbac_service():
        return rbac_service

    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_jwt_service] = get_test_jwt_service
    app.dependency_overrides[get_user_service] = get_test_user_service
    app.dependency_overrides[get_rbac_service] = get_test_rbac_service

    return app


@pytest.fixture
def test_client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestUserRegistrationFlow:
    """Test user registration flow."""

    def test_successful_registration(self, test_client, rbac_service):
        """Test successful user registration."""
        # Initialize default roles
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        # Register user
        response = test_client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "SecureP@ssw0rd123!",
                "first_name": "New",
                "last_name": "User",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "user_id" in data

    def test_registration_weak_password(self, test_client):
        """Test registration with weak password."""
        response = test_client.post(
            "/auth/register",
            json={
                "email": "user@example.com",
                "username": "testuser",
                "password": "weak",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in response.json()["detail"].lower()

    def test_registration_duplicate_email(self, test_client, rbac_service):
        """Test registration with duplicate email."""
        # Initialize and register first user
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        test_client.post(
            "/auth/register",
            json={
                "email": "existing@example.com",
                "username": "user1",
                "password": "SecureP@ssw0rd123!",
            },
        )

        # Try to register with same email
        response = test_client.post(
            "/auth/register",
            json={
                "email": "existing@example.com",
                "username": "user2",
                "password": "SecureP@ssw0rd123!",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in response.json()["detail"].lower()


class TestLoginFlow:
    """Test login and authentication flow."""

    @pytest.fixture(autouse=True)
    def setup_user(self, test_client, rbac_service):
        """Setup test user."""
        # Initialize roles and register user
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        response = test_client.post(
            "/auth/register",
            json={
                "email": "testuser@example.com",
                "username": "testuser",
                "password": "TestP@ssw0rd123!",
            },
        )
        assert response.status_code == status.HTTP_201_CREATED

    def test_successful_login(self, test_client):
        """Test successful login."""
        response = test_client.post(
            "/auth/login",
            data={
                "username": "testuser",
                "password": "TestP@ssw0rd123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"
        assert data["user_id"] is not None
        assert "trader" in data["roles"]  # Default role

    def test_login_with_email(self, test_client):
        """Test login using email instead of username."""
        response = test_client.post(
            "/auth/login",
            data={
                "username": "testuser@example.com",
                "password": "TestP@ssw0rd123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()

    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials."""
        response = test_client.post(
            "/auth/login",
            data={
                "username": "testuser",
                "password": "WrongPassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_account_lockout(self, test_client, user_service):
        """Test account lockout after failed attempts."""
        # Reduce lockout threshold for testing
        user_service.max_login_attempts = 3

        # Make multiple failed attempts
        for i in range(3):
            response = test_client.post(
                "/auth/login",
                data={
                    "username": "testuser",
                    "password": "WrongPassword",
                },
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Next attempt should indicate account locked
        response = test_client.post(
            "/auth/login",
            data={
                "username": "testuser",
                "password": "TestP@ssw0rd123!",  # Even correct password
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "locked" in response.json()["detail"].lower()


class TestTokenRefreshFlow:
    """Test token refresh flow."""

    @pytest.fixture
    def auth_tokens(self, test_client, rbac_service):
        """Get authentication tokens."""
        # Setup and login
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        test_client.post(
            "/auth/register",
            json={
                "email": "refresh@example.com",
                "username": "refreshuser",
                "password": "TestP@ssw0rd123!",
            },
        )

        response = test_client.post(
            "/auth/login",
            data={
                "username": "refreshuser",
                "password": "TestP@ssw0rd123!",
            },
        )

        return response.json()

    def test_refresh_token(self, test_client, auth_tokens):
        """Test refreshing access token."""
        response = test_client.post(
            "/auth/refresh",
            json={"refresh_token": auth_tokens["refresh_token"]},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        # New tokens should be different (rotation)
        assert data["access_token"] != auth_tokens["access_token"]
        assert data["refresh_token"] != auth_tokens["refresh_token"]

    def test_refresh_invalid_token(self, test_client):
        """Test refresh with invalid token."""
        response = test_client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid_token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestPasswordManagement:
    """Test password reset and change flows."""

    @pytest.fixture(autouse=True)
    def setup_user(self, test_client, rbac_service):
        """Setup test user."""
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        test_client.post(
            "/auth/register",
            json={
                "email": "pwuser@example.com",
                "username": "pwuser",
                "password": "OldP@ssw0rd123!",
            },
        )

    def test_password_reset_request(self, test_client):
        """Test requesting password reset."""
        response = test_client.post(
            "/auth/password/reset",
            json={"email": "pwuser@example.com"},
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        # Should not reveal if email exists
        assert "email exists" in response.json()["message"].lower()

    def test_password_reset_nonexistent_email(self, test_client):
        """Test password reset for non-existent email."""
        response = test_client.post(
            "/auth/password/reset",
            json={"email": "nonexistent@example.com"},
        )

        # Should return same response to prevent email enumeration
        assert response.status_code == status.HTTP_202_ACCEPTED

    @pytest.mark.asyncio
    async def test_password_reset_confirm(self, test_client, test_db):
        """Test confirming password reset."""
        # Get user and set reset token
        user = test_db.query(User).filter_by(email="pwuser@example.com").first()
        user.password_reset_token = "test_reset_token"
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        test_db.commit()

        # Reset password
        response = test_client.post(
            "/auth/password/reset/confirm",
            json={
                "reset_token": "test_reset_token",
                "new_password": "NewP@ssw0rd456!",
            },
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify can login with new password
        login_response = test_client.post(
            "/auth/login",
            data={
                "username": "pwuser",
                "password": "NewP@ssw0rd456!",
            },
        )
        assert login_response.status_code == status.HTTP_200_OK

    def test_change_password(self, test_client):
        """Test changing password while logged in."""
        # Login first
        login_response = test_client.post(
            "/auth/login",
            data={
                "username": "pwuser",
                "password": "OldP@ssw0rd123!",
            },
        )
        token = login_response.json()["access_token"]

        # Change password
        response = test_client.post(
            "/auth/password/change",
            json={
                "current_password": "OldP@ssw0rd123!",
                "new_password": "NewP@ssw0rd789!",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify can login with new password
        login_response = test_client.post(
            "/auth/login",
            data={
                "username": "pwuser",
                "password": "NewP@ssw0rd789!",
            },
        )
        assert login_response.status_code == status.HTTP_200_OK


class TestAPIKeyManagement:
    """Test API key creation and usage."""

    @pytest.fixture
    def user_token(self, test_client, rbac_service):
        """Get authenticated user token."""
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        test_client.post(
            "/auth/register",
            json={
                "email": "apiuser@example.com",
                "username": "apiuser",
                "password": "TestP@ssw0rd123!",
            },
        )

        response = test_client.post(
            "/auth/login",
            data={
                "username": "apiuser",
                "password": "TestP@ssw0rd123!",
            },
        )

        return response.json()["access_token"]

    def test_create_api_key(self, test_client, user_token):
        """Test creating API key."""
        response = test_client.post(
            "/auth/api-keys",
            json={
                "name": "Test API Key",
                "permissions": ["portfolio:read", "trades:read"],
                "rate_limit": 500,
                "expires_in_days": 30,
            },
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test API Key"
        assert "api_key" in data  # Raw key shown once
        assert data["rate_limit"] == 500
        assert len(data["permissions"]) == 2

    def test_list_api_keys(self, test_client, user_token):
        """Test listing user's API keys."""
        # Create a key first
        create_response = test_client.post(
            "/auth/api-keys",
            json={"name": "Key 1"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert create_response.status_code == status.HTTP_201_CREATED

        # List keys
        response = test_client.get(
            "/auth/api-keys",
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        keys = response.json()
        assert len(keys) >= 1
        assert keys[0]["name"] == "Key 1"
        assert keys[0]["api_key"] is None  # Never shown after creation

    def test_revoke_api_key(self, test_client, user_token):
        """Test revoking API key."""
        # Create a key
        create_response = test_client.post(
            "/auth/api-keys",
            json={"name": "To Revoke"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        key_id = create_response.json()["id"]

        # Revoke it
        response = test_client.delete(
            f"/auth/api-keys/{key_id}",
            params={"reason": "No longer needed"},
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT


class TestRoleBasedAccess:
    """Test role-based access control."""

    @pytest.fixture
    def setup_users(self, test_client, test_db, rbac_service):
        """Setup users with different roles."""
        asyncio.run(rbac_service.initialize_default_roles_and_permissions())

        # Create admin user
        test_client.post(
            "/auth/register",
            json={
                "email": "admin@example.com",
                "username": "admin",
                "password": "AdminP@ssw0rd123!",
            },
        )

        # Create trader user
        test_client.post(
            "/auth/register",
            json={
                "email": "trader@example.com",
                "username": "trader",
                "password": "TraderP@ssw0rd123!",
            },
        )

        # Assign admin role
        admin_user = test_db.query(User).filter_by(username="admin").first()
        asyncio.run(rbac_service.assign_role(str(admin_user.id), "admin"))
        test_db.commit()

        # Get tokens
        admin_response = test_client.post(
            "/auth/login",
            data={"username": "admin", "password": "AdminP@ssw0rd123!"},
        )

        trader_response = test_client.post(
            "/auth/login",
            data={"username": "trader", "password": "TraderP@ssw0rd123!"},
        )

        return {
            "admin_token": admin_response.json()["access_token"],
            "trader_token": trader_response.json()["access_token"],
        }

    def test_admin_only_endpoint(self, test_client, setup_users):
        """Test admin-only endpoint access."""
        # Admin should have access
        response = test_client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {setup_users['admin_token']}"},
        )
        assert response.status_code == status.HTTP_200_OK

        # Trader should not have access
        response = test_client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {setup_users['trader_token']}"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_permission_based_access(self, test_client, setup_users):
        """Test permission-based endpoint access."""
        # Trader has trades:execute permission
        response = test_client.post(
            "/trades/execute",
            headers={"Authorization": f"Bearer {setup_users['trader_token']}"},
        )
        assert response.status_code == status.HTTP_200_OK

        # Admin also has access (admin bypass)
        response = test_client.post(
            "/trades/execute",
            headers={"Authorization": f"Bearer {setup_users['admin_token']}"},
        )
        assert response.status_code == status.HTTP_200_OK
