"""
Unit tests for Role-Based Access Control (RBAC) service.
"""

import hashlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.orm import Session

from src.infrastructure.auth.models import APIKey, Role, User
from src.infrastructure.auth.rbac_service import RBACService


class TestRBACService:
    """Test RBAC service functionality."""

    @pytest.fixture
    def db_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.delete = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def rbac_service(self, db_session):
        """Create RBAC service instance."""
        return RBACService(db_session)

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        user = Mock(spec=User)
        user.id = "user123"
        user.email = "user@example.com"
        user.username = "testuser"
        user.roles = []
        user.has_role = Mock(return_value=False)
        user.has_permission = Mock(return_value=False)
        user.get_permissions = Mock(return_value=[])
        return user

    @pytest.fixture
    def mock_role(self):
        """Create mock role."""
        role = Mock(spec=Role)
        role.id = "role123"
        role.name = "trader"
        role.description = "Trader role"
        role.is_system = False
        role.permissions = []
        return role

    @pytest.mark.asyncio
    async def test_create_role(self, rbac_service, db_session):
        """Test creating a new role."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = (
            None  # No existing role
        )

        # Execute
        role = await rbac_service.create_role(
            name="custom_role",
            description="Custom role",
            permissions=["portfolio:read", "trades:read"],
        )

        # Assert
        assert role.name == "custom_role"
        assert role.description == "Custom role"
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_duplicate_role(self, rbac_service, db_session, mock_role):
        """Test creating duplicate role raises error."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_role

        # Execute and assert
        with pytest.raises(ValueError, match="already exists"):
            await rbac_service.create_role(name="trader")

    @pytest.mark.asyncio
    async def test_update_role(self, rbac_service, db_session, mock_role):
        """Test updating an existing role."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_role

        # Execute
        updated_role = await rbac_service.update_role(
            role_name="trader", description="Updated description", permissions=["portfolio:write"]
        )

        # Assert
        assert updated_role.description == "Updated description"
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_system_role(self, rbac_service, db_session, mock_role):
        """Test updating system role is not allowed."""
        # Setup
        mock_role.is_system = True
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_role

        # Execute and assert
        with pytest.raises(ValueError, match="Cannot modify system role"):
            await rbac_service.update_role(role_name="trader")

    @pytest.mark.asyncio
    async def test_delete_role(self, rbac_service, db_session, mock_role):
        """Test deleting a role."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_role

        # Execute
        result = await rbac_service.delete_role("trader")

        # Assert
        assert result is True
        db_session.delete.assert_called_once_with(mock_role)
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_system_role(self, rbac_service, db_session, mock_role):
        """Test deleting system role is not allowed."""
        # Setup
        mock_role.is_system = True
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_role

        # Execute and assert
        with pytest.raises(ValueError, match="Cannot delete system role"):
            await rbac_service.delete_role("trader")

    @pytest.mark.asyncio
    async def test_assign_role(self, rbac_service, db_session, mock_user, mock_role):
        """Test assigning role to user."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_user,
            mock_role,
        ]

        # Execute
        result = await rbac_service.assign_role(user_id="user123", role_name="trader")

        # Assert
        assert result is True
        mock_user.roles.append.assert_called_once_with(mock_role)
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_assign_duplicate_role(self, rbac_service, db_session, mock_user, mock_role):
        """Test assigning duplicate role returns False."""
        # Setup
        mock_user.roles = [mock_role]
        db_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_user,
            mock_role,
        ]

        # Execute
        result = await rbac_service.assign_role(user_id="user123", role_name="trader")

        # Assert
        assert result is False
        mock_user.roles.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_revoke_role(self, rbac_service, db_session, mock_user, mock_role):
        """Test revoking role from user."""
        # Setup
        mock_user.roles = [mock_role]
        db_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_user,
            mock_role,
        ]

        # Execute
        result = await rbac_service.revoke_role(user_id="user123", role_name="trader")

        # Assert
        assert result is True
        mock_user.roles.remove.assert_called_once_with(mock_role)
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_roles(self, rbac_service, db_session, mock_user):
        """Test getting user roles."""
        # Setup
        role1 = Mock(name="trader")
        role2 = Mock(name="viewer")
        mock_user.roles = [role1, role2]
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        # Execute
        roles = await rbac_service.get_user_roles("user123")

        # Assert
        assert roles == ["trader", "viewer"]

    @pytest.mark.asyncio
    async def test_check_permission(self, rbac_service, db_session, mock_user):
        """Test checking user permission."""
        # Setup
        mock_user.has_permission.return_value = True
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        # Execute
        has_perm = await rbac_service.check_permission("user123", "trades", "execute")

        # Assert
        assert has_perm is True
        mock_user.has_permission.assert_called_once_with("trades", "execute")

    @pytest.mark.asyncio
    async def test_check_permission_admin_bypass(self, rbac_service, db_session, mock_user):
        """Test admin role bypasses permission checks."""
        # Setup
        mock_user.has_role.return_value = True  # Has admin role
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        # Execute
        has_perm = await rbac_service.check_permission("user123", "any", "action")

        # Assert
        assert has_perm is True
        mock_user.has_role.assert_called_once_with("admin")

    @pytest.mark.asyncio
    async def test_get_user_permissions(self, rbac_service, db_session, mock_user):
        """Test getting all user permissions."""
        # Setup
        mock_user.get_permissions.return_value = ["portfolio:read", "trades:execute"]
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        # Execute
        permissions = await rbac_service.get_user_permissions("user123")

        # Assert
        assert permissions == ["portfolio:read", "trades:execute"]


class TestAPIKeyManagement:
    """Test API key management functionality."""

    @pytest.fixture
    def db_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def rbac_service(self, db_session):
        """Create RBAC service instance."""
        return RBACService(db_session)

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        user = Mock(spec=User)
        user.id = "user123"
        user.get_permissions = Mock(return_value=["portfolio:read", "trades:read"])
        return user

    @pytest.mark.asyncio
    async def test_create_api_key(self, rbac_service, db_session, mock_user):
        """Test creating API key."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        with patch(
            "src.infrastructure.auth.rbac_service.secrets.token_urlsafe", return_value="test_token"
        ):
            # Execute
            result = await rbac_service.create_api_key(
                user_id="user123",
                name="Test Key",
                permissions=["trades:execute"],
                rate_limit=500,
                expires_in_days=30,
            )

        # Assert
        assert result["name"] == "Test Key"
        assert "api_key" in result  # Raw key returned
        assert result["rate_limit"] == 500
        assert result["permissions"] == ["trades:execute"]
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_api_key_inherit_permissions(self, rbac_service, db_session, mock_user):
        """Test API key inherits user permissions when not specified."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_user

        with patch(
            "src.infrastructure.auth.rbac_service.secrets.token_urlsafe", return_value="test_token"
        ):
            # Execute
            result = await rbac_service.create_api_key(user_id="user123", name="Test Key")

        # Assert
        assert result["permissions"] == ["portfolio:read", "trades:read"]

    @pytest.mark.asyncio
    async def test_validate_api_key(self, rbac_service, db_session):
        """Test validating API key."""
        # Setup
        api_key = "sk_test_12345"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        mock_key = Mock(spec=APIKey)
        mock_key.id = "key123"
        mock_key.user_id = "user123"
        mock_key.name = "Test Key"
        mock_key.permissions = ["trades:read"]
        mock_key.rate_limit = 1000
        mock_key.expires_at = None
        mock_key.is_active = True
        mock_key.is_valid.return_value = True

        db_session.query.return_value.filter_by.return_value.first.return_value = mock_key

        # Execute
        key_info = await rbac_service.validate_api_key(api_key)

        # Assert
        assert key_info is not None
        assert key_info.id == "key123"
        assert key_info.user_id == "user123"
        assert key_info.permissions == ["trades:read"]

    @pytest.mark.asyncio
    async def test_validate_invalid_api_key(self, rbac_service, db_session):
        """Test validating invalid API key."""
        # Setup
        db_session.query.return_value.filter_by.return_value.first.return_value = None

        # Execute
        key_info = await rbac_service.validate_api_key("invalid_key")

        # Assert
        assert key_info is None

    @pytest.mark.asyncio
    async def test_validate_expired_api_key(self, rbac_service, db_session):
        """Test validating expired API key."""
        # Setup
        mock_key = Mock(spec=APIKey)
        mock_key.is_valid.return_value = False  # Expired or revoked

        db_session.query.return_value.filter_by.return_value.first.return_value = mock_key

        # Execute
        key_info = await rbac_service.validate_api_key("expired_key")

        # Assert
        assert key_info is None

    @pytest.mark.asyncio
    async def test_update_api_key_last_used(self, rbac_service, db_session):
        """Test updating API key last used timestamp."""
        # Setup
        mock_key = Mock(spec=APIKey)
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_key

        # Execute
        await rbac_service.update_api_key_last_used("key123", "127.0.0.1")

        # Assert
        assert mock_key.last_used_at is not None
        assert mock_key.last_used_ip == "127.0.0.1"
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_user_api_keys(self, rbac_service, db_session):
        """Test listing user API keys."""
        # Setup
        mock_key1 = Mock(spec=APIKey)
        mock_key1.id = "key1"
        mock_key1.name = "Key 1"
        mock_key1.last_four = "1234"
        mock_key1.permissions = ["trades:read"]
        mock_key1.rate_limit = 1000
        mock_key1.last_used_at = None
        mock_key1.expires_at = None
        mock_key1.is_active = True
        mock_key1.created_at = datetime.utcnow()

        mock_key2 = Mock(spec=APIKey)
        mock_key2.id = "key2"
        mock_key2.name = "Key 2"
        mock_key2.last_four = "5678"
        mock_key2.permissions = ["portfolio:read"]
        mock_key2.rate_limit = 500
        mock_key2.last_used_at = datetime.utcnow()
        mock_key2.expires_at = datetime.utcnow() + timedelta(days=30)
        mock_key2.is_active = True
        mock_key2.created_at = datetime.utcnow()

        db_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_key1,
            mock_key2,
        ]

        # Execute
        keys = await rbac_service.list_user_api_keys("user123")

        # Assert
        assert len(keys) == 2
        assert keys[0]["name"] == "Key 1"
        assert keys[1]["name"] == "Key 2"

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, rbac_service, db_session):
        """Test revoking API key."""
        # Setup
        mock_key = Mock(spec=APIKey)
        mock_key.revoke = Mock()
        db_session.query.return_value.filter_by.return_value.first.return_value = mock_key

        # Execute
        result = await rbac_service.revoke_api_key("user123", "key123", "No longer needed")

        # Assert
        assert result is True
        mock_key.revoke.assert_called_once_with("No longer needed")
        db_session.commit.assert_called_once()


class TestPermissionDecorators:
    """Test permission and role decorators."""

    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service with mocked methods."""
        service = RBACService(Mock())
        service.check_permission = AsyncMock(return_value=True)
        service.get_user_roles = AsyncMock(return_value=["trader"])
        return service

    @pytest.mark.asyncio
    async def test_require_permission_decorator_success(self, rbac_service):
        """Test require_permission decorator with valid permission."""

        # Setup
        @rbac_service.require_permission("trades", "execute")
        async def protected_function(user_id: str, data: str) -> str:
            return f"Executed for {user_id}: {data}"

        # Execute
        result = await protected_function("user123", "test_data")

        # Assert
        assert result == "Executed for user123: test_data"
        rbac_service.check_permission.assert_called_once_with("user123", "trades", "execute")

    @pytest.mark.asyncio
    async def test_require_permission_decorator_denied(self, rbac_service):
        """Test require_permission decorator with denied permission."""
        # Setup
        rbac_service.check_permission.return_value = False

        @rbac_service.require_permission("admin", "delete")
        async def protected_function(user_id: str) -> str:
            return "Should not execute"

        # Execute and assert
        with pytest.raises(PermissionError, match="admin:delete"):
            await protected_function("user123")

    @pytest.mark.asyncio
    async def test_require_role_decorator_success(self, rbac_service):
        """Test require_role decorator with valid role."""

        # Setup
        @rbac_service.require_role("trader", "admin")
        async def protected_function(user_id: str) -> str:
            return f"Executed for {user_id}"

        # Execute
        result = await protected_function("user123")

        # Assert
        assert result == "Executed for user123"
        rbac_service.get_user_roles.assert_called_once_with("user123")

    @pytest.mark.asyncio
    async def test_require_role_decorator_denied(self, rbac_service):
        """Test require_role decorator with missing role."""
        # Setup
        rbac_service.get_user_roles.return_value = ["viewer"]

        @rbac_service.require_role("admin")
        async def protected_function(user_id: str) -> str:
            return "Should not execute"

        # Execute and assert
        with pytest.raises(PermissionError, match="admin"):
            await protected_function("user123")


class TestSystemInitialization:
    """Test system initialization for roles and permissions."""

    @pytest.fixture
    def db_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def rbac_service(self, db_session):
        """Create RBAC service instance."""
        service = RBACService(db_session)
        service.create_role = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_initialize_default_roles_and_permissions(self, rbac_service, db_session):
        """Test initializing default roles and permissions."""
        # Setup - No existing permissions or roles
        db_session.query.return_value.filter_by.return_value.first.return_value = None

        # Execute
        await rbac_service.initialize_default_roles_and_permissions()

        # Assert
        # Should create default permissions
        assert db_session.add.call_count > 0  # Multiple permissions added

        # Should create default roles
        assert rbac_service.create_role.call_count >= 4  # At least admin, trader, viewer, api_user

        # Verify admin role created
        rbac_service.create_role.assert_any_call(
            name="admin",
            description="Full system administrator access",
            permissions=["system:admin"],
            is_system=True,
        )

        # Verify trader role created with appropriate permissions
        trader_call = None
        for call in rbac_service.create_role.call_args_list:
            if call[1]["name"] == "trader":
                trader_call = call
                break

        assert trader_call is not None
        assert "trades:execute" in trader_call[1]["permissions"]
        assert trader_call[1]["is_system"] is True

        db_session.commit.assert_called()
