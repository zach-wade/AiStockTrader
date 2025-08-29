"""
Role-Based Access Control (RBAC) service for authorization.

This module provides RBAC functionality including role management,
permission checking, and API key management.
"""

import hashlib
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

from sqlalchemy.orm import Session

from .models import APIKey, Permission, Role, User

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """API Key information."""

    id: str
    user_id: str
    name: str
    permissions: list[str]
    rate_limit: int
    expires_at: datetime | None
    is_active: bool


class RBACService:
    """
    Role-Based Access Control service.

    Manages roles, permissions, and API keys for authorization.
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize RBAC service.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self._permission_cache: dict[str, set[str]] = {}

    # Role Management

    async def create_role(
        self,
        name: str,
        description: str | None = None,
        permissions: list[str] | None = None,
        is_system: bool = False,
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name
            description: Role description
            permissions: List of permission strings (resource:action)
            is_system: Whether this is a system role

        Returns:
            Created role
        """
        # Check if role exists
        existing_role = self.db.query(Role).filter_by(name=name).first()
        if existing_role:
            raise ValueError(f"Role '{name}' already exists")

        # Create role
        role = Role(name=name, description=description, is_system=is_system)

        # Add permissions
        if permissions:
            for perm_string in permissions:
                permission = await self._get_or_create_permission(perm_string)
                if permission:
                    role.permissions.append(permission)

        self.db.add(role)
        self.db.commit()

        logger.info(f"Created role: {name}")
        return role

    async def update_role(
        self,
        role_name: str,
        description: str | None = None,
        permissions: list[str] | None = None,
    ) -> Role:
        """
        Update an existing role.

        Args:
            role_name: Role name to update
            description: New description
            permissions: New list of permissions

        Returns:
            Updated role
        """
        role = self.db.query(Role).filter_by(name=role_name).first()
        if not role:
            raise ValueError(f"Role '{role_name}' not found")

        if role.is_system:
            raise ValueError("Cannot modify system role")

        if description is not None:
            role.description = description  # type: ignore[assignment]

        if permissions is not None:
            # Clear existing permissions
            role.permissions.clear()

            # Add new permissions
            for perm_string in permissions:
                permission = await self._get_or_create_permission(perm_string)
                if permission:
                    role.permissions.append(permission)

        self.db.commit()

        # Clear permission cache
        self._clear_permission_cache()

        logger.info(f"Updated role: {role_name}")
        return role

    async def delete_role(self, role_name: str) -> bool:
        """
        Delete a role.

        Args:
            role_name: Role name to delete

        Returns:
            True if deleted
        """
        role = self.db.query(Role).filter_by(name=role_name).first()
        if not role:
            raise ValueError(f"Role '{role_name}' not found")

        if role.is_system:
            raise ValueError("Cannot delete system role")

        self.db.delete(role)
        self.db.commit()

        # Clear permission cache
        self._clear_permission_cache()

        logger.info(f"Deleted role: {role_name}")
        return True

    async def get_role(self, role_name: str) -> Role | None:
        """Get role by name."""
        return self.db.query(Role).filter_by(name=role_name).first()

    async def list_roles(self) -> list[Role]:
        """List all roles."""
        return self.db.query(Role).all()

    # User Role Assignment

    async def assign_role(
        self,
        user_id: str,
        role_name: str,
        granted_by: str | None = None,
        expires_at: datetime | None = None,
    ) -> bool:
        """
        Assign role to user.

        Args:
            user_id: User ID
            role_name: Role name to assign
            granted_by: ID of user granting the role
            expires_at: Optional expiration time

        Returns:
            True if assigned
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        role = self.db.query(Role).filter_by(name=role_name).first()
        if not role:
            raise ValueError(f"Role '{role_name}' not found")

        # Check if already assigned
        if role in user.roles:
            logger.warning(f"User {user_id} already has role {role_name}")
            return False

        # Assign role
        user.roles.append(role)
        self.db.commit()

        # Clear permission cache for user
        self._clear_user_permission_cache(user_id)

        logger.info(f"Assigned role {role_name} to user {user_id}")
        return True

    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """
        Revoke role from user.

        Args:
            user_id: User ID
            role_name: Role name to revoke

        Returns:
            True if revoked
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        role = self.db.query(Role).filter_by(name=role_name).first()
        if not role:
            raise ValueError(f"Role '{role_name}' not found")

        if role not in user.roles:
            logger.warning(f"User {user_id} doesn't have role {role_name}")
            return False

        # Revoke role
        user.roles.remove(role)
        self.db.commit()

        # Clear permission cache for user
        self._clear_user_permission_cache(user_id)

        logger.info(f"Revoked role {role_name} from user {user_id}")
        return True

    async def get_user_roles(self, user_id: str) -> list[str]:
        """Get all roles for a user."""
        from uuid import UUID

        try:
            user_id_uuid = UUID(user_id)
        except ValueError:
            return []

        user = self.db.query(User).filter(User.id == user_id_uuid).first()
        if not user:
            return []
        return [role.name for role in user.roles]

    # Permission Management

    async def _get_or_create_permission(self, permission_string: str) -> Permission | None:
        """Get or create permission from string format."""
        parts = permission_string.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid permission format: {permission_string}")
            return None

        resource, action = parts

        # Check if exists
        permission = self.db.query(Permission).filter_by(resource=resource, action=action).first()

        if not permission:
            # Create new permission
            permission = Permission(resource=resource, action=action)
            self.db.add(permission)
            self.db.commit()

        return permission

    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            resource: Resource name
            action: Action name

        Returns:
            True if user has permission
        """
        # Check cache first
        cache_key = f"{user_id}:{resource}:{action}"
        if cache_key in self._permission_cache:
            return cache_key in self._permission_cache.get(user_id, set())

        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            return False

        # Check if user has admin role (bypass all checks)
        if user.has_role("admin"):
            return True

        # Check specific permission
        return user.has_permission(resource, action)

    async def get_user_permissions(self, user_id: str) -> list[str]:
        """Get all permissions for a user."""
        from uuid import UUID

        try:
            user_id_uuid = UUID(user_id)
        except ValueError:
            return []

        user = self.db.query(User).filter(User.id == user_id_uuid).first()
        if not user:
            return []
        return user.get_permissions()

    # API Key Management

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: list[str] | None = None,
        rate_limit: int = 1000,
        expires_in_days: int | None = None,
    ) -> dict[str, Any]:
        """
        Create API key for user.

        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions for this key
            rate_limit: Rate limit per hour
            expires_in_days: Days until expiration

        Returns:
            API key information including the key (only shown once)
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Generate API key
        key_prefix = "sk_live_" if not self._is_test_environment() else "sk_test_"
        raw_key = key_prefix + secrets.token_urlsafe(32)

        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Get last 4 characters for identification
        last_four = raw_key[-4:]

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # If no permissions specified, inherit user permissions
        if not permissions:
            permissions = user.get_permissions()

        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            last_four=last_four,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

        self.db.add(api_key)
        self.db.commit()

        logger.info(f"Created API key '{name}' for user {user_id}")

        return {
            "id": str(api_key.id),
            "api_key": raw_key,  # Only returned once
            "name": name,
            "last_four": last_four,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": api_key.created_at.isoformat(),
        }

    async def validate_api_key(self, api_key: str) -> APIKeyInfo | None:
        """
        Validate API key and return info.

        Args:
            api_key: Raw API key

        Returns:
            API key information if valid
        """
        # Hash the key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Find key in database
        key_record = self.db.query(APIKey).filter_by(key_hash=key_hash).first()

        if not key_record:
            logger.warning("Invalid API key attempted")
            return None

        # Check if key is valid
        if not key_record.is_valid():
            logger.warning(f"Expired or revoked API key: {key_record.id}")
            return None

        return APIKeyInfo(
            id=str(key_record.id),
            user_id=str(key_record.user_id),
            name=key_record.name,  # type: ignore
            permissions=key_record.permissions,  # type: ignore
            rate_limit=key_record.rate_limit,  # type: ignore
            expires_at=key_record.expires_at,  # type: ignore
            is_active=key_record.is_active,  # type: ignore
        )

    async def update_api_key_last_used(self, key_id: str, ip_address: str | None = None) -> None:
        """Update API key last used timestamp."""
        api_key = self.db.query(APIKey).filter_by(id=key_id).first()
        if api_key:
            api_key.last_used_at = datetime.utcnow()  # type: ignore[assignment]
            if ip_address:
                api_key.last_used_ip = ip_address  # type: ignore
            self.db.commit()

    async def check_api_key_rate_limit(self, key_id: str) -> bool:
        """
        Check if API key is within rate limit.

        Args:
            key_id: API key ID

        Returns:
            True if within limit
        """
        # This would integrate with Redis for actual rate limiting
        # For now, return True
        return True

    async def list_user_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        """List all API keys for a user."""
        api_keys = self.db.query(APIKey).filter_by(user_id=user_id, is_active=True).all()

        return [
            {
                "id": str(key.id),
                "name": key.name,
                "last_four": key.last_four,
                "permissions": key.permissions,
                "rate_limit": key.rate_limit,
                "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "is_active": key.is_active,
                "created_at": key.created_at.isoformat(),
            }
            for key in api_keys
        ]

    async def revoke_api_key(self, user_id: str, key_id: str, reason: str | None = None) -> bool:
        """
        Revoke API key.

        Args:
            user_id: User ID (for authorization)
            key_id: API key ID to revoke
            reason: Revocation reason

        Returns:
            True if revoked
        """
        api_key = self.db.query(APIKey).filter_by(id=key_id, user_id=user_id).first()

        if not api_key:
            raise ValueError("API key not found")

        api_key.revoke(reason)
        self.db.commit()

        logger.info(f"Revoked API key {key_id} for user {user_id}")
        return True

    # Permission Decorators

    def require_permission(self, resource: str, action: str) -> Callable[..., Any]:
        """
        Decorator to require permission for a function.

        Usage:
            @rbac_service.require_permission('trades', 'execute')
            async def execute_trade(user_id: str, ...) -> Any:
                ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract user_id from kwargs or first arg
                user_id = kwargs.get("user_id") or (args[0] if args else None)
                if not user_id:
                    raise ValueError("User ID required for permission check")

                if not await self.check_permission(user_id, resource, action):
                    raise PermissionError(f"Permission denied: {resource}:{action}")

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def require_role(self, *role_names: str) -> Callable[..., Any]:
        """
        Decorator to require role for a function.

        Usage:
            @rbac_service.require_role('admin', 'trader')
            async def admin_function(user_id: str, ...) -> Any:
                ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract user_id from kwargs or first arg
                user_id = kwargs.get("user_id") or (args[0] if args else None)
                if not user_id:
                    raise ValueError("User ID required for role check")

                user_roles = await self.get_user_roles(user_id)
                if not any(role in user_roles for role in role_names):
                    raise PermissionError(f"Role required: {', '.join(role_names)}")

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    # Cache Management

    def _clear_permission_cache(self) -> Any:
        """Clear entire permission cache."""
        self._permission_cache.clear()

    def _clear_user_permission_cache(self, user_id: str) -> Any:
        """Clear permission cache for specific user."""
        self._permission_cache.pop(user_id, None)

    def _is_test_environment(self) -> bool:
        """Check if running in test environment."""
        import os

        return os.getenv("ENVIRONMENT", "development") in ["test", "development"]

    # System Initialization

    async def initialize_default_roles_and_permissions(self) -> None:
        """Initialize default roles and permissions for the system."""
        # Default permissions
        default_permissions = [
            ("portfolio", "read", "View portfolio data"),
            ("portfolio", "write", "Modify portfolio settings"),
            ("trades", "read", "View trade history"),
            ("trades", "execute", "Execute new trades"),
            ("trades", "cancel", "Cancel pending orders"),
            ("api_keys", "create", "Create new API keys"),
            ("api_keys", "revoke", "Revoke API keys"),
            ("users", "read", "View user profiles"),
            ("users", "write", "Modify user data"),
            ("users", "delete", "Delete user accounts"),
            ("system", "admin", "Full administrative access"),
            ("reports", "read", "View reports"),
            ("reports", "generate", "Generate new reports"),
            ("strategies", "read", "View trading strategies"),
            ("strategies", "write", "Modify trading strategies"),
            ("strategies", "execute", "Execute trading strategies"),
        ]

        for resource, action, description in default_permissions:
            existing = self.db.query(Permission).filter_by(resource=resource, action=action).first()

            if not existing:
                permission = Permission(resource=resource, action=action, description=description)
                self.db.add(permission)
                self.db.flush()  # Ensure it's committed immediately to avoid duplicates

        # Default roles
        default_roles = [
            ("admin", "Full system administrator access", ["system:admin"], True),
            (
                "trader",
                "Standard trading operations",
                [
                    "portfolio:read",
                    "portfolio:write",
                    "trades:read",
                    "trades:execute",
                    "trades:cancel",
                    "reports:read",
                    "strategies:read",
                ],
                True,
            ),
            (
                "viewer",
                "Read-only access to portfolio",
                ["portfolio:read", "trades:read", "reports:read"],
                True,
            ),
            (
                "api_user",
                "API-only access for automated trading",
                ["trades:read", "trades:execute", "portfolio:read", "strategies:execute"],
                True,
            ),
            (
                "analyst",
                "Data analysis and reporting",
                ["portfolio:read", "trades:read", "reports:read", "reports:generate"],
                False,
            ),
        ]

        for role_name, description, permissions, is_system in default_roles:
            existing_role = self.db.query(Role).filter_by(name=role_name).first()

            if not existing_role:
                await self.create_role(
                    name=role_name,
                    description=description,
                    permissions=permissions,
                    is_system=is_system,
                )

        self.db.commit()
        logger.info("Initialized default roles and permissions")
