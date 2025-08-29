"""
Database models for authentication and authorization.

This module defines SQLAlchemy models for users, roles, permissions,
sessions, and API keys used in the JWT-based authentication system.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import INET, UUID
from sqlalchemy.types import String as SQLString
from sqlalchemy.types import TypeDecorator


class IPAddress(TypeDecorator[str]):
    """Database-agnostic IP address field."""

    impl = SQLString
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(INET())
        else:
            return dialect.type_descriptor(SQLString(45))  # IPv6 max length


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# Association tables for many-to-many relationships
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("granted_at", DateTime, default=datetime.utcnow),
    Column("granted_by", UUID(as_uuid=True), ForeignKey("users.id")),
    Column("expires_at", DateTime, nullable=True),
    Index("idx_user_roles", "user_id"),
    Index("idx_role_users", "role_id"),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column(
        "role_id", UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "permission_id",
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("idx_role_perms", "role_id"),
    Index("idx_perm_roles", "permission_id"),
)


class User(Base):  # type: ignore[valid-type, misc]
    """User model with enhanced security fields."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))

    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255), index=True)
    email_verification_expires = Column(DateTime)

    # Password reset
    password_reset_token = Column(String(255), index=True)
    password_reset_expires = Column(DateTime)

    # MFA settings
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255))

    # Security features
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime)
    last_login_ip = Column(IPAddress)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    roles = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users",
        primaryjoin="User.id == user_roles.c.user_id",
        secondaryjoin="Role.id == user_roles.c.role_id",
    )
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    oauth_connections = relationship(
        "OAuthConnection", back_populates="user", cascade="all, delete-orphan"
    )
    mfa_backup_codes = relationship(
        "MFABackupCode", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs = relationship("AuthAuditLog", back_populates="user")

    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.locked_until:
            return bool(datetime.utcnow() < self.locked_until)
        return False

    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock account for specified duration."""
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)  # type: ignore[assignment]
        self.failed_login_attempts = 0  # type: ignore[assignment]

    def increment_failed_attempts(self) -> int:
        """Increment failed login attempts."""
        current = getattr(self, "failed_login_attempts", 0) or 0
        self.failed_login_attempts = current + 1  # type: ignore[assignment]
        return getattr(self, "failed_login_attempts", 0)

    def reset_failed_attempts(self) -> None:
        """Reset failed login attempts after successful login."""
        self.failed_login_attempts = 0  # type: ignore[assignment]
        self.locked_until = None  # type: ignore[assignment]

    def get_permissions(self) -> list[str]:
        """Get all permissions for this user through their roles."""
        permissions = set()
        for role in self.roles:
            for permission in role.permissions:
                permissions.add(f"{permission.resource}:{permission.action}")
        return list(permissions)

    def has_permission(self, resource: str, action: str) -> bool:
        """Check if user has specific permission."""
        permission_string = f"{resource}:{action}"
        return permission_string in self.get_permissions()

    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role."""
        return any(role.name == role_name for role in self.roles)


class Role(Base):  # type: ignore[valid-type, misc]
    """Role model for RBAC."""

    __tablename__ = "roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    is_system = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles",
        primaryjoin="Role.id == user_roles.c.role_id",
        secondaryjoin="User.id == user_roles.c.user_id",
    )
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):  # type: ignore[valid-type, misc]
    """Permission model for fine-grained access control."""

    __tablename__ = "permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

    __table_args__ = (
        UniqueConstraint("resource", "action", name="uq_resource_action"),
        Index("idx_resource_action", "resource", "action"),
    )

    def to_string(self) -> str:
        """Convert permission to string format."""
        return f"{self.resource}:{self.action}"


class APIKey(Base):  # type: ignore[valid-type, misc]
    """API Key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    last_four = Column(String(4), nullable=False)
    permissions = Column(JSON, default=list)
    rate_limit = Column(Integer, default=1000)

    # Expiration and usage
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    last_used_ip = Column(IPAddress)

    # Status
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)
    revoked_reason = Column(Text)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_key_user", "user_id"),
        Index("idx_api_key_active", "is_active", "expires_at"),
    )

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at:
            return bool(datetime.utcnow() > self.expires_at)
        return False

    def is_valid(self) -> bool:
        """Check if API key is valid for use."""
        return bool(self.is_active and not self.is_expired() and not self.revoked_at)

    def revoke(self, reason: str | None = None) -> None:
        """Revoke the API key."""
        self.is_active = False  # type: ignore[assignment]
        self.revoked_at = datetime.utcnow()  # type: ignore[assignment]
        self.revoked_reason = reason  # type: ignore[assignment]


class OAuthConnection(Base):  # type: ignore[valid-type, misc]
    """OAuth provider connection for SSO."""

    __tablename__ = "oauth_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)
    provider_user_id = Column(String(255), nullable=False)

    # OAuth tokens
    access_token = Column(Text)
    refresh_token = Column(Text)
    token_expires_at = Column(DateTime)

    # Provider-specific data
    provider_data = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="oauth_connections")

    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id", name="uq_provider_user"),
        Index("idx_oauth_user", "user_id"),
        Index("idx_oauth_provider", "provider", "provider_user_id"),
    )


class UserSession(Base):  # type: ignore[valid-type, misc]
    """User session tracking for security audit."""

    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token_hash = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token_hash = Column(String(255), unique=True, index=True)

    # Device and location info
    device_id = Column(String(255))
    user_agent = Column(Text)
    ip_address = Column(IPAddress)
    location = Column(JSON)

    # Expiration
    expires_at = Column(DateTime, nullable=False)
    refresh_expires_at = Column(DateTime)

    # Status
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)
    revoked_reason = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("idx_session_user", "user_id", "is_active"),
        Index("idx_session_expiry", "expires_at", "is_active"),
    )

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return bool(datetime.utcnow() > self.expires_at)

    def is_valid(self) -> bool:
        """Check if session is valid."""
        return bool(self.is_active and not self.is_expired() and not self.revoked_at)

    def extend(self, duration: timedelta) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.utcnow() + duration  # type: ignore[assignment]
        if self.refresh_expires_at:
            self.refresh_expires_at = datetime.utcnow() + timedelta(days=7)  # type: ignore[assignment]

    def revoke(self, reason: str | None = None) -> None:
        """Revoke the session."""
        self.is_active = False  # type: ignore[assignment]
        self.revoked_at = datetime.utcnow()  # type: ignore[assignment]
        self.revoked_reason = reason  # type: ignore[assignment]

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()  # type: ignore[assignment]


class AuthAuditLog(Base):  # type: ignore[valid-type, misc]
    """Audit log for security events."""

    __tablename__ = "auth_audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSON)
    ip_address = Column(IPAddress)
    user_agent = Column(Text)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")


class MFABackupCode(Base):  # type: ignore[valid-type, misc]
    """MFA backup codes for account recovery."""

    __tablename__ = "mfa_backup_codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    code_hash = Column(String(255), nullable=False, index=True)
    used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="mfa_backup_codes")

    __table_args__ = (Index("idx_mfa_user", "user_id"),)

    def is_used(self) -> bool:
        """Check if backup code has been used."""
        return self.used_at is not None

    def mark_used(self) -> None:
        """Mark backup code as used."""
        self.used_at = datetime.utcnow()  # type: ignore[assignment]
