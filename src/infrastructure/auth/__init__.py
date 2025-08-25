"""
JWT-based authentication system for AI trading platform.

This package provides a complete authentication and authorization solution
with JWT tokens, RBAC, MFA support, and API key management.
"""

from .jwt_service import (
    InvalidTokenException,
    JWTService,
    TokenExpiredException,
    TokenReuseException,
    TokenRevokedException,
)
from .middleware import (
    APIKeyAuth,
    AuditLoggingMiddleware,
    JWTBearer,
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequirePermission,
    RequireRole,
    SecurityHeadersMiddleware,
    SessionManagementMiddleware,
    get_current_user,
    get_current_user_permissions,
    get_current_user_roles,
)
from .models import (
    APIKey,
    AuthAuditLog,
    Base,
    MFABackupCode,
    OAuthConnection,
    Permission,
    Role,
    User,
    UserSession,
)
from .rbac_service import APIKeyInfo, RBACService
from .user_service import (
    AuthenticationResult,
    PasswordHasher,
    PasswordValidator,
    RegistrationResult,
    UserService,
)

__all__ = [
    # JWT Service
    "JWTService",
    "TokenExpiredException",
    "TokenRevokedException",
    "InvalidTokenException",
    "TokenReuseException",
    # User Service
    "UserService",
    "AuthenticationResult",
    "RegistrationResult",
    "PasswordHasher",
    "PasswordValidator",
    # RBAC Service
    "RBACService",
    "APIKeyInfo",
    # Middleware
    "JWTBearer",
    "APIKeyAuth",
    "RequirePermission",
    "RequireRole",
    "SecurityHeadersMiddleware",
    "RequestIDMiddleware",
    "AuditLoggingMiddleware",
    "RateLimitMiddleware",
    "SessionManagementMiddleware",
    "get_current_user",
    "get_current_user_permissions",
    "get_current_user_roles",
    # Models
    "User",
    "Role",
    "Permission",
    "APIKey",
    "OAuthConnection",
    "UserSession",
    "AuthAuditLog",
    "MFABackupCode",
    "Base",
]

__version__ = "1.0.0"
