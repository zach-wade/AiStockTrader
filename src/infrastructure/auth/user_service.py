"""
User management service for authentication.

This module handles user registration, login, password management,
and account security features.

This module has been refactored into focused components in the services/ subdirectory.
The classes here are maintained for backward compatibility.
"""

# Import refactored components
from .services import (
    AuthenticationService,
    MFAService,
    PasswordService,
    RegistrationService,
    SessionManager,
    UserService,
)

# Import data classes for backward compatibility
from .services.authentication import AuthenticationResult
from .services.password_service import PasswordHasher, PasswordValidator
from .services.registration import RegistrationResult

# Re-export for backward compatibility
__all__ = [
    "AuthenticationResult",
    "RegistrationResult",
    "PasswordHasher",
    "PasswordValidator",
    "PasswordService",
    "RegistrationService",
    "AuthenticationService",
    "SessionManager",
    "MFAService",
    "UserService",
]
