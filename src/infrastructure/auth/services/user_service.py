"""
Main user service orchestrator.

Provides a unified interface for all user-related operations by
orchestrating the various specialized services.
"""

import logging
from typing import Any

from sqlalchemy.orm import Session

from ..jwt_service import JWTService
from .authentication import AuthenticationResult, AuthenticationService
from .mfa_service import MFAService
from .password_service import PasswordService
from .registration import RegistrationResult, RegistrationService
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class UserService:
    """
    Main user management service.

    Orchestrates user registration, authentication, password management,
    MFA, and account security by delegating to specialized services.
    """

    def __init__(
        self,
        db_session: Session,
        jwt_service: JWTService,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        require_email_verification: bool = True,
    ):
        """
        Initialize user service with its dependencies.

        Args:
            db_session: Database session
            jwt_service: JWT token service
            max_login_attempts: Maximum failed login attempts before lockout
            lockout_duration_minutes: Account lockout duration
            require_email_verification: Whether to require email verification
        """
        self.db = db_session
        self.jwt_service = jwt_service

        # Initialize specialized services
        self.password_service = PasswordService()

        self.registration_service = RegistrationService(
            db_session, self.password_service, require_email_verification
        )

        self.auth_service = AuthenticationService(
            db_session,
            jwt_service,
            self.password_service,
            max_login_attempts,
            lockout_duration_minutes,
            require_email_verification,
        )

        self.session_manager = SessionManager(db_session, jwt_service)

        self.mfa_service = MFAService(db_session, self.password_service)

    # Registration operations
    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        roles: list[str] | None = None,
    ) -> RegistrationResult:
        """Register a new user."""
        return await self.registration_service.register_user(
            email, username, password, first_name, last_name, roles
        )

    async def verify_email(self, verification_token: str) -> bool:
        """Verify user email address."""
        return await self.registration_service.verify_email(verification_token)

    async def resend_verification_email(self, email: str) -> bool:
        """Resend email verification token."""
        return await self.registration_service.resend_verification_email(email)

    # Authentication operations
    async def authenticate(
        self,
        email_or_username: str,
        password: str,
        device_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthenticationResult:
        """Authenticate user and create session."""
        return await self.auth_service.authenticate(
            email_or_username, password, device_id, ip_address, user_agent
        )

    async def verify_mfa(
        self,
        mfa_session_token: str,
        mfa_code: str,
        device_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthenticationResult:
        """
        Verify MFA code and complete authentication with rate limiting.
        """
        # Get user ID from session token
        user_id = self.jwt_service.redis.get(f"mfa:session:{mfa_session_token}")
        if not user_id:
            raise ValueError("Invalid or expired MFA session")

        # Check MFA attempt rate limit (5 attempts per 5 minutes per user)
        rate_limit_key = f"mfa:attempts:{user_id}"
        attempts = self.jwt_service.redis.get(rate_limit_key)

        if attempts:
            attempts = int(attempts)
            if attempts >= 5:
                raise ValueError("Too many MFA attempts. Please try again later.")
        else:
            attempts = 0

        # Increment attempt counter with 5-minute expiry
        self.jwt_service.redis.setex(rate_limit_key, 300, str(attempts + 1))

        # Verify MFA code
        is_valid = self.mfa_service.verify_mfa_code(str(user_id), mfa_code)

        if not is_valid:
            raise ValueError("Invalid MFA code")

        # Delete MFA session token and reset rate limit counter on success
        self.jwt_service.redis.delete(f"mfa:session:{mfa_session_token}")
        self.jwt_service.redis.delete(rate_limit_key)

        # Get user and create session
        from ..models import User

        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        return await self.session_manager.create_user_session(
            user, device_id, ip_address, user_agent
        )

    # Password operations
    async def request_password_reset(self, email: str) -> bool:
        """Request password reset token."""
        return await self.auth_service.request_password_reset(email)

    async def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password using reset token."""
        return await self.auth_service.reset_password(reset_token, new_password)

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        return await self.auth_service.change_password(user_id, current_password, new_password)

    # Session operations
    async def logout(self, user_id: str, session_id: str, everywhere: bool = False):
        """Logout user from current session or all sessions."""
        await self.session_manager.logout(user_id, session_id, everywhere)

    async def refresh_token(
        self, refresh_token: str, device_id: str | None = None
    ) -> dict[str, Any]:
        """Refresh access token using refresh token."""
        return await self.session_manager.refresh_token(refresh_token, device_id)

    async def get_active_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get all active sessions for a user."""
        return await self.session_manager.get_active_sessions(user_id)

    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """Revoke a specific session."""
        return await self.session_manager.revoke_session(user_id, session_id)

    # MFA operations
    async def setup_mfa(self, user_id: str) -> dict[str, Any]:
        """Setup MFA for a user."""
        return await self.mfa_service.setup_mfa(user_id)

    async def confirm_mfa_setup(self, user_id: str, verification_code: str) -> bool:
        """Confirm MFA setup by verifying a code."""
        return await self.mfa_service.confirm_mfa_setup(user_id, verification_code)

    async def disable_mfa(self, user_id: str, password: str) -> bool:
        """Disable MFA for a user."""
        return await self.mfa_service.disable_mfa(user_id, password)

    async def regenerate_backup_codes(self, user_id: str, password: str) -> list[str]:
        """Regenerate backup codes for a user."""
        return await self.mfa_service.regenerate_backup_codes(user_id, password)

    async def get_backup_codes_status(self, user_id: str) -> dict[str, Any]:
        """Get backup codes status for a user."""
        return await self.mfa_service.get_backup_codes_status(user_id)

    # Utility methods for backward compatibility
    def _log_audit_event(
        self,
        event_type: str,
        user_id: str | None = None,
        event_data: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        success: bool = True,
    ) -> None:
        """Log audit event (delegates to auth service)."""
        self.auth_service._log_audit_event(
            event_type, user_id, event_data, ip_address, user_agent, success
        )
