"""
Authentication service.

Handles user authentication, login attempts, account lockout,
and security measures.
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..jwt_service import JWTService
from ..models import AuthAuditLog, User
from .password_service import PasswordService

logger = logging.getLogger(__name__)


@dataclass
class AuthenticationResult:
    """Authentication result data."""

    user_id: str
    access_token: str
    refresh_token: str
    expires_in: int
    roles: list[str]
    permissions: list[str]
    mfa_required: bool = False
    mfa_session_token: str | None = None


class AuthenticationService:
    """User authentication service."""

    def __init__(
        self,
        db_session: Session,
        jwt_service: JWTService,
        password_service: PasswordService,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        require_email_verification: bool = True,
    ):
        self.db = db_session
        self.jwt_service = jwt_service
        self.password_service = password_service
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        self.require_email_verification = require_email_verification

    async def authenticate(
        self,
        email_or_username: str,
        password: str,
        device_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthenticationResult:
        """
        Authenticate user and create session.

        Args:
            email_or_username: Email or username
            password: User password
            device_id: Device identifier
            ip_address: Client IP address
            user_agent: User agent string

        Returns:
            Authentication result with tokens

        Raises:
            ValueError: If authentication fails
        """
        # Find user
        user = self._find_user(email_or_username)

        # Always perform password verification to prevent timing attacks
        if not user:
            self._perform_dummy_verification(password)
            self._log_audit_event(
                event_type="login_failed",
                event_data={"identifier": email_or_username, "reason": "User not found"},
                ip_address=ip_address,
                success=False,
            )
            raise ValueError("Invalid credentials")

        # Check if account is locked
        if user.is_locked():
            # Still verify password to maintain constant time
            self.password_service.verify_password(password, str(user.password_hash))

            self._log_audit_event(
                event_type="login_failed",
                user_id=str(user.id),
                event_data={"reason": "Account locked"},
                ip_address=ip_address,
                success=False,
            )
            raise ValueError(f"Account locked until {user.locked_until}")

        # Verify password
        if not self.password_service.verify_password(password, str(user.password_hash)):
            self._handle_failed_login(user, ip_address)
            raise ValueError("Invalid credentials")

        # Check email verification
        if self.require_email_verification and not user.email_verified:
            raise ValueError("Email not verified")

        # Check if MFA is required
        if user.mfa_enabled:
            return self._create_mfa_session(user)

        # Create full session
        from .session_manager import SessionManager

        session_manager = SessionManager(self.db, self.jwt_service)
        return await session_manager.create_user_session(user, device_id, ip_address, user_agent)

    def _find_user(self, email_or_username: str) -> User | None:
        """Find user by email or username."""
        return (
            self.db.query(User)
            .filter((User.email == email_or_username) | (User.username == email_or_username))
            .first()
        )

    def _perform_dummy_verification(self, password: str) -> None:
        """Perform dummy password verification to prevent timing attacks."""
        dummy_hash = "$2b$12$dummy.hash.for.timing.attack.prevention"
        self.password_service.verify_password(password, dummy_hash)

    def _handle_failed_login(self, user: User, ip_address: str | None) -> None:
        """Handle failed login attempt."""
        # Increment failed attempts
        attempts = user.increment_failed_attempts()

        # Lock account if max attempts exceeded
        if attempts >= self.max_login_attempts:
            user.lock_account(self.lockout_duration.seconds // 60)
            self.db.commit()

            self._log_audit_event(
                event_type="account_locked",
                user_id=str(user.id),
                event_data={"attempts": attempts},
                ip_address=ip_address,
                success=False,
            )
            raise ValueError("Account locked due to too many failed attempts")

        self.db.commit()

        self._log_audit_event(
            event_type="login_failed",
            user_id=str(user.id),
            event_data={"reason": "Invalid password", "attempts": attempts},
            ip_address=ip_address,
            success=False,
        )

    def _create_mfa_session(self, user: User) -> AuthenticationResult:
        """Create MFA session for two-factor authentication."""
        # Generate temporary session token for MFA
        mfa_session_token = secrets.token_urlsafe(32)
        # Store in cache with 5 minute expiration
        self.jwt_service.redis.setex(f"mfa:session:{mfa_session_token}", 300, str(user.id))

        return AuthenticationResult(
            user_id=str(user.id),
            access_token="",
            refresh_token="",
            expires_in=0,
            roles=[],
            permissions=[],
            mfa_required=True,
            mfa_session_token=mfa_session_token,
        )

    def _log_audit_event(
        self,
        event_type: str,
        user_id: str | None = None,
        event_data: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        success: bool = True,
    ) -> None:
        """Log audit event."""
        audit_log = AuthAuditLog(
            user_id=user_id,
            event_type=event_type,
            event_data=event_data or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
        )
        self.db.add(audit_log)
        self.db.commit()

    async def request_password_reset(self, email: str) -> bool:
        """
        Request password reset token.

        Args:
            email: User email address

        Returns:
            True if reset token was generated (but don't reveal if email exists)
        """
        user = self.db.query(User).filter_by(email=email).first()

        if user:
            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            user.password_reset_token = reset_token
            user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
            self.db.commit()

            self._log_audit_event(
                event_type="password_reset_requested", user_id=str(user.id), success=True
            )

            logger.info(f"Password reset requested for {email}")

        # Always return False to not reveal if email exists
        return False

    async def reset_password(self, reset_token: str, new_password: str) -> bool:
        """
        Reset password using reset token.

        Args:
            reset_token: Password reset token
            new_password: New password

        Returns:
            True if password was reset
        """
        # Validate new password
        is_valid, errors = self.password_service.validate_password(new_password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Find user with valid reset token
        user = (
            self.db.query(User)
            .filter(
                User.password_reset_token == reset_token,
                User.password_reset_expires > datetime.utcnow(),
            )
            .first()
        )

        if not user:
            self._log_audit_event(
                event_type="password_reset_failed",
                event_data={"reason": "Invalid or expired token"},
                success=False,
            )
            raise ValueError("Invalid or expired reset token")

        # Update password
        user.password_hash = self.password_service.hash_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires = None

        # Revoke all sessions
        for session in user.sessions:
            if session.is_active:
                session.revoke("Password reset")

        self.db.commit()

        # Revoke all tokens
        self.jwt_service.revoke_all_user_tokens(str(user.id))

        self._log_audit_event(
            event_type="password_reset_success", user_id=str(user.id), success=True
        )

        logger.info(f"Password reset for user {user.username}")
        return True

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: User identifier
            current_password: Current password
            new_password: New password

        Returns:
            True if password was changed
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Verify current password
        if not self.password_service.verify_password(current_password, str(user.password_hash)):
            self._log_audit_event(
                event_type="password_change_failed",
                user_id=user_id,
                event_data={"reason": "Invalid current password"},
                success=False,
            )
            raise ValueError("Invalid current password")

        # Validate new password
        is_valid, errors = self.password_service.validate_password(new_password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Update password
        user.password_hash = self.password_service.hash_password(new_password)
        self.db.commit()

        self._log_audit_event(event_type="password_changed", user_id=user_id, success=True)

        logger.info(f"Password changed for user {user.username}")
        return True
