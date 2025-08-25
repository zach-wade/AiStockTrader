"""
User management service for authentication.

This module handles user registration, login, password management,
and account security features.
"""

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import bcrypt
import pyotp
from email_validator import EmailNotValidError, validate_email
from sqlalchemy.orm import Session

from .jwt_service import JWTService
from .models import AuthAuditLog, MFABackupCode, Role, User, UserSession

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


@dataclass
class RegistrationResult:
    """User registration result."""

    user_id: str
    email: str
    username: str
    email_verification_required: bool
    verification_token: str | None = None


class PasswordHasher:
    """Bcrypt password hashing utility."""

    def __init__(self, rounds: int = 12) -> None:
        """Initialize with bcrypt rounds (cost factor)."""
        self.rounds = rounds

    def hash(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """Check if password needs rehashing with updated rounds."""
        try:
            # Extract rounds from hash
            hash_parts = password_hash.split("$")
            if len(hash_parts) >= 3:
                current_rounds = int(hash_parts[2])
                return current_rounds < self.rounds
        except Exception:
            pass
        return False


class PasswordValidator:
    """Password strength validator."""

    MIN_LENGTH = 12
    MAX_LENGTH = 128

    COMMON_PASSWORDS = {
        "password",
        "123456",
        "password123",
        "admin",
        "letmein",
        "qwerty",
        "monkey",
        "dragon",
        "baseball",
        "iloveyou",
        "trustno1",
        "1234567",
        "welcome",
        "login",
        "admin123",
    }

    @classmethod
    def validate(cls, password: str) -> tuple[bool, list[str]]:
        """
        Validate password strength.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Length check
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters long")
        if len(password) > cls.MAX_LENGTH:
            errors.append(f"Password must not exceed {cls.MAX_LENGTH} characters")

        # Complexity checks
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one number")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password):
            errors.append("Password must contain at least one special character")

        # Common password check
        if password.lower() in cls.COMMON_PASSWORDS:
            errors.append("Password is too common")

        # Sequential character check
        if cls._has_sequential_chars(password):
            errors.append("Password contains sequential characters")

        return len(errors) == 0, errors

    @staticmethod
    def _has_sequential_chars(password: str, threshold: int = 3) -> bool:
        """Check for sequential characters."""
        for i in range(len(password) - threshold + 1):
            substr = password[i : i + threshold]
            if substr.isdigit():
                nums = [int(c) for c in substr]
                if all(nums[j] + 1 == nums[j + 1] for j in range(len(nums) - 1)):
                    return True
            elif substr.isalpha():
                if all(ord(substr[j]) + 1 == ord(substr[j + 1]) for j in range(len(substr) - 1)):
                    return True
        return False


class UserService:
    """
    User management service.

    Handles user registration, authentication, password management,
    and account security.
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
        Initialize user service.

        Args:
            db_session: Database session
            jwt_service: JWT token service
            max_login_attempts: Maximum failed login attempts before lockout
            lockout_duration_minutes: Account lockout duration
            require_email_verification: Whether to require email verification
        """
        self.db = db_session
        self.jwt_service = jwt_service
        self.password_hasher = PasswordHasher()
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        self.require_email_verification = require_email_verification

    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        roles: list[str] | None = None,
    ) -> RegistrationResult:
        """
        Register a new user.

        Args:
            email: User email address
            username: Unique username
            password: User password
            first_name: First name
            last_name: Last name
            roles: Initial roles to assign

        Returns:
            Registration result

        Raises:
            ValueError: If validation fails
            IntegrityError: If user already exists
        """
        # Validate email
        try:
            valid_email = validate_email(email)
            email = valid_email.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e!s}")

        # Validate username
        if not re.match(r"^[a-zA-Z0-9_-]{3,30}$", username):
            raise ValueError("Username must be 3-30 characters, alphanumeric with _ or -")

        # Validate password
        is_valid, errors = PasswordValidator.validate(password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Check if user exists
        existing_user = (
            self.db.query(User).filter((User.email == email) | (User.username == username)).first()
        )

        if existing_user:
            if existing_user.email == email:
                raise ValueError("Email already registered")
            else:
                raise ValueError("Username already taken")

        # Create user
        user = User(
            email=email,
            username=username,
            password_hash=self.password_hasher.hash(password),
            first_name=first_name,
            last_name=last_name,
            email_verified=not self.require_email_verification,
        )

        # Generate email verification token if required
        verification_token = None
        if self.require_email_verification:
            verification_token = secrets.token_urlsafe(32)
            user.email_verification_token = verification_token
            user.email_verification_expires = datetime.utcnow() + timedelta(hours=24)

        # Assign default role
        if not roles:
            roles = ["trader"]  # Default role

        for role_name in roles:
            role = self.db.query(Role).filter_by(name=role_name).first()
            if role:
                user.roles.append(role)

        # Save user
        self.db.add(user)
        self.db.commit()

        # Log registration
        self._log_audit_event(
            event_type="user_registration",
            user_id=str(user.id),
            event_data={"email": email, "username": username},
            success=True,
        )

        logger.info(f"User registered: {username} ({email})")

        return RegistrationResult(
            user_id=str(user.id),
            email=email,
            username=username,
            email_verification_required=self.require_email_verification,
            verification_token=verification_token,
        )

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
        user = (
            self.db.query(User)
            .filter((User.email == email_or_username) | (User.username == email_or_username))
            .first()
        )

        # Always perform password verification to prevent timing attacks
        # Use a dummy hash if user not found to maintain constant time
        if not user:
            # Create a dummy hash to verify against (maintains constant time)
            dummy_hash = "$2b$12$dummy.hash.for.timing.attack.prevention"
            self.password_hasher.verify(password, dummy_hash)

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
            self.password_hasher.verify(password, str(user.password_hash))

            self._log_audit_event(
                event_type="login_failed",
                user_id=str(user.id),
                event_data={"reason": "Account locked"},
                ip_address=ip_address,
                success=False,
            )
            raise ValueError(f"Account locked until {user.locked_until}")

        # Verify password
        if not self.password_hasher.verify(password, str(user.password_hash)):
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
            raise ValueError("Invalid credentials")

        # Check email verification
        if self.require_email_verification and not user.email_verified:
            raise ValueError("Email not verified")

        # Check if MFA is required
        if user.mfa_enabled:
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

        # Create session
        return await self._create_user_session(user, device_id, ip_address, user_agent)

    async def _create_user_session(
        self,
        user: User,
        device_id: str | None,
        ip_address: str | None,
        user_agent: str | None,
    ) -> AuthenticationResult:
        """Create user session and generate tokens."""
        # Reset failed attempts
        user.reset_failed_attempts()

        # Update last login
        user.last_login_at = datetime.utcnow()  # type: ignore[assignment]
        user.last_login_ip = ip_address

        # Create session
        session = UserSession(
            user_id=user.id,
            session_token_hash=secrets.token_urlsafe(32),
            device_id=device_id or secrets.token_urlsafe(16),
            user_agent=user_agent,
            ip_address=ip_address,
            expires_at=datetime.utcnow() + timedelta(hours=24),
            refresh_expires_at=datetime.utcnow() + timedelta(days=7),
        )

        self.db.add(session)
        self.db.commit()

        # Get roles and permissions
        roles = [role.name for role in user.roles]
        permissions = user.get_permissions()

        # Generate tokens
        access_token = self.jwt_service.create_access_token(
            user_id=str(user.id),
            email=str(user.email),
            username=str(user.username),
            roles=roles,
            permissions=permissions,
            session_id=str(session.id),
            device_id=device_id,
            ip_address=ip_address,
            mfa_verified=bool(user.mfa_enabled),
        )

        refresh_token = self.jwt_service.create_refresh_token(
            user_id=str(user.id), session_id=str(session.id), device_id=device_id
        )

        # Update session with token hashes
        session.refresh_token_hash = bcrypt.hashpw(
            refresh_token.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        self.db.commit()

        # Log successful login
        self._log_audit_event(
            event_type="login_success",
            user_id=str(user.id),
            event_data={"session_id": str(session.id), "device_id": device_id},
            ip_address=ip_address,
            success=True,
        )

        logger.info(f"User logged in: {user.username} from {ip_address}")

        return AuthenticationResult(
            user_id=str(user.id),
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=900,  # 15 minutes
            roles=roles,
            permissions=permissions,
            mfa_required=False,
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

        Args:
            mfa_session_token: Temporary session token from initial auth
            mfa_code: MFA code from authenticator app
            device_id: Device identifier
            ip_address: Client IP address
            user_agent: User agent string

        Returns:
            Authentication result with tokens

        Raises:
            ValueError: If MFA verification fails or rate limit exceeded
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
                self._log_audit_event(
                    event_type="mfa_rate_limit_exceeded",
                    user_id=str(user_id),
                    ip_address=ip_address,
                    success=False,
                )
                raise ValueError("Too many MFA attempts. Please try again later.")
        else:
            attempts = 0

        # Increment attempt counter with 5-minute expiry
        self.jwt_service.redis.setex(rate_limit_key, 300, str(attempts + 1))

        # Get user
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Try to verify as TOTP code first, then as backup code
        is_valid = False

        # Check if it's a TOTP code (6 digits)
        if len(mfa_code) == 6 and mfa_code.isdigit():
            is_valid = self._verify_mfa_code(user, mfa_code)
        # Check if it's a backup code (8 hex characters)
        elif len(mfa_code) == 8:
            is_valid = await self.verify_backup_code(str(user.id), mfa_code)

        if not is_valid:
            self._log_audit_event(
                event_type="mfa_failed", user_id=str(user.id), ip_address=ip_address, success=False
            )
            raise ValueError("Invalid MFA code")

        # Delete MFA session token and reset rate limit counter on success
        self.jwt_service.redis.delete(f"mfa:session:{mfa_session_token}")
        self.jwt_service.redis.delete(rate_limit_key)

        # Create session
        return await self._create_user_session(user, device_id, ip_address, user_agent)

    def _verify_mfa_code(self, user: User, code: str) -> bool:
        """
        Verify MFA code using TOTP.

        Args:
            user: User object with MFA secret
            code: 6-digit TOTP code from authenticator app

        Returns:
            True if code is valid
        """
        if not user.mfa_secret:
            logger.warning(f"User {user.id} has MFA enabled but no secret stored")
            return False

        try:
            # Create TOTP instance with user's secret
            totp = pyotp.TOTP(user.mfa_secret)

            # Verify the code with a window of 1 to allow for time drift
            # This accepts codes from 30 seconds before/after current time
            is_valid = totp.verify(code, valid_window=1)

            if not is_valid:
                logger.warning(f"Invalid MFA code attempt for user {user.id}")

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying MFA code for user {user.id}: {e}")
            return False

    async def logout(self, user_id: str, session_id: str, everywhere: bool = False):
        """
        Logout user from current session or all sessions.

        Args:
            user_id: User identifier
            session_id: Current session ID
            everywhere: Logout from all devices
        """
        if everywhere:
            # Revoke all user sessions
            sessions = self.db.query(UserSession).filter_by(user_id=user_id, is_active=True).all()

            for session in sessions:
                session.revoke("User logged out from all devices")

            # Revoke all JWT tokens
            self.jwt_service.revoke_all_user_tokens(user_id)

            self._log_audit_event(event_type="logout_all", user_id=user_id, success=True)

            logger.info(f"User {user_id} logged out from all devices")
        else:
            # Revoke current session
            session = (
                self.db.query(UserSession)
                .filter_by(id=session_id, user_id=user_id, is_active=True)
                .first()
            )

            if session:
                session.revoke("User logged out")
                self.jwt_service.revoke_session_tokens(session_id)

            self._log_audit_event(
                event_type="logout",
                user_id=user_id,
                event_data={"session_id": session_id},
                success=True,
            )

            logger.info(f"User {user_id} logged out from session {session_id}")

        self.db.commit()

    async def request_password_reset(self, email: str) -> bool:
        """
        Request password reset token.

        Args:
            email: User email address

        Returns:
            True if reset token was generated
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
            return True

        # Don't reveal if email exists
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
        is_valid, errors = PasswordValidator.validate(new_password)
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
        user.password_hash = self.password_hasher.hash(new_password)
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

    async def verify_email(self, verification_token: str) -> bool:
        """
        Verify user email address.

        Args:
            verification_token: Email verification token

        Returns:
            True if email was verified
        """
        user = (
            self.db.query(User)
            .filter(
                User.email_verification_token == verification_token,
                User.email_verification_expires > datetime.utcnow(),
            )
            .first()
        )

        if not user:
            raise ValueError("Invalid or expired verification token")

        user.email_verified = True
        user.email_verification_token = None
        user.email_verification_expires = None
        self.db.commit()

        self._log_audit_event(event_type="email_verified", user_id=str(user.id), success=True)

        logger.info(f"Email verified for user {user.username}")
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
        if not self.password_hasher.verify(current_password, str(user.password_hash)):
            self._log_audit_event(
                event_type="password_change_failed",
                user_id=user_id,
                event_data={"reason": "Invalid current password"},
                success=False,
            )
            raise ValueError("Invalid current password")

        # Validate new password
        is_valid, errors = PasswordValidator.validate(new_password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Update password
        user.password_hash = self.password_hasher.hash(new_password)
        self.db.commit()

        self._log_audit_event(event_type="password_changed", user_id=user_id, success=True)

        logger.info(f"Password changed for user {user.username}")
        return True

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

    async def setup_mfa(self, user_id: str) -> dict[str, Any]:
        """
        Setup MFA for a user by generating a new TOTP secret.

        Args:
            user_id: User ID

        Returns:
            Dict containing secret, QR code URI, and backup codes
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        if user.mfa_enabled:
            raise ValueError("MFA is already enabled for this user")

        # Generate a new base32 secret
        secret = pyotp.random_base32()

        # Generate provisioning URI for QR code
        # This URI can be converted to a QR code for scanning
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=str(user.email), issuer_name="AI Trading System"
        )

        # Generate backup codes
        backup_codes = self._generate_backup_codes(user)

        # Store secret (should be encrypted in production)
        user.mfa_secret = secret
        self.db.commit()

        self._log_audit_event(event_type="mfa_setup_initiated", user_id=str(user.id), success=True)

        return {"secret": secret, "qr_code_uri": totp_uri, "backup_codes": backup_codes}

    async def confirm_mfa_setup(self, user_id: str, verification_code: str) -> bool:
        """
        Confirm MFA setup by verifying a code from the authenticator app.

        Args:
            user_id: User ID
            verification_code: Code from authenticator app

        Returns:
            True if MFA was successfully enabled
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        if user.mfa_enabled:
            raise ValueError("MFA is already enabled")

        if not user.mfa_secret:
            raise ValueError("MFA setup not initiated")

        # Verify the code to ensure user has successfully set up their authenticator
        if self._verify_mfa_code(user, verification_code):
            user.mfa_enabled = True
            self.db.commit()

            self._log_audit_event(event_type="mfa_enabled", user_id=str(user.id), success=True)

            logger.info(f"MFA enabled for user {user.id}")
            return True
        else:
            self._log_audit_event(
                event_type="mfa_setup_failed", user_id=str(user.id), success=False
            )
            return False

    async def disable_mfa(self, user_id: str, password: str) -> bool:
        """
        Disable MFA for a user (requires password confirmation).

        Args:
            user_id: User ID
            password: User's password for confirmation

        Returns:
            True if MFA was disabled
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        if not user.mfa_enabled:
            raise ValueError("MFA is not enabled")

        # Verify password before disabling MFA
        if not self.password_hasher.verify(password, str(user.password_hash)):
            self._log_audit_event(
                event_type="mfa_disable_failed",
                user_id=str(user.id),
                event_data={"reason": "invalid_password"},
                success=False,
            )
            raise ValueError("Invalid password")

        # Disable MFA and clear secret
        user.mfa_enabled = False
        user.mfa_secret = None

        # Delete all backup codes
        self.db.query(MFABackupCode).filter_by(user_id=user.id).delete()

        self.db.commit()

        self._log_audit_event(event_type="mfa_disabled", user_id=str(user.id), success=True)

        logger.info(f"MFA disabled for user {user.id}")
        return True

    def _generate_backup_codes(self, user: User, count: int = 10) -> list[str]:
        """
        Generate MFA backup codes for a user.

        Args:
            user: User object
            count: Number of backup codes to generate

        Returns:
            List of backup codes
        """
        # Delete existing backup codes
        self.db.query(MFABackupCode).filter_by(user_id=user.id).delete()

        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = secrets.token_hex(4).upper()

            # Hash the code before storing
            code_hash = self.password_hasher.hash(code)

            backup_code = MFABackupCode(user_id=user.id, code_hash=code_hash, used=False)
            self.db.add(backup_code)
            codes.append(code)

        self.db.commit()
        return codes

    async def verify_backup_code(self, user_id: str, backup_code: str) -> bool:
        """
        Verify and consume a backup code for MFA.

        Args:
            user_id: User ID
            backup_code: Backup code to verify

        Returns:
            True if backup code is valid
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Get unused backup codes for the user
        backup_codes = self.db.query(MFABackupCode).filter_by(user_id=user.id, used=False).all()

        for code_record in backup_codes:
            if self.password_hasher.verify(backup_code.upper(), str(code_record.code_hash)):
                # Mark code as used
                code_record.used = True
                code_record.used_at = datetime.utcnow()  # type: ignore[assignment]
                self.db.commit()

                self._log_audit_event(
                    event_type="mfa_backup_code_used", user_id=str(user.id), success=True
                )

                logger.info(f"Backup code used for user {user.id}")
                return True

        self._log_audit_event(
            event_type="mfa_backup_code_failed", user_id=str(user.id), success=False
        )

        return False
