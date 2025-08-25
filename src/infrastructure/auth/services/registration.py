"""
User registration service.

Handles user registration, email validation, username validation,
and initial role assignment.
"""

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from email_validator import EmailNotValidError, validate_email
from sqlalchemy.orm import Session

from ..models import Role, User
from .password_service import PasswordService


@dataclass
class RegistrationResult:
    """User registration result."""

    user_id: str
    email: str
    username: str
    email_verification_required: bool
    verification_token: str | None = None


class RegistrationService:
    """User registration service."""

    def __init__(
        self,
        db_session: Session,
        password_service: PasswordService,
        require_email_verification: bool = True,
    ):
        self.db = db_session
        self.password_service = password_service
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
        validated_email = self._validate_email(email)

        # Validate username
        self._validate_username(username)

        # Validate password
        is_valid, errors = self.password_service.validate_password(password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Check if user exists
        self._check_user_exists(validated_email, username)

        # Create user
        user = self._create_user(validated_email, username, password, first_name, last_name)

        # Generate email verification token if required
        verification_token = self._setup_email_verification(user)

        # Assign roles
        self._assign_roles(user, roles or ["trader"])

        # Save user
        self.db.add(user)
        self.db.commit()

        return RegistrationResult(
            user_id=str(user.id),
            email=validated_email,
            username=username,
            email_verification_required=self.require_email_verification,
            verification_token=verification_token,
        )

    def _validate_email(self, email: str) -> str:
        """Validate and normalize email address."""
        try:
            valid_email = validate_email(email)
            return valid_email.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e!s}")

    def _validate_username(self, username: str) -> None:
        """Validate username format."""
        if not re.match(r"^[a-zA-Z0-9_-]{3,30}$", username):
            raise ValueError("Username must be 3-30 characters, alphanumeric with _ or -")

    def _check_user_exists(self, email: str, username: str) -> None:
        """Check if user with email or username already exists."""
        existing_user = (
            self.db.query(User).filter((User.email == email) | (User.username == username)).first()
        )

        if existing_user:
            if existing_user.email == email:
                raise ValueError("Email already registered")
            else:
                raise ValueError("Username already taken")

    def _create_user(
        self,
        email: str,
        username: str,
        password: str,
        first_name: str | None,
        last_name: str | None,
    ) -> User:
        """Create user object."""
        return User(
            email=email,
            username=username,
            password_hash=self.password_service.hash_password(password),
            first_name=first_name,
            last_name=last_name,
            email_verified=not self.require_email_verification,
        )

    def _setup_email_verification(self, user: User) -> str | None:
        """Setup email verification if required."""
        if not self.require_email_verification:
            return None

        verification_token = secrets.token_urlsafe(32)
        user.email_verification_token = verification_token
        user.email_verification_expires = datetime.utcnow() + timedelta(hours=24)
        return verification_token

    def _assign_roles(self, user: User, role_names: list[str]) -> None:
        """Assign roles to user."""
        for role_name in role_names:
            role = self.db.query(Role).filter_by(name=role_name).first()
            if role:
                user.roles.append(role)

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

        return True

    async def resend_verification_email(self, email: str) -> bool:
        """
        Resend email verification token.

        Args:
            email: User email address

        Returns:
            True if verification email was sent
        """
        user = self.db.query(User).filter_by(email=email, email_verified=False).first()

        if not user:
            return False  # Don't reveal if email exists

        # Generate new token
        verification_token = secrets.token_urlsafe(32)
        user.email_verification_token = verification_token
        user.email_verification_expires = datetime.utcnow() + timedelta(hours=24)

        self.db.commit()

        return True
