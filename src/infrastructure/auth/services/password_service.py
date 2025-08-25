"""
Password management service.

Handles password hashing, validation, strength checking,
and password-related operations.
"""

import logging
import re

import bcrypt

logger = logging.getLogger(__name__)


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


class PasswordService:
    """Password management service."""

    def __init__(self) -> None:
        self.hasher = PasswordHasher()
        self.validator = PasswordValidator()

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.hasher.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return self.hasher.verify(password, password_hash)

    def validate_password(self, password: str) -> tuple[bool, list[str]]:
        """Validate password strength."""
        return self.validator.validate(password)

    def needs_rehash(self, password_hash: str) -> bool:
        """Check if password needs rehashing."""
        return self.hasher.needs_rehash(password_hash)

    def rehash_if_needed(self, password: str, password_hash: str) -> str | None:
        """Rehash password if needed and password is correct."""
        if self.verify_password(password, password_hash) and self.needs_rehash(password_hash):
            return self.hash_password(password)
        return None
