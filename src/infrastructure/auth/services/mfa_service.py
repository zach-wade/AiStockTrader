"""
Multi-factor authentication service.

Handles MFA setup, verification, backup codes generation,
and MFA-related operations.
"""

import logging
import secrets
from datetime import datetime
from typing import Any

import pyotp
from sqlalchemy.orm import Session

from ..models import AuthAuditLog, MFABackupCode, User
from .password_service import PasswordService

logger = logging.getLogger(__name__)


class MFAService:
    """Multi-factor authentication service."""

    def __init__(self, db_session: Session, password_service: PasswordService):
        self.db = db_session
        self.password_service = password_service

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
        user.mfa_secret = secret  # type: ignore[assignment]
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
            user.mfa_enabled = True  # type: ignore[assignment]
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
        if not self.password_service.verify_password(password, str(user.password_hash)):
            self._log_audit_event(
                event_type="mfa_disable_failed",
                user_id=str(user.id),
                event_data={"reason": "invalid_password"},
                success=False,
            )
            raise ValueError("Invalid password")

        # Disable MFA and clear secret
        user.mfa_enabled = False  # type: ignore[assignment]
        user.mfa_secret = None  # type: ignore[assignment]

        # Delete all backup codes
        self.db.query(MFABackupCode).filter_by(user_id=user.id).delete()

        self.db.commit()

        self._log_audit_event(event_type="mfa_disabled", user_id=str(user.id), success=True)

        logger.info(f"MFA disabled for user {user.id}")
        return True

    def verify_mfa_code(self, user_id: str, code: str) -> bool:
        """
        Verify MFA code (TOTP or backup code).

        Args:
            user_id: User ID
            code: MFA code to verify

        Returns:
            True if code is valid
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Check if it's a TOTP code (6 digits)
        if len(code) == 6 and code.isdigit():
            return self._verify_mfa_code(user, code)
        # Check if it's a backup code (8 hex characters)
        elif len(code) == 8:
            return self._verify_backup_code(user, code)

        return False

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
            totp = pyotp.TOTP(user.mfa_secret)  # type: ignore[arg-type]

            # Verify the code with a window of 1 to allow for time drift
            # This accepts codes from 30 seconds before/after current time
            is_valid = totp.verify(code, valid_window=1)

            if not is_valid:
                logger.warning(f"Invalid MFA code attempt for user {user.id}")

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying MFA code for user {user.id}: {e}")
            return False

    def _verify_backup_code(self, user: User, backup_code: str) -> bool:
        """
        Verify and consume a backup code for MFA.

        Args:
            user: User object
            backup_code: Backup code to verify

        Returns:
            True if backup code is valid
        """
        # Get unused backup codes for the user
        backup_codes = self.db.query(MFABackupCode).filter_by(user_id=user.id, used=False).all()

        for code_record in backup_codes:
            if self.password_service.verify_password(
                backup_code.upper(), str(code_record.code_hash)
            ):
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
            code_hash = self.password_service.hash_password(code)

            backup_code = MFABackupCode(user_id=user.id, code_hash=code_hash, used=False)
            self.db.add(backup_code)
            codes.append(code)

        self.db.commit()
        return codes

    async def regenerate_backup_codes(self, user_id: str, password: str) -> list[str]:
        """
        Regenerate backup codes for a user.

        Args:
            user_id: User ID
            password: User's password for confirmation

        Returns:
            New list of backup codes
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        if not user.mfa_enabled:
            raise ValueError("MFA is not enabled")

        # Verify password before regenerating codes
        if not self.password_service.verify_password(password, str(user.password_hash)):
            self._log_audit_event(
                event_type="mfa_backup_codes_regen_failed",
                user_id=str(user.id),
                event_data={"reason": "invalid_password"},
                success=False,
            )
            raise ValueError("Invalid password")

        # Generate new backup codes
        backup_codes = self._generate_backup_codes(user)

        self._log_audit_event(
            event_type="mfa_backup_codes_regenerated", user_id=str(user.id), success=True
        )

        logger.info(f"Backup codes regenerated for user {user.id}")
        return backup_codes

    async def get_backup_codes_status(self, user_id: str) -> dict[str, Any]:
        """
        Get backup codes status for a user.

        Args:
            user_id: User ID

        Returns:
            Status of backup codes
        """
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        if not user.mfa_enabled:
            return {"mfa_enabled": False, "total_codes": 0, "used_codes": 0, "remaining_codes": 0}

        backup_codes = self.db.query(MFABackupCode).filter_by(user_id=user.id).all()
        used_codes = [code for code in backup_codes if code.used]

        return {
            "mfa_enabled": True,
            "total_codes": len(backup_codes),
            "used_codes": len(used_codes),
            "remaining_codes": len(backup_codes) - len(used_codes),
        }

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
