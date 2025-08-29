"""
MFA enforcement system for critical trading operations.

Provides mandatory multi-factor authentication for high-risk operations
with bypass protection, session management, and audit logging.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from .models import AuthAuditLog, User
from .services.mfa_service import MFAService

logger = logging.getLogger(__name__)


class MFARequiredOperation(Enum):
    """Operations that require MFA verification."""

    # Trading operations
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_ORDER = "modify_order"
    BULK_ORDER_ACTION = "bulk_order_action"
    POSITION_CLOSE = "position_close"
    PORTFOLIO_TRANSFER = "portfolio_transfer"

    # Risk management
    RISK_LIMIT_CHANGE = "risk_limit_change"
    STOP_LOSS_DISABLE = "stop_loss_disable"

    # Account management
    ACCOUNT_SETTINGS_CHANGE = "account_settings_change"
    API_KEY_GENERATION = "api_key_generation"
    WITHDRAWAL_REQUEST = "withdrawal_request"

    # System administration
    SYSTEM_CONFIGURATION = "system_configuration"
    USER_ROLE_CHANGE = "user_role_change"
    EMERGENCY_SYSTEM_STOP = "emergency_system_stop"


class MFAVerificationMethod(Enum):
    """MFA verification methods."""

    TOTP = "totp"  # Time-based OTP
    SMS = "sms"  # SMS verification
    BACKUP_CODE = "backup_code"  # Backup recovery codes
    HARDWARE_TOKEN = "hardware_token"  # Hardware security key


@dataclass
class MFASession:
    """MFA verification session."""

    session_id: str
    user_id: str
    operation: MFARequiredOperation
    verified_at: datetime
    expires_at: datetime
    verification_method: MFAVerificationMethod
    ip_address: str | None = None
    user_agent: str | None = None
    operation_data: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if MFA session is still valid."""
        return datetime.utcnow() < self.expires_at

    def is_expired(self) -> bool:
        """Check if MFA session has expired."""
        return datetime.utcnow() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "operation": self.operation.value,
            "verified_at": self.verified_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "verification_method": self.verification_method.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "operation_data": self.operation_data,
        }


class MFABypassAttemptError(Exception):
    """Raised when MFA bypass is attempted without authorization."""

    pass


class MFAEnforcementService:
    """
    MFA enforcement service for critical operations.

    Features:
    - Operation-specific MFA requirements
    - Session-based MFA verification
    - Bypass protection and monitoring
    - Audit logging for all MFA events
    - Configurable session timeouts
    - Risk-based MFA requirements
    """

    def __init__(
        self,
        db_session: Session,
        mfa_service: MFAService,
        default_session_timeout_minutes: int = 15,
        high_risk_timeout_minutes: int = 5,
        bypass_protection_enabled: bool = True,
        audit_all_attempts: bool = True,
    ):
        self.db = db_session
        self.mfa_service = mfa_service
        self.default_session_timeout_minutes = default_session_timeout_minutes
        self.high_risk_timeout_minutes = high_risk_timeout_minutes
        self.bypass_protection_enabled = bypass_protection_enabled
        self.audit_all_attempts = audit_all_attempts

        # Active MFA sessions
        self.active_sessions: dict[str, MFASession] = {}

        # Operation risk levels
        self.operation_risk_levels = {
            # High risk - short timeout
            MFARequiredOperation.PLACE_ORDER: "high",
            MFARequiredOperation.BULK_ORDER_ACTION: "high",
            MFARequiredOperation.WITHDRAWAL_REQUEST: "high",
            MFARequiredOperation.EMERGENCY_SYSTEM_STOP: "high",
            MFARequiredOperation.USER_ROLE_CHANGE: "high",
            # Medium risk - default timeout
            MFARequiredOperation.CANCEL_ORDER: "medium",
            MFARequiredOperation.MODIFY_ORDER: "medium",
            MFARequiredOperation.POSITION_CLOSE: "medium",
            MFARequiredOperation.RISK_LIMIT_CHANGE: "medium",
            MFARequiredOperation.STOP_LOSS_DISABLE: "medium",
            # Low risk - longer timeout
            MFARequiredOperation.API_KEY_GENERATION: "low",
            MFARequiredOperation.ACCOUNT_SETTINGS_CHANGE: "low",
            MFARequiredOperation.SYSTEM_CONFIGURATION: "low",
        }

        # Bypass attempts tracking
        self.bypass_attempts: dict[str, list[datetime]] = {}

        logger.info("MFA Enforcement Service initialized")

    def require_mfa(
        self,
        operation: MFARequiredOperation,
        user_id: str,
        request: Request,
        operation_data: dict[str, Any] | None = None,
        custom_timeout_minutes: int | None = None,
    ) -> MFASession:
        """
        Require MFA verification for an operation.

        Returns:
            MFASession if verification is valid

        Raises:
            HTTPException: If MFA verification is required or invalid
        """
        # Check if user has MFA enabled
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        if not user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="MFA must be enabled for this operation",
            )

        # Check for existing valid MFA session
        existing_session = self._get_valid_mfa_session(user_id, operation)
        if existing_session:
            # Update operation data if provided
            if operation_data:
                existing_session.operation_data.update(operation_data)

            self._audit_mfa_event(
                "mfa_session_reused",
                user_id,
                request,
                {
                    "operation": operation.value,
                    "session_id": existing_session.session_id,
                },
            )

            return existing_session

        # Check for MFA verification in request
        mfa_code = self._extract_mfa_code(request)
        if not mfa_code:
            self._audit_mfa_event(
                "mfa_required",
                user_id,
                request,
                {
                    "operation": operation.value,
                    "reason": "no_mfa_code_provided",
                },
            )

            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="MFA verification required for this operation",
                headers={"X-MFA-Required": "true", "X-MFA-Operation": operation.value},
            )

        # Verify MFA code
        if not self.mfa_service.verify_mfa_code(user_id, mfa_code):
            self._audit_mfa_event(
                "mfa_verification_failed",
                user_id,
                request,
                {
                    "operation": operation.value,
                    "mfa_code_length": len(mfa_code),
                },
            )

            # Check for bypass attempts
            if self.bypass_protection_enabled:
                self._check_bypass_attempts(user_id, request)

            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA code")

        # Create MFA session
        session = self._create_mfa_session(
            user_id=user_id,
            operation=operation,
            request=request,
            operation_data=operation_data or {},
            verification_method=self._determine_verification_method(mfa_code),
            custom_timeout_minutes=custom_timeout_minutes,
        )

        self._audit_mfa_event(
            "mfa_verification_success",
            user_id,
            request,
            {
                "operation": operation.value,
                "session_id": session.session_id,
                "verification_method": session.verification_method.value,
            },
        )

        logger.info(f"MFA verification successful: user={user_id}, operation={operation.value}")

        return session

    def verify_mfa_session(
        self,
        user_id: str,
        operation: MFARequiredOperation,
        session_id: str | None = None,
    ) -> bool:
        """
        Verify if user has valid MFA session for operation.

        Returns:
            True if valid session exists
        """
        session = self._get_valid_mfa_session(user_id, operation, session_id)
        return session is not None

    def invalidate_mfa_session(
        self,
        user_id: str,
        session_id: str | None = None,
        operation: MFARequiredOperation | None = None,
    ) -> int:
        """
        Invalidate MFA sessions.

        Returns:
            Number of sessions invalidated
        """
        invalidated = 0
        sessions_to_remove = []

        for sid, session in self.active_sessions.items():
            should_invalidate = False

            # Match user
            if session.user_id != user_id:
                continue

            # Match session ID if provided
            if session_id and session.session_id != session_id:
                continue

            # Match operation if provided
            if operation and session.operation != operation:
                continue

            sessions_to_remove.append(sid)
            invalidated += 1

        # Remove sessions
        for sid in sessions_to_remove:
            del self.active_sessions[sid]

        if invalidated > 0:
            logger.info(f"Invalidated {invalidated} MFA sessions for user {user_id}")

        return invalidated

    def get_mfa_status(self, user_id: str) -> dict[str, Any]:
        """Get MFA status for user."""
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            return {"error": "User not found"}

        # Get active sessions
        user_sessions = []
        for session in self.active_sessions.values():
            if session.user_id == user_id and session.is_valid():
                user_sessions.append(
                    {
                        "session_id": session.session_id,
                        "operation": session.operation.value,
                        "expires_at": session.expires_at.isoformat(),
                        "verification_method": session.verification_method.value,
                    }
                )

        return {
            "mfa_enabled": user.mfa_enabled,
            "active_sessions": user_sessions,
            "bypass_attempts_recent": len(self.bypass_attempts.get(user_id, [])),
        }

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired MFA sessions."""
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)

        # Remove expired sessions
        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired MFA sessions")

        return len(expired_sessions)

    def _get_valid_mfa_session(
        self,
        user_id: str,
        operation: MFARequiredOperation,
        session_id: str | None = None,
    ) -> MFASession | None:
        """Get valid MFA session for user and operation."""
        for session in self.active_sessions.values():
            if session.user_id == user_id and session.operation == operation and session.is_valid():
                if session_id and session.session_id != session_id:
                    continue

                return session

        return None

    def _create_mfa_session(
        self,
        user_id: str,
        operation: MFARequiredOperation,
        request: Request,
        operation_data: dict[str, Any],
        verification_method: MFAVerificationMethod,
        custom_timeout_minutes: int | None = None,
    ) -> MFASession:
        """Create new MFA session."""
        # Determine timeout based on operation risk
        if custom_timeout_minutes:
            timeout_minutes = custom_timeout_minutes
        else:
            risk_level = self.operation_risk_levels.get(operation, "medium")
            if risk_level == "high":
                timeout_minutes = self.high_risk_timeout_minutes
            else:
                timeout_minutes = self.default_session_timeout_minutes

        # Generate session ID
        import secrets

        session_id = f"mfa_{user_id}_{int(time.time())}_{secrets.token_hex(8)}"

        # Create session
        session = MFASession(
            session_id=session_id,
            user_id=user_id,
            operation=operation,
            verified_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=timeout_minutes),
            verification_method=verification_method,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            operation_data=operation_data,
        )

        # Store session
        self.active_sessions[session_id] = session

        return session

    def _extract_mfa_code(self, request: Request) -> str | None:
        """Extract MFA code from request headers or body."""
        # Try header first
        mfa_code = request.headers.get("X-MFA-Code")
        if mfa_code:
            return mfa_code

        # Try query parameter
        mfa_code = request.query_params.get("mfa_code")
        if mfa_code:
            return mfa_code

        # Try to extract from JSON body (for POST requests)
        if hasattr(request, "_json"):
            try:
                body = getattr(request, "_json", {})
                return body.get("mfa_code")
            except Exception:
                pass

        return None

    def _determine_verification_method(self, mfa_code: str) -> MFAVerificationMethod:
        """Determine verification method based on code format."""
        if len(mfa_code) == 6 and mfa_code.isdigit():
            return MFAVerificationMethod.TOTP
        elif len(mfa_code) == 8 and all(c in "0123456789ABCDEF" for c in mfa_code.upper()):
            return MFAVerificationMethod.BACKUP_CODE
        else:
            # Default to TOTP for unknown formats
            return MFAVerificationMethod.TOTP

    def _check_bypass_attempts(self, user_id: str, request: Request) -> None:
        """Check for MFA bypass attempts and take action."""
        now = datetime.utcnow()

        # Track bypass attempt
        if user_id not in self.bypass_attempts:
            self.bypass_attempts[user_id] = []

        self.bypass_attempts[user_id].append(now)

        # Clean old attempts (last hour)
        cutoff_time = now - timedelta(hours=1)
        self.bypass_attempts[user_id] = [
            attempt for attempt in self.bypass_attempts[user_id] if attempt > cutoff_time
        ]

        # Check for excessive attempts
        recent_attempts = len(self.bypass_attempts[user_id])

        if recent_attempts >= 5:  # 5+ failed attempts in an hour
            self._audit_mfa_event(
                "mfa_bypass_attempt_detected",
                user_id,
                request,
                {
                    "attempt_count": recent_attempts,
                    "severity": "high",
                },
            )

            logger.warning(
                f"Excessive MFA bypass attempts detected: user={user_id}, attempts={recent_attempts}"
            )

            # Temporarily lock user account or increase security measures
            # This would integrate with your user management system

            raise MFABypassAttemptError("Excessive MFA verification failures detected")

    def _audit_mfa_event(
        self,
        event_type: str,
        user_id: str,
        request: Request,
        event_data: dict[str, Any],
    ) -> None:
        """Audit MFA events."""
        if not self.audit_all_attempts and event_type not in [
            "mfa_verification_failed",
            "mfa_bypass_attempt_detected",
        ]:
            return

        audit_log = AuthAuditLog(
            user_id=user_id,
            event_type=event_type,
            event_data=event_data,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=event_type not in ["mfa_verification_failed", "mfa_bypass_attempt_detected"],
        )

        self.db.add(audit_log)
        self.db.commit()


def require_mfa_for_operation(
    operation: MFARequiredOperation,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to require MFA for specific operations.

    Usage:
        @require_mfa_for_operation(MFARequiredOperation.PLACE_ORDER)
        async def place_order(request: Request, ...):
            # Operation implementation
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract request and get MFA enforcement service
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found for MFA enforcement",
                )

            # Get user ID from request state
            user_id = getattr(request.state, "user_id", None)
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
                )

            # Get MFA enforcement service (this would be injected via dependency injection)
            # For now, we'll assume it's available in the request state
            mfa_service = getattr(request.state, "mfa_enforcement", None)
            if not mfa_service:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="MFA enforcement service not available",
                )

            # Require MFA verification
            try:
                mfa_session = mfa_service.require_mfa(
                    operation=operation,
                    user_id=user_id,
                    request=request,
                )

                # Store MFA session in request state for operation use
                request.state.mfa_session = mfa_session

            except HTTPException:
                # Re-raise HTTP exceptions (MFA required, invalid code, etc.)
                raise
            except Exception as e:
                logger.error(f"MFA enforcement error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="MFA enforcement failed",
                )

            # Execute the original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators for common operations
def require_mfa_for_trading() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Require MFA for trading operations."""
    return require_mfa_for_operation(MFARequiredOperation.PLACE_ORDER)


def require_mfa_for_risk_management() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Require MFA for risk management operations."""
    return require_mfa_for_operation(MFARequiredOperation.RISK_LIMIT_CHANGE)


def require_mfa_for_account_changes() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Require MFA for account setting changes."""
    return require_mfa_for_operation(MFARequiredOperation.ACCOUNT_SETTINGS_CHANGE)
