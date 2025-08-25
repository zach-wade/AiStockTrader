"""
Session management service.

Handles user session creation, management, token generation,
and session lifecycle operations.
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import bcrypt
from sqlalchemy.orm import Session

from ..jwt_service import JWTService
from ..models import AuthAuditLog, User, UserSession

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


class SessionManager:
    """Session management service."""

    def __init__(self, db_session: Session, jwt_service: JWTService):
        self.db = db_session
        self.jwt_service = jwt_service

    async def create_user_session(
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

    async def refresh_token(
        self, refresh_token: str, device_id: str | None = None
    ) -> dict[str, Any]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token
            device_id: Device identifier

        Returns:
            New access token and refresh token
        """
        # Decode refresh token
        payload = self.jwt_service.verify_refresh_token(refresh_token)

        user_id = payload.get("user_id")
        session_id = payload.get("session_id")

        # Find active session
        session = (
            self.db.query(UserSession)
            .filter_by(id=session_id, user_id=user_id, is_active=True)
            .first()
        )

        if not session or session.refresh_expires_at < datetime.utcnow():
            raise ValueError("Invalid or expired refresh token")

        # Verify refresh token hash
        if not bcrypt.checkpw(
            refresh_token.encode("utf-8"), session.refresh_token_hash.encode("utf-8")
        ):
            raise ValueError("Invalid refresh token")

        # Get user
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            raise ValueError("User not found")

        # Generate new tokens
        roles = [role.name for role in user.roles]
        permissions = user.get_permissions()

        new_access_token = self.jwt_service.create_access_token(
            user_id=str(user.id),
            email=str(user.email),
            username=str(user.username),
            roles=roles,
            permissions=permissions,
            session_id=str(session.id),
            device_id=device_id,
            ip_address=session.ip_address,
            mfa_verified=bool(user.mfa_enabled),
        )

        new_refresh_token = self.jwt_service.create_refresh_token(
            user_id=str(user.id), session_id=str(session.id), device_id=device_id
        )

        # Update session with new refresh token hash
        session.refresh_token_hash = bcrypt.hashpw(
            new_refresh_token.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # Extend session expiry
        session.expires_at = datetime.utcnow() + timedelta(hours=24)
        session.refresh_expires_at = datetime.utcnow() + timedelta(days=7)

        self.db.commit()

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "expires_in": 900,  # 15 minutes
        }

    async def get_active_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get all active sessions for a user."""
        sessions = self.db.query(UserSession).filter_by(user_id=user_id, is_active=True).all()

        return [
            {
                "session_id": str(session.id),
                "device_id": session.device_id,
                "user_agent": session.user_agent,
                "ip_address": session.ip_address,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "last_activity": (
                    session.last_activity_at.isoformat() if session.last_activity_at else None
                ),
            }
            for session in sessions
        ]

    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """Revoke a specific session."""
        session = (
            self.db.query(UserSession)
            .filter_by(id=session_id, user_id=user_id, is_active=True)
            .first()
        )

        if not session:
            return False

        session.revoke("Session revoked by user")
        self.jwt_service.revoke_session_tokens(session_id)
        self.db.commit()

        self._log_audit_event(
            event_type="session_revoked",
            user_id=user_id,
            event_data={"session_id": session_id},
            success=True,
        )

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
