"""
Shared authentication types.

Common types used across the authentication system.
"""

from dataclasses import dataclass


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
