"""
Authentication API endpoints for the AI Trading System.

This module provides FastAPI endpoints for user authentication,
registration, password management, and MFA operations.
"""

import logging
from collections.abc import Generator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr
from sqlalchemy.orm import Session

from .jwt_service import JWTService
from .middleware import RequireRole
from .rbac_service import RBACService
from .services.registration import RegistrationResult
from .services.user_service import UserService
from .types import AuthenticationResult

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response models
class UserRegistrationRequest(BaseModel):
    """User registration request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=30, pattern="^[a-zA-Z0-9_-]+$")
    password: SecretStr = Field(..., min_length=12, max_length=128)
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)


class UserRegistrationResponse(BaseModel):
    """User registration response."""

    user_id: str
    email: str
    username: str
    message: str
    email_verification_required: bool


class LoginRequest(BaseModel):
    """Login request."""

    username: str  # Can be email or username
    password: SecretStr
    device_id: str | None = None


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user_id: str
    roles: list[str]
    permissions: list[str]
    mfa_required: bool = False
    mfa_session_token: str | None = None


class MFAVerificationRequest(BaseModel):
    """MFA verification request."""

    mfa_session_token: str
    mfa_code: str = Field(..., min_length=6, max_length=8)
    device_id: str | None = None


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation."""

    reset_token: str
    new_password: SecretStr = Field(..., min_length=12, max_length=128)


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: SecretStr
    new_password: SecretStr = Field(..., min_length=12, max_length=128)


class CreateAPIKeyRequest(BaseModel):
    """API key creation request."""

    name: str = Field(..., min_length=1, max_length=100)
    permissions: list[str] | None = None
    rate_limit: int = Field(default=1000, ge=1, le=10000)
    expires_in_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response."""

    id: str
    api_key: str | None = None  # Only shown on creation
    name: str
    last_four: str
    permissions: list[str]
    rate_limit: int
    expires_at: str | None
    created_at: str


class MFASetupResponse(BaseModel):
    """MFA setup response."""

    secret: str
    qr_code_uri: str
    backup_codes: list[str]


class UserProfileResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    username: str
    first_name: str | None
    last_name: str | None
    roles: list[str]
    permissions: list[str]
    email_verified: bool
    mfa_enabled: bool
    created_at: str
    last_login_at: str | None


# Dependency injection functions


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    # TODO: This should be injected from application layer
    # For now, create a session factory based on environment
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Get database URL from environment or use SQLite for development
    database_url = os.getenv("DATABASE_URL", "sqlite:///./auth.db")

    # Create engine and session factory
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Ensure tables exist
    from .models import Base

    Base.metadata.create_all(bind=engine)

    # Create and yield session
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_jwt_service() -> JWTService:
    """Get JWT service instance."""
    return JWTService()


def get_user_service(
    db: Session = Depends(get_db), jwt_service: JWTService = Depends(get_jwt_service)
) -> UserService:
    """Get user service instance."""
    return UserService(db_session=db, jwt_service=jwt_service)


def get_rbac_service(db: Session = Depends(get_db)) -> RBACService:
    """Get RBAC service instance."""
    return RBACService(db_session=db)


def get_current_user(request: Request) -> str:
    """Get current authenticated user ID."""
    return str(request.state.user_id)


# Public endpoints (no authentication required)
@router.post(
    "/register", response_model=UserRegistrationResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    request: UserRegistrationRequest,
    user_service: UserService = Depends(get_user_service),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> UserRegistrationResponse:
    """
    Register a new user account.

    Requires:
    - Valid email address
    - Unique username (3-30 characters, alphanumeric with _ or -)
    - Strong password (12+ characters with complexity requirements)
    """
    try:
        result: RegistrationResult = await user_service.register_user(
            email=request.email,
            username=request.username,
            password=request.password.get_secret_value(),
            first_name=request.first_name,
            last_name=request.last_name,
        )

        # Queue email verification in background if required
        if result.email_verification_required and result.verification_token:
            # This would send verification email
            # background_tasks.add_task(send_verification_email, request.email, result.verification_token)
            pass

        return UserRegistrationResponse(
            user_id=result.user_id,
            email=result.email,
            username=result.username,
            message="Registration successful. Please check your email for verification.",
            email_verification_required=result.email_verification_required,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed"
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service),
) -> LoginResponse:
    """
    Authenticate user and create session.

    Accepts username (or email) and password.
    Returns JWT tokens if successful, or MFA session if MFA is enabled.
    """
    try:
        # Get client info
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        result: AuthenticationResult = await user_service.authenticate(
            email_or_username=form_data.username,
            password=form_data.password,
            device_id=form_data.client_id,  # OAuth2 client_id can be used as device_id
            ip_address=client_ip,
            user_agent=user_agent,
        )

        return LoginResponse(
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=result.expires_in,
            user_id=result.user_id,
            roles=result.roles,
            permissions=result.permissions,
            mfa_required=result.mfa_required,
            mfa_session_token=result.mfa_session_token,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication failed"
        )


@router.post("/mfa/verify", response_model=LoginResponse)
async def verify_mfa(
    request: Request,
    mfa_request: MFAVerificationRequest,
    user_service: UserService = Depends(get_user_service),
) -> LoginResponse:
    """
    Verify MFA code and complete authentication.

    Requires MFA session token from initial login and 6-digit TOTP code.
    """
    try:
        # Get client info
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        result = await user_service.verify_mfa(
            mfa_session_token=mfa_request.mfa_session_token,
            mfa_code=mfa_request.mfa_code,
            device_id=mfa_request.device_id,
            ip_address=client_ip,
            user_agent=user_agent,
        )

        return LoginResponse(
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=result.expires_in,
            user_id=result.user_id,
            roles=result.roles,
            permissions=result.permissions,
            mfa_required=False,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"MFA verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA verification failed"
        )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    jwt_service: JWTService = Depends(get_jwt_service),
    user_service: UserService = Depends(get_user_service),
    db: Session = Depends(get_db),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> LoginResponse:
    """
    Refresh access token using refresh token.

    Implements token rotation for enhanced security.
    """
    try:
        # First verify the refresh token to get user_id
        payload = jwt_service.verify_refresh_token(refresh_request.refresh_token)
        user_id_str = payload["sub"]

        # Import UUID and User model
        from uuid import UUID

        from .models import User

        # Log for debugging
        logger.info(f"Refreshing token for user_id: {user_id_str}")

        # Convert string user_id to UUID for database query
        try:
            user_id_uuid = UUID(user_id_str)
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to convert user_id to UUID: {user_id_str}, error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user ID format"
            )

        # Fetch user details from database
        user = db.query(User).filter(User.id == user_id_uuid).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        # Get roles and permissions
        user_roles = await rbac_service.get_user_roles(str(user.id))
        user_permissions = await rbac_service.get_user_permissions(str(user.id))

        # Prepare user details
        user_details = {
            "email": user.email,
            "username": user.username,
            "roles": user_roles,
            "permissions": user_permissions,
            "mfa_verified": user.mfa_enabled,
        }

        # Rotate refresh token with user details
        new_access_token, new_refresh_token = jwt_service.rotate_refresh_token(
            refresh_request.refresh_token, user_details=user_details
        )

        # Decode new access token to get user info
        payload = jwt_service.verify_access_token(new_access_token)

        return LoginResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=900,  # 15 minutes
            user_id=payload["sub"],
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
        )

    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )


@router.post("/password/reset", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset(
    reset_request: PasswordResetRequest,
    user_service: UserService = Depends(get_user_service),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict[str, str]:
    """
    Request password reset token.

    Sends reset instructions to the registered email if it exists.
    """
    # Always return success to prevent email enumeration
    await user_service.request_password_reset(reset_request.email)

    # Queue email sending in background
    # background_tasks.add_task(send_password_reset_email, reset_request.email)

    return {"message": "If the email exists, reset instructions have been sent"}


@router.post("/password/reset/confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(
    confirm_request: PasswordResetConfirmRequest,
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Reset password using reset token.

    Requires valid reset token and new password meeting complexity requirements.
    """
    try:
        success = await user_service.reset_password(
            reset_token=confirm_request.reset_token,
            new_password=confirm_request.new_password.get_secret_value(),
        )

        if success:
            return {"message": "Password has been reset successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Password reset failed"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password reset failed"
        )


@router.get("/email/verify/{token}", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str,
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Verify email address using verification token.
    """
    try:
        success = await user_service.verify_email(token)
        if success:
            return {"message": "Email verified successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email verification failed"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Email verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Email verification failed"
        )


# Protected endpoints (authentication required)
@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: Request,
    everywhere: bool = False,
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Logout current session or all sessions.

    Set everywhere=true to logout from all devices.
    """
    session_id = request.state.session_id
    await user_service.logout(user_id=current_user, session_id=session_id, everywhere=everywhere)

    return {"message": "Logged out successfully"}


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> UserProfileResponse:
    """
    Get current user profile.
    """
    from .models import User

    db = user_service.db
    user = db.query(User).filter_by(id=current_user).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    roles = await rbac_service.get_user_roles(current_user)
    permissions = await rbac_service.get_user_permissions(current_user)

    return UserProfileResponse(
        id=str(user.id),
        email=str(user.email),
        username=str(user.username),
        first_name=str(user.first_name) if user.first_name else None,
        last_name=str(user.last_name) if user.last_name else None,
        roles=roles,
        permissions=permissions,
        email_verified=bool(user.email_verified),
        mfa_enabled=bool(user.mfa_enabled),
        created_at=user.created_at.isoformat(),
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


@router.post("/password/change", status_code=status.HTTP_200_OK)
async def change_password(
    change_request: ChangePasswordRequest,
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Change current user password.

    Requires current password for verification.
    """
    try:
        success = await user_service.change_password(
            user_id=current_user,
            current_password=change_request.current_password.get_secret_value(),
            new_password=change_request.new_password.get_secret_value(),
        )

        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Password change failed"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password change failed"
        )


# MFA endpoints
@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> MFASetupResponse:
    """
    Initialize MFA setup for current user.

    Returns TOTP secret, QR code URI, and backup codes.
    """
    try:
        result = await user_service.setup_mfa(current_user)

        return MFASetupResponse(
            secret=result["secret"],
            qr_code_uri=result["qr_code_uri"],
            backup_codes=result["backup_codes"],
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"MFA setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA setup failed"
        )


@router.post("/mfa/confirm", status_code=status.HTTP_200_OK)
async def confirm_mfa_setup(
    verification_code: str,
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Confirm MFA setup by verifying a code from authenticator app.
    """
    try:
        success = await user_service.confirm_mfa_setup(current_user, verification_code)

        if success:
            return {"message": "MFA has been enabled successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification code"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"MFA confirmation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA confirmation failed"
        )


@router.post("/mfa/disable", status_code=status.HTTP_200_OK)
async def disable_mfa(
    password: SecretStr,
    current_user: str = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> dict[str, str]:
    """
    Disable MFA for current user.

    Requires password confirmation for security.
    """
    try:
        success = await user_service.disable_mfa(current_user, password.get_secret_value())

        if success:
            return {"message": "MFA has been disabled"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to disable MFA"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"MFA disable failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to disable MFA"
        )


# API Key endpoints
@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_request: CreateAPIKeyRequest,
    current_user: str = Depends(get_current_user),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> APIKeyResponse:
    """
    Create a new API key for programmatic access.

    Returns the API key only once - store it securely!
    """
    try:
        result = await rbac_service.create_api_key(
            user_id=current_user,
            name=key_request.name,
            permissions=key_request.permissions,
            rate_limit=key_request.rate_limit,
            expires_in_days=key_request.expires_in_days,
        )

        return APIKeyResponse(
            id=result["id"],
            api_key=result["api_key"],  # Only shown once!
            name=result["name"],
            last_four=result["last_four"],
            permissions=result["permissions"],
            rate_limit=result["rate_limit"],
            expires_at=result["expires_at"],
            created_at=result["created_at"],
        )

    except Exception as e:
        logger.error(f"API key creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key creation failed"
        )


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: str = Depends(get_current_user),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> list[APIKeyResponse]:
    """
    List all API keys for current user.
    """
    keys = await rbac_service.list_user_api_keys(current_user)

    return [
        APIKeyResponse(
            id=key["id"],
            api_key=None,  # Never show key after creation
            name=key["name"],
            last_four=key["last_four"],
            permissions=key["permissions"],
            rate_limit=key["rate_limit"],
            expires_at=key["expires_at"],
            created_at=key["created_at"],
        )
        for key in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    reason: str | None = None,
    current_user: str = Depends(get_current_user),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> None:
    """
    Revoke an API key.
    """
    try:
        await rbac_service.revoke_api_key(user_id=current_user, key_id=key_id, reason=reason)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"API key revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key revocation failed"
        )


# Admin endpoints (require admin role)
@router.post("/admin/roles/{user_id}/assign", dependencies=[Depends(RequireRole("admin"))])
async def assign_role_to_user(
    user_id: str,
    role_name: str,
    current_user: str = Depends(get_current_user),
    rbac_service: RBACService = Depends(get_rbac_service),
) -> dict[str, str]:
    """
    Assign a role to a user (admin only).
    """
    try:
        success = await rbac_service.assign_role(
            user_id=user_id, role_name=role_name, granted_by=current_user
        )

        if success:
            return {"message": f"Role {role_name} assigned to user {user_id}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Role assignment failed"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Role assignment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Role assignment failed"
        )


@router.post("/admin/roles/{user_id}/revoke", dependencies=[Depends(RequireRole("admin"))])
async def revoke_role_from_user(
    user_id: str,
    role_name: str,
    rbac_service: RBACService = Depends(get_rbac_service),
) -> dict[str, str]:
    """
    Revoke a role from a user (admin only).
    """
    try:
        success = await rbac_service.revoke_role(user_id=user_id, role_name=role_name)

        if success:
            return {"message": f"Role {role_name} revoked from user {user_id}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Role revocation failed"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Role revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Role revocation failed"
        )


# Initialize default roles and permissions on startup
async def initialize_auth_system(rbac_service: RBACService) -> None:
    """Initialize default roles and permissions."""
    try:
        await rbac_service.initialize_default_roles_and_permissions()
        logger.info("Authentication system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize auth system: {e}")
        raise
