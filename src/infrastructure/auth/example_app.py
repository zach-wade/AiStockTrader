"""
Example FastAPI application with JWT authentication.

This module demonstrates how to integrate the authentication system
into a FastAPI application.
"""

import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any

import redis
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .jwt_service import JWTService
from .middleware import (
    RequestIDMiddleware,
    RequirePermission,
    RequireRole,
    SecurityHeadersMiddleware,
    get_current_user,
    get_current_user_permissions,
)
from .models import Base
from .rbac_service import RBACService
from .user_service import UserService

# Pydantic models for request/response


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    username: str
    password: str
    first_name: str | None = None
    last_name: str | None = None


class LoginRequest(BaseModel):
    """User login request."""

    email_or_username: str
    password: str
    device_id: str | None = None
    remember_me: bool = False


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""

    mfa_session_token: str
    code: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""

    token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str
    new_password: str


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""

    name: str
    permissions: list[str] | None = None
    expires_in_days: int | None = None


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/trading_platform")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis setup
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# Initialize services
jwt_service = JWTService(redis_client=redis_client)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print("Starting authentication service...")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Initialize default roles and permissions
    db = SessionLocal()
    rbac_service = RBACService(db)
    await rbac_service.initialize_default_roles_and_permissions()
    await db.close()

    yield

    # Shutdown
    print("Shutting down authentication service...")
    await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="Trading Platform Authentication API",
    description="JWT-based authentication system for AI trading platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    max_age=3600,
)


# Authentication endpoints


@app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    user_service = UserService(db, jwt_service)

    try:
        result = await user_service.register_user(
            email=request.email,
            username=request.username,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
        )

        return {
            "id": result.user_id,
            "email": result.email,
            "username": result.username,
            "email_verified": not result.email_verification_required,
            "message": (
                "Registration successful. Please verify your email."
                if result.email_verification_required
                else "Registration successful."
            ),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login user and get tokens."""
    user_service = UserService(db, jwt_service)

    try:
        result = await user_service.authenticate(
            email_or_username=request.email_or_username,
            password=request.password,
            device_id=request.device_id,
        )

        if result.mfa_required:
            return {"mfa_required": True, "mfa_session_token": result.mfa_session_token}

        return {
            "access_token": result.access_token,
            "refresh_token": result.refresh_token,
            "token_type": "Bearer",
            "expires_in": result.expires_in,
            "user": {
                "id": result.user_id,
                "roles": result.roles,
                "permissions": result.permissions,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@app.post("/api/v1/auth/mfa/verify")
async def verify_mfa(request: MFAVerifyRequest, db: Session = Depends(get_db)):
    """Verify MFA code."""
    user_service = UserService(db, jwt_service)

    try:
        result = await user_service.verify_mfa(
            mfa_session_token=request.mfa_session_token, mfa_code=request.code
        )

        return {
            "access_token": result.access_token,
            "refresh_token": result.refresh_token,
            "token_type": "Bearer",
            "expires_in": result.expires_in,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@app.post("/api/v1/auth/refresh")
async def refresh_token(request: TokenRefreshRequest) -> Any:
    """Refresh access token."""
    try:
        access_token, refresh_token = jwt_service.rotate_refresh_token(request.refresh_token)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 900,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    everywhere: bool = False,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Logout user."""
    user_service = UserService(db, jwt_service)

    # Get session ID from request context
    # This would come from the JWT token
    session_id = "current_session_id"  # Placeholder

    await user_service.logout(user_id=current_user, session_id=session_id, everywhere=everywhere)


@app.post("/api/v1/auth/password/reset-request")
async def request_password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    """Request password reset."""
    user_service = UserService(db, jwt_service)

    # Always return success to prevent user enumeration
    await user_service.request_password_reset(request.email)

    return {"message": "Password reset email sent if account exists"}


@app.post("/api/v1/auth/password/reset-confirm")
async def reset_password(request: PasswordResetConfirm, db: Session = Depends(get_db)):
    """Reset password with token."""
    user_service = UserService(db, jwt_service)

    try:
        await user_service.reset_password(
            reset_token=request.token, new_password=request.new_password
        )

        return {"message": "Password successfully reset"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/auth/password/change")
async def change_password(
    request: ChangePasswordRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change user password."""
    user_service = UserService(db, jwt_service)

    try:
        await user_service.change_password(
            user_id=current_user,
            current_password=request.current_password,
            new_password=request.new_password,
        )

        return {"message": "Password successfully changed"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/auth/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify email address."""
    user_service = UserService(db, jwt_service)

    try:
        await user_service.verify_email(token)

        return {"message": "Email successfully verified"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# User management endpoints


@app.get("/api/v1/users/me")
async def get_current_user_info(
    current_user: str = Depends(get_current_user),
    permissions: list[str] = Depends(get_current_user_permissions),
    db: Session = Depends(get_db),
):
    """Get current user information."""
    from .models import User

    user = db.query(User).filter_by(id=current_user).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return {
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "roles": [role.name for role in user.roles],
        "permissions": permissions,
        "mfa_enabled": user.mfa_enabled,
        "email_verified": user.email_verified,
        "created_at": user.created_at.isoformat(),
    }


# API Key management endpoints


@app.post("/api/v1/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: str = Depends(get_current_user),
    _: bool = Depends(RequirePermission("api_keys", "create")),
    db: Session = Depends(get_db),
):
    """Create new API key."""
    rbac_service = RBACService(db)

    try:
        key_info = await rbac_service.create_api_key(
            user_id=current_user,
            name=request.name,
            permissions=request.permissions,
            expires_in_days=request.expires_in_days,
        )

        return key_info
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/api-keys")
async def list_api_keys(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List user's API keys."""
    rbac_service = RBACService(db)

    keys = await rbac_service.list_user_api_keys(current_user)

    return {"api_keys": keys}


@app.delete("/api/v1/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    reason: str | None = None,
    current_user: str = Depends(get_current_user),
    _: bool = Depends(RequirePermission("api_keys", "revoke")),
    db: Session = Depends(get_db),
):
    """Revoke API key."""
    rbac_service = RBACService(db)

    try:
        await rbac_service.revoke_api_key(user_id=current_user, key_id=key_id, reason=reason)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# Protected endpoints example


@app.get("/api/v1/trades")
async def get_trades(
    current_user: str = Depends(get_current_user),
    _: bool = Depends(RequirePermission("trades", "read")),
):
    """Get user trades (protected endpoint)."""
    return {"user_id": current_user, "trades": []}  # Placeholder


@app.post("/api/v1/trades")
async def execute_trade(
    trade_data: dict[str, Any],
    current_user: str = Depends(get_current_user),
    _: bool = Depends(RequirePermission("trades", "execute")),
):
    """Execute trade (protected endpoint)."""
    return {"user_id": current_user, "trade": trade_data, "status": "executed"}


@app.get("/api/v1/admin/users")
async def list_users(_: bool = Depends(RequireRole("admin")), db: Session = Depends(get_db)):
    """List all users (admin only)."""
    from .models import User

    users = db.query(User).all()

    return {
        "users": [
            {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "created_at": user.created_at.isoformat(),
            }
            for user in users
        ]
    }


# Health check
@app.get("/health")
async def health_check() -> Any:
    """Health check endpoint."""
    return {"status": "healthy", "service": "authentication", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
