"""
JWT token management service for authentication.

This module handles JWT token creation, validation, and revocation
for the authentication system.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

import jwt
import redis
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)


class TokenExpiredException(Exception):
    """Raised when a token has expired."""

    pass


class TokenRevokedException(Exception):
    """Raised when a token has been revoked."""

    pass


class InvalidTokenException(Exception):
    """Raised when a token is invalid."""

    pass


class TokenReuseException(Exception):
    """Raised when refresh token reuse is detected."""

    pass


class JWTService:
    """
    JWT token service for creating and validating tokens.

    Supports:
    - Access tokens (15 minutes default)
    - Refresh tokens (7 days default)
    - Token blacklisting
    - Token rotation
    - Key rotation support
    """

    def __init__(
        self,
        private_key_path: str | None = None,
        public_key_path: str | None = None,
        redis_client: redis.Redis | None = None,
        issuer: str = "https://api.tradingplatform.com",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7,
    ):
        """
        Initialize JWT service.

        Args:
            private_key_path: Path to RSA private key for signing
            public_key_path: Path to RSA public key for verification
            redis_client: Redis client for token blacklisting
            issuer: Token issuer identifier
            access_token_expire_minutes: Access token expiration in minutes
            refresh_token_expire_days: Refresh token expiration in days
        """
        self.issuer = issuer
        self.algorithm = "RS256"
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)

        # Load keys - require persistent keys for production security
        if private_key_path and os.path.exists(private_key_path):
            self.private_key = self._load_private_key(private_key_path)
        elif os.getenv("ENVIRONMENT", "development") == "production":
            raise InvalidTokenException(
                "JWT private key is required for production. "
                "Please generate RSA keys and set the private_key_path parameter. "
                "Use: openssl genrsa -out private_key.pem 2048"
            )
        else:
            # Only allow key generation in development/testing
            logger.warning(
                "No private key found - generating ephemeral keys for DEVELOPMENT ONLY. "
                "These keys will be lost on restart and all tokens will be invalidated!"
            )
            self.private_key, self.public_key = self._generate_key_pair()

        if public_key_path and os.path.exists(public_key_path):
            self.public_key = self._load_public_key(public_key_path)
        elif not hasattr(self, "public_key"):
            # If we didn't generate keys above, we need the public key
            if os.getenv("ENVIRONMENT", "development") == "production":
                raise InvalidTokenException(
                    "JWT public key is required for production. "
                    "Please generate RSA keys and set the public_key_path parameter. "
                    "Use: openssl rsa -in private_key.pem -pubout -out public_key.pem"
                )

        # Redis for token management
        self.redis = redis_client or self._create_redis_client()

        # Key rotation support
        self.key_id = f"{datetime.utcnow().year}-{datetime.utcnow().month:02d}-key-1"

    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with default settings."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url, decode_responses=True)
        return client  # type: ignore[no-any-return]

    def _load_private_key(self, path: str) -> Any:
        """Load RSA private key from file."""
        with open(path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(), password=None, backend=default_backend()
            )

    def _load_public_key(self, path: str) -> Any:
        """Load RSA public key from file."""
        with open(path, "rb") as key_file:
            return serialization.load_pem_public_key(key_file.read(), backend=default_backend())

    def _generate_key_pair(self) -> tuple[Any, Any]:
        """Generate new RSA key pair for development."""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def _get_private_key_pem(self) -> bytes:
        """Get private key in PEM format."""
        pem_bytes: bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pem_bytes

    def _get_public_key_pem(self) -> bytes:
        """Get public key in PEM format."""
        pem_bytes: bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem_bytes

    def create_access_token(
        self,
        user_id: str,
        email: str,
        username: str,
        roles: list[str],
        permissions: list[str],
        session_id: str,
        device_id: str | None = None,
        ip_address: str | None = None,
        mfa_verified: bool = False,
        custom_claims: dict[str, Any] | None = None,
    ) -> str:
        """
        Create JWT access token.

        Args:
            user_id: User identifier
            email: User email
            username: Username
            roles: List of user roles
            permissions: List of user permissions
            session_id: Session identifier
            device_id: Device identifier
            ip_address: Client IP address
            mfa_verified: Whether MFA was verified
            custom_claims: Additional claims to include

        Returns:
            Signed JWT access token
        """
        now = datetime.utcnow()
        jti = f"jwt_{secrets.token_urlsafe(16)}"

        payload = {
            # Standard claims
            "iss": self.issuer,
            "sub": user_id,
            "aud": [self.issuer],
            "exp": now + self.access_token_expire,
            "nbf": now,
            "iat": now,
            "jti": jti,
            # Session info
            "sid": session_id,
            # User info
            "email": email,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            # Security context
            "scope": "trading api",
            "mfa_verified": mfa_verified,
        }

        # Add optional claims
        if device_id:
            payload["device_id"] = device_id
        if ip_address:
            payload["ip"] = ip_address
        if custom_claims:
            payload.update(custom_claims)

        # Create token
        token = jwt.encode(
            payload,
            self._get_private_key_pem(),
            algorithm=self.algorithm,
            headers={"kid": self.key_id},
        )

        # Store JTI in Redis for revocation checking
        self.redis.setex(f"jwt:valid:{jti}", int(self.access_token_expire.total_seconds()), "1")

        # Track token for user (for bulk revocation)
        self.redis.sadd(f"user:tokens:{user_id}", jti)
        self.redis.expire(f"user:tokens:{user_id}", int(self.access_token_expire.total_seconds()))

        logger.info(f"Created access token for user {user_id} with JTI {jti}")
        return token

    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        device_id: str | None = None,
        token_family: str | None = None,
    ) -> str:
        """
        Create JWT refresh token with rotation support.

        Args:
            user_id: User identifier
            session_id: Session identifier
            device_id: Device identifier
            token_family: Token family for rotation tracking

        Returns:
            Signed JWT refresh token
        """
        now = datetime.utcnow()
        jti = f"refresh_{secrets.token_urlsafe(16)}"
        token_family = token_family or f"family_{secrets.token_urlsafe(8)}"

        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": [f"{self.issuer}/refresh"],
            "exp": now + self.refresh_token_expire,
            "iat": now,
            "jti": jti,
            "sid": session_id,
            "token_family": token_family,
            "type": "refresh",
        }

        if device_id:
            payload["device_id"] = device_id

        # Create token
        token = jwt.encode(
            payload,
            self._get_private_key_pem(),
            algorithm=self.algorithm,
            headers={"kid": self.key_id},
        )

        # Store refresh token family for rotation
        self.redis.setex(
            f"refresh:family:{token_family}", int(self.refresh_token_expire.total_seconds()), jti
        )

        # Track refresh token
        self.redis.setex(
            f"refresh:token:{jti}", int(self.refresh_token_expire.total_seconds()), user_id
        )

        logger.info(f"Created refresh token for user {user_id} with family {token_family}")
        return token

    def verify_access_token(self, token: str) -> dict[str, Any]:
        """
        Verify and decode access token.

        Args:
            token: JWT access token

        Returns:
            Decoded token payload

        Raises:
            TokenExpiredException: If token is expired
            TokenRevokedException: If token is revoked
            InvalidTokenException: If token is invalid
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                token,
                self._get_public_key_pem(),
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=[self.issuer],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            # Check if token is revoked
            jti = payload.get("jti")
            if not self.redis.get(f"jwt:valid:{jti}"):
                logger.warning(f"Token {jti} has been revoked")
                raise TokenRevokedException("Token has been revoked")

            # Check if token is blacklisted
            if self.redis.get(f"jwt:blacklist:{jti}"):
                logger.warning(f"Token {jti} is blacklisted")
                raise TokenRevokedException("Token is blacklisted")

            return dict(payload)

        except jwt.ExpiredSignatureError:
            raise TokenExpiredException("Access token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenException(f"Invalid access token: {e!s}")

    def verify_refresh_token(self, token: str) -> dict[str, Any]:
        """
        Verify refresh token with rotation detection.

        Args:
            token: JWT refresh token

        Returns:
            Decoded token payload

        Raises:
            TokenExpiredException: If token is expired
            TokenReuseException: If token reuse is detected
            InvalidTokenException: If token is invalid
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                token,
                self._get_public_key_pem(),
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=[f"{self.issuer}/refresh"],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            jti = payload.get("jti")
            token_family = payload.get("token_family")

            # Check if this is the current token in the family
            current_jti = self.redis.get(f"refresh:family:{token_family}")

            if not current_jti:
                # Family doesn't exist - possible attack
                logger.error(f"Token family {token_family} not found - possible reuse attack")
                raise TokenReuseException("Invalid refresh token family")

            if current_jti != jti:
                # Token reuse detected - revoke entire family
                logger.error(f"Refresh token reuse detected for family {token_family}")
                self.revoke_token_family(token_family)
                raise TokenReuseException("Refresh token reuse detected - all tokens revoked")

            # Check if token is explicitly revoked
            if self.redis.get(f"refresh:blacklist:{jti}"):
                raise TokenRevokedException("Refresh token is revoked")

            return dict(payload)

        except jwt.ExpiredSignatureError:
            raise TokenExpiredException("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenException(f"Invalid refresh token: {e!s}")

    def rotate_refresh_token(self, old_token: str) -> tuple[str, str]:
        """
        Rotate refresh token and generate new access token.

        Args:
            old_token: Current refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        # Verify old token
        payload = self.verify_refresh_token(old_token)

        user_id = payload["sub"]
        session_id = payload["sid"]
        device_id = payload.get("device_id")
        token_family = payload["token_family"]

        # Revoke old token
        old_jti = payload["jti"]
        self.redis.setex(f"refresh:blacklist:{old_jti}", 86400, "1")

        # Create new refresh token in same family
        new_refresh_token = self.create_refresh_token(
            user_id=user_id, session_id=session_id, device_id=device_id, token_family=token_family
        )

        # Get user details for access token (would come from database in production)
        # For now, we'll create a minimal access token
        new_access_token = self.create_access_token(
            user_id=user_id,
            email=payload.get("email", ""),
            username=payload.get("username", ""),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
            session_id=session_id,
            device_id=device_id,
            mfa_verified=payload.get("mfa_verified", False),
        )

        return new_access_token, new_refresh_token

    def revoke_token(self, jti: str) -> None:
        """
        Revoke a specific token by JTI.

        Args:
            jti: JWT ID to revoke
        """
        # Remove from valid tokens
        self.redis.delete(f"jwt:valid:{jti}")

        # Add to blacklist
        self.redis.setex(f"jwt:blacklist:{jti}", 86400, "1")

        logger.info(f"Revoked token {jti}")

    def revoke_token_family(self, token_family: str) -> None:
        """
        Revoke entire token family (for refresh token reuse detection).

        Args:
            token_family: Token family identifier
        """
        self.redis.delete(f"refresh:family:{token_family}")
        logger.warning(f"Revoked entire token family {token_family}")

    async def revoke_all_user_tokens(self, user_id: str) -> None:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User identifier
        """
        # Get all user tokens
        token_key = f"user:tokens:{user_id}"
        tokens_result = self.redis.smembers(token_key)
        if hasattr(tokens_result, "__await__"):
            tokens = await tokens_result or set()
        else:
            tokens = tokens_result or set()

        # Revoke each token
        for jti in tokens:
            self.revoke_token(jti)

        # Clear user token set
        self.redis.delete(token_key)

        logger.info(f"Revoked all tokens for user {user_id}")

    def revoke_session_tokens(self, session_id: str) -> None:
        """
        Revoke all tokens for a specific session.

        Args:
            session_id: Session identifier
        """
        # This would require tracking tokens by session
        # Implementation depends on specific requirements
        pattern = f"jwt:session:{session_id}:*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)

        logger.info(f"Revoked all tokens for session {session_id}")

    def cleanup_expired_tokens(self) -> Any:
        """Clean up expired token records from Redis."""
        # Redis handles expiration automatically with TTL
        # This method can be used for additional cleanup if needed
        pass

    def get_token_info(self, jti: str) -> dict[str, Any] | None:
        """
        Get information about a token.

        Args:
            jti: JWT ID

        Returns:
            Token information if found
        """
        if self.redis.get(f"jwt:valid:{jti}"):
            return {"status": "valid", "jti": jti}
        elif self.redis.get(f"jwt:blacklist:{jti}"):
            return {"status": "revoked", "jti": jti}
        else:
            return None
