"""
Comprehensive tests for JWT authentication service.

This test suite covers:
- Token generation and validation
- Token expiration handling
- Token revocation and blacklisting
- Refresh token rotation
- Key management and rotation
- Security edge cases
- Performance under load
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
import pytest
import redis
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.infrastructure.auth.jwt_service import (
    InvalidTokenException,
    JWTService,
    TokenExpiredException,
    TokenReuseException,
    TokenRevokedException,
)


class TestJWTService:
    """Test JWT service functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock(spec=redis.Redis)
        mock.get.return_value = None
        mock.set.return_value = True
        mock.exists.return_value = False
        mock.setex.return_value = True
        return mock

    @pytest.fixture
    def jwt_service(self, mock_redis):
        """Create JWT service instance for testing."""
        # Patch the redis module before importing JWTService
        with patch("src.infrastructure.auth.jwt_service.redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis
            # Create service with mock redis client to avoid connection attempts
            service = JWTService(redis_client=mock_redis)
            return service

    @pytest.fixture
    def sample_user_claims(self):
        """Sample user claims for token generation."""
        return {
            "user_id": "user123",
            "email": "user@example.com",
            "roles": ["trader"],
            "permissions": ["read:portfolio", "write:orders"],
        }

    def test_service_initialization(self, mock_redis):
        """Test JWT service initialization."""
        service = JWTService(
            redis_client=mock_redis,
            issuer="test-issuer",
            access_token_expire_minutes=30,
            refresh_token_expire_days=14,
        )

        assert service.issuer == "test-issuer"
        assert service.algorithm == "RS256"
        assert service.access_token_expire == timedelta(minutes=30)
        assert service.refresh_token_expire == timedelta(days=14)
        assert service.redis is mock_redis
        assert service.private_key is not None
        assert service.public_key is not None

    def test_key_generation(self, mock_redis):
        """Test RSA key pair generation."""
        service = JWTService(redis_client=mock_redis)

        # Keys should be generated
        assert service.private_key is not None
        assert service.public_key is not None

        # Verify keys are valid RSA keys
        private_pem = service.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        assert b"BEGIN PRIVATE KEY" in private_pem

    def test_load_keys_from_files(self, mock_redis, tmp_path):
        """Test loading keys from files."""
        # Generate test keys
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()

        # Save keys to files
        private_path = tmp_path / "private.pem"
        public_path = tmp_path / "public.pem"

        private_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

        public_path.write_bytes(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

        # Load keys in service
        service = JWTService(
            private_key_path=str(private_path),
            public_key_path=str(public_path),
            redis_client=mock_redis,
        )

        assert service.private_key is not None
        assert service.public_key is not None

    def test_create_access_token(self, jwt_service, sample_user_claims):
        """Test access token creation."""
        token = jwt_service.create_access_token(sample_user_claims)

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify token
        decoded = jwt.decode(
            token,
            jwt_service.public_key,
            algorithms=[jwt_service.algorithm],
            issuer=jwt_service.issuer,
        )

        assert decoded["user_id"] == sample_user_claims["user_id"]
        assert decoded["email"] == sample_user_claims["email"]
        assert decoded["roles"] == sample_user_claims["roles"]
        assert decoded["type"] == "access"
        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded

    def test_create_refresh_token(self, jwt_service, sample_user_claims):
        """Test refresh token creation."""
        token = jwt_service.create_refresh_token(sample_user_claims)

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify token
        decoded = jwt.decode(
            token,
            jwt_service.public_key,
            algorithms=[jwt_service.algorithm],
            issuer=jwt_service.issuer,
        )

        assert decoded["user_id"] == sample_user_claims["user_id"]
        assert decoded["type"] == "refresh"
        assert "exp" in decoded
        assert decoded["exp"] > datetime.utcnow().timestamp()

    def test_verify_valid_token(self, jwt_service, sample_user_claims):
        """Test verifying a valid token."""
        token = jwt_service.create_access_token(sample_user_claims)
        decoded = jwt_service.verify_token(token)

        assert decoded is not None
        assert decoded["user_id"] == sample_user_claims["user_id"]
        assert decoded["email"] == sample_user_claims["email"]

    def test_verify_expired_token(self, jwt_service, sample_user_claims):
        """Test verifying an expired token."""
        # Create token with very short expiration
        jwt_service.access_token_expire = timedelta(seconds=-1)
        token = jwt_service.create_access_token(sample_user_claims)

        with pytest.raises(TokenExpiredException):
            jwt_service.verify_token(token)

    def test_verify_invalid_signature(self, jwt_service, sample_user_claims):
        """Test verifying token with invalid signature."""
        token = jwt_service.create_access_token(sample_user_claims)
        # Tamper with token
        tampered_token = token[:-10] + "corrupted!"

        with pytest.raises(InvalidTokenException):
            jwt_service.verify_token(tampered_token)

    def test_verify_malformed_token(self, jwt_service):
        """Test verifying malformed token."""
        malformed_token = "not.a.valid.token"

        with pytest.raises(InvalidTokenException):
            jwt_service.verify_token(malformed_token)

    def test_revoke_token(self, jwt_service, mock_redis, sample_user_claims):
        """Test token revocation."""
        token = jwt_service.create_access_token(sample_user_claims)
        decoded = jwt_service.verify_token(token)

        # Revoke token
        jwt_service.revoke_token(token)

        # Verify Redis was called to store revoked token
        jti = decoded["jti"]
        mock_redis.setex.assert_called()
        call_args = mock_redis.setex.call_args
        assert f"revoked_token:{jti}" in call_args[0][0]

    def test_verify_revoked_token(self, jwt_service, mock_redis, sample_user_claims):
        """Test verifying a revoked token."""
        token = jwt_service.create_access_token(sample_user_claims)
        decoded = jwt_service.verify_token(token)

        # Revoke token
        jwt_service.revoke_token(token)

        # Mock Redis to return True for revoked token check
        mock_redis.exists.return_value = True

        with pytest.raises(TokenRevokedException):
            jwt_service.verify_token(token)

    def test_refresh_token_rotation(self, jwt_service, mock_redis, sample_user_claims):
        """Test refresh token rotation."""
        # Create initial refresh token
        refresh_token = jwt_service.create_refresh_token(sample_user_claims)

        # Rotate refresh token
        new_access, new_refresh = jwt_service.rotate_refresh_token(refresh_token)

        assert new_access is not None
        assert new_refresh is not None
        assert new_refresh != refresh_token

        # Old refresh token should be revoked
        mock_redis.setex.assert_called()

    def test_refresh_token_reuse_detection(self, jwt_service, mock_redis, sample_user_claims):
        """Test detection of refresh token reuse."""
        refresh_token = jwt_service.create_refresh_token(sample_user_claims)

        # First rotation succeeds
        new_access, new_refresh = jwt_service.rotate_refresh_token(refresh_token)

        # Mark old token as used
        mock_redis.exists.return_value = True

        # Attempting to reuse old refresh token should fail
        with pytest.raises(TokenReuseException):
            jwt_service.rotate_refresh_token(refresh_token)

    def test_token_with_custom_claims(self, jwt_service):
        """Test token creation with custom claims."""
        custom_claims = {
            "user_id": "user456",
            "org_id": "org789",
            "subscription_tier": "premium",
            "feature_flags": ["advanced_analytics", "api_access"],
        }

        token = jwt_service.create_access_token(custom_claims)
        decoded = jwt_service.verify_token(token)

        assert decoded["org_id"] == "org789"
        assert decoded["subscription_tier"] == "premium"
        assert "advanced_analytics" in decoded["feature_flags"]

    def test_token_expiration_boundary(self, jwt_service, sample_user_claims):
        """Test token expiration at exact boundary."""
        # Set very short expiration
        jwt_service.access_token_expire = timedelta(seconds=1)
        token = jwt_service.create_access_token(sample_user_claims)

        # Token should be valid immediately
        decoded = jwt_service.verify_token(token)
        assert decoded is not None

        # Wait for expiration
        time.sleep(2)

        # Token should now be expired
        with pytest.raises(TokenExpiredException):
            jwt_service.verify_token(token)

    def test_concurrent_token_operations(self, jwt_service, sample_user_claims):
        """Test concurrent token operations."""
        import threading

        tokens = []
        errors = []

        def create_and_verify():
            try:
                token = jwt_service.create_access_token(sample_user_claims)
                decoded = jwt_service.verify_token(token)
                tokens.append(token)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_and_verify)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(tokens) == 10
        # All tokens should be unique
        assert len(set(tokens)) == 10

    def test_token_audience_validation(self, jwt_service, sample_user_claims):
        """Test token audience validation."""
        # Create token with specific audience
        claims_with_audience = {**sample_user_claims, "aud": "mobile-app"}
        token = jwt_service.create_access_token(claims_with_audience)

        # Verify with correct audience
        decoded = jwt_service.verify_token(token, audience="mobile-app")
        assert decoded is not None

        # Verify with wrong audience should fail
        with pytest.raises(InvalidTokenException):
            jwt_service.verify_token(token, audience="web-app")

    def test_key_rotation(self, jwt_service, mock_redis, sample_user_claims):
        """Test key rotation support."""
        # Create token with current key
        old_key_id = jwt_service.key_id
        token1 = jwt_service.create_access_token(sample_user_claims)

        # Simulate key rotation
        jwt_service.rotate_keys()
        new_key_id = jwt_service.key_id

        assert old_key_id != new_key_id

        # Create token with new key
        token2 = jwt_service.create_access_token(sample_user_claims)

        # Both tokens should still be verifiable
        decoded1 = jwt_service.verify_token(token1)
        decoded2 = jwt_service.verify_token(token2)

        assert decoded1 is not None
        assert decoded2 is not None

    def test_token_size_limits(self, jwt_service):
        """Test token size with large claims."""
        large_claims = {
            "user_id": "user123",
            "permissions": ["perm" + str(i) for i in range(1000)],  # Large permission list
        }

        token = jwt_service.create_access_token(large_claims)

        # Token should still be created but size should be reasonable
        assert len(token) < 10000  # Reasonable size limit

        # Should still be verifiable
        decoded = jwt_service.verify_token(token)
        assert len(decoded["permissions"]) == 1000

    def test_token_jti_uniqueness(self, jwt_service, sample_user_claims):
        """Test that JTI (JWT ID) is unique for each token."""
        tokens = []
        jtis = []

        for _ in range(100):
            token = jwt_service.create_access_token(sample_user_claims)
            decoded = jwt_service.verify_token(token)
            tokens.append(token)
            jtis.append(decoded["jti"])

        # All JTIs should be unique
        assert len(set(jtis)) == 100

    def test_redis_connection_failure(self, sample_user_claims):
        """Test handling Redis connection failure."""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.setex.side_effect = redis.ConnectionError("Connection failed")

        service = JWTService(redis_client=mock_redis)
        token = service.create_access_token(sample_user_claims)

        # Should still create token even if Redis fails
        assert token is not None

        # Revocation should handle Redis failure gracefully
        with pytest.raises(redis.ConnectionError):
            service.revoke_token(token)

    def test_token_not_before_claim(self, jwt_service, sample_user_claims):
        """Test token with 'not before' claim."""
        future_time = datetime.utcnow() + timedelta(minutes=5)
        claims_with_nbf = {**sample_user_claims, "nbf": future_time.timestamp()}

        token = jwt_service.create_access_token(claims_with_nbf)

        # Token should not be valid yet
        with pytest.raises(InvalidTokenException):
            jwt_service.verify_token(token)

    def test_batch_token_revocation(self, jwt_service, mock_redis, sample_user_claims):
        """Test revoking multiple tokens at once."""
        tokens = []
        for i in range(5):
            claims = {**sample_user_claims, "session_id": f"session_{i}"}
            token = jwt_service.create_access_token(claims)
            tokens.append(token)

        # Revoke all tokens
        jwt_service.revoke_tokens(tokens)

        # Verify all tokens were revoked
        assert mock_redis.setex.call_count >= 5

    def test_token_decode_without_verification(self, jwt_service, sample_user_claims):
        """Test decoding token without verification."""
        token = jwt_service.create_access_token(sample_user_claims)

        # Decode without verification (for logging/debugging)
        decoded = jwt_service.decode_token_unsafe(token)

        assert decoded["user_id"] == sample_user_claims["user_id"]
        # Should work even with expired token
        jwt_service.access_token_expire = timedelta(seconds=-1)
        expired_token = jwt_service.create_access_token(sample_user_claims)
        decoded = jwt_service.decode_token_unsafe(expired_token)
        assert decoded is not None
