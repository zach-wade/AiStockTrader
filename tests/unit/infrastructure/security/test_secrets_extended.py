"""
Extended unit tests for secrets management module to achieve 80%+ coverage.

Focuses on uncovered areas including encryption, rate limiting, rotation,
and advanced provider features.
"""

import os
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from cryptography.fernet import Fernet

from src.infrastructure.security.secrets import (
    AWSSecretsManagerProvider,
    EnvironmentSecretProvider,
    RateLimiter,
    SecretConfig,
    SecretEncryption,
    SecretEncryptionError,
    SecretProvider,
    SecretRateLimitError,
    SecretsManager,
)


class TestSecretEncryption:
    """Test encryption/decryption functionality"""

    def test_init_with_key(self):
        """Test initialization with provided key"""
        key = Fernet.generate_key().decode()
        encryption = SecretEncryption(encryption_key=key)
        assert encryption._fernet is not None

    def test_init_with_bytes_key(self):
        """Test initialization with bytes key"""
        key = Fernet.generate_key()  # bytes
        encryption = SecretEncryption(encryption_key=key)
        assert encryption._fernet is not None

    @patch.dict(os.environ, {"SECRET_MASTER_PASSWORD": "test_password", "SECRET_SALT": "test_salt"})
    def test_init_without_key(self):
        """Test initialization deriving key from password"""
        encryption = SecretEncryption()
        assert encryption._fernet is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_default_password(self):
        """Test initialization with default password when env vars missing"""
        encryption = SecretEncryption()
        assert encryption._fernet is not None

    def test_encrypt_decrypt(self):
        """Test successful encryption and decryption"""
        key = Fernet.generate_key().decode()
        encryption = SecretEncryption(encryption_key=key)

        original = "sensitive_data"
        encrypted = encryption.encrypt(original)

        assert encrypted != original
        assert isinstance(encrypted, str)

        decrypted = encryption.decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_error(self):
        """Test encryption error handling"""
        encryption = SecretEncryption()
        # Mock fernet to raise an error
        encryption._fernet = Mock()
        encryption._fernet.encrypt.side_effect = Exception("Encryption failed")

        with pytest.raises(SecretEncryptionError) as exc_info:
            encryption.encrypt("test")
        assert "Failed to encrypt secret" in str(exc_info)

    def test_decrypt_error(self):
        """Test decryption error handling"""
        encryption = SecretEncryption()

        with pytest.raises(SecretEncryptionError) as exc_info:
            encryption.decrypt("invalid_encrypted_data")
        assert "Failed to decrypt secret" in str(exc_info)

    def test_generate_key(self):
        """Test key generation"""
        key = SecretEncryption.generate_key()
        assert isinstance(key, str)
        # Verify it's a valid Fernet key
        Fernet(key.encode())


class TestRateLimiter:
    """Test rate limiting functionality"""

    def test_is_allowed_under_limit(self):
        """Test request allowed when under limit"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed("test_id", rule) is True

    def test_is_allowed_over_limit(self):
        """Test request denied when over limit"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("test_id", rule) is True
        assert limiter.is_allowed("test_id", rule) is True
        assert limiter.is_allowed("test_id", rule) is False

    @patch("time.time")
    def test_window_expiry(self, mock_time):
        """Test that old requests expire from window"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=2, window_seconds=10)

        # Make requests at time 100
        mock_time.return_value = 100
        assert limiter.is_allowed("test_id", rule) is True
        assert limiter.is_allowed("test_id", rule) is True
        assert limiter.is_allowed("test_id", rule) is False

        # Move time forward past window
        mock_time.return_value = 111  # 11 seconds later
        assert limiter.is_allowed("test_id", rule) is True

    def test_get_remaining_requests(self):
        """Test getting remaining requests count"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=5, window_seconds=60, burst_allowance=2)

        assert limiter.get_remaining_requests("test_id", rule) == 5

        limiter.is_allowed("test_id", rule)
        assert limiter.get_remaining_requests("test_id", rule) == 4

        # Test with burst tokens
        limiter._burst_tokens["test_id"] = 2
        assert limiter.get_remaining_requests("test_id", rule) == 6

    @patch("time.time")
    def test_get_remaining_requests_with_expired(self, mock_time):
        """Test remaining requests with expired entries"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=3, window_seconds=10)

        mock_time.return_value = 100
        limiter.is_allowed("test_id", rule)
        limiter.is_allowed("test_id", rule)

        mock_time.return_value = 111  # Past window
        remaining = limiter.get_remaining_requests("test_id", rule)
        assert remaining == 3  # All requests expired

    def test_reset_limit(self):
        """Test resetting rate limit for an identifier"""
        from src.infrastructure.security.hardening import RateLimitRule

        limiter = RateLimiter()
        rule = RateLimitRule(max_requests=1, window_seconds=60)

        # Use up the limit
        assert limiter.is_allowed("test_id", rule) is True
        assert limiter.is_allowed("test_id", rule) is False

        # Set burst tokens and cooldown
        limiter._burst_tokens["test_id"] = 5
        limiter._cooldowns["test_id"] = 1000

        # Reset the limit
        limiter.reset_limit("test_id")

        # Should be allowed again
        assert limiter.is_allowed("test_id", rule) is True
        assert "test_id" not in limiter._burst_tokens
        assert "test_id" not in limiter._cooldowns


class TestSecretConfigExtended:
    """Extended tests for SecretConfig"""

    @patch.dict(
        os.environ,
        {
            "SECRET_ENCRYPTION_ENABLED": "false",
            "SECRET_ENCRYPTION_KEY": "test_key",
            "SECRET_RATE_LIMIT_REQUESTS": "200",
            "SECRET_RATE_LIMIT_WINDOW": "120",
            "SECRET_ROTATION_ENABLED": "true",
            "SECRET_ROTATION_INTERVAL": "7200",
        },
    )
    def test_from_env_extended(self):
        """Test loading extended configuration from environment"""
        config = SecretConfig.from_env()

        assert config.enable_encryption is False
        assert config.encryption_key == "test_key"
        assert config.rate_limit_requests == 200
        assert config.rate_limit_window == 120
        assert config.enable_rotation is True
        assert config.rotation_interval == 7200

    def test_secret_validation_patterns(self):
        """Test secret validation patterns configuration"""
        patterns = {"API_KEY": r"^[A-Z0-9]{32}$", "PASSWORD": r"^.{8,}$"}
        config = SecretConfig(secret_validation_patterns=patterns)
        assert config.secret_validation_patterns == patterns


class TestEnvironmentSecretProviderExtended:
    """Extended tests for EnvironmentSecretProvider"""

    def test_rotate_secret(self):
        """Test rotation (not supported by environment provider)"""
        provider = EnvironmentSecretProvider()

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            result = provider.rotate_secret("KEY")
            assert result is False
            mock_logger.warning.assert_called_once()

    @patch("src.domain.services.secrets_validation_service.SecretsValidationService")
    def test_check_format(self, mock_service):
        """Test format checking delegation to domain service"""
        provider = EnvironmentSecretProvider()
        mock_service.validate_secret_format.return_value = True

        result = provider.check_format("API_KEY", "test_value")

        assert result is True
        mock_service.validate_secret_format.assert_called_once_with("API_KEY", "test_value")


class TestAWSSecretsManagerProviderExtended:
    """Extended tests for AWS provider"""

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_rotate_secret_success(self):
        """Test successful secret rotation"""
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock secrets module for token generation
        with patch("secrets.token_urlsafe", return_value="new_secret_value"):
            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            provider._cached_secrets = {"key": "old_value"}

            with patch("src.infrastructure.security.secrets.logger") as mock_logger:
                result = provider.rotate_secret("key")

                assert result is True
                mock_logger.info.assert_called_once()
                # Cache should be cleared
                assert provider._cached_secrets is None

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_rotate_secret_failure(self):
        """Test failed secret rotation"""
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.update_secret.side_effect = Exception("Update failed")

        provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
        provider._cached_secrets = {"key": "value"}

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            with patch("secrets.token_urlsafe", return_value="new_value"):
                result = provider.rotate_secret("key")

                assert result is False
                mock_logger.error.assert_called()

    @patch("src.domain.services.secrets_validation_service.SecretsValidationService")
    def test_check_format_aws(self, mock_service):
        """Test format checking for AWS provider"""
        provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
        mock_service.validate_secret_format.return_value = False

        result = provider.check_format("API_KEY", "invalid")

        assert result is False
        mock_service.validate_secret_format.assert_called_once_with("API_KEY", "invalid")


class TestSecretsManagerExtended:
    """Extended tests for SecretsManager"""

    def test_init_with_encryption(self):
        """Test initialization with encryption enabled"""
        config = SecretConfig(enable_encryption=True, encryption_key=Fernet.generate_key().decode())
        manager = SecretsManager(config)

        assert manager._encryption is not None
        assert manager._rate_limiter is not None

    def test_init_with_rate_limiting(self):
        """Test initialization with custom rate limiting"""
        config = SecretConfig(rate_limit_requests=50, rate_limit_window=30)
        manager = SecretsManager(config)

        assert manager._rate_limiter is not None

    @patch("time.time")
    def test_get_secret_with_rate_limiting(self, mock_time):
        """Test get_secret with rate limiting"""
        config = SecretConfig(enable_encryption=False, rate_limit_requests=2, rate_limit_window=10)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "value"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        mock_time.return_value = 100

        # First two requests should succeed
        assert manager.get_secret("KEY", required=False) == "value"
        assert manager.get_secret("KEY", required=False) == "value"

        # Third request should be rate limited
        with pytest.raises(SecretRateLimitError) as exc_info:
            manager.get_secret("KEY", required=False)
        assert "Rate limit exceeded" in str(exc_info)

    @patch("time.time")
    def test_get_secret_with_rotation_check(self, mock_time):
        """Test automatic rotation check"""
        config = SecretConfig(enable_rotation=True, rotation_interval=100, enable_encryption=False)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "value"
        mock_provider.check_format.return_value = True
        mock_provider.rotate_secret.return_value = True
        manager._provider = mock_provider

        # Set last rotation time
        mock_time.return_value = 1000
        manager._rotation_timestamps["KEY"] = 800  # 200 seconds ago

        # Should trigger rotation
        value = manager.get_secret("KEY", required=False)

        assert value == "value"
        mock_provider.rotate_secret.assert_called_once_with("KEY")
        assert manager._rotation_timestamps["KEY"] == 1000

    @patch("time.time")
    def test_get_secret_rotation_failure(self, mock_time):
        """Test handling of rotation failure"""
        config = SecretConfig(enable_rotation=True, rotation_interval=100, enable_encryption=False)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "value"
        mock_provider.check_format.return_value = True
        mock_provider.rotate_secret.side_effect = Exception("Rotation failed")
        manager._provider = mock_provider

        mock_time.return_value = 1000
        manager._rotation_timestamps["KEY"] = 800

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            # Should log error but continue
            value = manager.get_secret("KEY", required=False)

            assert value == "value"
            mock_logger.error.assert_called()

    def test_get_secret_with_encryption(self):
        """Test getting encrypted secret"""
        key = Fernet.generate_key().decode()
        config = SecretConfig(enable_encryption=True, encryption_key=key)
        manager = SecretsManager(config)

        # Encrypt a test value
        fernet = Fernet(key.encode())
        encrypted_value = fernet.encrypt(b"decrypted_value").decode()

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = encrypted_value
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        # Mock the _is_encrypted_secret method
        with patch.object(manager, "_is_encrypted_secret", return_value=True):
            value = manager.get_secret("ENCRYPTED_KEY", required=False, decrypt=True)
            assert value == "decrypted_value"

    def test_get_secret_encryption_error_required(self):
        """Test encryption error with required secret"""
        config = SecretConfig(enable_encryption=True, encryption_key=Fernet.generate_key().decode())
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "invalid_encrypted_data"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch.object(manager, "_is_encrypted_secret", return_value=True):
            with patch("src.infrastructure.security.secrets.logger"):
                with pytest.raises(SecretEncryptionError):
                    manager.get_secret("KEY", required=True, decrypt=True)

    def test_get_secret_encryption_error_optional(self):
        """Test encryption error with optional secret"""
        config = SecretConfig(enable_encryption=True, encryption_key=Fernet.generate_key().decode())
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "invalid_encrypted_data"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch.object(manager, "_is_encrypted_secret", return_value=True):
            with patch("src.infrastructure.security.secrets.logger"):
                value = manager.get_secret("KEY", required=False, decrypt=True)
                assert value is None

    def test_get_secret_no_decrypt(self):
        """Test getting secret without decryption"""
        config = SecretConfig(enable_encryption=True)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "encrypted_value"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        value = manager.get_secret("KEY", required=False, decrypt=False)
        assert value == "encrypted_value"

    def test_get_secret_format_check_warning(self):
        """Test warning when secret fails format check"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "invalid_format"
        mock_provider.check_format.return_value = False
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            value = manager.get_secret("KEY", required=False)
            assert value == "invalid_format"
            mock_logger.warning.assert_called()

    def test_set_secret_success(self):
        """Test successful secret setting"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.check_format.return_value = True
        mock_provider.set_secret.return_value = True
        manager._provider = mock_provider

        # Add to cache first
        manager._cache["KEY"] = ("old_value", time.time())

        result = manager.set_secret("KEY", "new_value", encrypt=False)

        assert result is True
        assert "KEY" not in manager._cache  # Cache cleared
        mock_provider.set_secret.assert_called_once_with("KEY", "new_value")

    def test_set_secret_format_check_failure(self):
        """Test set_secret with format check failure"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.check_format.return_value = False
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            result = manager.set_secret("KEY", "invalid_value")
            assert result is False
            mock_logger.warning.assert_called()

    def test_set_secret_with_encryption(self):
        """Test setting secret with encryption"""
        key = Fernet.generate_key().decode()
        config = SecretConfig(enable_encryption=True, encryption_key=key)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.check_format.return_value = True
        mock_provider.set_secret.return_value = True
        manager._provider = mock_provider

        with patch.object(manager, "_is_encrypted_secret", return_value=True):
            result = manager.set_secret("ENCRYPTED_KEY", "plain_value", encrypt=True)

            assert result is True
            # Verify the value was encrypted
            call_args = mock_provider.set_secret.call_args
            encrypted_value = call_args[0][1]
            assert encrypted_value != "plain_value"

            # Verify we can decrypt it
            fernet = Fernet(key.encode())
            decrypted = fernet.decrypt(encrypted_value.encode()).decode()
            assert decrypted == "plain_value"

    def test_set_secret_encryption_error(self):
        """Test set_secret with encryption error"""
        config = SecretConfig(enable_encryption=True)
        manager = SecretsManager(config)

        # Mock encryption to fail
        manager._encryption = Mock()
        manager._encryption.encrypt.side_effect = SecretEncryptionError("Encryption failed")

        mock_provider = MagicMock()
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch.object(manager, "_is_encrypted_secret", return_value=True):
            with patch("src.infrastructure.security.secrets.logger"):
                with pytest.raises(SecretEncryptionError):
                    manager.set_secret("KEY", "value", encrypt=True)

    def test_rotate_secret_disabled(self):
        """Test rotation when disabled"""
        config = SecretConfig(enable_rotation=False)
        manager = SecretsManager(config)

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            result = manager.rotate_secret("KEY")
            assert result is False
            mock_logger.warning.assert_called()

    def test_rotate_secret_success(self):
        """Test successful secret rotation"""
        config = SecretConfig(enable_rotation=True)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.rotate_secret.return_value = True
        manager._provider = mock_provider

        # Add to cache
        manager._cache["KEY"] = ("old_value", time.time())

        result = manager.rotate_secret("KEY")

        assert result is True
        assert "KEY" not in manager._cache  # Cache cleared
        assert "KEY" in manager._rotation_timestamps

    def test_is_encrypted_secret(self):
        """Test checking if secret should be encrypted"""
        manager = SecretsManager()

        with patch(
            "src.domain.services.secrets_validation_service.SecretsValidationService"
        ) as mock_service:
            mock_service.should_encrypt_secret.return_value = True
            assert manager._is_encrypted_secret("PASSWORD") is True

            mock_service.should_encrypt_secret.return_value = False
            assert manager._is_encrypted_secret("PUBLIC_KEY") is False

    def test_get_rate_limit_status(self):
        """Test getting rate limit status"""
        config = SecretConfig(rate_limit_requests=100, rate_limit_window=60)
        manager = SecretsManager(config)

        # Make some requests
        manager._rate_limiter.is_allowed("test_key", manager._rate_limiter)

        status = manager.get_rate_limit_status("test_key")

        assert "remaining" in status
        assert "limit" in status
        assert status["limit"] == 100

    def test_get_rate_limit_status_disabled(self):
        """Test rate limit status when disabled"""
        manager = SecretsManager()
        manager._rate_limiter = None

        status = manager.get_rate_limit_status("test_key")

        assert status == {"remaining": -1, "limit": -1}

    def test_clear_rate_limits(self):
        """Test clearing rate limits"""
        manager = SecretsManager()

        # Add some rate limit data
        manager._rate_limiter.requests["key1"] = [1, 2, 3]
        manager._rate_limiter.requests["key2"] = [4, 5, 6]

        manager.clear_rate_limits()

        assert len(manager._rate_limiter.requests) == 0

    def test_clear_rate_limits_disabled(self):
        """Test clearing rate limits when disabled"""
        manager = SecretsManager()
        manager._rate_limiter = None

        # Should not raise error
        manager.clear_rate_limits()

    def test_get_secrets_batch(self):
        """Test batch secret retrieval"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            if key == "KEY2":
                return None
            return f"value_{key}"

        mock_provider.get_secret.side_effect = mock_get_secret
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        result = manager.get_secrets_batch(["KEY1", "KEY2", "KEY3"], decrypt=False)

        assert result == {"KEY1": "value_KEY1", "KEY2": None, "KEY3": "value_KEY3"}

    def test_get_secrets_batch_with_errors(self):
        """Test batch retrieval with rate limit errors"""
        config = SecretConfig(rate_limit_requests=1, rate_limit_window=60)
        manager = SecretsManager(config)

        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "value"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            # First key should succeed, second should be rate limited
            result = manager.get_secrets_batch(["KEY1", "KEY2"], decrypt=False)

            assert result["KEY1"] == "value"
            assert result["KEY2"] is None
            mock_logger.error.assert_called()

    def test_check_all_formats(self):
        """Test checking all cached secret formats"""
        manager = SecretsManager()

        # Add secrets to cache
        manager._cache = {
            "KEY1": ("value1", time.time()),
            "KEY2": ("value2", time.time()),
            "KEY3": (None, time.time()),
        }

        mock_provider = MagicMock()

        def mock_check_format(key, value):
            return key != "KEY2"  # KEY2 fails format check

        mock_provider.check_format.side_effect = mock_check_format
        manager._provider = mock_provider

        results = manager.check_all_formats()

        assert results == {"KEY1": True, "KEY2": False, "KEY3": False}  # None value

    def test_get_encryption_status(self):
        """Test getting encryption status"""
        config = SecretConfig(
            enable_encryption=True,
            enable_rotation=True,
            rotation_interval=7200,
            cache_ttl=600,
            rate_limit_requests=200,
            provider=SecretProvider.AWS_SECRETS_MANAGER,
        )
        manager = SecretsManager(config)

        status = manager.get_encryption_status()

        assert status == {
            "encryption_enabled": True,
            "rotation_enabled": True,
            "rotation_interval": 7200,
            "cache_ttl": 600,
            "rate_limit_requests": 200,
            "provider": "aws_secrets_manager",
        }

    @patch("src.domain.services.secrets_validation_service.SecretsValidationService")
    def test_get_database_config_with_validation(self, mock_service):
        """Test database config with validation service"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            secrets = {"DB_USER": "user", "DB_PASSWORD": "pass"}
            return secrets.get(key)

        mock_provider.get_secret.side_effect = mock_get_secret
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        mock_service.apply_defaults_to_database_config.return_value = {
            "host": "localhost",
            "port": 5432,
            "database": "ai_trader",
            "user": "user",
            "password": "pass",
        }

        config = manager.get_database_config()

        assert config["user"] == "user"
        mock_service.apply_defaults_to_database_config.assert_called_once()

    @patch("src.domain.services.secrets_validation_service.SecretsValidationService")
    def test_get_broker_config_with_validation(self, mock_service):
        """Test broker config with validation service"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        mock_service.apply_defaults_to_broker_config.return_value = {
            "alpaca_api_key": None,
            "alpaca_api_secret": None,
            "alpaca_base_url": "https://paper-api.alpaca.markets",
            "polygon_api_key": None,
        }

        config = manager.get_broker_config()

        assert config["alpaca_base_url"] == "https://paper-api.alpaca.markets"
        mock_service.apply_defaults_to_broker_config.assert_called_once_with(
            "alpaca",
            {
                "alpaca_api_key": None,
                "alpaca_api_secret": None,
                "alpaca_base_url": None,
                "polygon_api_key": None,
            },
        )


class TestExceptionTypes:
    """Test custom exception types"""

    def test_secret_encryption_error(self):
        """Test SecretEncryptionError exception"""
        error = SecretEncryptionError("Encryption failed")
        assert str(error) == "Encryption failed"
        assert isinstance(error, Exception)

    def test_secret_rate_limit_error(self):
        """Test SecretRateLimitError exception"""
        error = SecretRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, Exception)
