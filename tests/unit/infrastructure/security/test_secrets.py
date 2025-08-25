"""
Comprehensive unit tests for secrets management module.

Tests all secret providers, caching, error handling, and configuration.
"""

# Standard library imports
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest

# Local imports
from src.infrastructure.security.secrets import (
    AWSSecretsManagerProvider,
    EnvironmentSecretProvider,
    SecretConfig,
    SecretNotFoundError,
    SecretProvider,
    SecretProviderError,
    SecretsManager,
)


class TestSecretConfig:
    """Test SecretConfig configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SecretConfig()

        assert config.provider == SecretProvider.ENVIRONMENT
        assert config.aws_region is None
        assert config.aws_secret_name is None
        assert config.vault_url is None
        assert config.vault_token is None
        assert config.azure_vault_url is None
        assert config.cache_ttl == 300

    def test_custom_config(self):
        """Test custom configuration values"""
        config = SecretConfig(
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            aws_region="us-east-1",
            aws_secret_name="my-secrets",
            cache_ttl=600,
        )

        assert config.provider == SecretProvider.AWS_SECRETS_MANAGER
        assert config.aws_region == "us-east-1"
        assert config.aws_secret_name == "my-secrets"
        assert config.cache_ttl == 600

    @patch.dict(
        os.environ,
        {
            "SECRET_PROVIDER": "aws_secrets_manager",
            "AWS_REGION": "us-west-2",
            "AWS_SECRET_NAME": "test-secrets",
            "VAULT_URL": "https://vault.example.com",
            "VAULT_TOKEN": "test-token",
            "AZURE_VAULT_URL": "https://keyvault.azure.net",
            "SECRET_CACHE_TTL": "600",
        },
    )
    def test_from_env(self):
        """Test loading configuration from environment variables"""
        config = SecretConfig.from_env()

        assert config.provider == SecretProvider.AWS_SECRETS_MANAGER
        assert config.aws_region == "us-west-2"
        assert config.aws_secret_name == "test-secrets"
        assert config.vault_url == "https://vault.example.com"
        assert config.vault_token == "test-token"
        assert config.azure_vault_url == "https://keyvault.azure.net"
        assert config.cache_ttl == 600

    @patch.dict(os.environ, {"SECRET_PROVIDER": "invalid_provider"})
    def test_from_env_invalid_provider(self):
        """Test handling of invalid provider in environment"""
        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            config = SecretConfig.from_env()

            assert config.provider == SecretProvider.ENVIRONMENT
            mock_logger.warning.assert_called_once()

    @patch.dict(os.environ, {"SECRET_CACHE_TTL": "not_a_number"})
    def test_from_env_invalid_cache_ttl(self):
        """Test handling of invalid cache TTL in environment"""
        with pytest.raises(ValueError):
            SecretConfig.from_env()

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_empty(self):
        """Test loading configuration with no environment variables"""
        config = SecretConfig.from_env()

        assert config.provider == SecretProvider.ENVIRONMENT
        assert config.aws_region is None
        assert config.cache_ttl == 300


class TestEnvironmentSecretProvider:
    """Test EnvironmentSecretProvider"""

    def test_get_secret_without_prefix(self):
        """Test getting secret without prefix"""
        provider = EnvironmentSecretProvider()

        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            value = provider.get_secret("TEST_KEY")
            assert value == "test_value"

    def test_get_secret_with_prefix(self):
        """Test getting secret with prefix"""
        provider = EnvironmentSecretProvider(prefix="APP_")

        with patch.dict(os.environ, {"APP_TEST_KEY": "test_value"}):
            value = provider.get_secret("TEST_KEY")
            assert value == "test_value"

    def test_get_secret_fallback_prefixes(self):
        """Test fallback to common prefixes when key not found"""
        provider = EnvironmentSecretProvider()

        with patch.dict(os.environ, {"TRADING_DB_PASSWORD": "secret123"}):
            value = provider.get_secret("DB_PASSWORD")
            assert value == "secret123"

    def test_get_secret_not_found(self):
        """Test getting non-existent secret returns None"""
        provider = EnvironmentSecretProvider()

        with patch.dict(os.environ, {}, clear=True):
            value = provider.get_secret("NONEXISTENT_KEY")
            assert value is None

    def test_get_secrets_batch(self):
        """Test getting multiple secrets at once"""
        provider = EnvironmentSecretProvider()

        with patch.dict(os.environ, {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}):
            secrets = provider.get_secrets_batch(["KEY1", "KEY2", "KEY3", "KEY4"])

            assert secrets == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3", "KEY4": None}

    def test_set_secret(self):
        """Test setting a secret in environment"""
        provider = EnvironmentSecretProvider()

        with patch.dict(os.environ, {}, clear=True):
            result = provider.set_secret("NEW_KEY", "new_value")

            assert result is True
            assert os.environ.get("NEW_KEY") == "new_value"

    def test_set_secret_with_prefix(self):
        """Test setting a secret with prefix"""
        provider = EnvironmentSecretProvider(prefix="TEST_")

        with patch.dict(os.environ, {}, clear=True):
            result = provider.set_secret("KEY", "value")

            assert result is True
            assert os.environ.get("TEST_KEY") == "value"


class TestAWSSecretsManagerProvider:
    """Test AWSSecretsManagerProvider"""

    def test_init(self):
        """Test provider initialization"""
        provider = AWSSecretsManagerProvider(region="us-east-1", secret_name="test-secret")

        assert provider.region == "us-east-1"
        assert provider.secret_name == "test-secret"
        assert provider._client is None
        assert provider._cached_secrets is None

    def test_get_client_success(self):
        """Test successful boto3 client creation"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            client = provider._get_client()

            assert client == mock_client
            mock_boto3.client.assert_called_once_with("secretsmanager", region_name="us-east-1")

            # Test that client is cached
            client2 = provider._get_client()
            assert client2 == mock_client
            assert mock_boto3.client.call_count == 1

    def test_get_client_import_error(self):
        """Test handling of missing boto3 library"""
        provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
        provider._client = None  # Ensure client is not cached

        # Mock the import to fail
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(SecretProviderError) as exc_info:
                provider._get_client()

            assert "boto3 is required" in str(exc_info)

    def test_load_secrets_json(self):
        """Test loading JSON secrets from AWS"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {
                "SecretString": json.dumps({"key1": "value1", "key2": "value2"})
            }

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            secrets = provider._load_secrets()

            assert secrets == {"key1": "value1", "key2": "value2"}
            assert provider._cached_secrets == secrets

            # Test caching
            secrets2 = provider._load_secrets()
            assert secrets2 == secrets
            assert mock_client.get_secret_value.call_count == 1

    def test_load_secrets_plain_text(self):
        """Test loading plain text secret from AWS"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {"SecretString": "plain_secret_value"}

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            secrets = provider._load_secrets()

            assert secrets == {"value": "plain_secret_value"}

    def test_load_secrets_binary_error(self):
        """Test error handling for binary secrets"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {"SecretBinary": b"binary_data"}

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")

            with pytest.raises(SecretProviderError) as exc_info:
                provider._load_secrets()

            assert "Binary secrets are not supported" in str(exc_info)

    def test_load_secrets_aws_error(self):
        """Test handling of AWS API errors"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.side_effect = Exception("AWS error")

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")

            with patch("src.infrastructure.security.secrets.logger") as mock_logger:
                with pytest.raises(SecretProviderError) as exc_info:
                    provider._load_secrets()

                assert "AWS Secrets Manager error" in str(exc_info)
                mock_logger.error.assert_called_once()

    def test_get_secret(self):
        """Test getting a single secret"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {
                "SecretString": json.dumps({"key1": "value1", "key2": "value2"})
            }

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")

            assert provider.get_secret("key1") == "value1"
            assert provider.get_secret("key2") == "value2"
            assert provider.get_secret("nonexistent") is None

    def test_get_secrets_batch(self):
        """Test getting multiple secrets"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {
                "SecretString": json.dumps({"key1": "value1", "key2": "value2"})
            }

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            secrets = provider.get_secrets_batch(["key1", "key2", "key3"])

            assert secrets == {"key1": "value1", "key2": "value2", "key3": None}

    def test_set_secret_success(self):
        """Test successfully updating a secret"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {
                "SecretString": json.dumps({"existing": "value"})
            }

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")
            result = provider.set_secret("new_key", "new_value")

            assert result is True

            # Verify update was called with correct data
            mock_client.update_secret.assert_called_once()
            call_args = mock_client.update_secret.call_args
            assert call_args[1]["SecretId"] == "test-secret"

            updated_secrets = json.loads(call_args[1]["SecretString"])
            assert updated_secrets == {"existing": "value", "new_key": "new_value"}

            # Verify cache was cleared
            assert provider._cached_secrets is None

    def test_set_secret_failure(self):
        """Test handling of secret update failure"""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            mock_boto3 = sys.modules["boto3"]
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            mock_client.get_secret_value.return_value = {"SecretString": json.dumps({})}
            mock_client.update_secret.side_effect = Exception("Update failed")

            provider = AWSSecretsManagerProvider("us-east-1", "test-secret")

            with patch("src.infrastructure.security.secrets.logger") as mock_logger:
                result = provider.set_secret("key", "value")

                assert result is False
                mock_logger.error.assert_called_once()


class TestSecretsManager:
    """Test SecretsManager main class"""

    def test_singleton_pattern(self):
        """Test that SecretsManager follows singleton pattern"""
        SecretsManager._instance = None  # Reset singleton

        manager1 = SecretsManager.get_instance()
        manager2 = SecretsManager.get_instance()

        assert manager1 is manager2

        # Cleanup
        SecretsManager._instance = None

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        manager = SecretsManager()

        assert manager.config.provider == SecretProvider.ENVIRONMENT
        assert isinstance(manager._provider, EnvironmentSecretProvider)
        assert manager._cache == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        config = SecretConfig(provider=SecretProvider.ENVIRONMENT, cache_ttl=600)

        manager = SecretsManager(config)

        assert manager.config == config
        assert manager.config.cache_ttl == 600

    @patch("src.infrastructure.security.secrets.AWSSecretsManagerProvider")
    def test_create_provider_aws(self, mock_aws_provider_class):
        """Test creating AWS provider"""
        mock_provider = MagicMock()
        mock_aws_provider_class.return_value = mock_provider

        config = SecretConfig(
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            aws_region="us-east-1",
            aws_secret_name="test-secret",
        )

        manager = SecretsManager(config)

        assert manager._provider == mock_provider
        mock_aws_provider_class.assert_called_once_with("us-east-1", "test-secret")

    def test_create_provider_aws_missing_config(self):
        """Test fallback to environment when AWS config is incomplete"""
        config = SecretConfig(
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            aws_region="us-east-1",
            # Missing aws_secret_name
        )

        with patch("src.infrastructure.security.secrets.logger") as mock_logger:
            manager = SecretsManager(config)

            assert isinstance(manager._provider, EnvironmentSecretProvider)
            mock_logger.warning.assert_called_once()

    def test_get_secret_required_found(self):
        """Test getting a required secret that exists"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "secret_value"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        value = manager.get_secret("TEST_KEY", required=True)

        assert value == "secret_value"
        mock_provider.get_secret.assert_called_once_with("TEST_KEY")

    def test_get_secret_required_not_found(self):
        """Test getting a required secret that doesn't exist"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        manager._provider = mock_provider

        with pytest.raises(SecretNotFoundError) as exc_info:
            manager.get_secret("TEST_KEY", required=True)

        assert "Required secret 'TEST_KEY' not found" in str(exc_info)

    def test_get_secret_optional_not_found(self):
        """Test getting an optional secret that doesn't exist"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        manager._provider = mock_provider

        value = manager.get_secret("TEST_KEY", required=False)

        assert value is None

    @patch("time.time")
    def test_get_secret_caching(self, mock_time):
        """Test secret caching behavior"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "cached_value"
        manager._provider = mock_provider

        # First call - should hit provider
        mock_time.return_value = 1000
        value1 = manager.get_secret("CACHE_KEY", required=False)

        assert value1 == "cached_value"
        assert mock_provider.get_secret.call_count == 1

        # Second call within cache TTL - should use cache
        mock_time.return_value = 1100  # 100 seconds later
        value2 = manager.get_secret("CACHE_KEY", required=False)

        assert value2 == "cached_value"
        assert mock_provider.get_secret.call_count == 1  # Still 1

        # Third call after cache TTL - should hit provider again
        mock_time.return_value = 1400  # 400 seconds later
        value3 = manager.get_secret("CACHE_KEY", required=False)

        assert value3 == "cached_value"
        assert mock_provider.get_secret.call_count == 2

    @patch("time.time")
    def test_get_secret_cache_with_none_value(self, mock_time):
        """Test that None values are also cached"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        manager._provider = mock_provider

        mock_time.return_value = 1000

        # First call
        value1 = manager.get_secret("MISSING_KEY", required=False)
        assert value1 is None
        assert mock_provider.get_secret.call_count == 1

        # Second call - should use cached None
        mock_time.return_value = 1100
        value2 = manager.get_secret("MISSING_KEY", required=False)
        assert value2 is None
        assert mock_provider.get_secret.call_count == 1

    def test_get_database_config(self):
        """Test getting database configuration"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            secrets = {
                "DB_HOST": "db.example.com",
                "DB_PORT": "5432",
                "DB_NAME": "testdb",
                "DB_USER": "testuser",
                "DB_PASSWORD": "testpass",
            }
            return secrets.get(key)

        mock_provider.get_secret.side_effect = mock_get_secret
        manager._provider = mock_provider

        config = manager.get_database_config()

        assert config == {
            "host": "db.example.com",
            "port": 5432,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        }

    def test_get_database_config_defaults(self):
        """Test database config with default values"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            secrets = {"DB_USER": "user", "DB_PASSWORD": "pass"}
            return secrets.get(key)

        mock_provider.get_secret.side_effect = mock_get_secret
        manager._provider = mock_provider

        config = manager.get_database_config()

        assert config == {
            "host": "localhost",
            "port": 5432,
            "database": "ai_trader",
            "user": "user",
            "password": "pass",
        }

    def test_get_database_config_missing_required(self):
        """Test database config with missing required fields"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        manager._provider = mock_provider

        with pytest.raises(SecretNotFoundError):
            manager.get_database_config()

    def test_get_broker_config(self):
        """Test getting broker configuration"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            secrets = {
                "ALPACA_API_KEY": "test_key",
                "ALPACA_API_SECRET": "test_secret",
                "ALPACA_BASE_URL": "https://test.alpaca.com",
                "POLYGON_API_KEY": "polygon_key",
            }
            return secrets.get(key)

        mock_provider.get_secret.side_effect = mock_get_secret
        manager._provider = mock_provider

        config = manager.get_broker_config()

        assert config == {
            "alpaca_api_key": "test_key",
            "alpaca_api_secret": "test_secret",
            "alpaca_base_url": "https://test.alpaca.com",
            "polygon_api_key": "polygon_key",
        }

    def test_get_broker_config_defaults(self):
        """Test broker config with default values"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        manager._provider = mock_provider

        config = manager.get_broker_config()

        assert config == {
            "alpaca_api_key": None,
            "alpaca_api_secret": None,
            "alpaca_base_url": "https://paper-api.alpaca.markets",
            "polygon_api_key": None,
        }

    def test_clear_cache(self):
        """Test clearing the secrets cache"""
        manager = SecretsManager()
        manager._cache = {"key1": ("value1", 1000), "key2": ("value2", 1000)}

        manager.clear_cache()

        assert manager._cache == {}

    def test_check_secrets_exist_all_present(self):
        """Test validating all required secrets are present"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = "value"
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.SecretsValidationService") as mock_service:
            mock_service.validate_required_secrets.return_value = []  # No missing secrets
            result = manager.check_secrets_exist(["KEY1", "KEY2", "KEY3"])

        assert result is True
        assert mock_provider.get_secret.call_count == 3

    def test_check_secrets_exist_some_missing(self):
        """Test validating with some missing secrets"""
        manager = SecretsManager()
        mock_provider = MagicMock()

        def mock_get_secret(key):
            if key == "KEY2":
                return None
            return "value"

        mock_provider.get_secret.side_effect = mock_get_secret
        mock_provider.check_format.return_value = True
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.SecretsValidationService") as mock_service:
            mock_service.validate_required_secrets.return_value = ["KEY2"]  # KEY2 is missing
            with pytest.raises(SecretNotFoundError) as exc_info:
                manager.check_secrets_exist(["KEY1", "KEY2", "KEY3"])

        assert "Missing required secrets: KEY2" in str(exc_info)

    def test_check_secrets_exist_multiple_missing(self):
        """Test validating with multiple missing secrets"""
        manager = SecretsManager()
        mock_provider = MagicMock()
        mock_provider.get_secret.return_value = None
        mock_provider.check_format.return_value = False
        manager._provider = mock_provider

        with patch("src.infrastructure.security.secrets.SecretsValidationService") as mock_service:
            mock_service.validate_required_secrets.return_value = [
                "KEY1",
                "KEY2",
                "KEY3",
            ]  # All missing
            with pytest.raises(SecretNotFoundError) as exc_info:
                manager.check_secrets_exist(["KEY1", "KEY2", "KEY3"])

        assert "Missing required secrets: KEY1, KEY2, KEY3" in str(exc_info)

    def test_check_secrets_exist_empty_list(self):
        """Test validating with empty list of required secrets"""
        manager = SecretsManager()

        with patch("src.infrastructure.security.secrets.SecretsValidationService") as mock_service:
            mock_service.validate_required_secrets.return_value = []  # No missing secrets
            result = manager.check_secrets_exist([])

        assert result is True


class TestSecretProviderEnum:
    """Test SecretProvider enum"""

    def test_enum_values(self):
        """Test all enum values are defined correctly"""
        assert SecretProvider.ENVIRONMENT == "environment"
        assert SecretProvider.AWS_SECRETS_MANAGER == "aws_secrets_manager"
        assert SecretProvider.HASHICORP_VAULT == "hashicorp_vault"
        assert SecretProvider.AZURE_KEY_VAULT == "azure_key_vault"

    def test_enum_from_string(self):
        """Test creating enum from string value"""
        assert SecretProvider("environment") == SecretProvider.ENVIRONMENT
        assert SecretProvider("aws_secrets_manager") == SecretProvider.AWS_SECRETS_MANAGER

    def test_enum_invalid_value(self):
        """Test invalid enum value raises error"""
        with pytest.raises(ValueError):
            SecretProvider("invalid_provider")


class TestExceptions:
    """Test custom exception classes"""

    def test_secret_not_found_error(self):
        """Test SecretNotFoundError exception"""
        error = SecretNotFoundError("Secret 'API_KEY' not found")

        assert str(error) == "Secret 'API_KEY' not found"
        assert isinstance(error, Exception)

    def test_secret_provider_error(self):
        """Test SecretProviderError exception"""
        error = SecretProviderError("Failed to connect to AWS")

        assert str(error) == "Failed to connect to AWS"
        assert isinstance(error, Exception)
