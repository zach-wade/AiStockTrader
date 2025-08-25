"""
Comprehensive unit tests for Secrets Management.

Tests the secrets management system including multiple providers,
caching, configuration, and error handling with full coverage.
"""

# Standard library imports
import json
import os
import time
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


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    env_vars = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "testdb",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
        "ALPACA_API_KEY": "test_api_key",
        "ALPACA_API_SECRET": "test_api_secret",
        "SECRET_PROVIDER": "environment",
        "AWS_REGION": "us-east-1",
        "AWS_SECRET_NAME": "test-secret",
        "VAULT_URL": "https://vault.example.com",
        "VAULT_TOKEN": "vault-token",
        "AZURE_VAULT_URL": "https://keyvault.azure.com",
        "SECRET_CACHE_TTL": "60",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def secret_config():
    """Sample secret configuration."""
    return SecretConfig(
        provider=SecretProvider.ENVIRONMENT,
        aws_region="us-east-1",
        aws_secret_name="test-secret",
        cache_ttl=60,
    )


@pytest.fixture
def env_provider():
    """Environment secret provider."""
    return EnvironmentSecretProvider()


@pytest.fixture
def aws_provider():
    """AWS Secrets Manager provider."""
    return AWSSecretsManagerProvider("us-east-1", "test-secret")


@pytest.mark.unit
class TestSecretConfig:
    """Test secret configuration."""

    def test_secret_provider_enum(self):
        """Test secret provider enum values."""
        assert SecretProvider.ENVIRONMENT == "environment"
        assert SecretProvider.AWS_SECRETS_MANAGER == "aws_secrets_manager"
        assert SecretProvider.HASHICORP_VAULT == "hashicorp_vault"
        assert SecretProvider.AZURE_KEY_VAULT == "azure_key_vault"

    def test_secret_config_defaults(self):
        """Test secret config default values."""
        config = SecretConfig()

        assert config.provider == SecretProvider.ENVIRONMENT
        assert config.aws_region is None
        assert config.aws_secret_name is None
        assert config.vault_url is None
        assert config.vault_token is None
        assert config.azure_vault_url is None
        assert config.cache_ttl == 300

    def test_secret_config_from_env(self, mock_env):
        """Test loading secret config from environment."""
        config = SecretConfig.from_env()

        assert config.provider == SecretProvider.ENVIRONMENT
        assert config.aws_region == "us-east-1"
        assert config.aws_secret_name == "test-secret"
        assert config.vault_url == "https://vault.example.com"
        assert config.vault_token == "vault-token"
        assert config.azure_vault_url == "https://keyvault.azure.com"
        assert config.cache_ttl == 60

    def test_secret_config_from_env_unknown_provider(self):
        """Test loading config with unknown provider."""
        with patch.dict(os.environ, {"SECRET_PROVIDER": "unknown"}, clear=True):
            config = SecretConfig.from_env()
            assert config.provider == SecretProvider.ENVIRONMENT

    def test_secret_config_from_env_aws_provider(self):
        """Test loading config with AWS provider."""
        with patch.dict(os.environ, {"SECRET_PROVIDER": "aws_secrets_manager"}, clear=True):
            config = SecretConfig.from_env()
            assert config.provider == SecretProvider.AWS_SECRETS_MANAGER

    def test_secret_config_from_env_no_cache_ttl(self):
        """Test loading config without cache TTL."""
        with patch.dict(os.environ, {}, clear=True):
            config = SecretConfig.from_env()
            assert config.cache_ttl == 300  # Default value


@pytest.mark.unit
class TestEnvironmentSecretProvider:
    """Test environment secret provider."""

    def test_get_secret_exists(self, env_provider, mock_env):
        """Test getting existing secret from environment."""
        result = env_provider.get_secret("DB_HOST")
        assert result == "localhost"

    def test_get_secret_not_exists(self, env_provider):
        """Test getting non-existent secret."""
        with patch.dict(os.environ, {}, clear=True):
            result = env_provider.get_secret("NONEXISTENT")
            assert result is None

    def test_get_secret_with_prefix(self, mock_env):
        """Test getting secret with provider prefix."""
        provider = EnvironmentSecretProvider(prefix="DB_")
        result = provider.get_secret("USER")
        assert result == "testuser"

    def test_get_secret_with_fallback_prefixes(self):
        """Test getting secret with fallback prefixes."""
        with patch.dict(os.environ, {"APP_TEST_KEY": "app_value"}, clear=True):
            provider = EnvironmentSecretProvider()
            result = provider.get_secret("TEST_KEY")
            assert result == "app_value"

        with patch.dict(os.environ, {"TRADING_TEST_KEY": "trading_value"}, clear=True):
            provider = EnvironmentSecretProvider()
            result = provider.get_secret("TEST_KEY")
            assert result == "trading_value"

    def test_get_secrets_batch(self, env_provider, mock_env):
        """Test getting multiple secrets at once."""
        keys = ["DB_HOST", "DB_PORT", "DB_NAME", "NONEXISTENT"]
        results = env_provider.get_secrets_batch(keys)

        assert results["DB_HOST"] == "localhost"
        assert results["DB_PORT"] == "5432"
        assert results["DB_NAME"] == "testdb"
        assert results["NONEXISTENT"] is None

    def test_set_secret(self, env_provider):
        """Test setting environment variable."""
        result = env_provider.set_secret("TEST_SECRET", "test_value")

        assert result is True
        assert os.environ["TEST_SECRET"] == "test_value"

        # Cleanup
        del os.environ["TEST_SECRET"]

    def test_set_secret_with_prefix(self):
        """Test setting secret with prefix."""
        provider = EnvironmentSecretProvider(prefix="TEST_")
        result = provider.set_secret("KEY", "value")

        assert result is True
        assert os.environ["TEST_KEY"] == "value"

        # Cleanup
        del os.environ["TEST_KEY"]


@pytest.mark.unit
class TestAWSSecretsManagerProvider:
    """Test AWS Secrets Manager provider."""

    def test_initialization(self, aws_provider):
        """Test AWS provider initialization."""
        assert aws_provider.region == "us-east-1"
        assert aws_provider.secret_name == "test-secret"
        assert aws_provider._client is None
        assert aws_provider._cached_secrets is None

    def test_get_client_boto3_not_installed(self, aws_provider):
        """Test getting client when boto3 is not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(SecretProviderError, match="boto3 is required"):
                aws_provider._get_client()

    def test_get_client_success(self, aws_provider):
        """Test successful client creation."""
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch("src.infrastructure.security.secrets.boto3", mock_boto3):
            client = aws_provider._get_client()

            assert client is mock_client
            mock_boto3.client.assert_called_once_with("secretsmanager", region_name="us-east-1")

            # Second call should return cached client
            client2 = aws_provider._get_client()
            assert client2 is mock_client
            assert mock_boto3.client.call_count == 1

    def test_load_secrets_json_format(self, aws_provider):
        """Test loading secrets in JSON format."""
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "db_password": "secret123",
                    "api_key": "key456",
                }
            )
        }

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            secrets = aws_provider._load_secrets()

            assert secrets["db_password"] == "secret123"
            assert secrets["api_key"] == "key456"

            # Second call should return cached secrets
            secrets2 = aws_provider._load_secrets()
            assert secrets2 is secrets
            assert mock_client.get_secret_value.call_count == 1

    def test_load_secrets_plain_text(self, aws_provider):
        """Test loading secrets in plain text format."""
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "plain_secret_value"}

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            secrets = aws_provider._load_secrets()

            assert secrets["value"] == "plain_secret_value"

    def test_load_secrets_binary_not_supported(self, aws_provider):
        """Test that binary secrets are not supported."""
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretBinary": b"binary_data"}

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            with pytest.raises(SecretProviderError, match="Binary secrets are not supported"):
                aws_provider._load_secrets()

    def test_load_secrets_error(self, aws_provider):
        """Test error handling when loading secrets."""
        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("AWS error")

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            with pytest.raises(SecretProviderError, match="AWS Secrets Manager error"):
                aws_provider._load_secrets()

    def test_get_secret(self, aws_provider):
        """Test getting single secret from AWS."""
        with patch.object(
            aws_provider, "_load_secrets", return_value={"key1": "value1", "key2": "value2"}
        ):
            result = aws_provider.get_secret("key1")
            assert result == "value1"

            result = aws_provider.get_secret("nonexistent")
            assert result is None

    def test_get_secrets_batch(self, aws_provider):
        """Test getting multiple secrets from AWS."""
        with patch.object(
            aws_provider, "_load_secrets", return_value={"key1": "value1", "key2": "value2"}
        ):
            results = aws_provider.get_secrets_batch(["key1", "key2", "key3"])

            assert results["key1"] == "value1"
            assert results["key2"] == "value2"
            assert results["key3"] is None

    def test_set_secret_success(self, aws_provider):
        """Test setting secret in AWS."""
        mock_client = MagicMock()
        aws_provider._cached_secrets = {"existing": "value"}

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            with patch.object(aws_provider, "_load_secrets", return_value={"existing": "value"}):
                result = aws_provider.set_secret("new_key", "new_value")

                assert result is True
                mock_client.update_secret.assert_called_once()
                call_args = mock_client.update_secret.call_args
                assert call_args[1]["SecretId"] == "test-secret"
                secret_data = json.loads(call_args[1]["SecretString"])
                assert secret_data["new_key"] == "new_value"
                assert aws_provider._cached_secrets is None  # Cache cleared

    def test_set_secret_error(self, aws_provider):
        """Test error handling when setting secret."""
        mock_client = MagicMock()
        mock_client.update_secret.side_effect = Exception("Update failed")

        with patch.object(aws_provider, "_get_client", return_value=mock_client):
            with patch.object(aws_provider, "_load_secrets", return_value={}):
                result = aws_provider.set_secret("key", "value")
                assert result is False


@pytest.mark.unit
class TestSecretsManager:
    """Test main secrets manager."""

    def test_singleton_instance(self):
        """Test singleton pattern."""
        # Clear any existing instance
        SecretsManager._instance = None

        instance1 = SecretsManager.get_instance()
        instance2 = SecretsManager.get_instance()

        assert instance1 is instance2

    def test_initialization_with_config(self, secret_config):
        """Test initialization with provided config."""
        manager = SecretsManager(config=secret_config)

        assert manager.config is secret_config
        assert isinstance(manager._provider, EnvironmentSecretProvider)
        assert manager._cache == {}

    def test_initialization_without_config(self):
        """Test initialization with default config."""
        with patch.object(SecretConfig, "from_env") as mock_from_env:
            mock_from_env.return_value = SecretConfig()
            manager = SecretsManager()

            mock_from_env.assert_called_once()
            assert isinstance(manager._provider, EnvironmentSecretProvider)

    def test_create_provider_environment(self, secret_config):
        """Test creating environment provider."""
        manager = SecretsManager(config=secret_config)
        provider = manager._create_provider()

        assert isinstance(provider, EnvironmentSecretProvider)

    def test_create_provider_aws(self):
        """Test creating AWS provider."""
        config = SecretConfig(
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            aws_region="us-east-1",
            aws_secret_name="test-secret",
        )
        manager = SecretsManager(config=config)
        provider = manager._create_provider()

        assert isinstance(provider, AWSSecretsManagerProvider)

    def test_create_provider_aws_missing_config(self):
        """Test AWS provider fallback when config is missing."""
        config = SecretConfig(
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            aws_region=None,
            aws_secret_name=None,
        )
        manager = SecretsManager(config=config)
        provider = manager._create_provider()

        assert isinstance(provider, EnvironmentSecretProvider)

    def test_get_secret_from_cache(self, secret_config):
        """Test getting secret from cache."""
        manager = SecretsManager(config=secret_config)
        manager._cache["test_key"] = ("cached_value", time.time())

        with patch.object(manager._provider, "get_secret") as mock_get:
            result = manager.get_secret("test_key", required=False, decrypt=False)

            assert result == "cached_value"
            mock_get.assert_not_called()

    def test_get_secret_cache_expired(self, secret_config):
        """Test getting secret when cache is expired."""
        manager = SecretsManager(config=secret_config)
        manager._cache["test_secret"] = (
            "old_value_long_enough_1234567890",
            time.time() - 3600,
        )  # Old timestamp

        with patch.object(
            manager._provider, "get_secret", return_value="new_value_long_enough_1234567890"
        ):
            result = manager.get_secret("test_secret", required=False, decrypt=False)

            assert result == "new_value_long_enough_1234567890"
            assert manager._cache["test_secret"][0] == "new_value_long_enough_1234567890"

    def test_get_secret_not_cached(self, secret_config):
        """Test getting secret not in cache."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager._provider, "get_secret", return_value="fetched_value"):
            result = manager.get_secret("test_key", required=False)

            assert result == "fetched_value"
            assert "test_key" in manager._cache
            assert manager._cache["test_key"][0] == "fetched_value"

    def test_get_secret_required_missing(self, secret_config):
        """Test getting required secret that doesn't exist."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager._provider, "get_secret", return_value=None):
            with pytest.raises(
                SecretNotFoundError, match="Required secret 'missing_key' not found"
            ):
                manager.get_secret("missing_key", required=True)

    def test_get_secret_not_required_missing(self, secret_config):
        """Test getting non-required secret that doesn't exist."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager._provider, "get_secret", return_value=None):
            result = manager.get_secret("missing_key", required=False)
            assert result is None

    def test_check_required(self, secret_config):
        """Test _check_required helper method."""
        manager = SecretsManager(config=secret_config)

        # Required and present
        result = manager._check_required("value", "key", True)
        assert result == "value"

        # Not required and missing
        result = manager._check_required(None, "key", False)
        assert result is None

        # Required and missing
        with pytest.raises(SecretNotFoundError):
            manager._check_required(None, "key", True)

    def test_get_database_config(self, secret_config):
        """Test getting database configuration."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager, "get_secret") as mock_get:
            mock_get.side_effect = ["localhost", "5432", "testdb", "user", "password"]

            with patch(
                "src.domain.services.secrets_validation_service.SecretsValidationService"
            ) as mock_service:
                mock_service.apply_defaults_to_database_config.return_value = {
                    "host": "localhost",
                    "port": 5432,
                    "database": "testdb",
                    "user": "user",
                    "password": "password",
                }

                config = manager.get_database_config()

                assert config["host"] == "localhost"
                assert config["port"] == 5432
                assert config["user"] == "user"
                assert config["password"] == "password"

    def test_get_broker_config(self, secret_config):
        """Test getting broker configuration."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager, "get_secret") as mock_get:
            mock_get.side_effect = [
                "api_key",
                "api_secret",
                "https://paper-api.alpaca.markets",
                "polygon_key",
            ]

            with patch(
                "src.domain.services.secrets_validation_service.SecretsValidationService"
            ) as mock_service:
                mock_service.apply_defaults_to_broker_config.return_value = {
                    "alpaca_api_key": "api_key",
                    "alpaca_api_secret": "api_secret",
                    "alpaca_base_url": "https://paper-api.alpaca.markets",
                    "polygon_api_key": "polygon_key",
                }

                config = manager.get_broker_config()

                assert config["alpaca_api_key"] == "api_key"
                assert config["alpaca_api_secret"] == "api_secret"

    def test_clear_cache(self, secret_config):
        """Test clearing the cache."""
        manager = SecretsManager(config=secret_config)
        manager._cache = {"key1": ("value1", time.time()), "key2": ("value2", time.time())}

        manager.clear_cache()

        assert manager._cache == {}

    def test_check_secrets_exist_all_present(self, secret_config):
        """Test checking secrets when all are present."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager, "get_secret", return_value="value"):
            with patch(
                "src.domain.services.secrets_validation_service.SecretsValidationService"
            ) as mock_service:
                mock_service.validate_required_secrets.return_value = []  # No missing

                result = manager.check_secrets_exist(["key1", "key2"])
                assert result is True

    def test_check_secrets_exist_some_missing(self, secret_config):
        """Test checking secrets when some are missing."""
        manager = SecretsManager(config=secret_config)

        with patch.object(manager, "get_secret", side_effect=[None, "value"]):
            with patch(
                "src.domain.services.secrets_validation_service.SecretsValidationService"
            ) as mock_service:
                mock_service.validate_required_secrets.return_value = ["key1"]  # Missing

                with pytest.raises(SecretNotFoundError, match="Missing required secrets: key1"):
                    manager.check_secrets_exist(["key1", "key2"])


@pytest.mark.unit
class TestExceptions:
    """Test custom exceptions."""

    def test_secret_not_found_error(self):
        """Test SecretNotFoundError exception."""
        error = SecretNotFoundError("Secret 'test' not found")
        assert str(error) == "Secret 'test' not found"

    def test_secret_provider_error(self):
        """Test SecretProviderError exception."""
        error = SecretProviderError("Provider failed")
        assert str(error) == "Provider failed"
