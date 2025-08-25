"""
Secure secrets management using environment variables and optional cloud providers.

This module provides a secure abstraction for managing sensitive configuration
like database passwords, API keys, and other credentials. It supports multiple
backends including environment variables, AWS Secrets Manager, and HashiCorp Vault.
"""

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeVar

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SecretProvider(Enum):
    """Supported secret storage providers."""

    ENVIRONMENT = "environment"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    AZURE_KEY_VAULT = "azure_key_vault"


class SecretNotFoundError(Exception):
    """Raised when a required secret cannot be found."""

    pass


class SecretProviderError(Exception):
    """Raised when there's an error with the secret provider."""

    pass


class SecretEncryptionError(Exception):
    """Raised when there's an error with secret encryption/decryption."""

    pass


class SecretRateLimitError(Exception):
    """Raised when secret access rate limit is exceeded."""

    pass


@dataclass
class SecretConfig:
    """Configuration for secrets management."""

    provider: SecretProvider = SecretProvider.ENVIRONMENT
    aws_region: str | None = None
    aws_secret_name: str | None = None
    vault_url: str | None = None
    vault_token: str | None = None
    azure_vault_url: str | None = None
    cache_ttl: int = 300  # Cache secrets for 5 minutes by default
    enable_encryption: bool = True  # Enable encryption for sensitive secrets
    encryption_key: str | None = None  # Base64 encoded encryption key
    rate_limit_requests: int = 100  # Max requests per minute
    rate_limit_window: int = 60  # Rate limit window in seconds
    enable_rotation: bool = False  # Enable automatic secret rotation
    rotation_interval: int = 3600  # Rotation interval in seconds
    secret_validation_patterns: dict[str, str] = field(
        default_factory=dict
    )  # Validation patterns for secrets

    @classmethod
    def from_env(cls) -> "SecretConfig":
        """Load configuration from environment variables."""
        provider_str = os.getenv("SECRET_PROVIDER", "environment").lower()

        try:
            provider = SecretProvider(provider_str)
        except ValueError:
            logger.warning(f"Unknown secret provider: {provider_str}, defaulting to environment")
            provider = SecretProvider.ENVIRONMENT

        return cls(
            provider=provider,
            aws_region=os.getenv("AWS_REGION"),
            aws_secret_name=os.getenv("AWS_SECRET_NAME"),
            vault_url=os.getenv("VAULT_URL"),
            vault_token=os.getenv("VAULT_TOKEN"),
            azure_vault_url=os.getenv("AZURE_VAULT_URL"),
            cache_ttl=int(os.getenv("SECRET_CACHE_TTL", "300")),
            enable_encryption=os.getenv("SECRET_ENCRYPTION_ENABLED", "true").lower() == "true",
            encryption_key=os.getenv("SECRET_ENCRYPTION_KEY"),
            rate_limit_requests=int(os.getenv("SECRET_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("SECRET_RATE_LIMIT_WINDOW", "60")),
            enable_rotation=os.getenv("SECRET_ROTATION_ENABLED", "false").lower() == "true",
            rotation_interval=int(os.getenv("SECRET_ROTATION_INTERVAL", "3600")),
        )


class ISecretProvider(ABC):
    """Interface for secret providers."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret value by key."""
        pass

    @abstractmethod
    def get_secrets_batch(self, keys: list[str]) -> dict[str, str | None]:
        """Retrieve multiple secrets at once."""
        pass

    @abstractmethod
    def set_secret(self, key: str, value: str) -> bool:
        """Store a secret (if supported by provider)."""
        pass

    @abstractmethod
    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret (if supported by provider)."""
        pass

    @abstractmethod
    def check_format(self, key: str, value: str) -> bool:
        """Check secret format using domain service."""
        pass


class EnvironmentSecretProvider(ISecretProvider):
    """Secret provider that reads from environment variables."""

    def __init__(self, prefix: str = "") -> None:
        """
        Initialize environment provider.

        Args:
            prefix: Optional prefix for all environment variables
        """
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable."""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        value = os.getenv(env_key)

        if value is None:
            # Try with common prefixes if not found
            for prefix in ["APP_", "TRADING_", ""]:
                alt_key = f"{prefix}{key}"
                value = os.getenv(alt_key)
                if value is not None:
                    break

        return value

    def get_secrets_batch(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple secrets from environment."""
        return {key: self.get_secret(key) for key in keys}

    def set_secret(self, key: str, value: str) -> bool:
        """Set environment variable (for testing only)."""
        env_key = f"{self.prefix}{key}" if self.prefix else key
        os.environ[env_key] = value
        return True

    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret (environment provider doesn't support rotation)."""
        logger.warning("Environment provider does not support secret rotation")
        return False

    def check_format(self, key: str, value: str) -> bool:
        """Delegate format checking to domain service."""
        from src.domain.services.secrets_validation_service import SecretsValidationService

        return SecretsValidationService.validate_secret_format(key, value)


class AWSSecretsManagerProvider(ISecretProvider):
    """Secret provider using AWS Secrets Manager."""

    def __init__(self, region: str, secret_name: str) -> None:
        """
        Initialize AWS Secrets Manager provider.

        Args:
            region: AWS region
            secret_name: Name of the secret in AWS Secrets Manager
        """
        self.region = region
        self.secret_name = secret_name
        self._client = None
        self._cached_secrets: dict[str, str] | None = None

    def _get_client(self) -> Any:
        """Lazy load boto3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("secretsmanager", region_name=self.region)
            except ImportError:
                raise SecretProviderError(
                    "boto3 is required for AWS Secrets Manager. Install with: pip install boto3"
                )
        return self._client

    def _load_secrets(self) -> dict[str, str]:
        """Load all secrets from AWS Secrets Manager."""
        if self._cached_secrets is not None:
            return self._cached_secrets

        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=self.secret_name)

            # Secrets can be stored as JSON or plain text
            if "SecretString" in response:
                secret_string = response["SecretString"]
                try:
                    self._cached_secrets = json.loads(secret_string)
                except json.JSONDecodeError:
                    # If not JSON, treat as single value
                    self._cached_secrets = {"value": secret_string}
            else:
                # Binary secret (not supported for this use case)
                raise SecretProviderError("Binary secrets are not supported")

            return self._cached_secrets

        except Exception as e:
            logger.error(f"Failed to retrieve secrets from AWS: {e}")
            raise SecretProviderError(f"AWS Secrets Manager error: {e}")

    def get_secret(self, key: str) -> str | None:
        """Get secret from AWS Secrets Manager."""
        secrets = self._load_secrets()
        return secrets.get(key)

    def get_secrets_batch(self, keys: list[str]) -> dict[str, str | None]:
        """Get multiple secrets from AWS."""
        secrets = self._load_secrets()
        return {key: secrets.get(key) for key in keys}

    def set_secret(self, key: str, value: str) -> bool:
        """Update secret in AWS Secrets Manager."""
        try:
            secrets = self._load_secrets()
            secrets[key] = value

            client = self._get_client()
            client.update_secret(SecretId=self.secret_name, SecretString=json.dumps(secrets))

            # Clear cache to force reload
            self._cached_secrets = None
            return True

        except Exception as e:
            logger.error(f"Failed to update secret in AWS: {e}")
            return False

    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret in AWS Secrets Manager."""
        try:
            client = self._get_client()
            # Generate new secret value (simplified - in production use proper rotation)
            import secrets

            new_value = secrets.token_urlsafe(32)

            # Update the secret
            result = self.set_secret(key, new_value)
            if result:
                logger.info(f"Successfully rotated secret {key}")
            return result

        except Exception as e:
            logger.error(f"Failed to rotate secret {key} in AWS: {e}")
            return False

    def check_format(self, key: str, value: str) -> bool:
        """Delegate format checking to domain service."""
        from src.domain.services.secrets_validation_service import SecretsValidationService

        return SecretsValidationService.validate_secret_format(key, value)


class SecretEncryption:
    """Handles encryption/decryption of sensitive secrets."""

    def __init__(self, encryption_key: str | None = None) -> None:
        """Initialize encryption with key."""
        if encryption_key:
            self._fernet = Fernet(
                encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
            )
        else:
            # Require proper configuration - no defaults for security
            password = os.getenv("SECRET_MASTER_PASSWORD")
            salt = os.getenv("SECRET_SALT")

            if not password or not salt:
                raise SecretEncryptionError(
                    "SECRET_MASTER_PASSWORD and SECRET_SALT environment variables are required "
                    "for encryption. Please set these securely before running the application."
                )

            # Validate minimum security requirements
            if len(password) < 16:
                raise SecretEncryptionError(
                    "SECRET_MASTER_PASSWORD must be at least 16 characters for security"
                )

            if len(salt) < 16:
                raise SecretEncryptionError(
                    "SECRET_SALT must be at least 16 characters for security"
                )

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self._fernet = Fernet(key)

    def encrypt(self, value: str) -> str:
        """Encrypt a string value."""
        try:
            return self._fernet.encrypt(value.encode()).decode()
        except Exception as e:
            raise SecretEncryptionError(f"Failed to encrypt secret: {e}")

    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value."""
        try:
            return self._fernet.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            raise SecretEncryptionError(f"Failed to decrypt secret: {e}")

    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


class RateLimiter:
    """Rate limiter for secret access."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        client_requests = self.requests[key]

        # Remove old requests outside the window
        client_requests[:] = [
            req_time for req_time in client_requests if now - req_time < self.window_seconds
        ]

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Record this request
        client_requests.append(now)
        return True

    def get_remaining_requests(self, key: str) -> int:
        """Get remaining requests for a key."""
        now = time.time()
        client_requests = self.requests[key]
        client_requests[:] = [
            req_time for req_time in client_requests if now - req_time < self.window_seconds
        ]
        return max(0, self.max_requests - len(client_requests))


class SecretsManager:
    """
    Main secrets manager that provides a unified interface for accessing secrets.

    This class automatically selects the appropriate provider based on configuration
    and provides caching, validation, encryption, rate limiting, and rotation mechanisms.
    """

    _instance: Optional["SecretsManager"] = None

    def __init__(self, config: SecretConfig | None = None) -> None:
        """
        Initialize secrets manager.

        Args:
            config: Secret management configuration
        """
        self.config = config or SecretConfig.from_env()
        self._provider = self._create_provider()
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._encryption: SecretEncryption | None = None
        self._rate_limiter: RateLimiter | None = None
        self._rotation_timestamps: dict[str, float] = {}

        # Initialize encryption if enabled
        if self.config.enable_encryption:
            self._encryption = SecretEncryption(self.config.encryption_key)

        # Initialize rate limiter
        self._rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window,
        )

    @classmethod
    def get_instance(cls) -> "SecretsManager":
        """Get singleton instance of secrets manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _create_provider(self) -> ISecretProvider:
        """Create the appropriate secret provider based on configuration."""
        if self.config.provider == SecretProvider.AWS_SECRETS_MANAGER:
            if not self.config.aws_region or not self.config.aws_secret_name:
                logger.warning(
                    "AWS Secrets Manager configured but missing region/name, falling back to environment"
                )
                return EnvironmentSecretProvider()
            return AWSSecretsManagerProvider(self.config.aws_region, self.config.aws_secret_name)

        # Default to environment provider
        return EnvironmentSecretProvider()

    def get_secret(self, key: str, required: bool = True, decrypt: bool = True) -> str | None:
        """
        Get a secret value with caching, rate limiting, and optional decryption.

        Args:
            key: Secret key
            required: If True, raise exception if secret not found
            decrypt: If True and encryption is enabled, decrypt the secret

        Returns:
            Secret value or None if not found and not required

        Raises:
            SecretNotFoundError: If secret is required but not found
            SecretRateLimitError: If rate limit is exceeded
            SecretEncryptionError: If decryption fails
        """
        # Check rate limit
        if self._rate_limiter and not self._rate_limiter.is_allowed(key):
            raise SecretRateLimitError(f"Rate limit exceeded for secret '{key}'")

        # Check if rotation is needed
        self._check_rotation_needed(key)

        # Try cache
        cached = self._cache.get(key)
        if cached:
            value, timestamp = cached
            expired = time.time() - timestamp >= self.config.cache_ttl
            if not expired:
                result = self._check_required(value, key, required)
                if result and decrypt and self._encryption and self._is_encrypted_secret(key):
                    try:
                        result = self._encryption.decrypt(result)
                    except SecretEncryptionError as e:
                        logger.error(f"Failed to decrypt secret '{key}': {e}")
                        if required:
                            raise
                        return None
                return result

        # Get from provider and cache
        value = self._provider.get_secret(key)
        if value:
            # Check secret format
            if not self._provider.check_format(key, value):
                logger.warning(f"Secret '{key}' failed format check")

        self._cache[key] = (value, time.time())

        result = self._check_required(value, key, required)
        if result and decrypt and self._encryption and self._is_encrypted_secret(key):
            try:
                result = self._encryption.decrypt(result)
            except SecretEncryptionError as e:
                logger.error(f"Failed to decrypt secret '{key}': {e}")
                if required:
                    raise
                return None

        return result

    def _check_required(self, value: str | None, key: str, required: bool) -> str | None:
        """Simple required check helper."""
        if required and value is None:
            raise SecretNotFoundError(f"Required secret '{key}' not found")
        return value

    def get_database_config(self) -> dict[str, Any]:
        """Get database configuration from secrets - simple retrieval."""
        from src.domain.services.secrets_validation_service import SecretsValidationService

        # Simple retrieval of secrets
        config = {
            "host": self.get_secret("DB_HOST", required=False),
            "port": self.get_secret("DB_PORT", required=False),
            "database": self.get_secret("DB_NAME", required=False),
            "user": self.get_secret("DB_USER", required=True),
            "password": self.get_secret("DB_PASSWORD", required=True),
        }

        # Use domain service for applying defaults and validation
        return SecretsValidationService.apply_defaults_to_database_config(config)

    def get_broker_config(self) -> dict[str, Any]:
        """Get broker configuration from secrets - simple retrieval."""
        from src.domain.services.secrets_validation_service import SecretsValidationService

        # Simple retrieval of secrets
        config = {
            "alpaca_api_key": self.get_secret("ALPACA_API_KEY", required=False),
            "alpaca_api_secret": self.get_secret("ALPACA_API_SECRET", required=False),
            "alpaca_base_url": self.get_secret("ALPACA_BASE_URL", required=False),
            "polygon_api_key": self.get_secret("POLYGON_API_KEY", required=False),
        }

        # Use domain service for applying defaults
        return SecretsValidationService.apply_defaults_to_broker_config("alpaca", config)

    def set_secret(self, key: str, value: str, encrypt: bool = True) -> bool:
        """
        Set a secret value with optional encryption.

        Args:
            key: Secret key
            value: Secret value
            encrypt: If True and encryption is enabled, encrypt the secret

        Returns:
            True if successful

        Raises:
            SecretEncryptionError: If encryption fails
        """
        # Check secret format
        if not self._provider.check_format(key, value):
            logger.warning(f"Secret '{key}' failed format check")
            return False

        # Encrypt if needed
        if encrypt and self._encryption and self._is_encrypted_secret(key):
            try:
                value = self._encryption.encrypt(value)
            except SecretEncryptionError as e:
                logger.error(f"Failed to encrypt secret '{key}': {e}")
                raise

        # Store in provider
        result = self._provider.set_secret(key, value)

        # Clear cache for this key
        if key in self._cache:
            del self._cache[key]

        return result

    def rotate_secret(self, key: str) -> bool:
        """
        Rotate a secret and update rotation timestamp.

        Args:
            key: Secret key to rotate

        Returns:
            True if successful
        """
        if not self.config.enable_rotation:
            logger.warning("Secret rotation is disabled")
            return False

        result = self._provider.rotate_secret(key)
        if result:
            self._rotation_timestamps[key] = time.time()
            # Clear cache for this key
            if key in self._cache:
                del self._cache[key]

        return result

    def _check_rotation_needed(self, key: str) -> None:
        """Check if secret needs rotation and rotate if necessary."""
        if not self.config.enable_rotation:
            return

        last_rotation = self._rotation_timestamps.get(key, 0)
        if time.time() - last_rotation >= self.config.rotation_interval:
            try:
                self.rotate_secret(key)
            except Exception as e:
                logger.error(f"Failed to auto-rotate secret '{key}': {e}")

    def _is_encrypted_secret(self, key: str) -> bool:
        """Check if a secret should be encrypted using domain service."""
        from src.domain.services.secrets_validation_service import SecretsValidationService

        return SecretsValidationService.should_encrypt_secret(key)

    def get_rate_limit_status(self, key: str) -> dict[str, int]:
        """Get rate limit status for a key."""
        if not self._rate_limiter:
            return {"remaining": -1, "limit": -1}

        return {
            "remaining": self._rate_limiter.get_remaining_requests(key),
            "limit": self.config.rate_limit_requests,
        }

    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._cache.clear()

    def clear_rate_limits(self) -> None:
        """Clear all rate limit counters."""
        if self._rate_limiter:
            self._rate_limiter.requests.clear()

    def get_secrets_batch(self, keys: list[str], decrypt: bool = True) -> dict[str, str | None]:
        """Get multiple secrets with rate limiting and decryption."""
        result = {}
        for key in keys:
            try:
                result[key] = self.get_secret(key, required=False, decrypt=decrypt)
            except (SecretRateLimitError, SecretEncryptionError) as e:
                logger.error(f"Failed to get secret '{key}': {e}")
                result[key] = None
        return result

    def check_secrets_exist(self, required_keys: list[str]) -> bool:
        """
        Check that required secrets exist with validation.

        Args:
            required_keys: List of required secret keys

        Returns:
            True if all secrets are present and valid

        Raises:
            SecretNotFoundError: If any required secret is missing
        """
        from src.domain.services.secrets_validation_service import SecretsValidationService

        # Get all secrets
        secrets = {key: self.get_secret(key, required=False, decrypt=True) for key in required_keys}

        # Delegate to domain service for validation
        missing = SecretsValidationService.validate_required_secrets(secrets, required_keys)

        if missing:
            raise SecretNotFoundError(f"Missing required secrets: {', '.join(missing)}")

        return True

    def check_all_formats(self) -> dict[str, bool]:
        """
        Check all cached secrets against their format requirements.

        Returns:
            Dictionary mapping secret keys to format check status
        """
        check_results = {}
        for key, (value, _) in self._cache.items():
            if value is not None:
                check_results[key] = self._provider.check_format(key, value)
            else:
                check_results[key] = False
        return check_results

    def get_encryption_status(self) -> dict[str, Any]:
        """
        Get encryption status and configuration.

        Returns:
            Dictionary with encryption configuration
        """
        return {
            "encryption_enabled": self.config.enable_encryption,
            "rotation_enabled": self.config.enable_rotation,
            "rotation_interval": self.config.rotation_interval,
            "cache_ttl": self.config.cache_ttl,
            "rate_limit_requests": self.config.rate_limit_requests,
            "provider": self.config.provider.value,
        }
