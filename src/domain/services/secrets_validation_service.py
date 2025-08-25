"""
Secrets Validation Service - Domain service for secrets validation logic.

This service handles business logic related to secrets validation,
moved from the infrastructure layer to maintain clean architecture.
"""

from typing import Any


class SecretsValidationService:
    """
    Domain service for secrets validation logic.

    This service contains business rules for validating secrets and configurations
    that was previously in the infrastructure layer.
    """

    # Business rules for secret validation
    REQUIRED_DB_SECRETS = ["DB_USER", "DB_PASSWORD"]
    OPTIONAL_DB_SECRETS = ["DB_HOST", "DB_PORT", "DB_NAME"]

    REQUIRED_ALPACA_SECRETS = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    OPTIONAL_ALPACA_SECRETS = ["ALPACA_BASE_URL"]

    @staticmethod
    def validate_required_secrets(
        secrets: dict[str, str | None], required_keys: list[str]
    ) -> list[str]:
        """
        Validate that required secrets are present.

        This is business logic about which secrets are required.

        Args:
            secrets: Dictionary of secret values
            required_keys: List of required secret keys

        Returns:
            List of missing secret keys (empty if all present)
        """
        missing = []
        for key in required_keys:
            value = secrets.get(key)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(key)
        return missing

    @staticmethod
    def validate_database_config(config: dict[str, Any]) -> bool:
        """
        Validate database configuration according to business rules.

        Args:
            config: Database configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Required fields
        if not config.get("user"):
            raise ValueError("Database user is required")
        if not config.get("password"):
            raise ValueError("Database password is required")

        # Validate port if provided
        port = config.get("port")
        if port is not None:
            try:
                port_num = int(port)
                if port_num < 1 or port_num > 65535:
                    raise ValueError(f"Invalid database port: {port_num}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid database port: {port}") from e

        return True

    @staticmethod
    def validate_broker_secrets(broker_type: str, secrets: dict[str, str | None]) -> list[str]:
        """
        Validate broker-specific secrets.

        Args:
            broker_type: Type of broker
            secrets: Dictionary of secret values

        Returns:
            List of missing required secrets
        """
        if broker_type.lower() == "alpaca":
            return SecretsValidationService.validate_required_secrets(
                secrets, SecretsValidationService.REQUIRED_ALPACA_SECRETS
            )

        # Paper and backtest brokers don't require secrets
        return []

    @staticmethod
    def apply_defaults_to_database_config(config: dict[str, Any]) -> dict[str, Any]:
        """
        Apply default values to database configuration.

        This encapsulates business rules about default values.

        Args:
            config: Database configuration dictionary

        Returns:
            Configuration with defaults applied
        """
        defaults = {"host": "localhost", "port": 5432, "database": "ai_trader"}

        result = defaults.copy()
        result.update(config)

        # Ensure port is an integer
        if "port" in result:
            try:
                port_value = result["port"]
                if isinstance(port_value, (str, int, float)):
                    result["port"] = int(port_value)
                else:
                    result["port"] = defaults["port"]
            except (ValueError, TypeError):
                result["port"] = defaults["port"]

        return result

    @staticmethod
    def apply_defaults_to_broker_config(broker_type: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Apply default values to broker configuration.

        Args:
            broker_type: Type of broker
            config: Broker configuration dictionary

        Returns:
            Configuration with defaults applied
        """
        result = config.copy()

        if broker_type.lower() == "alpaca":
            # Default to paper trading URL if not specified
            if "alpaca_base_url" not in result or not result["alpaca_base_url"]:
                result["alpaca_base_url"] = "https://paper-api.alpaca.markets"

        return result

    @staticmethod
    def sanitize_secret_for_logging(key: str, value: str | None) -> str:
        """
        Sanitize a secret value for safe logging.

        This ensures sensitive values are not exposed in logs.

        Args:
            key: Secret key
            value: Secret value

        Returns:
            Sanitized string safe for logging
        """
        if value is None:
            return f"{key}=<not set>"

        # List of keys that should be fully masked
        sensitive_keys = ["PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL", "PRIVATE", "API"]

        # Check if key contains sensitive words
        key_upper = key.upper()
        for sensitive in sensitive_keys:
            if sensitive in key_upper:
                # Mask all but first 2 characters
                if len(value) > 2:
                    return f"{key}={value[:2]}{'*' * (len(value) - 2)}"
                else:
                    return f"{key}={'*' * len(value)}"

        # For non-sensitive keys, show partial value
        if len(value) > 10:
            return f"{key}={value[:5]}...{value[-2:]}"

        return f"{key}={value}"

    @staticmethod
    def validate_secret_format(key: str, value: str) -> bool:
        """
        Validate secret format according to business rules.

        This method contains the business logic for determining if a secret
        value meets the requirements for its type.

        Args:
            key: Secret key (used to determine type)
            value: Secret value to validate

        Returns:
            True if the secret format is valid, False otherwise
        """
        if not value:
            return False

        key_upper = key.upper()

        # Business rules for API keys
        if key_upper.endswith("_KEY") or "API_KEY" in key_upper:
            # API keys should be at least 20 characters
            if len(value) < 20:
                return False
            # Should contain mix of letters and numbers/symbols
            has_letter = any(c.isalpha() for c in value)
            has_other = any(not c.isalpha() for c in value)
            if not (has_letter and has_other):
                return False

        # Business rules for tokens
        if key_upper.endswith("_TOKEN") or "TOKEN" in key_upper:
            # Tokens should be at least 16 characters
            if len(value) < 16:
                return False

        # Business rules for passwords
        if "PASSWORD" in key_upper or "PASSWD" in key_upper:
            # Passwords should be at least 8 characters
            if len(value) < 8:
                return False

        # Business rules for secrets
        if key_upper.endswith("_SECRET") or "SECRET" in key_upper:
            # Secrets should be at least 16 characters
            if len(value) < 16:
                return False

        # Business rules for database configuration
        if key_upper.startswith("DB_"):
            if key_upper == "DB_PORT":
                # Port should be numeric
                try:
                    port = int(value)
                    if port < 1 or port > 65535:
                        return False
                except ValueError:
                    return False
            elif key_upper == "DB_HOST":
                # Host should not be empty and should be valid
                if len(value) < 1:
                    return False
            elif key_upper in ["DB_USER", "DB_NAME", "DB_DATABASE"]:
                # Database identifiers should be at least 1 character
                if len(value) < 1:
                    return False

        # Business rules for URLs
        if key_upper.endswith("_URL") or "URL" in key_upper:
            # Should start with http:// or https://
            if not (value.startswith("http://") or value.startswith("https://")):
                return False
            # Should have more than just the protocol
            if len(value) < 10:
                return False

        # Business rules for broker-specific secrets
        if "ALPACA" in key_upper:
            if "API_KEY" in key_upper or "API_SECRET" in key_upper:
                # Alpaca keys are typically 20+ characters
                if len(value) < 20:
                    return False

        if "POLYGON" in key_upper and "API_KEY" in key_upper:
            # Polygon API keys are typically 32 characters
            if len(value) < 30:
                return False

        # Default: consider valid if we don't have specific rules
        return True

    @staticmethod
    def get_secret_type(key: str) -> str:
        """
        Determine the type of secret based on its key.

        This is a business rule about categorizing secrets.

        Args:
            key: Secret key

        Returns:
            Type of secret (e.g., 'api_key', 'password', 'token', 'url', 'generic')
        """
        key_upper = key.upper()

        if "PASSWORD" in key_upper or "PASSWD" in key_upper:
            return "password"
        elif key_upper.endswith("_KEY") or "API_KEY" in key_upper:
            return "api_key"
        elif key_upper.endswith("_TOKEN") or "TOKEN" in key_upper:
            return "token"
        elif key_upper.endswith("_SECRET") or "SECRET" in key_upper:
            return "secret"
        elif key_upper.endswith("_URL") or "URL" in key_upper:
            return "url"
        elif key_upper.startswith("DB_"):
            return "database"
        else:
            return "generic"

    @staticmethod
    def should_encrypt_secret(key: str) -> bool:
        """
        Determine if a secret should be encrypted based on its type.

        This is a business rule about which secrets require encryption.

        Args:
            key: Secret key

        Returns:
            True if the secret should be encrypted
        """
        secret_type = SecretsValidationService.get_secret_type(key)

        # Business rule: These types should always be encrypted
        encrypted_types = ["password", "api_key", "token", "secret"]

        return secret_type in encrypted_types
