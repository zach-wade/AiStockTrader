"""
Comprehensive tests for SecretsValidationService
"""

import pytest

from src.domain.services.secrets_validation_service import SecretsValidationService


class TestSecretsValidationService:
    """Test suite for SecretsValidationService"""

    @pytest.fixture
    def service(self):
        """Create a SecretsValidationService instance"""
        return SecretsValidationService()

    def test_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "validate_secret_format")
        assert hasattr(service, "get_secret_type")

    def test_validate_api_key_valid(self, service):
        """Test validation of valid API keys"""
        # Valid API key (20+ chars)
        valid_key = "abcdefghij1234567890"
        assert service.validate_secret_format("api_key", valid_key) is True
        assert service.validate_secret_format("alpaca_api_key", valid_key) is True

    def test_validate_api_key_invalid(self, service):
        """Test validation of invalid API keys"""
        # Too short
        with pytest.raises(SecretValidationError) as exc:
            service.validate_secret_format("api_key", "short")
        assert "at least 20 characters" in str(exc)

        # Empty
        with pytest.raises(SecretValidationError) as exc:
            service.validate_secret_format("api_key", "")
        assert "at least 20 characters" in str(exc)

    def test_validate_secret_key_valid(self, service):
        """Test validation of valid secret keys"""
        valid_secret = "abcdefghij1234567890abcdefghij12"  # 32+ chars
        assert service.validate_secret_format("api_secret", valid_secret) is True
        assert service.validate_secret_format("secret_key", valid_secret) is True

    def test_validate_secret_key_invalid(self, service):
        """Test validation of invalid secret keys"""
        # Too short
        with pytest.raises(SecretValidationError) as exc:
            service.validate_secret_format("api_secret", "short")
        assert "at least 32 characters" in str(exc)

    def test_validate_token_valid(self, service):
        """Test validation of valid tokens"""
        valid_token = "1234567890abcdef"  # 16+ chars
        assert service.validate_secret_format("access_token", valid_token) is True
        assert service.validate_secret_format("refresh_token", valid_token) is True

    def test_validate_token_invalid(self, service):
        """Test validation of invalid tokens"""
        with pytest.raises(SecretValidationError) as exc:
            service.validate_secret_format("access_token", "short")
        assert "at least 16 characters" in str(exc)

    def test_validate_password_valid(self, service):
        """Test validation of valid passwords"""
        valid_password = "MyP@ssw0rd"  # 8+ chars
        assert service.validate_secret_format("password", valid_password) is True
        assert service.validate_secret_format("db_password", valid_password) is True

    def test_validate_password_invalid(self, service):
        """Test validation of invalid passwords"""
        with pytest.raises(SecretValidationError) as exc:
            service.validate_secret_format("password", "short")
        assert "at least 8 characters" in str(exc)

    def test_validate_url_valid(self, service):
        """Test validation of valid URLs"""
        valid_urls = [
            "https://api.example.com",
            "http://localhost:8080",
            "postgresql://user:pass@localhost/db",
            "redis://localhost:6379",
        ]
        for url in valid_urls:
            assert service.validate_secret_format("base_url", url) is True
            assert service.validate_secret_format("api_url", url) is True

    def test_validate_url_invalid(self, service):
        """Test validation of invalid URLs"""
        invalid_urls = [
            "not-a-url",
            "ftp://invalid",  # Not http/https/postgresql/redis
            "http://",  # Incomplete
            "",
        ]
        for url in invalid_urls:
            with pytest.raises(SecretValidationError):
                service.validate_secret_format("base_url", url)

    def test_validate_database_config(self, service):
        """Test validation of database configuration"""
        valid_config = {
            "host": "localhost",
            "port": 5432,
            "database": "trading_db",
            "user": "trader",
            "password": "SecurePass123",
        }
        assert service.validate_database_config(valid_config) is True

    def test_validate_database_config_missing_fields(self, service):
        """Test validation of incomplete database config"""
        invalid_config = {
            "host": "localhost",
            "port": 5432,
            # Missing database, user, password
        }
        with pytest.raises(SecretValidationError) as exc:
            service.validate_database_config(invalid_config)
        assert "Missing required database configuration" in str(exc)

    def test_validate_database_config_invalid_port(self, service):
        """Test validation of database config with invalid port"""
        invalid_config = {
            "host": "localhost",
            "port": -1,  # Invalid port
            "database": "trading_db",
            "user": "trader",
            "password": "SecurePass123",
        }
        with pytest.raises(SecretValidationError) as exc:
            service.validate_database_config(invalid_config)
        assert "Invalid port" in str(exc)

    def test_get_secret_type_api_key(self, service):
        """Test identifying API key type secrets"""
        api_keys = ["api_key", "alpaca_api_key", "polygon_api_key"]
        for key in api_keys:
            assert service.get_secret_type(key) == SecretType.API_KEY

    def test_get_secret_type_secret_key(self, service):
        """Test identifying secret key type secrets"""
        secret_keys = ["api_secret", "secret_key", "alpaca_secret"]
        for key in secret_keys:
            assert service.get_secret_type(key) == SecretType.SECRET_KEY

    def test_get_secret_type_token(self, service):
        """Test identifying token type secrets"""
        tokens = ["access_token", "refresh_token", "auth_token"]
        for token in tokens:
            assert service.get_secret_type(token) == SecretType.TOKEN

    def test_get_secret_type_password(self, service):
        """Test identifying password type secrets"""
        passwords = ["password", "db_password", "redis_password"]
        for password in passwords:
            assert service.get_secret_type(password) == SecretType.PASSWORD

    def test_get_secret_type_url(self, service):
        """Test identifying URL type secrets"""
        urls = ["base_url", "api_url", "webhook_url"]
        for url in urls:
            assert service.get_secret_type(url) == SecretType.URL

    def test_get_secret_type_config(self, service):
        """Test identifying config type secrets"""
        configs = ["database_config", "broker_config", "redis_config"]
        for config in configs:
            assert service.get_secret_type(config) == SecretType.CONFIG

    def test_get_secret_type_other(self, service):
        """Test identifying other type secrets"""
        others = ["some_value", "random_setting", "misc_data"]
        for other in others:
            assert service.get_secret_type(other) == SecretType.OTHER

    def test_should_encrypt_secret_sensitive(self, service):
        """Test encryption requirement for sensitive secrets"""
        sensitive = ["api_key", "api_secret", "password", "access_token", "private_key"]
        for secret in sensitive:
            assert service.should_encrypt_secret(secret) is True

    def test_should_encrypt_secret_non_sensitive(self, service):
        """Test encryption not required for non-sensitive data"""
        non_sensitive = ["base_url", "environment", "port", "database_name", "timeout"]
        for secret in non_sensitive:
            assert service.should_encrypt_secret(secret) is False

    def test_validate_all_secrets(self, service):
        """Test validation of multiple secrets"""
        secrets = {
            "api_key": "abcdefghij1234567890",
            "api_secret": "abcdefghij1234567890abcdefghij12",
            "base_url": "https://api.example.com",
            "password": "SecurePass123",
        }

        results = service.validate_all_secrets(secrets)
        assert all(results.values())
        assert len(results) == 4

    def test_validate_all_secrets_with_failures(self, service):
        """Test validation with some invalid secrets"""
        secrets = {
            "api_key": "short",  # Invalid
            "api_secret": "abcdefghij1234567890abcdefghij12",  # Valid
            "base_url": "not-a-url",  # Invalid
            "password": "SecurePass123",  # Valid
        }

        results = service.validate_all_secrets(secrets)
        assert results["api_key"] is False
        assert results["api_secret"] is True
        assert results["base_url"] is False
        assert results["password"] is True

    def test_get_validation_rules(self, service):
        """Test getting validation rules for a secret type"""
        api_key_rules = service.get_validation_rules("api_key")
        assert "min_length" in api_key_rules
        assert api_key_rules["min_length"] == 20

        password_rules = service.get_validation_rules("password")
        assert "min_length" in password_rules
        assert password_rules["min_length"] == 8

        url_rules = service.get_validation_rules("base_url")
        assert "format" in url_rules
        assert url_rules["format"] == "url"

    def test_validate_broker_credentials(self, service):
        """Test validation of broker credentials"""
        valid_creds = {
            "api_key": "abcdefghij1234567890",
            "api_secret": "abcdefghij1234567890abcdefghij12",
            "base_url": "https://api.alpaca.markets",
        }
        assert service.validate_broker_credentials(valid_creds) is True

    def test_validate_broker_credentials_invalid(self, service):
        """Test validation of invalid broker credentials"""
        invalid_creds = {"api_key": "short", "api_secret": "also_short", "base_url": "not-a-url"}
        with pytest.raises(SecretValidationError):
            service.validate_broker_credentials(invalid_creds)

    def test_mask_secret_value(self, service):
        """Test masking of secret values for logging"""
        # API key - show first 4 and last 4
        masked = service.mask_secret_value("abcdefghij1234567890")
        assert masked == "abcd...7890"

        # Short value - all masked
        masked = service.mask_secret_value("short")
        assert masked == "*****"

        # Empty value
        masked = service.mask_secret_value("")
        assert masked == ""

    def test_validate_encryption_key(self, service):
        """Test validation of encryption keys"""
        # Valid 32-byte key (base64 encoded)
        valid_key = "dGhpc2lzYTMyYnl0ZWtleWZvcmVuY3J5cHRpb24="
        assert service.validate_encryption_key(valid_key) is True

        # Invalid key (too short)
        with pytest.raises(SecretValidationError):
            service.validate_encryption_key("short")

    def test_validate_jwt_token(self, service):
        """Test validation of JWT tokens"""
        # Mock JWT format (header.payload.signature)
        valid_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature"
        assert service.validate_jwt_token(valid_jwt) is True

        # Invalid format
        with pytest.raises(SecretValidationError):
            service.validate_jwt_token("not.a.jwt")

    def test_validate_api_endpoint(self, service):
        """Test validation of API endpoints"""
        valid_endpoints = [
            "https://api.example.com/v1/orders",
            "http://localhost:8080/health",
            "https://broker.com/api/v2/positions",
        ]
        for endpoint in valid_endpoints:
            assert service.validate_api_endpoint(endpoint) is True

        # Invalid endpoints
        invalid_endpoints = [
            "not-a-url",
            "ftp://server.com/file",  # Wrong protocol
            "https://",  # Incomplete
        ]
        for endpoint in invalid_endpoints:
            with pytest.raises(SecretValidationError):
                service.validate_api_endpoint(endpoint)
