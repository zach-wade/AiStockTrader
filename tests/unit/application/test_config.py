"""
Comprehensive tests for application configuration module.

Tests all configuration classes, environment handling, and factory methods.
"""

import os
from decimal import Decimal
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.application.config import (
    ApplicationConfig,
    BrokerConfig,
    DatabaseConfig,
    Environment,
    FeatureFlags,
    LoggingConfig,
    RiskConfig,
    get_config,
    reset_config,
    set_config,
)
from src.application.config_loader import ConfigLoader


class TestEnvironment:
    """Test Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_environment_from_string(self):
        """Test creating environment from string."""
        assert Environment("development") == Environment.DEVELOPMENT
        assert Environment("production") == Environment.PRODUCTION


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_default_values(self):
        """Test default database configuration values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "ai_trader"
        assert config.user == "zachwade"
        assert config.password == ""
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is False

    def test_custom_values(self):
        """Test custom database configuration values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="trading_db",
            user="trader",
            password="secret",
            pool_size=20,
            max_overflow=40,
            echo=True,
        )
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "trading_db"
        assert config.user == "trader"
        assert config.password == "secret"
        assert config.pool_size == 20
        assert config.max_overflow == 40
        assert config.echo is True

    @patch.dict(
        os.environ,
        {
            "DB_HOST": "env-host",
            "DB_PORT": "5434",
            "DB_NAME": "env-db",
            "DB_USER": "env-user",
            "DB_PASSWORD": "env-pass",
            "DB_POOL_SIZE": "15",
            "DB_MAX_OVERFLOW": "30",
            "DB_ECHO": "true",
        },
    )
    def test_from_env(self):
        """Test creating database config from environment variables."""
        config = DatabaseConfig.from_env()
        assert config.host == "env-host"
        assert config.port == 5434
        assert config.database == "env-db"
        assert config.user == "env-user"
        assert config.password == "env-pass"
        assert config.pool_size == 15
        assert config.max_overflow == 30
        assert config.echo is True

    @patch.dict(os.environ, {"DB_ECHO": "false"}, clear=True)
    def test_from_env_echo_false(self):
        """Test echo=false from environment."""
        config = DatabaseConfig.from_env()
        assert config.echo is False

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        config = DatabaseConfig.from_env()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "ai_trader"


class TestBrokerConfig:
    """Test BrokerConfig class."""

    def test_default_values(self):
        """Test default broker configuration values."""
        config = BrokerConfig()
        assert config.type == "paper"
        assert config.auto_connect is True
        assert config.api_key == ""
        assert config.api_secret == ""
        assert config.base_url == ""
        assert config.data_feed == "iex"
        assert config.enable_fractional is False

    def test_custom_values(self):
        """Test custom broker configuration values."""
        config = BrokerConfig(
            type="alpaca",
            auto_connect=False,
            api_key="key123",
            api_secret="secret456",
            base_url="https://api.alpaca.markets",
            data_feed="sip",
            enable_fractional=True,
        )
        assert config.type == "alpaca"
        assert config.auto_connect is False
        assert config.api_key == "key123"
        assert config.api_secret == "secret456"
        assert config.base_url == "https://api.alpaca.markets"
        assert config.data_feed == "sip"
        assert config.enable_fractional is True

    @patch.dict(
        os.environ,
        {
            "BROKER_TYPE": "alpaca",
            "BROKER_AUTO_CONNECT": "false",
            "BROKER_API_KEY": "env-key",
            "BROKER_API_SECRET": "env-secret",
            "BROKER_BASE_URL": "https://paper-api.alpaca.markets",
            "BROKER_DATA_FEED": "sip",
            "BROKER_ENABLE_FRACTIONAL": "true",
        },
    )
    def test_from_env(self):
        """Test creating broker config from environment variables."""
        config = BrokerConfig.from_env()
        assert config.type == "alpaca"
        assert config.auto_connect is False
        assert config.api_key == "env-key"
        assert config.api_secret == "env-secret"
        assert config.base_url == "https://paper-api.alpaca.markets"
        assert config.data_feed == "sip"
        assert config.enable_fractional is True

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        config = BrokerConfig.from_env()
        assert config.type == "paper"
        assert config.auto_connect is True
        assert config.api_key == ""


class TestRiskConfig:
    """Test RiskConfig class."""

    def test_default_values(self):
        """Test default risk configuration values."""
        config = RiskConfig()
        assert config.max_position_size_pct == Decimal("10")
        assert config.max_total_exposure_pct == Decimal("100")
        assert config.max_daily_loss_pct == Decimal("5")
        assert config.min_cash_balance == Decimal("10000")
        assert config.enable_stop_loss is True
        assert config.default_stop_loss_pct == Decimal("2")

    def test_custom_values(self):
        """Test custom risk configuration values."""
        config = RiskConfig(
            max_position_size_pct=Decimal("20"),
            max_total_exposure_pct=Decimal("80"),
            max_daily_loss_pct=Decimal("3"),
            min_cash_balance=Decimal("50000"),
            enable_stop_loss=False,
            default_stop_loss_pct=Decimal("5"),
        )
        assert config.max_position_size_pct == Decimal("20")
        assert config.max_total_exposure_pct == Decimal("80")
        assert config.max_daily_loss_pct == Decimal("3")
        assert config.min_cash_balance == Decimal("50000")
        assert config.enable_stop_loss is False
        assert config.default_stop_loss_pct == Decimal("5")

    @patch.dict(
        os.environ,
        {
            "RISK_MAX_POSITION_SIZE_PCT": "15",
            "RISK_MAX_TOTAL_EXPOSURE_PCT": "90",
            "RISK_MAX_DAILY_LOSS_PCT": "4",
            "RISK_MIN_CASH_BALANCE": "25000",
            "RISK_ENABLE_STOP_LOSS": "false",
            "RISK_DEFAULT_STOP_LOSS_PCT": "3",
        },
    )
    def test_from_env(self):
        """Test creating risk config from environment variables."""
        config = RiskConfig.from_env()
        assert config.max_position_size_pct == Decimal("15")
        assert config.max_total_exposure_pct == Decimal("90")
        assert config.max_daily_loss_pct == Decimal("4")
        assert config.min_cash_balance == Decimal("25000")
        assert config.enable_stop_loss is False
        assert config.default_stop_loss_pct == Decimal("3")

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        config = RiskConfig.from_env()
        assert config.max_position_size_pct == Decimal("10")
        assert config.enable_stop_loss is True


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file is None
        assert config.max_bytes == 10485760  # 10MB
        assert config.backup_count == 5

    def test_custom_values(self):
        """Test custom logging configuration values."""
        config = LoggingConfig(
            level="DEBUG",
            format="%(levelname)s: %(message)s",
            file="/var/log/trader.log",
            max_bytes=5242880,  # 5MB
            backup_count=10,
        )
        assert config.level == "DEBUG"
        assert config.format == "%(levelname)s: %(message)s"
        assert config.file == "/var/log/trader.log"
        assert config.max_bytes == 5242880
        assert config.backup_count == 10

    @patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "%(message)s",
            "LOG_FILE": "custom.log",
            "LOG_MAX_BYTES": "1048576",
            "LOG_BACKUP_COUNT": "3",
        },
    )
    def test_from_env(self):
        """Test creating logging config from environment variables."""
        config = LoggingConfig.from_env()
        assert config.level == "WARNING"
        assert config.format == "%(message)s"
        assert config.file == "custom.log"
        assert config.max_bytes == 1048576
        assert config.backup_count == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        config = LoggingConfig.from_env()
        assert config.level == "INFO"
        assert config.file is None


class TestFeatureFlags:
    """Test FeatureFlags class."""

    def test_default_values(self):
        """Test default feature flags."""
        flags = FeatureFlags()
        assert flags.enable_caching is True
        assert flags.enable_metrics is True
        assert flags.enable_tracing is False
        assert flags.enable_backtesting is True
        assert flags.enable_paper_trading is True
        assert flags.enable_live_trading is False
        assert flags.enable_notifications is False
        assert flags.enable_web_ui is False

    def test_custom_values(self):
        """Test custom feature flags."""
        flags = FeatureFlags(
            enable_caching=False,
            enable_metrics=False,
            enable_tracing=True,
            enable_backtesting=False,
            enable_paper_trading=False,
            enable_live_trading=True,
            enable_notifications=True,
            enable_web_ui=True,
        )
        assert flags.enable_caching is False
        assert flags.enable_metrics is False
        assert flags.enable_tracing is True
        assert flags.enable_backtesting is False
        assert flags.enable_paper_trading is False
        assert flags.enable_live_trading is True
        assert flags.enable_notifications is True
        assert flags.enable_web_ui is True

    @patch.dict(
        os.environ,
        {
            "FEATURE_CACHING": "false",
            "FEATURE_METRICS": "false",
            "FEATURE_TRACING": "true",
            "FEATURE_BACKTESTING": "false",
            "FEATURE_PAPER_TRADING": "false",
            "FEATURE_LIVE_TRADING": "true",
            "FEATURE_NOTIFICATIONS": "true",
            "FEATURE_WEB_UI": "true",
        },
    )
    def test_from_env(self):
        """Test creating feature flags from environment variables."""
        flags = FeatureFlags.from_env()
        assert flags.enable_caching is False
        assert flags.enable_metrics is False
        assert flags.enable_tracing is True
        assert flags.enable_backtesting is False
        assert flags.enable_paper_trading is False
        assert flags.enable_live_trading is True
        assert flags.enable_notifications is True
        assert flags.enable_web_ui is True

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        flags = FeatureFlags.from_env()
        assert flags.enable_caching is True
        assert flags.enable_live_trading is False


class TestApplicationConfig:
    """Test ApplicationConfig class."""

    def test_default_values(self):
        """Test default application configuration."""
        config = ApplicationConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.broker, BrokerConfig)
        assert isinstance(config.risk, RiskConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.features, FeatureFlags)

    def test_custom_values(self):
        """Test custom application configuration."""
        config = ApplicationConfig(
            environment=Environment.PRODUCTION,
            database=DatabaseConfig(host="prod-db"),
            broker=BrokerConfig(type="alpaca"),
            risk=RiskConfig(max_position_size_pct=Decimal("25")),
            logging=LoggingConfig(level="ERROR"),
            features=FeatureFlags(enable_live_trading=True),
        )
        assert config.environment == Environment.PRODUCTION
        assert config.database.host == "prod-db"
        assert config.broker.type == "alpaca"
        assert config.risk.max_position_size_pct == Decimal("25")
        assert config.logging.level == "ERROR"
        assert config.features.enable_live_trading is True

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_from_env(self):
        """Test creating application config from environment."""
        config = ConfigLoader.from_env()
        assert config.environment == Environment.PRODUCTION

    @patch.dict(os.environ, {"ENVIRONMENT": "invalid"})
    def test_from_env_invalid_environment(self):
        """Test from_env with invalid environment."""
        with pytest.raises(ValueError):
            ConfigLoader.from_env()

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_default_environment(self):
        """Test from_env with no environment set."""
        config = ConfigLoader.from_env()
        assert config.environment == Environment.DEVELOPMENT

    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
environment: staging
database:
  host: staging-db
  port: 5432
  database: staging_trader
broker:
  type: alpaca
  api_key: test-key
risk:
  max_position_size_pct: 15
  max_daily_loss_pct: 3
logging:
  level: DEBUG
  file: staging.log
features:
  enable_caching: true
  enable_live_trading: false
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = ConfigLoader.from_yaml("config.yaml")
            assert config.environment == Environment.STAGING
            assert config.database.host == "staging-db"
            assert config.broker.type == "alpaca"
            assert config.risk.max_position_size_pct == Decimal("15")
            assert config.logging.level == "DEBUG"
            assert config.features.enable_caching is True

    def test_from_yaml_partial_config(self):
        """Test loading partial configuration from YAML."""
        yaml_content = """
environment: testing
database:
  host: test-db
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = ConfigLoader.from_yaml("config.yaml")
            assert config.environment == Environment.TESTING
            assert config.database.host == "test-db"
            # Check defaults are preserved
            assert config.broker.type == "paper"
            assert config.risk.max_position_size_pct == Decimal("10")

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ApplicationConfig(
            environment=Environment.TESTING,
            database=DatabaseConfig(host="test-db", port=5433),
            broker=BrokerConfig(type="paper", api_key="test"),
        )
        config_dict = config.to_dict()

        assert config_dict["environment"] == "testing"
        assert config_dict["database"]["host"] == "test-db"
        assert config_dict["database"]["port"] == 5433
        assert config_dict["broker"]["type"] == "paper"
        assert config_dict["broker"]["api_key"] == "test"
        # Check nested configs are included
        assert "risk" in config_dict
        assert "logging" in config_dict
        assert "features" in config_dict

    def test_to_yaml(self):
        """Test converting configuration to YAML string."""
        config = ApplicationConfig(
            environment=Environment.DEVELOPMENT, database=DatabaseConfig(host="dev-db", port=5432)
        )

        yaml_str = ConfigLoader.to_yaml(config)
        assert "environment: development" in yaml_str
        assert "host: dev-db" in yaml_str
        assert "port: 5432" in yaml_str

        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert parsed["environment"] == "development"
        assert parsed["database"]["host"] == "dev-db"

    def test_validate_production_config(self):
        """Test validation for production configuration."""
        # Valid production config
        config = ApplicationConfig(
            environment=Environment.PRODUCTION,
            database=DatabaseConfig(password="secure_pass"),
            broker=BrokerConfig(type="alpaca", api_key="key", api_secret="secret"),
        )
        # Should not raise
        config.validate()

        # Invalid production config (no DB password)
        config = ApplicationConfig(
            environment=Environment.PRODUCTION, database=DatabaseConfig(password="")
        )
        with pytest.raises(ValueError, match="password"):
            config.validate()

    def test_validate_broker_credentials(self):
        """Test validation of broker credentials."""
        # Paper broker doesn't need credentials
        config = ApplicationConfig(broker=BrokerConfig(type="paper"))
        config.validate()

        # Alpaca broker needs credentials
        config = ApplicationConfig(broker=BrokerConfig(type="alpaca", api_key="", api_secret=""))
        with pytest.raises(ValueError, match="API credentials"):
            config.validate()


class TestConfigurationFunctions:
    """Test configuration utility functions."""

    def test_get_config_singleton(self):
        """Test configuration singleton pattern."""
        reset_config()  # Reset first
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test setting configuration."""
        reset_config()  # Reset first
        new_config = ApplicationConfig(
            environment=Environment.TESTING, database=DatabaseConfig(host="test-db")
        )

        set_config(new_config)
        retrieved = get_config()

        assert retrieved is new_config
        assert retrieved.environment == Environment.TESTING
        assert retrieved.database.host == "test-db"

    def test_reset_config(self):
        """Test resetting configuration."""
        # Set custom config
        custom_config = ApplicationConfig(environment=Environment.PRODUCTION)
        set_config(custom_config)

        # Reset
        reset_config()

        # Should get new default config
        new_config = get_config()
        assert new_config is not custom_config
        assert new_config.environment == Environment.DEVELOPMENT

    @patch.dict(os.environ, {"ENVIRONMENT": "staging"})
    def test_get_config_from_env(self):
        """Test get_config loads from environment on first call."""
        reset_config()  # Reset to force reload
        config = get_config()
        assert config.environment == Environment.STAGING

    def test_config_immutability(self):
        """Test that config modifications don't affect singleton."""
        reset_config()
        config1 = get_config()
        original_host = config1.database.host

        # Try to modify
        config1.database.host = "modified"

        # Get config again
        config2 = get_config()

        # Both should be modified (same instance)
        assert config2.database.host == "modified"

        # Reset and verify clean state
        reset_config()
        config3 = get_config()
        assert config3.database.host == original_host
