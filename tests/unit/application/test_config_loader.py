"""
Comprehensive unit tests for Configuration Loader.

Tests all configuration loading functionality including:
- Loading from environment variables
- Loading from YAML files
- Saving to YAML files
- Handling various data types and edge cases
"""

import os
import tempfile
from decimal import Decimal
from unittest.mock import patch

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
)
from src.application.config_loader import ConfigLoader


class TestConfigLoaderFromEnv:
    """Test loading configuration from environment variables."""

    @patch.dict(os.environ, clear=True)
    def test_from_env_default_values(self):
        """Test loading with default values when no env vars set."""
        config = ConfigLoader.from_env()

        assert config.environment == Environment.DEVELOPMENT
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.database.database == "ai_trader"
        assert config.broker.type == "paper"
        assert config.risk.max_position_size_pct == Decimal("10")
        assert config.logging.level == "INFO"
        assert config.features.enable_caching is True

    @patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "production",
            "DB_HOST": "prod-db.example.com",
            "DB_PORT": "5433",
            "DB_NAME": "prod_trading",
            "DB_USER": "prod_user",
            "DB_PASSWORD": "secret123",
            "BROKER_TYPE": "alpaca",
            "BROKER_API_KEY": "test_key",
            "BROKER_API_SECRET": "test_secret",
            "RISK_MAX_POSITION_SIZE_PCT": "0.05",
            "LOG_LEVEL": "WARNING",
            "FEATURE_CACHING": "false",
        },
    )
    def test_from_env_with_values(self):
        """Test loading with environment variables set."""
        config = ConfigLoader.from_env()

        assert config.environment == Environment.PRODUCTION
        assert config.database.host == "prod-db.example.com"
        assert config.database.port == 5433
        assert config.database.database == "prod_trading"
        assert config.database.user == "prod_user"
        assert config.database.password == "secret123"
        assert config.broker.type == "alpaca"
        assert config.broker.api_key == "test_key"
        assert config.broker.api_secret == "test_secret"
        assert config.risk.max_position_size_pct == Decimal("0.05")
        assert config.logging.level == "WARNING"
        assert config.features.enable_caching is False

    @patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"})
    def test_from_env_invalid_environment(self):
        """Test loading with invalid environment value."""
        with pytest.raises(ValueError, match="Invalid environment: invalid_env"):
            ConfigLoader.from_env()

    @patch.dict(
        os.environ,
        {"ENVIRONMENT": "staging", "DB_POOL_SIZE": "20", "DB_MAX_OVERFLOW": "5", "DB_ECHO": "true"},
    )
    def test_from_env_database_pool_settings(self):
        """Test loading database pool settings from environment."""
        config = ConfigLoader.from_env()

        assert config.environment == Environment.STAGING
        assert config.database.pool_size == 20
        assert config.database.max_overflow == 5
        assert config.database.echo is True

    @patch.dict(
        os.environ,
        {
            "BROKER_TYPE": "paper",
            "PAPER_INITIAL_CAPITAL": "100000.50",
            "PAPER_SLIPPAGE_PCT": "0.002",
            "PAPER_COMMISSION_PER_SHARE": "0.005",
            "PAPER_MIN_COMMISSION": "0.50",
        },
    )
    def test_from_env_paper_broker_settings(self):
        """Test loading paper broker settings from environment."""
        config = ConfigLoader.from_env()

        assert config.broker.type == "paper"
        assert config.broker.paper_initial_capital == Decimal("100000.50")
        assert config.broker.paper_slippage_pct == Decimal("0.002")
        assert config.broker.paper_commission_per_share == Decimal("0.005")
        assert config.broker.paper_min_commission == Decimal("0.50")

    @patch.dict(
        os.environ,
        {
            "RISK_MAX_TOTAL_EXPOSURE_PCT": "0.95",
            "RISK_MAX_DAILY_LOSS_PCT": "0.03",
            "RISK_MIN_CASH_BALANCE": "5000",
            "RISK_ENABLE_STOP_LOSS": "false",
            "RISK_DEFAULT_STOP_LOSS_PCT": "0.05",
        },
    )
    def test_from_env_risk_settings(self):
        """Test loading risk settings from environment."""
        config = ConfigLoader.from_env()

        assert config.risk.max_total_exposure_pct == Decimal("0.95")
        assert config.risk.max_daily_loss_pct == Decimal("0.03")
        assert config.risk.min_cash_balance == Decimal("5000")
        assert config.risk.enable_stop_loss is False
        assert config.risk.default_stop_loss_pct == Decimal("0.05")

    @patch.dict(
        os.environ,
        {
            "LOG_FORMAT": "%(asctime)s - %(levelname)s - %(message)s",
            "LOG_FILE": "/var/log/trading.log",
            "LOG_MAX_BYTES": "10485760",
            "LOG_BACKUP_COUNT": "10",
        },
    )
    def test_from_env_logging_settings(self):
        """Test loading logging settings from environment."""
        config = ConfigLoader.from_env()

        assert config.logging.format == "%(asctime)s - %(levelname)s - %(message)s"
        assert config.logging.file == "/var/log/trading.log"
        assert config.logging.max_bytes == 10485760
        assert config.logging.backup_count == 10

    @patch.dict(
        os.environ,
        {
            "FEATURE_METRICS": "true",
            "FEATURE_TRACING": "true",
            "FEATURE_BACKTESTING": "false",
            "FEATURE_PAPER_TRADING": "true",
            "FEATURE_LIVE_TRADING": "false",
            "FEATURE_NOTIFICATIONS": "true",
            "FEATURE_WEB_UI": "false",
        },
    )
    def test_from_env_feature_flags(self):
        """Test loading feature flags from environment."""
        config = ConfigLoader.from_env()

        assert config.features.enable_metrics is True
        assert config.features.enable_tracing is True
        assert config.features.enable_backtesting is False
        assert config.features.enable_paper_trading is True
        assert config.features.enable_live_trading is False
        assert config.features.enable_notifications is True
        assert config.features.enable_web_ui is False


class TestConfigLoaderFromYAML:
    """Test loading configuration from YAML files."""

    def test_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            # Should return default configuration
            assert config.environment == Environment.DEVELOPMENT
            assert config.database.host == "localhost"
            assert config.broker.type == "paper"
        finally:
            os.unlink(temp_path)

    def test_from_yaml_null_content(self):
        """Test loading from YAML file with null content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("null")
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            # Should return default configuration
            assert config.environment == Environment.DEVELOPMENT
            assert config.database.host == "localhost"
        finally:
            os.unlink(temp_path)

    def test_from_yaml_full_configuration(self):
        """Test loading complete configuration from YAML."""
        yaml_content = """
environment: production

database:
  host: prod-db.example.com
  port: 5433
  database: prod_trading
  user: prod_user
  password: secret_password
  pool_size: 30
  max_overflow: 10
  echo: false

broker:
  type: alpaca
  auto_connect: true
  api_key: alpaca_key_123
  api_secret: alpaca_secret_456
  base_url: https://api.alpaca.markets
  data_feed: sip
  enable_fractional: true
  alpaca_paper: false

risk:
  max_position_size_pct: 0.05
  max_total_exposure_pct: 0.9
  max_daily_loss_pct: 0.02
  min_cash_balance: 10000
  enable_stop_loss: true
  default_stop_loss_pct: 0.03

logging:
  level: DEBUG
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  file: /var/log/prod_trading.log
  max_bytes: 52428800
  backup_count: 20

features:
  enable_caching: true
  enable_metrics: true
  enable_tracing: true
  enable_backtesting: false
  enable_paper_trading: false
  enable_live_trading: true
  enable_notifications: true
  enable_web_ui: true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            assert config.environment == Environment.PRODUCTION

            # Database settings
            assert config.database.host == "prod-db.example.com"
            assert config.database.port == 5433
            assert config.database.database == "prod_trading"
            assert config.database.user == "prod_user"
            assert config.database.password == "secret_password"
            assert config.database.pool_size == 30
            assert config.database.max_overflow == 10
            assert config.database.echo is False

            # Broker settings
            assert config.broker.type == "alpaca"
            assert config.broker.auto_connect is True
            assert config.broker.api_key == "alpaca_key_123"
            assert config.broker.api_secret == "alpaca_secret_456"
            assert config.broker.base_url == "https://api.alpaca.markets"
            assert config.broker.data_feed == "sip"
            assert config.broker.enable_fractional is True
            assert config.broker.alpaca_paper is False

            # Risk settings
            assert config.risk.max_position_size_pct == Decimal("0.05")
            assert config.risk.max_total_exposure_pct == Decimal("0.9")
            assert config.risk.max_daily_loss_pct == Decimal("0.02")
            assert config.risk.min_cash_balance == Decimal("10000")
            assert config.risk.enable_stop_loss is True
            assert config.risk.default_stop_loss_pct == Decimal("0.03")

            # Logging settings
            assert config.logging.level == "DEBUG"
            assert config.logging.format == "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            assert config.logging.file == "/var/log/prod_trading.log"
            assert config.logging.max_bytes == 52428800
            assert config.logging.backup_count == 20

            # Feature flags
            assert config.features.enable_caching is True
            assert config.features.enable_metrics is True
            assert config.features.enable_tracing is True
            assert config.features.enable_backtesting is False
            assert config.features.enable_paper_trading is False
            assert config.features.enable_live_trading is True
            assert config.features.enable_notifications is True
            assert config.features.enable_web_ui is True

        finally:
            os.unlink(temp_path)

    def test_from_yaml_partial_configuration(self):
        """Test loading partial configuration from YAML."""
        yaml_content = """
environment: staging

database:
  host: staging-db.example.com
  port: 5432

broker:
  type: paper
  paper_initial_capital: 50000
  paper_slippage_pct: 0.001
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            assert config.environment == Environment.STAGING

            # Specified values
            assert config.database.host == "staging-db.example.com"
            assert config.database.port == 5432
            assert config.broker.type == "paper"
            assert config.broker.paper_initial_capital == Decimal("50000")
            assert config.broker.paper_slippage_pct == Decimal("0.001")

            # Default values
            assert config.database.database == "ai_trader"
            assert config.database.user == "zachwade"
            assert config.broker.paper_commission_per_share == Decimal("0.01")
            assert config.risk.max_position_size_pct == Decimal("10")
            assert config.logging.level == "INFO"

        finally:
            os.unlink(temp_path)

    def test_from_yaml_legacy_broker_attributes(self):
        """Test loading YAML with legacy broker attribute names."""
        yaml_content = """
broker:
  type: alpaca
  alpaca_api_key: legacy_key
  alpaca_secret_key: legacy_secret
  alpaca_paper: true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            assert config.broker.type == "alpaca"
            assert config.broker.alpaca_api_key == "legacy_key"
            assert config.broker.alpaca_secret_key == "legacy_secret"
            assert config.broker.alpaca_paper is True

        finally:
            os.unlink(temp_path)

    def test_from_yaml_decimal_conversion(self):
        """Test proper decimal conversion for numeric fields."""
        yaml_content = """
risk:
  max_position_size_pct: 0.123456789
  max_total_exposure_pct: 0.987654321
  min_cash_balance: 12345.67

broker:
  paper_initial_capital: 99999.99
  paper_slippage_pct: 0.0012345
  paper_commission_per_share: 0.00123
  paper_min_commission: 0.12
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)

            # Check decimal precision is preserved
            assert config.risk.max_position_size_pct == Decimal("0.123456789")
            assert config.risk.max_total_exposure_pct == Decimal("0.987654321")
            assert config.risk.min_cash_balance == Decimal("12345.67")
            assert config.broker.paper_initial_capital == Decimal("99999.99")
            assert config.broker.paper_slippage_pct == Decimal("0.0012345")
            assert config.broker.paper_commission_per_share == Decimal("0.00123")
            assert config.broker.paper_min_commission == Decimal("0.12")

        finally:
            os.unlink(temp_path)

    def test_from_yaml_invalid_environment(self):
        """Test loading YAML with invalid environment value."""
        yaml_content = """
environment: invalid_environment
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                ConfigLoader.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_yaml("/non/existent/file.yaml")


class TestConfigLoaderToYAML:
    """Test saving configuration to YAML."""

    def test_to_yaml_default_config(self):
        """Test converting default config to YAML string."""
        config = ApplicationConfig()
        yaml_str = ConfigLoader.to_yaml(config)

        # Parse the YAML to verify structure
        parsed = yaml.safe_load(yaml_str)

        assert parsed["environment"] == "development"
        assert parsed["database"]["host"] == "localhost"
        assert parsed["database"]["port"] == 5432
        assert parsed["broker"]["type"] == "paper"
        assert parsed["risk"]["max_position_size_pct"] == "10"
        assert parsed["logging"]["level"] == "INFO"
        assert parsed["features"]["enable_caching"] is True

    def test_to_yaml_custom_config(self):
        """Test converting custom config to YAML string."""
        config = ApplicationConfig(
            environment=Environment.PRODUCTION,
            database=DatabaseConfig(
                host="prod.example.com",
                port=5433,
                database="prod_db",
                user="prod_user",
                password="secret",
            ),
            broker=BrokerConfig(type="alpaca", api_key="key123", api_secret="secret456"),
            risk=RiskConfig(max_position_size_pct=Decimal("0.05"), enable_stop_loss=True),
        )

        yaml_str = ConfigLoader.to_yaml(config)
        parsed = yaml.safe_load(yaml_str)

        assert parsed["environment"] == "production"
        assert parsed["database"]["host"] == "prod.example.com"
        assert parsed["database"]["port"] == 5433
        assert parsed["database"]["user"] == "prod_user"
        assert parsed["broker"]["type"] == "alpaca"
        assert parsed["broker"]["api_key"] == "key123"
        assert parsed["risk"]["max_position_size_pct"] == "0.05"
        assert parsed["risk"]["enable_stop_loss"] is True

    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = ApplicationConfig(
            environment=Environment.STAGING,
            database=DatabaseConfig(host="staging.example.com", port=5432),
            features=FeatureFlags(enable_metrics=True, enable_live_trading=False),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            ConfigLoader.save_to_yaml(config, temp_path)

            # Read back and verify
            with open(temp_path) as f:
                content = f.read()
                parsed = yaml.safe_load(content)

            assert parsed["environment"] == "staging"
            assert parsed["database"]["host"] == "staging.example.com"
            assert parsed["features"]["enable_metrics"] is True
            assert parsed["features"]["enable_live_trading"] is False

        finally:
            os.unlink(temp_path)

    def test_round_trip_configuration(self):
        """Test that config survives save and load cycle."""
        original_config = ApplicationConfig(
            environment=Environment.PRODUCTION,
            database=DatabaseConfig(
                host="db.example.com",
                port=5433,
                database="trading",
                user="trader",
                password="pass123",
                pool_size=25,
                max_overflow=5,
                echo=True,
            ),
            broker=BrokerConfig(
                type="alpaca",
                auto_connect=False,
                api_key="test_key",
                api_secret="test_secret",
                base_url="https://api.test.com",
                data_feed="iex",
                enable_fractional=True,
                paper_initial_capital=Decimal("75000"),
                paper_slippage_pct=Decimal("0.0015"),
                paper_commission_per_share=Decimal("0.008"),
                paper_min_commission=Decimal("0.75"),
            ),
            risk=RiskConfig(
                max_position_size_pct=Decimal("0.08"),
                max_total_exposure_pct=Decimal("0.85"),
                max_daily_loss_pct=Decimal("0.025"),
                min_cash_balance=Decimal("7500"),
                enable_stop_loss=True,
                default_stop_loss_pct=Decimal("0.04"),
            ),
            logging=LoggingConfig(
                level="DEBUG",
                format="custom format",
                file="/custom/log.txt",
                max_bytes=1048576,
                backup_count=15,
            ),
            features=FeatureFlags(
                enable_caching=False,
                enable_metrics=True,
                enable_tracing=False,
                enable_backtesting=True,
                enable_paper_trading=False,
                enable_live_trading=True,
                enable_notifications=False,
                enable_web_ui=True,
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Save and reload
            ConfigLoader.save_to_yaml(original_config, temp_path)
            loaded_config = ConfigLoader.from_yaml(temp_path)

            # Compare all fields
            assert loaded_config.environment == original_config.environment

            # Database
            assert loaded_config.database.host == original_config.database.host
            assert loaded_config.database.port == original_config.database.port
            assert loaded_config.database.database == original_config.database.database
            assert loaded_config.database.user == original_config.database.user
            assert loaded_config.database.password == original_config.database.password
            assert loaded_config.database.pool_size == original_config.database.pool_size
            assert loaded_config.database.max_overflow == original_config.database.max_overflow
            assert loaded_config.database.echo == original_config.database.echo

            # Broker
            assert loaded_config.broker.type == original_config.broker.type
            assert loaded_config.broker.auto_connect == original_config.broker.auto_connect
            assert loaded_config.broker.api_key == original_config.broker.api_key
            assert loaded_config.broker.api_secret == original_config.broker.api_secret
            assert loaded_config.broker.base_url == original_config.broker.base_url
            assert loaded_config.broker.data_feed == original_config.broker.data_feed
            assert (
                loaded_config.broker.enable_fractional == original_config.broker.enable_fractional
            )
            assert (
                loaded_config.broker.paper_initial_capital
                == original_config.broker.paper_initial_capital
            )
            assert (
                loaded_config.broker.paper_slippage_pct == original_config.broker.paper_slippage_pct
            )
            assert (
                loaded_config.broker.paper_commission_per_share
                == original_config.broker.paper_commission_per_share
            )
            assert (
                loaded_config.broker.paper_min_commission
                == original_config.broker.paper_min_commission
            )

            # Risk
            assert (
                loaded_config.risk.max_position_size_pct
                == original_config.risk.max_position_size_pct
            )
            assert (
                loaded_config.risk.max_total_exposure_pct
                == original_config.risk.max_total_exposure_pct
            )
            assert loaded_config.risk.max_daily_loss_pct == original_config.risk.max_daily_loss_pct
            assert loaded_config.risk.min_cash_balance == original_config.risk.min_cash_balance
            assert loaded_config.risk.enable_stop_loss == original_config.risk.enable_stop_loss
            assert (
                loaded_config.risk.default_stop_loss_pct
                == original_config.risk.default_stop_loss_pct
            )

            # Logging
            assert loaded_config.logging.level == original_config.logging.level
            assert loaded_config.logging.format == original_config.logging.format
            assert loaded_config.logging.file == original_config.logging.file
            assert loaded_config.logging.max_bytes == original_config.logging.max_bytes
            assert loaded_config.logging.backup_count == original_config.logging.backup_count

            # Features
            assert loaded_config.features.enable_caching == original_config.features.enable_caching
            assert loaded_config.features.enable_metrics == original_config.features.enable_metrics
            assert loaded_config.features.enable_tracing == original_config.features.enable_tracing
            assert (
                loaded_config.features.enable_backtesting
                == original_config.features.enable_backtesting
            )
            assert (
                loaded_config.features.enable_paper_trading
                == original_config.features.enable_paper_trading
            )
            assert (
                loaded_config.features.enable_live_trading
                == original_config.features.enable_live_trading
            )
            assert (
                loaded_config.features.enable_notifications
                == original_config.features.enable_notifications
            )
            assert loaded_config.features.enable_web_ui == original_config.features.enable_web_ui

        finally:
            os.unlink(temp_path)
