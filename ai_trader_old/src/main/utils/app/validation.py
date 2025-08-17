"""
Application Configuration Validation

This module provides comprehensive validation for application-level configuration,
ensuring that all required settings are present and valid before app execution.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any

# Local imports
from main.utils.auth import CredentialType, CredentialValidator, validate_credential
from main.utils.core import ErrorHandlingMixin, get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    validated_components: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)

    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the validation result."""
        self.recommendations.append(recommendation)

    def merge(self, other: "ValidationResult"):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.recommendations.extend(other.recommendations)
        self.validated_components.extend(other.validated_components)
        if other.errors:
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "validated_components": self.validated_components,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


class AppConfigValidator(ErrorHandlingMixin):
    """
    Comprehensive application configuration validator.

    This class validates various aspects of the application configuration
    including credentials, paths, network settings, and component configurations.
    """

    def __init__(self, config: Any):
        """
        Initialize AppConfigValidator.

        Args:
            config: Application configuration to validate
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(f"{__name__}.AppConfigValidator")
        self.credential_validator = CredentialValidator()

    def validate_all(self) -> ValidationResult:
        """
        Validate all configuration components.

        Returns:
            ValidationResult with comprehensive validation results
        """
        result = ValidationResult(passed=True)

        try:
            # Validate different components
            components = [
                ("credentials", self.validate_credentials),
                ("paths", self.validate_paths),
                ("database", self.validate_database_config),
                ("data_sources", self.validate_data_sources),
                ("trading", self.validate_trading_config),
                ("monitoring", self.validate_monitoring_config),
                ("features", self.validate_features_config),
                ("universe", self.validate_universe_config),
            ]

            for component_name, validator_func in components:
                try:
                    component_result = validator_func()
                    result.merge(component_result)
                    result.validated_components.append(component_name)
                except Exception as e:
                    result.add_error(f"Failed to validate {component_name}: {e!s}")
                    self.handle_error(e, f"validating {component_name}")

            # Overall validation summary
            if result.passed:
                self.logger.info(
                    f"✅ Configuration validation passed for {len(result.validated_components)} components"
                )
            else:
                self.logger.error(
                    f"❌ Configuration validation failed with {len(result.errors)} errors"
                )

            return result

        except Exception as e:
            self.handle_error(e, "performing configuration validation")
            result.add_error(f"Validation process failed: {e!s}")
            return result

    def validate_credentials(self) -> ValidationResult:
        """Validate API credentials and authentication settings."""
        result = ValidationResult(passed=True)

        # Alpaca credentials
        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        alpaca_base_url = os.getenv("ALPACA_BASE_URL")

        if not alpaca_key:
            result.add_error("ALPACA_API_KEY environment variable is not set")
        elif not self._validate_api_key_format(alpaca_key):
            result.add_error("ALPACA_API_KEY format is invalid")

        if not alpaca_secret:
            result.add_error("ALPACA_SECRET_KEY environment variable is not set")
        elif not self._validate_api_secret_format(alpaca_secret):
            result.add_error("ALPACA_SECRET_KEY format is invalid")

        if not alpaca_base_url:
            result.add_warning("ALPACA_BASE_URL not set, using default")
        elif not self._validate_url_format(alpaca_base_url):
            result.add_error("ALPACA_BASE_URL format is invalid")

        # Polygon credentials
        polygon_key = os.getenv("POLYGON_API_KEY")

        if not polygon_key:
            result.add_warning("POLYGON_API_KEY environment variable is not set")
        elif not self._validate_api_key_format(polygon_key):
            result.add_error("POLYGON_API_KEY format is invalid")

        # Reddit credentials (optional)
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")

        if reddit_client_id and not reddit_client_secret:
            result.add_error("REDDIT_CLIENT_SECRET is required when REDDIT_CLIENT_ID is set")
        elif reddit_client_secret and not reddit_client_id:
            result.add_error("REDDIT_CLIENT_ID is required when REDDIT_CLIENT_SECRET is set")

        # Validate credential security
        if alpaca_key and alpaca_secret:
            cred_result = validate_credential(
                credential_type=CredentialType.API_KEY, credential_value=alpaca_key
            )
            if not cred_result.is_valid:
                result.add_warning(f"Alpaca API key security issue: {cred_result.message}")

        return result

    def validate_paths(self) -> ValidationResult:
        """Validate file system paths and directory structure."""
        result = ValidationResult(passed=True)

        # Check data lake paths
        data_lake_root = self.config.get("data_lake.root_path", "data_lake")
        data_lake_path = Path(data_lake_root)

        if not data_lake_path.exists():
            result.add_warning(f"Data lake root path does not exist: {data_lake_path}")
            result.add_recommendation(f"Create data lake directory: mkdir -p {data_lake_path}")

        # Check required subdirectories
        required_dirs = ["raw", "processed", "archive", "features"]
        for dir_name in required_dirs:
            dir_path = data_lake_path / dir_name
            if not dir_path.exists():
                result.add_warning(f"Data lake subdirectory missing: {dir_path}")

        # Check logs directory
        logs_path = Path("logs")
        if not logs_path.exists():
            result.add_warning(f"Logs directory does not exist: {logs_path}")

        # Check models directory
        models_path = Path("models")
        if not models_path.exists():
            result.add_warning(f"Models directory does not exist: {models_path}")

        # Check configuration files
        config_files = [
            "src/main/data_pipeline/config/layer_definitions.yaml",
            "src/main/config/universe.yaml",
            "src/main/config/strategies.yaml",
        ]

        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                result.add_error(f"Required configuration file missing: {config_path}")

        # Check write permissions
        write_paths = [data_lake_path, logs_path, models_path]
        for path in write_paths:
            if path.exists() and not os.access(path, os.W_OK):
                result.add_error(f"No write permission for: {path}")

        return result

    def validate_database_config(self) -> ValidationResult:
        """Validate database configuration."""
        result = ValidationResult(passed=True)

        database_config = self.config.get("database", {})

        # Required database settings
        required_settings = ["host", "port", "database", "user"]
        for setting in required_settings:
            if not database_config.get(setting):
                result.add_error(f"Database {setting} is not configured")

        # Validate database URL format
        db_url = database_config.get("url")
        if db_url and not self._validate_database_url(db_url):
            result.add_error("Database URL format is invalid")

        # Check connection pool settings
        pool_config = database_config.get("pool", {})
        min_size = pool_config.get("min_size", 1)
        max_size = pool_config.get("max_size", 10)

        if min_size < 1:
            result.add_error("Database pool min_size must be at least 1")
        if max_size < min_size:
            result.add_error("Database pool max_size must be greater than min_size")
        if max_size > 100:
            result.add_warning("Database pool max_size is very high, consider reducing")

        # Check timeout settings
        timeout = database_config.get("timeout", 30)
        if timeout < 5:
            result.add_warning("Database timeout is very low, may cause issues")
        elif timeout > 300:
            result.add_warning("Database timeout is very high, consider reducing")

        return result

    def validate_data_sources(self) -> ValidationResult:
        """Validate data source configurations."""
        result = ValidationResult(passed=True)

        data_sources = self.config.get("data.sources", [])

        if not data_sources:
            result.add_error("No data sources configured")
            return result

        # Validate each data source
        valid_sources = ["alpaca", "polygon", "yahoo", "benzinga", "reddit"]
        for source in data_sources:
            if source not in valid_sources:
                result.add_error(f"Invalid data source: {source}")

        # Check for at least one market data source
        market_sources = ["alpaca", "polygon", "yahoo"]
        if not any(source in market_sources for source in data_sources):
            result.add_error("At least one market data source (alpaca, polygon, yahoo) is required")

        # Validate source-specific configurations
        for source in data_sources:
            source_config = self.config.get(f"data_sources.{source}", {})

            if source == "alpaca":
                result.merge(self._validate_alpaca_config(source_config))
            elif source == "polygon":
                result.merge(self._validate_polygon_config(source_config))
            elif source == "yahoo":
                result.merge(self._validate_yahoo_config(source_config))

        return result

    def validate_trading_config(self) -> ValidationResult:
        """Validate trading-related configuration."""
        result = ValidationResult(passed=True)

        trading_config = self.config.get("trading", {})

        # Validate trading mode
        mode = trading_config.get("mode", "paper")
        if mode not in ["paper", "live"]:
            result.add_error(f"Invalid trading mode: {mode}")

        if mode == "live":
            result.add_warning("Live trading mode enabled - ensure proper risk management")

        # Validate risk management settings
        risk_config = trading_config.get("risk_management", {})

        max_position_size = risk_config.get("max_position_size", 0.05)
        if max_position_size > 0.2:
            result.add_warning("Maximum position size is very high (>20%)")
        elif max_position_size < 0.01:
            result.add_warning("Maximum position size is very low (<1%)")

        max_portfolio_exposure = risk_config.get("max_portfolio_exposure", 0.8)
        if max_portfolio_exposure > 0.95:
            result.add_error("Maximum portfolio exposure is too high (>95%)")
        elif max_portfolio_exposure < 0.5:
            result.add_warning("Maximum portfolio exposure is very conservative (<50%)")

        # Validate universe settings
        universe_config = trading_config.get("universe", {})
        max_symbols = universe_config.get("max_symbols", 100)

        if max_symbols > 1000:
            result.add_warning("Maximum symbols is very high, may impact performance")
        elif max_symbols < 10:
            result.add_warning("Maximum symbols is very low, may limit opportunities")

        return result

    def validate_monitoring_config(self) -> ValidationResult:
        """Validate monitoring and alerting configuration."""
        result = ValidationResult(passed=True)

        monitoring_config = self.config.get("monitoring", {})

        # Check alerting configuration
        alerting_config = monitoring_config.get("alerting", {})

        # Email alerts
        email_config = alerting_config.get("email", {})
        if email_config.get("enabled", False):
            if not email_config.get("smtp_host"):
                result.add_error("Email alerting enabled but SMTP host not configured")
            if not email_config.get("recipient"):
                result.add_error("Email alerting enabled but recipient not configured")

        # Slack alerts
        slack_config = alerting_config.get("slack", {})
        if slack_config.get("enabled", False):
            if not slack_config.get("webhook_url"):
                result.add_error("Slack alerting enabled but webhook URL not configured")

        # Discord alerts
        discord_config = alerting_config.get("discord", {})
        if discord_config.get("enabled", False):
            if not discord_config.get("webhook_url"):
                result.add_error("Discord alerting enabled but webhook URL not configured")

        # Health check configuration
        health_config = monitoring_config.get("health_checks", {})
        interval = health_config.get("interval", 300)
        if interval < 60:
            result.add_warning("Health check interval is very frequent (<60s)")
        elif interval > 3600:
            result.add_warning("Health check interval is very long (>1h)")

        return result

    def validate_features_config(self) -> ValidationResult:
        """Validate feature engineering configuration."""
        result = ValidationResult(passed=True)

        features_config = self.config.get("features", {})

        # Validate timeframes
        timeframes = features_config.get("timeframes", ["1day"])
        valid_timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "1day"]

        for timeframe in timeframes:
            if timeframe not in valid_timeframes:
                result.add_error(f"Invalid timeframe: {timeframe}")

        # Validate feature calculators
        calculators = features_config.get("calculators", {})
        if not calculators:
            result.add_warning("No feature calculators configured")

        # Check for essential calculators
        essential_calculators = ["technical_indicators", "market_regime", "sentiment"]
        for calculator in essential_calculators:
            if calculator not in calculators:
                result.add_recommendation(f"Consider enabling {calculator} calculator")

        return result

    def validate_universe_config(self) -> ValidationResult:
        """Validate universe configuration."""
        result = ValidationResult(passed=True)

        universe_config = self.config.get("universe", {})

        # Validate universe data
        universe_data = self.config.get("universe_data", {})

        if not universe_data:
            result.add_error("Universe data is not configured")
            return result

        # Check for required universe components
        required_components = ["popular_stocks", "sector_mappings"]
        for component in required_components:
            if component not in universe_data:
                result.add_warning(f"Universe component missing: {component}")

        # Validate popular stocks
        popular_stocks = universe_data.get("popular_stocks", [])
        if len(popular_stocks) < 10:
            result.add_warning("Very few popular stocks configured (<10)")

        # Validate sector mappings
        sector_mappings = universe_data.get("sector_mappings", {})
        if len(sector_mappings) < 5:
            result.add_warning("Very few sectors configured (<5)")

        return result

    def _validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False

        # Basic format validation
        if len(api_key) < 20:
            return False

        # Check for common patterns
        if re.match(r"^[A-Z0-9]{20,}$", api_key):
            return True

        return False

    def _validate_api_secret_format(self, api_secret: str) -> bool:
        """Validate API secret format."""
        if not api_secret:
            return False

        # Basic format validation
        if len(api_secret) < 30:
            return False

        return True

    def _validate_url_format(self, url: str) -> bool:
        """Validate URL format."""
        if not url:
            return False

        # Basic URL validation
        return url.startswith(("http://", "https://"))

    def _validate_database_url(self, db_url: str) -> bool:
        """Validate database URL format."""
        if not db_url:
            return False

        # Basic PostgreSQL URL validation
        return db_url.startswith("postgresql://") or db_url.startswith("postgres://")

    def _validate_alpaca_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate Alpaca-specific configuration."""
        result = ValidationResult(passed=True)

        # Validate rate limits
        rate_limits = config.get("rate_limits", {})
        if rate_limits.get("calls_per_minute", 200) > 200:
            result.add_warning("Alpaca rate limit exceeds recommended maximum")

        return result

    def _validate_polygon_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate Polygon-specific configuration."""
        result = ValidationResult(passed=True)

        # Validate subscription tier
        tier = config.get("subscription_tier", "basic")
        if tier not in ["basic", "starter", "developer", "advanced"]:
            result.add_error(f"Invalid Polygon subscription tier: {tier}")

        return result

    def _validate_yahoo_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate Yahoo-specific configuration."""
        result = ValidationResult(passed=True)

        # Yahoo Finance has rate limits
        rate_limits = config.get("rate_limits", {})
        if rate_limits.get("calls_per_minute", 60) > 60:
            result.add_warning("Yahoo Finance rate limit may be too high")

        return result


def validate_trading_config(config: Any) -> ValidationResult:
    """
    Validate trading-specific configuration.

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult for trading configuration
    """
    validator = AppConfigValidator(config)
    return validator.validate_trading_config()


def validate_data_pipeline_config(config: Any) -> ValidationResult:
    """
    Validate data pipeline configuration.

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult for data pipeline configuration
    """
    validator = AppConfigValidator(config)
    result = ValidationResult(passed=True)

    # Validate components relevant to data pipeline
    result.merge(validator.validate_credentials())
    result.merge(validator.validate_paths())
    result.merge(validator.validate_database_config())
    result.merge(validator.validate_data_sources())
    result.merge(validator.validate_features_config())
    result.merge(validator.validate_universe_config())

    return result


def ensure_critical_config(config: Any, required_keys: list[str]) -> None:
    """
    Ensure critical configuration keys exist or raise an error.

    Args:
        config: Configuration to check
        required_keys: List of required configuration keys

    Raises:
        ConfigValidationError: If any critical configuration is missing
    """
    missing_keys = []

    for key in required_keys:
        if not config.get(key):
            missing_keys.append(key)

    if missing_keys:
        result = ValidationResult(passed=False)
        result.errors = [f"Missing critical configuration: {key}" for key in missing_keys]

        raise ConfigValidationError(
            f"Critical configuration missing: {', '.join(missing_keys)}", result
        )


def validate_app_startup_config(config: Any) -> ValidationResult:
    """
    Validate configuration required for app startup.

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult for startup configuration
    """
    validator = AppConfigValidator(config)
    result = ValidationResult(passed=True)

    # Validate essential components for startup
    result.merge(validator.validate_credentials())
    result.merge(validator.validate_database_config())
    result.merge(validator.validate_data_sources())

    return result
