"""
Production Configuration Management

Environment-aware configuration system with feature flags, validation,
runtime updates, and integration with resilience infrastructure.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Configuration file formats."""

    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error."""

    field_name: str
    expected_type: type[Any]
    actual_value: Any
    message: str = ""

    def __str__(self) -> str:
        return (
            f"Config validation failed for '{self.field_name}': "
            f"expected {self.expected_type.__name__}, got {type(self.actual_value).__name__}"
            + (f" - {self.message}" if self.message else "")
        )


@dataclass
class ResilienceConfig:
    """Resilience-specific configuration."""

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_max_calls: int = 3

    # Retry settings
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_backoff_multiplier: float = 2.0
    retry_jitter: bool = True

    # Health check settings
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0

    # Fallback settings
    fallback_enabled: bool = True
    fallback_cache_ttl: float = 300.0
    fallback_timeout: float = 10.0

    def validate(self) -> None:
        """Validate resilience configuration."""
        if self.circuit_breaker_failure_threshold <= 0:
            raise ConfigValidationError(
                "circuit_breaker_failure_threshold",
                int,
                self.circuit_breaker_failure_threshold,
                "must be positive",
            )

        if self.retry_max_attempts < 0:
            raise ConfigValidationError(
                "retry_max_attempts", int, self.retry_max_attempts, "must be non-negative"
            )

        if self.retry_initial_delay <= 0:
            raise ConfigValidationError(
                "retry_initial_delay", float, self.retry_initial_delay, "must be positive"
            )


@dataclass
class TradingConfig:
    """Trading system specific configuration."""

    # Market hours
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    timezone: str = "America/New_York"

    # Order settings
    default_order_timeout: float = 30.0
    max_order_size: float = 10000.0
    min_order_size: float = 1.0

    # Risk management
    max_position_size: float = 100000.0
    max_portfolio_value: float = 1000000.0
    risk_check_enabled: bool = True

    # Data refresh rates
    price_update_interval: float = 1.0
    portfolio_update_interval: float = 5.0

    def validate(self) -> None:
        """Validate trading configuration."""
        if not (0 <= self.market_open_hour <= 23):
            raise ConfigValidationError(
                "market_open_hour", int, self.market_open_hour, "must be between 0 and 23"
            )

        if not (0 <= self.market_open_minute <= 59):
            raise ConfigValidationError(
                "market_open_minute", int, self.market_open_minute, "must be between 0 and 59"
            )

        if self.max_order_size <= self.min_order_size:
            raise ConfigValidationError(
                "max_order_size", float, self.max_order_size, "must be greater than min_order_size"
            )


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ai_trader"
    user: str = ""
    password: str = ""

    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    max_idle_time: float = 300.0
    command_timeout: float = 60.0

    # SSL settings
    ssl_mode: str = "prefer"
    ssl_cert_file: str | None = None
    ssl_key_file: str | None = None
    ssl_ca_file: str | None = None

    def validate(self) -> None:
        """Validate database configuration."""
        if not self.host:
            raise ConfigValidationError("host", str, self.host, "cannot be empty")

        if not (1 <= self.port <= 65535):
            raise ConfigValidationError("port", int, self.port, "must be between 1 and 65535")

        if self.min_pool_size <= 0:
            raise ConfigValidationError(
                "min_pool_size", int, self.min_pool_size, "must be positive"
            )

        if self.max_pool_size < self.min_pool_size:
            raise ConfigValidationError(
                "max_pool_size", int, self.max_pool_size, "must be >= min_pool_size"
            )


@dataclass
class ExternalAPIConfig:
    """External API configuration."""

    # Polygon.io
    polygon_api_key: str = ""
    polygon_base_url: str = "https://api.polygon.io"
    polygon_rate_limit: int = 5  # requests per second
    polygon_timeout: float = 30.0

    # Alpha Vantage
    alpha_vantage_api_key: str = ""
    alpha_vantage_base_url: str = "https://www.alphavantage.co"
    alpha_vantage_rate_limit: int = 5
    alpha_vantage_timeout: float = 30.0

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    alpaca_timeout: float = 30.0

    def validate(self) -> None:
        """Validate external API configuration."""
        # Check required API keys based on enabled features
        if not self.polygon_api_key:
            logger.warning("Polygon API key not configured")

        if not self.alpaca_api_key or not self.alpaca_secret_key:
            logger.warning("Alpaca API credentials not configured")

        # Validate URLs
        for url_field in ["polygon_base_url", "alpha_vantage_base_url", "alpaca_base_url"]:
            url = getattr(self, url_field)
            if not url.startswith(("http://", "https://")):
                raise ConfigValidationError(url_field, str, url, "must be a valid URL")


@dataclass
class FeatureFlags:
    """Feature flag configuration."""

    # Core features
    paper_trading_enabled: bool = True
    live_trading_enabled: bool = False

    # Data sources
    real_time_data_enabled: bool = True
    historical_data_caching: bool = True

    # Risk management
    position_sizing_enabled: bool = True
    risk_limits_enforced: bool = True
    stop_loss_enabled: bool = True

    # Monitoring
    metrics_collection_enabled: bool = True
    health_checks_enabled: bool = True
    error_reporting_enabled: bool = True

    # Experimental features
    machine_learning_enabled: bool = False
    advanced_analytics_enabled: bool = False
    beta_features_enabled: bool = False

    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return getattr(self, feature_name, False)

    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature flag."""
        if hasattr(self, feature_name):
            setattr(self, feature_name, True)
            logger.info(f"Enabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature flag: {feature_name}")

    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature flag."""
        if hasattr(self, feature_name):
            setattr(self, feature_name, False)
            logger.info(f"Disabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature flag: {feature_name}")


@dataclass
class ApplicationConfig:
    """Main application configuration."""

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    external_apis: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Application settings
    app_name: str = "AI Trading System"
    version: str = "1.0.0"

    def validate(self) -> None:
        """Validate entire configuration."""
        logger.info("Validating application configuration...")

        # Validate sub-configurations
        self.resilience.validate()
        self.trading.validate()
        self.database.validate()
        self.external_apis.validate()

        # Environment-specific validations
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                logger.warning("Debug mode enabled in production environment")

            if self.features.live_trading_enabled and not self.features.risk_limits_enforced:
                raise ConfigValidationError(
                    "risk_limits_enforced",
                    bool,
                    False,
                    "must be enabled for live trading in production",
                )

        logger.info("Configuration validation completed successfully")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load(self, source: str) -> dict[str, Any]:
        """Load configuration from source."""
        pass


class EnvironmentConfigLoader(ConfigLoader):
    """Load configuration from environment variables."""

    def __init__(self, prefix: str = "AI_TRADER_") -> None:
        self.prefix = prefix

    def load(self, source: str = "") -> dict[str, Any]:
        """Load configuration from environment variables."""
        config: dict[str, Any] = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix) :].lower()

                # Convert types
                if value.lower() in ("true", "false"):
                    config[config_key] = value.lower() == "true"
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    try:
                        config[config_key] = float(value)
                    except ValueError:
                        config[config_key] = value

        return config


class FileConfigLoader(ConfigLoader):
    """Load configuration from files."""

    def load(self, source: str) -> dict[str, Any]:
        """Load configuration from file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")

        content = path.read_text()

        if path.suffix.lower() == ".json":
            result = json.loads(content)
            return result if isinstance(result, dict) else {}
        elif path.suffix.lower() in (".yaml", ".yml"):
            result = yaml.safe_load(content)
            return result if isinstance(result, dict) else {}
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")


class ConfigManager:
    """
    Production configuration manager.

    Features:
    - Environment-aware loading
    - Configuration validation
    - Runtime updates
    - Feature flag management
    """

    def __init__(self, config_dir: str | None = None) -> None:
        self.config_dir = Path(config_dir or "config")
        self.loaders: dict[str, ConfigLoader] = {
            "env": EnvironmentConfigLoader(),
            "file": FileConfigLoader(),
        }
        self._config: ApplicationConfig | None = None
        self._watchers: list[Callable[[ApplicationConfig], None]] = []

    def load_config(
        self, environment: Environment | None = None, config_file: str | None = None
    ) -> ApplicationConfig:
        """
        Load configuration with environment precedence.

        Args:
            environment: Target environment
            config_file: Specific configuration file

        Returns:
            ApplicationConfig instance
        """
        # Determine environment
        env = environment or Environment(os.getenv("AI_TRADER_ENV", "development"))

        logger.info(f"Loading configuration for environment: {env.value}")

        # Start with defaults
        config_data = {}

        # Load from base config file
        base_file = self.config_dir / "config.yaml"
        if base_file.exists():
            base_config = self.loaders["file"].load(str(base_file))
            config_data.update(base_config)
            logger.debug(f"Loaded base configuration from {base_file}")

        # Load environment-specific config
        env_file = self.config_dir / f"config.{env.value}.yaml"
        if env_file.exists():
            env_config = self.loaders["file"].load(str(env_file))
            config_data.update(env_config)
            logger.debug(f"Loaded environment configuration from {env_file}")

        # Load specific config file if provided
        if config_file:
            file_config = self.loaders["file"].load(config_file)
            config_data.update(file_config)
            logger.debug(f"Loaded specific configuration from {config_file}")

        # Override with environment variables
        env_config = self.loaders["env"].load("")
        config_data.update(env_config)

        # Create configuration object
        config = self._create_config_from_dict(config_data, env)

        # Validate configuration
        config.validate()

        self._config = config

        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Config watcher failed: {e}")

        logger.info(f"Configuration loaded successfully for {env.value}")
        return config

    def _create_config_from_dict(self, data: dict[str, Any], env: Environment) -> ApplicationConfig:
        """Create ApplicationConfig from dictionary."""

        # Extract nested configurations
        resilience_data = data.get("resilience", {})
        trading_data = data.get("trading", {})
        database_data = data.get("database", {})
        external_apis_data = data.get("external_apis", {})
        features_data = data.get("features", {})

        return ApplicationConfig(
            environment=env,
            debug=data.get("debug", env != Environment.PRODUCTION),
            log_level=data.get("log_level", "DEBUG" if env == Environment.DEVELOPMENT else "INFO"),
            resilience=ResilienceConfig(**resilience_data),
            trading=TradingConfig(**trading_data),
            database=DatabaseConfig(**database_data),
            external_apis=ExternalAPIConfig(**external_apis_data),
            features=FeatureFlags(**features_data),
            app_name=data.get("app_name", "AI Trading System"),
            version=data.get("version", "1.0.0"),
        )

    def get_config(self) -> ApplicationConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config

    def reload_config(self) -> ApplicationConfig:
        """Reload configuration from sources."""
        logger.info("Reloading configuration...")
        return self.load_config(self._config.environment if self._config else None)

    def update_feature_flag(self, feature_name: str, enabled: bool) -> None:
        """Update a feature flag at runtime."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        if hasattr(self._config.features, feature_name):
            setattr(self._config.features, feature_name, enabled)
            logger.info(f"Updated feature flag {feature_name}: {enabled}")

            # Notify watchers
            for watcher in self._watchers:
                try:
                    watcher(self._config)
                except Exception as e:
                    logger.error(f"Config watcher failed after feature update: {e}")
        else:
            logger.warning(f"Unknown feature flag: {feature_name}")

    def add_config_watcher(self, watcher: Callable[[ApplicationConfig], None]) -> None:
        """Add configuration change watcher."""
        self._watchers.append(watcher)
        logger.debug("Added configuration watcher")

    def remove_config_watcher(self, watcher: Callable[[ApplicationConfig], None]) -> None:
        """Remove configuration change watcher."""
        if watcher in self._watchers:
            self._watchers.remove(watcher)
            logger.debug("Removed configuration watcher")

    def export_config(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """Export current configuration to string."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        # Convert to dictionary (simplified for export)
        config_dict = {
            "environment": self._config.environment.value,
            "debug": self._config.debug,
            "log_level": self._config.log_level,
            "app_name": self._config.app_name,
            "version": self._config.version,
        }

        if format == ConfigFormat.JSON:
            return json.dumps(config_dict, indent=2)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global configuration manager
config_manager = ConfigManager()


def get_config() -> ApplicationConfig:
    """Get current application configuration."""
    return config_manager.get_config()


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature flag is enabled."""
    try:
        config = get_config()
        return config.features.is_enabled(feature_name)
    except RuntimeError:
        # Configuration not loaded, assume disabled
        return False
