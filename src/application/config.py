"""
Application Configuration - Central configuration management.

This module provides configuration management for the application,
including environment variables, feature flags, and runtime settings.
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class Environment(Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ai_trader"
    user: str = "zachwade"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ai_trader"),
            user=os.getenv("DB_USER", "zachwade"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )


@dataclass
class BrokerConfig:
    """Broker configuration."""

    type: str = "paper"
    auto_connect: bool = True
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    data_feed: str = "iex"
    enable_fractional: bool = False

    # Legacy attributes for compatibility
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_paper: bool = True

    # Paper broker-specific
    paper_initial_capital: Decimal = Decimal("100000")
    paper_slippage_pct: Decimal = Decimal("0.001")
    paper_commission_per_share: Decimal = Decimal("0.01")
    paper_min_commission: Decimal = Decimal("1.0")

    @classmethod
    def from_env(cls) -> "BrokerConfig":
        """Create configuration from environment variables."""
        return cls(
            type=os.getenv("BROKER_TYPE", "paper"),
            auto_connect=os.getenv("BROKER_AUTO_CONNECT", "true").lower() == "true",
            api_key=os.getenv("BROKER_API_KEY", ""),
            api_secret=os.getenv("BROKER_API_SECRET", ""),
            base_url=os.getenv("BROKER_BASE_URL", ""),
            data_feed=os.getenv("BROKER_DATA_FEED", "iex"),
            enable_fractional=os.getenv("BROKER_ENABLE_FRACTIONAL", "false").lower() == "true",
            alpaca_api_key=os.getenv("BROKER_API_KEY"),
            alpaca_secret_key=os.getenv("BROKER_API_SECRET"),
            alpaca_paper=os.getenv("BROKER_PAPER", "true").lower() == "true",
            # Paper broker specific settings from environment
            paper_initial_capital=Decimal(os.getenv("PAPER_INITIAL_CAPITAL", "100000")),
            paper_slippage_pct=Decimal(os.getenv("PAPER_SLIPPAGE_PCT", "0.001")),
            paper_commission_per_share=Decimal(os.getenv("PAPER_COMMISSION_PER_SHARE", "0.01")),
            paper_min_commission=Decimal(os.getenv("PAPER_MIN_COMMISSION", "1.0")),
        )


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size_pct: Decimal = Decimal("10")  # Max 10% per position
    max_total_exposure_pct: Decimal = Decimal("100")  # Max 100% exposure
    max_daily_loss_pct: Decimal = Decimal("5")  # Max 5% daily loss
    min_cash_balance: Decimal = Decimal("10000")  # Min $10k cash
    enable_stop_loss: bool = True
    default_stop_loss_pct: Decimal = Decimal("2")  # 2% stop loss

    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Create configuration from environment variables."""
        return cls(
            max_position_size_pct=Decimal(os.getenv("RISK_MAX_POSITION_SIZE_PCT", "10")),
            max_total_exposure_pct=Decimal(os.getenv("RISK_MAX_TOTAL_EXPOSURE_PCT", "100")),
            max_daily_loss_pct=Decimal(os.getenv("RISK_MAX_DAILY_LOSS_PCT", "5")),
            min_cash_balance=Decimal(os.getenv("RISK_MIN_CASH_BALANCE", "10000")),
            enable_stop_loss=os.getenv("RISK_ENABLE_STOP_LOSS", "true").lower() == "true",
            default_stop_loss_pct=Decimal(os.getenv("RISK_DEFAULT_STOP_LOSS_PCT", "2")),
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Create configuration from environment variables."""
        file_path = os.getenv("LOG_FILE")
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=file_path if file_path else None,
            max_bytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        )


@dataclass
class FeatureFlags:
    """Feature flags for the application."""

    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_backtesting: bool = True
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    enable_notifications: bool = False
    enable_web_ui: bool = False

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """Create configuration from environment variables."""
        return cls(
            enable_caching=os.getenv("FEATURE_CACHING", "true").lower() == "true",
            enable_metrics=os.getenv("FEATURE_METRICS", "true").lower() == "true",
            enable_tracing=os.getenv("FEATURE_TRACING", "false").lower() == "true",
            enable_backtesting=os.getenv("FEATURE_BACKTESTING", "true").lower() == "true",
            enable_paper_trading=os.getenv("FEATURE_PAPER_TRADING", "true").lower() == "true",
            enable_live_trading=os.getenv("FEATURE_LIVE_TRADING", "false").lower() == "true",
            enable_notifications=os.getenv("FEATURE_NOTIFICATIONS", "false").lower() == "true",
            enable_web_ui=os.getenv("FEATURE_WEB_UI", "false").lower() == "true",
        )


@dataclass
class ApplicationConfig:
    """Main application configuration."""

    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "user": self.database.user,
                "password": self.database.password,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo,
            },
            "broker": {
                "type": self.broker.type,
                "auto_connect": self.broker.auto_connect,
                "api_key": self.broker.api_key,
                "api_secret": self.broker.api_secret,
                "base_url": self.broker.base_url,
                "data_feed": self.broker.data_feed,
                "enable_fractional": self.broker.enable_fractional,
                "alpaca_paper": self.broker.alpaca_paper,
                "paper_initial_capital": str(self.broker.paper_initial_capital),
                "paper_slippage_pct": str(self.broker.paper_slippage_pct),
                "paper_commission_per_share": str(self.broker.paper_commission_per_share),
                "paper_min_commission": str(self.broker.paper_min_commission),
            },
            "risk": {
                "max_position_size_pct": str(self.risk.max_position_size_pct),
                "max_total_exposure_pct": str(self.risk.max_total_exposure_pct),
                "max_daily_loss_pct": str(self.risk.max_daily_loss_pct),
                "min_cash_balance": str(self.risk.min_cash_balance),
                "enable_stop_loss": self.risk.enable_stop_loss,
                "default_stop_loss_pct": str(self.risk.default_stop_loss_pct),
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count,
            },
            "features": {
                "enable_caching": self.features.enable_caching,
                "enable_metrics": self.features.enable_metrics,
                "enable_tracing": self.features.enable_tracing,
                "enable_backtesting": self.features.enable_backtesting,
                "enable_paper_trading": self.features.enable_paper_trading,
                "enable_live_trading": self.features.enable_live_trading,
                "enable_notifications": self.features.enable_notifications,
                "enable_web_ui": self.features.enable_web_ui,
            },
        }

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if valid, raises exception otherwise
        """
        # Validate environment
        if self.environment == Environment.PRODUCTION:
            if not self.database.password:
                raise ValueError("Database password required for production")
            if self.database.echo:
                raise ValueError("Database echo should be disabled in production")
            if self.broker.type == "paper" and self.features.enable_live_trading:
                raise ValueError("Cannot enable live trading with paper broker")

        # Validate broker config
        if self.broker.type == "alpaca":
            if not self.broker.api_key and not self.broker.api_secret:
                if not self.broker.alpaca_api_key or not self.broker.alpaca_secret_key:
                    raise ValueError("API credentials required for Alpaca broker")

        # Validate risk config
        if self.risk.max_position_size_pct > 100:
            raise ValueError("Max position size cannot exceed 100%")
        if self.risk.max_daily_loss_pct > 100:
            raise ValueError("Max daily loss cannot exceed 100%")

        return True


# Global configuration singleton
_config: ApplicationConfig | None = None


def get_config() -> ApplicationConfig:
    """
    Get the application configuration singleton.

    Returns:
        ApplicationConfig: The application configuration
    """
    global _config
    if _config is None:
        # Avoid circular import by using sys.modules
        import sys

        if "src.application.config_loader" in sys.modules:
            loader = sys.modules["src.application.config_loader"].ConfigLoader
            _config = loader.from_env()
        else:
            # First time import - import module then get class
            from src.application import config_loader

            _config = config_loader.ConfigLoader.from_env()
    return _config


def set_config(config: ApplicationConfig) -> None:
    """
    Set the application configuration.

    Args:
        config: The new configuration
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the configuration singleton."""
    global _config
    _config = None
