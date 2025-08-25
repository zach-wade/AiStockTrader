"""
Configuration Management - Loads settings from secure secrets management
"""

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from dotenv import load_dotenv

from src.infrastructure.security import SecretNotFoundError, SecretsManager

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls, prefix: str = "") -> "DatabaseConfig":
        """Load database config from secure secrets management"""
        secrets_manager = SecretsManager.get_instance()

        try:
            # Try to get from secrets manager first
            db_config = secrets_manager.get_database_config()

            # Apply prefix if provided
            if prefix:
                # For test database, use different credentials
                db_config["user"] = (
                    secrets_manager.get_secret(f"{prefix}DB_USER", required=False)
                    or db_config["user"]
                )
                db_config["password"] = (
                    secrets_manager.get_secret(f"{prefix}DB_PASSWORD", required=False)
                    or db_config["password"]
                )
                db_config["database"] = (
                    secrets_manager.get_secret(f"{prefix}DB_NAME", required=False)
                    or db_config["database"]
                )

            return cls(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
            )

        except SecretNotFoundError as e:
            logger.error(f"Required database credentials not found: {e}")
            raise RuntimeError(
                "Database credentials not configured. Please set DB_USER and DB_PASSWORD "
                "environment variables or configure a secrets provider."
            )

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class BrokerConfig:
    """Broker configuration settings"""

    alpaca_api_key: str | None
    alpaca_api_secret: str | None
    alpaca_base_url: str
    polygon_api_key: str | None

    @classmethod
    def from_env(cls) -> "BrokerConfig":
        """Load broker config from environment variables"""
        return cls(
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_api_secret=os.getenv("ALPACA_API_SECRET"),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
        )


@dataclass
class TradingConfig:
    """Trading configuration settings"""

    initial_capital: Decimal
    max_position_size: Decimal
    max_portfolio_concentration: Decimal
    environment: str
    log_level: str

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load trading config from environment variables"""
        return cls(
            initial_capital=Decimal(os.getenv("INITIAL_CAPITAL", "100000")),
            max_position_size=Decimal(os.getenv("MAX_POSITION_SIZE", "10000")),
            max_portfolio_concentration=Decimal(os.getenv("MAX_PORTFOLIO_CONCENTRATION", "0.20")),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@dataclass
class CacheConfig:
    """Cache configuration settings"""

    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: str | None
    redis_ssl: bool
    redis_max_connections: int
    cache_default_ttl: int
    cache_key_prefix: str

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load cache config from environment variables"""
        return cls(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            redis_max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            cache_default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
            cache_key_prefix=os.getenv("CACHE_KEY_PREFIX", "trading"),
        )


@dataclass
class Config:
    """Application configuration"""

    database: DatabaseConfig
    test_database: DatabaseConfig
    broker: BrokerConfig
    trading: TradingConfig
    cache: CacheConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load all configuration from environment variables"""
        return cls(
            database=DatabaseConfig.from_env(),
            test_database=DatabaseConfig.from_env(prefix="TEST_"),
            broker=BrokerConfig.from_env(),
            trading=TradingConfig.from_env(),
            cache=CacheConfig.from_env(),
        )


# Global configuration instance (lazy-loaded)
_config: Config | None = None


def _get_config() -> Config:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def get_database_config(test: bool = False) -> DatabaseConfig:
    """Get database configuration"""
    config = _get_config()
    return config.test_database if test else config.database


def get_broker_config() -> BrokerConfig:
    """Get broker configuration"""
    return _get_config().broker


def get_trading_config() -> TradingConfig:
    """Get trading configuration"""
    return _get_config().trading


def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return _get_config().cache


class DSNBuilder:
    """
    Builder for database connection strings (DSN).
    Handles the construction of PostgreSQL connection strings with various options.
    """

    def __init__(self) -> None:
        """Initialize DSN builder with default values."""
        self.host = "localhost"
        self.port = 5432
        self.database = "ai_trader"
        self.user = ""
        self.password: str | None = None
        self.ssl_mode = "prefer"
        self.ssl_cert: str | None = None
        self.ssl_key: str | None = None
        self.ssl_ca: str | None = None
        self.additional_params: dict[str, Any] = {}

    def with_host(self, host: str) -> "DSNBuilder":
        """Set the host."""
        self.host = host
        return self

    def with_port(self, port: int) -> "DSNBuilder":
        """Set the port."""
        self.port = port
        return self

    def with_database(self, database: str) -> "DSNBuilder":
        """Set the database name."""
        self.database = database
        return self

    def with_credentials(self, user: str, password: str | None = None) -> "DSNBuilder":
        """Set user credentials."""
        self.user = user
        self.password = password
        return self

    def with_ssl(
        self,
        mode: str = "prefer",
        cert: str | None = None,
        key: str | None = None,
        ca: str | None = None,
    ) -> "DSNBuilder":
        """Configure SSL settings."""
        self.ssl_mode = mode
        self.ssl_cert = cert
        self.ssl_key = key
        self.ssl_ca = ca
        return self

    def with_param(self, key: str, value: str) -> "DSNBuilder":
        """Add additional connection parameter."""
        self.additional_params[key] = value
        return self

    def build(self) -> str:
        """
        Build the DSN string.

        Returns:
            Complete PostgreSQL DSN string
        """
        # Build base URL
        if self.password:
            auth = f"{self.user}:{self.password}"
        else:
            auth = self.user

        dsn = f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"

        # Collect all parameters
        params = dict(self.additional_params)

        # Add SSL parameters if not default
        if self.ssl_mode != "prefer":
            params["sslmode"] = self.ssl_mode
        if self.ssl_cert:
            params["sslcert"] = self.ssl_cert
        if self.ssl_key:
            params["sslkey"] = self.ssl_key
        if self.ssl_ca:
            params["sslrootcert"] = self.ssl_ca

        # Append parameters if any
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            dsn = f"{dsn}?{param_str}"

        return dsn

    @classmethod
    def from_config(
        cls,
        config: DatabaseConfig,
        ssl_mode: str = "prefer",
        ssl_cert: str | None = None,
        ssl_key: str | None = None,
        ssl_ca: str | None = None,
    ) -> str:
        """
        Build DSN from DatabaseConfig with optional SSL settings.

        Args:
            config: Database configuration
            ssl_mode: SSL mode (prefer, require, disable, etc.)
            ssl_cert: Path to SSL certificate file
            ssl_key: Path to SSL key file
            ssl_ca: Path to SSL CA file

        Returns:
            Complete PostgreSQL DSN string
        """
        builder = cls()
        return (
            builder.with_host(config.host)
            .with_port(config.port)
            .with_database(config.database)
            .with_credentials(config.user, config.password)
            .with_ssl(ssl_mode, ssl_cert, ssl_key, ssl_ca)
            .build()
        )
