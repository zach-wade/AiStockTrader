"""
Configuration Loader - Handles IO operations for configuration management.

This module is responsible for loading and saving configuration from/to
various sources (YAML files, environment variables) while keeping the
ApplicationConfig class focused on data representation and validation.
"""

import os
from decimal import Decimal

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


class ConfigLoader:
    """Handles loading and saving of configuration from various sources."""

    @classmethod
    def from_env(cls) -> ApplicationConfig:
        """
        Create configuration from environment variables.

        Returns:
            ApplicationConfig: Configuration loaded from environment
        """
        env_str = os.getenv("ENVIRONMENT", "development")
        try:
            environment = Environment(env_str)
        except ValueError:
            raise ValueError(f"Invalid environment: {env_str}")

        return ApplicationConfig(
            environment=environment,
            database=DatabaseConfig.from_env(),
            broker=BrokerConfig.from_env(),
            risk=RiskConfig.from_env(),
            logging=LoggingConfig.from_env(),
            features=FeatureFlags.from_env(),
        )

    @classmethod
    def from_yaml(cls, path: str) -> ApplicationConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ApplicationConfig: Configuration loaded from YAML file
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        config = ApplicationConfig()

        # Handle empty or null YAML files
        if not data:
            return config

        if "environment" in data:
            config.environment = Environment(data["environment"])

        if "database" in data:
            db_data = data["database"]
            config.database = DatabaseConfig(
                host=db_data.get("host", config.database.host),
                port=db_data.get("port", config.database.port),
                database=db_data.get("database", config.database.database),
                user=db_data.get("user", config.database.user),
                password=db_data.get("password", config.database.password),
                pool_size=db_data.get("pool_size", config.database.pool_size),
                max_overflow=db_data.get("max_overflow", config.database.max_overflow),
                echo=db_data.get("echo", config.database.echo),
            )

        if "broker" in data:
            broker_data = data["broker"]
            config.broker = BrokerConfig(
                type=broker_data.get("type", config.broker.type),
                auto_connect=broker_data.get("auto_connect", config.broker.auto_connect),
                api_key=broker_data.get("api_key", config.broker.api_key),
                api_secret=broker_data.get("api_secret", config.broker.api_secret),
                base_url=broker_data.get("base_url", config.broker.base_url),
                data_feed=broker_data.get("data_feed", config.broker.data_feed),
                enable_fractional=broker_data.get(
                    "enable_fractional", config.broker.enable_fractional
                ),
                # Handle legacy attributes if present
                alpaca_api_key=broker_data.get("alpaca_api_key", broker_data.get("api_key")),
                alpaca_secret_key=broker_data.get(
                    "alpaca_secret_key", broker_data.get("api_secret")
                ),
                alpaca_paper=broker_data.get("alpaca_paper", True),
                # Paper broker specific
                paper_initial_capital=Decimal(
                    str(
                        broker_data.get(
                            "paper_initial_capital", config.broker.paper_initial_capital
                        )
                    )
                ),
                paper_slippage_pct=Decimal(
                    str(broker_data.get("paper_slippage_pct", config.broker.paper_slippage_pct))
                ),
                paper_commission_per_share=Decimal(
                    str(
                        broker_data.get(
                            "paper_commission_per_share", config.broker.paper_commission_per_share
                        )
                    )
                ),
                paper_min_commission=Decimal(
                    str(broker_data.get("paper_min_commission", config.broker.paper_min_commission))
                ),
            )

        if "risk" in data:
            risk_data = data["risk"]
            config.risk = RiskConfig(
                max_position_size_pct=Decimal(
                    str(risk_data.get("max_position_size_pct", config.risk.max_position_size_pct))
                ),
                max_total_exposure_pct=Decimal(
                    str(risk_data.get("max_total_exposure_pct", config.risk.max_total_exposure_pct))
                ),
                max_daily_loss_pct=Decimal(
                    str(risk_data.get("max_daily_loss_pct", config.risk.max_daily_loss_pct))
                ),
                min_cash_balance=Decimal(
                    str(risk_data.get("min_cash_balance", config.risk.min_cash_balance))
                ),
                enable_stop_loss=risk_data.get("enable_stop_loss", config.risk.enable_stop_loss),
                default_stop_loss_pct=Decimal(
                    str(risk_data.get("default_stop_loss_pct", config.risk.default_stop_loss_pct))
                ),
            )

        if "logging" in data:
            log_data = data["logging"]
            config.logging = LoggingConfig(
                level=log_data.get("level", config.logging.level),
                format=log_data.get("format", config.logging.format),
                file=log_data.get("file", config.logging.file),
                max_bytes=log_data.get("max_bytes", config.logging.max_bytes),
                backup_count=log_data.get("backup_count", config.logging.backup_count),
            )

        if "features" in data:
            feat_data = data["features"]
            config.features = FeatureFlags(
                enable_caching=feat_data.get("enable_caching", config.features.enable_caching),
                enable_metrics=feat_data.get("enable_metrics", config.features.enable_metrics),
                enable_tracing=feat_data.get("enable_tracing", config.features.enable_tracing),
                enable_backtesting=feat_data.get(
                    "enable_backtesting", config.features.enable_backtesting
                ),
                enable_paper_trading=feat_data.get(
                    "enable_paper_trading", config.features.enable_paper_trading
                ),
                enable_live_trading=feat_data.get(
                    "enable_live_trading", config.features.enable_live_trading
                ),
                enable_notifications=feat_data.get(
                    "enable_notifications", config.features.enable_notifications
                ),
                enable_web_ui=feat_data.get("enable_web_ui", config.features.enable_web_ui),
            )

        return config

    @classmethod
    def to_yaml(cls, config: ApplicationConfig) -> str:
        """
        Convert configuration to YAML string.

        Args:
            config: ApplicationConfig instance to convert

        Returns:
            str: YAML representation of the configuration
        """
        return yaml.dump(config.to_dict(), default_flow_style=False)

    @classmethod
    def save_to_yaml(cls, config: ApplicationConfig, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: ApplicationConfig instance to save
            path: Path to save the YAML file to
        """
        yaml_content = cls.to_yaml(config)
        with open(path, "w") as f:
            f.write(yaml_content)
