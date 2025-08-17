"""
Main configuration model that brings together all validation models.
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
from pydantic import BaseModel, Field, ValidationError, model_validator

from .core import ApiKeysConfig, Environment
from .data import DataConfig, FeaturesConfig, PathsConfig, TrainingConfig, UniverseMainConfig
from .services import EnvironmentOverrides, MonitoringConfig, OrchestratorConfig
from .trading import BrokerConfig, RiskConfig, SystemConfig, TradingConfig

logger = logging.getLogger(__name__)


class AITraderConfig(BaseModel):
    """
    Main AI Trader configuration model with comprehensive validation.

    This model provides fail-fast validation for all configuration parameters
    to prevent runtime errors and ensure system reliability.
    """

    # Core configuration sections
    system: SystemConfig = Field(default_factory=SystemConfig, description="System configuration")
    api_keys: ApiKeysConfig = Field(..., description="API keys configuration")
    broker: BrokerConfig = Field(default_factory=BrokerConfig, description="Broker configuration")
    database: dict[str, Any] | None = Field(default=None, description="Database configuration")
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    trading: TradingConfig = Field(
        default_factory=TradingConfig, description="Trading configuration"
    )
    risk: RiskConfig = Field(
        default_factory=RiskConfig, description="Risk management configuration"
    )
    features: FeaturesConfig = Field(
        default_factory=FeaturesConfig, description="Features configuration"
    )
    strategies: dict[str, Any] = Field(default_factory=dict, description="Strategies configuration")
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    universe: UniverseMainConfig = Field(
        default_factory=UniverseMainConfig, description="Universe configuration"
    )
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Paths configuration")
    orchestrator: OrchestratorConfig = Field(
        default_factory=OrchestratorConfig, description="Orchestrator configuration"
    )

    # Environment-specific overrides
    environments: dict[str, EnvironmentOverrides] | None = Field(
        default=None, description="Environment overrides"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "allow",  # Allow unknown fields for backward compatibility
        "json_schema_extra": {
            "example": {
                "system": {"environment": "paper"},
                "api_keys": {
                    "alpaca": {"key": "${ALPACA_API_KEY}", "secret": "${ALPACA_SECRET_KEY}"}
                },
                "broker": {"paper_trading": True},
                "risk": {
                    "position_sizing": {"max_position_size": 5.0},
                    "circuit_breaker": {"daily_loss_limit": 3.0},
                },
            }
        },
    }

    @model_validator(mode="after")
    def validate_environment_consistency(self):
        """Ensure environment-specific settings are consistent."""
        # In paper environment, ensure paper trading is enabled
        if self.system.environment == Environment.PAPER and not self.broker.paper_trading:
            raise ValueError("Paper trading must be enabled in paper environment")

        # In live environment, warn about paper trading
        if self.system.environment == Environment.LIVE:
            if self.broker.paper_trading:
                logger.warning("Paper trading is enabled in live environment")
            else:
                logger.warning("LIVE TRADING ENABLED - Ensure this is intentional!")

        return self

    @model_validator(mode="after")
    def validate_risk_consistency(self):
        """Ensure risk settings are consistent across sections."""
        # Ensure position sizing is consistent
        risk_max = self.risk.position_sizing.max_position_size
        trading_max = self.trading.position_sizing.max_position_size

        # Convert percentage to dollar amount for comparison
        starting_cash = self.trading.starting_cash
        # Handle both decimal (0.05) and percentage (5.0) formats
        if risk_max <= 1.0:
            # Decimal format (0.05 = 5%)
            risk_max_dollars = risk_max * starting_cash
        else:
            # Percentage format (5.0 = 5%)
            risk_max_dollars = (risk_max / 100.0) * starting_cash

        if trading_max > risk_max_dollars:
            logger.warning(
                f"Trading max position size (${trading_max}) exceeds risk limit (${risk_max_dollars})"
            )

        return self

    def get_environment_config(self) -> "AITraderConfig":
        """Get configuration with environment-specific overrides applied."""
        if not self.environments:
            return self

        env_name = self.system.environment.value
        if env_name not in self.environments:
            return self

        # Create a copy and apply environment overrides using simple overlay
        config_dict = self.model_dump()
        env_overrides = self.environments[env_name]

        # Apply overrides using a simplified approach
        merged_config = self._apply_environment_overrides(config_dict, env_overrides)

        return AITraderConfig(**merged_config)

    def _apply_environment_overrides(
        self, base_config: dict, overrides: EnvironmentOverrides
    ) -> dict:
        """
        Apply environment overrides in a predictable, documented manner.

        This method replaces the complex deep merge with a simpler overlay system
        that applies overrides at specific, well-defined paths.

        Args:
            base_config: Base configuration dictionary
            overrides: Environment-specific overrides

        Returns:
            Configuration with overrides applied
        """
        # Convert overrides to dict, excluding unset values
        override_dict = overrides.model_dump(exclude_unset=True)

        # Apply each override explicitly at the correct path
        for section, values in override_dict.items():
            if section in base_config and isinstance(values, dict):
                # Merge section-level overrides
                if isinstance(base_config[section], dict):
                    base_config[section].update(values)
                else:
                    base_config[section] = values
            else:
                # Set new section
                base_config[section] = values

        return base_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Backward compatibility method for legacy code that expects Config.get() method.

        Args:
            key: Dot-separated path to the configuration value (e.g., 'api_keys.polygon.key')
            default: Default value to return if key not found

        Returns:
            The configuration value or default if not found
        """
        try:
            keys = key.split(".")
            value = self
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default


# Validation helper functions
def validate_config_file(config_path: str) -> AITraderConfig:
    """
    Validate a configuration file and return the validated config.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated AITraderConfig instance

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If config file not found
    """
    # Standard library imports
    from pathlib import Path

    # Third-party imports
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    try:
        return AITraderConfig(**config_dict)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def get_validation_errors(config_dict: dict[str, Any]) -> list[str]:
    """
    Get detailed validation errors for a configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        List of validation error messages
    """
    try:
        AITraderConfig(**config_dict)
        return []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")
        return errors
