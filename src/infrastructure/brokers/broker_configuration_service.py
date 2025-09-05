"""
Broker Configuration Service - Domain service for broker configuration logic.

This service handles business logic related to broker configuration and setup,
moved from the infrastructure layer to maintain clean architecture.
"""

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


class BrokerType(Enum):
    """Supported broker types."""

    ALPACA = "alpaca"
    PAPER = "paper"
    BACKTEST = "backtest"


class BrokerConfigurationService:
    """
    Domain service for broker configuration logic.

    This service contains business rules and logic for broker configuration
    that was previously in the infrastructure layer.
    """

    # Default configuration values - business rules
    DEFAULT_INITIAL_CAPITAL = Decimal("100000")
    DEFAULT_PAPER_TRADING = True
    DEFAULT_AUTO_CONNECT = True

    @staticmethod
    def determine_broker_type(broker_type: str | None, fallback_type: str = "paper") -> BrokerType:
        """
        Determine the broker type based on input and business rules.

        Args:
            broker_type: Optional broker type string
            fallback_type: Default type if none provided

        Returns:
            Validated BrokerType enum

        Raises:
            ValueError: If broker type is invalid
        """
        if broker_type is None:
            broker_type = fallback_type

        broker_type = broker_type.lower().strip()

        try:
            return BrokerType(broker_type)
        except ValueError:
            raise ValueError(
                f"Unsupported broker type: {broker_type}. "
                f"Supported types: {', '.join([t.value for t in BrokerType])}"
            )

    @staticmethod
    def normalize_initial_capital(capital: Any) -> Decimal:
        """
        Normalize and validate initial capital amount.

        Args:
            capital: Initial capital amount

        Returns:
            Validated capital as Decimal

        Raises:
            ValueError: If capital is invalid
        """
        if capital is None:
            return BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL

        try:
            decimal_capital = Decimal(str(capital))
        except (ValueError, TypeError, InvalidOperation) as e:
            raise ValueError(f"Invalid initial capital: {capital}") from e

        if decimal_capital <= 0:
            raise ValueError(f"Initial capital must be positive: {decimal_capital}")

        return decimal_capital

    @staticmethod
    def determine_paper_mode(paper: bool | None, default_paper: bool = True) -> bool:
        """
        Determine whether to use paper trading mode.

        Args:
            paper: Optional paper mode setting
            default_paper: Default value if not specified

        Returns:
            Boolean indicating paper mode
        """
        if paper is None:
            return default_paper
        return bool(paper)

    @staticmethod
    def get_default_config(broker_type: BrokerType) -> dict[str, Any]:
        """
        Get default configuration for a broker type.

        This encapsulates the business logic for default configurations.

        Args:
            broker_type: Type of broker

        Returns:
            Default configuration dictionary
        """
        if broker_type == BrokerType.ALPACA:
            return {
                "type": broker_type.value,
                "paper": BrokerConfigurationService.DEFAULT_PAPER_TRADING,
                "auto_connect": BrokerConfigurationService.DEFAULT_AUTO_CONNECT,
                "api_key": None,  # Must be provided
                "secret_key": None,  # Must be provided
            }
        elif broker_type == BrokerType.PAPER or broker_type == BrokerType.BACKTEST:
            return {
                "type": broker_type.value,
                "auto_connect": BrokerConfigurationService.DEFAULT_AUTO_CONNECT,
                "initial_capital": str(BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL),
                "exchange": "NYSE",  # Default exchange
            }
        else:
            raise ValueError(f"Unknown broker type: {broker_type}")

    @staticmethod
    def validate_alpaca_config(api_key: str | None, secret_key: str | None) -> bool:
        """
        Validate Alpaca broker configuration.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are required")

        # Additional validation could be added here
        # e.g., check key format, length, etc.

        return True

    @staticmethod
    def process_broker_config(config: dict[str, Any]) -> dict[str, Any]:
        """
        Process and validate broker configuration.

        This method applies business rules to the configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            Processed configuration dictionary
        """
        processed = config.copy()

        # Extract and validate broker type
        broker_type = BrokerConfigurationService.determine_broker_type(config.get("type"))
        processed["type"] = broker_type.value

        # Apply broker-specific processing
        if broker_type == BrokerType.ALPACA:
            processed["paper"] = BrokerConfigurationService.determine_paper_mode(
                config.get("paper")
            )
        elif broker_type in (BrokerType.PAPER, BrokerType.BACKTEST):
            if "initial_capital" in config:
                processed["initial_capital"] = BrokerConfigurationService.normalize_initial_capital(
                    config.get("initial_capital")
                )

        return processed
