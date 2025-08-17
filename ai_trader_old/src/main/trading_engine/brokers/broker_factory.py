"""
Broker Factory Pattern

This module provides a factory for creating broker instances based on
configuration, allowing easy switching between different broker implementations.
"""

# Standard library imports
from enum import Enum
from typing import Any

# Local imports
from main.trading_engine.brokers.alpaca_broker import AlpacaBroker
from main.trading_engine.brokers.backtest_broker import BacktestBroker
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.trading_engine.brokers.ib_broker import IBBroker
from main.trading_engine.brokers.mock_broker import MockBroker
from main.trading_engine.brokers.paper_broker import PaperBroker
from main.utils.core import get_logger
from main.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


class BrokerType(Enum):
    """Supported broker types."""

    ALPACA = "alpaca"
    PAPER = "paper"
    BACKTEST = "backtest"
    INTERACTIVE_BROKERS = "ib"
    MOCK = "mock"


class BrokerFactory:
    """
    Factory for creating broker instances.

    This factory pattern allows:
    - Easy switching between broker implementations
    - Consistent broker initialization
    - Configuration-based broker selection
    - Registration of custom broker implementations
    """

    # Registry of broker implementations
    _broker_registry: dict[BrokerType, type[BrokerInterface]] = {
        BrokerType.ALPACA: AlpacaBroker,
        BrokerType.PAPER: PaperBroker,
        BrokerType.BACKTEST: BacktestBroker,
        BrokerType.INTERACTIVE_BROKERS: IBBroker,
        BrokerType.MOCK: MockBroker,
    }

    @classmethod
    def create_broker(cls, broker_type: str, config: Any) -> BrokerInterface:
        """
        Create a broker instance based on type and configuration.

        Args:
            broker_type: Type of broker to create (e.g., 'alpaca', 'paper')
            config: Configuration object or dict

        Returns:
            BrokerInterface implementation

        Raises:
            ConfigurationError: If broker type is not supported
        """
        try:
            # Convert string to enum
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            supported_types = [bt.value for bt in BrokerType]
            raise ConfigurationError(
                f"Unknown broker type: '{broker_type}'. " f"Supported types: {supported_types}"
            )

        # Get broker class
        broker_class = cls._broker_registry.get(broker_enum)
        if not broker_class:
            raise ConfigurationError(f"No implementation registered for broker type: {broker_type}")

        # Create and return broker instance
        logger.info(f"Creating {broker_type} broker")
        broker = broker_class(config)

        # Log broker configuration (without sensitive data)
        cls._log_broker_info(broker_type, broker)

        return broker

    @classmethod
    def create_from_config(cls, config: Any) -> BrokerInterface:
        """
        Create a broker instance from configuration object.

        The configuration should have a 'broker' section with 'type' field.

        Args:
            config: Configuration object with broker settings

        Returns:
            BrokerInterface implementation
        """
        # Extract broker configuration
        if hasattr(config, "broker"):
            broker_config = config.broker
            broker_type = broker_config.get("type", "paper")
        elif isinstance(config, dict):
            broker_config = config.get("broker", {})
            broker_type = broker_config.get("type", "paper")
        else:
            raise ConfigurationError("Configuration must have a 'broker' section with 'type' field")

        return cls.create_broker(broker_type, config)

    @classmethod
    def register_broker(cls, broker_type: BrokerType, broker_class: type[BrokerInterface]):
        """
        Register a custom broker implementation.

        Args:
            broker_type: Type identifier for the broker
            broker_class: Broker class implementing BrokerInterface
        """
        if not issubclass(broker_class, BrokerInterface):
            raise ValueError(f"{broker_class.__name__} must inherit from BrokerInterface")

        cls._broker_registry[broker_type] = broker_class
        logger.info(f"Registered broker: {broker_type.value} -> {broker_class.__name__}")

    @classmethod
    def get_available_brokers(cls) -> list[str]:
        """Get list of available broker types."""
        return [bt.value for bt in cls._broker_registry.keys()]

    @classmethod
    def _log_broker_info(cls, broker_type: str, broker: BrokerInterface):
        """Log broker information without sensitive data."""
        info = {
            "type": broker_type,
            "class": broker.__class__.__name__,
            "connected": False,  # Not connected yet
        }

        # Add broker-specific non-sensitive info
        if broker_type == "paper":
            if hasattr(broker, "account_balance"):
                info["initial_balance"] = broker.account_balance
        elif broker_type == "alpaca":
            if hasattr(broker, "base_url"):
                info["api_endpoint"] = "paper" if "paper" in broker.base_url else "live"

        logger.info(f"Broker created: {info}")


def create_broker(broker_type: str | None = None, config: Any | None = None) -> BrokerInterface:
    """
    Convenience function to create a broker.

    Args:
        broker_type: Type of broker (if None, uses config)
        config: Configuration object (if None, uses default config)

    Returns:
        BrokerInterface implementation
    """
    # Use default config if not provided
    if config is None:
        # Local imports
        from main.config.config_manager import get_config

        config = get_config()

    # Create broker
    if broker_type:
        return BrokerFactory.create_broker(broker_type, config)
    else:
        return BrokerFactory.create_from_config(config)


# Export commonly used brokers for convenience
def create_paper_broker(config: Any | None = None) -> PaperBroker:
    """Create a paper trading broker."""
    return create_broker("paper", config)


def create_alpaca_broker(config: Any | None = None) -> AlpacaBroker:
    """Create an Alpaca broker."""
    return create_broker("alpaca", config)


def create_mock_broker(config: Any | None = None) -> MockBroker:
    """Create a mock broker for testing."""
    return create_broker("mock", config)


def create_backtest_broker(config: Any | None = None) -> BacktestBroker:
    """Create a backtest broker."""
    return create_broker("backtest", config)


# Create global broker registry instance for backward compatibility
broker_registry = BrokerFactory
