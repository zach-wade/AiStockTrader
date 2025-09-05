"""
Broker Factory - Creates broker instances based on configuration.

This is a simplified factory that creates thin broker adapters.
Business logic has been moved to domain services.
"""

import logging
import os
from decimal import Decimal
from typing import Any

from src.application.interfaces.broker import IBroker
from src.domain.services.trading_calendar import Exchange
from src.infrastructure.brokers.broker_configuration_service import (
    BrokerConfigurationService,
    BrokerType,
)

from .alpaca_broker import AlpacaBroker
from .paper_broker import PaperBroker

logger = logging.getLogger(__name__)


class BrokerFactory:
    """
    Simple factory for creating broker instances.

    This factory creates thin broker adapters using domain services for business logic.
    Complex configuration logic is handled by BrokerConfigurationService.
    """

    def __init__(self, config_service: BrokerConfigurationService | None = None) -> None:
        """Initialize with optional configuration service."""
        self.config_service = config_service or BrokerConfigurationService()

    def create_broker(self, broker_type: str | None = None, **kwargs: Any) -> IBroker:
        """
        Create a broker instance based on type.

        Args:
            broker_type: Type of broker to create ("alpaca", "paper", "backtest")
                        If None, will check BROKER_TYPE environment variable
            **kwargs: Broker-specific configuration

        Returns:
            Configured broker instance

        Raises:
            ValueError: If broker type is not supported
        """
        # Use domain service to determine broker type
        type_str = broker_type or os.getenv("BROKER_TYPE")
        broker_enum = self.config_service.determine_broker_type(type_str, "paper")

        logger.info(f"Creating {broker_enum.value} broker")

        # Simple switch based on type - no complex logic
        if broker_enum == BrokerType.ALPACA:
            return self._create_alpaca_broker(**kwargs)
        elif broker_enum == BrokerType.PAPER:
            return self._create_paper_broker(**kwargs)
        elif broker_enum == BrokerType.BACKTEST:
            return self._create_backtest_broker(**kwargs)
        else:
            # This should never happen due to enum validation
            raise ValueError(f"Unexpected broker type: {broker_enum}")

    def _create_alpaca_broker(self, **kwargs: Any) -> AlpacaBroker:
        """Create Alpaca broker instance - simple factory method."""
        # Get configuration from environment or kwargs
        api_key = kwargs.get("api_key") or os.getenv("ALPACA_API_KEY")
        secret_key = kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY")

        # Use domain service for paper mode determination
        paper = self.config_service.determine_paper_mode(
            kwargs.get("paper"), default_paper=os.getenv("ALPACA_PAPER", "true").lower() == "true"
        )

        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        # Simple auto-connect check
        if kwargs.get("auto_connect", True):
            broker.connect()

        return broker

    def _create_paper_broker(self, **kwargs: Any) -> PaperBroker:
        """Create paper trading broker instance - simple factory method."""
        # Use domain service for capital normalization
        initial_capital = self.config_service.normalize_initial_capital(
            kwargs.get("initial_capital") or os.getenv("PAPER_INITIAL_CAPITAL")
        )

        # Get exchange for trading calendar
        exchange = kwargs.get("exchange", Exchange.NYSE)

        broker = PaperBroker(
            initial_capital=initial_capital,
            exchange=exchange,
        )

        # Simple auto-connect check
        if kwargs.get("auto_connect", True):
            broker.connect()

        return broker

    def _create_backtest_broker(self, **kwargs: Any) -> PaperBroker:
        """Create backtesting broker instance - delegates to paper broker."""
        # Set defaults for backtesting
        kwargs.setdefault("initial_capital", Decimal("100000"))
        kwargs.setdefault("exchange", Exchange.NYSE)

        # Create paper broker for backtesting
        broker = self._create_paper_broker(**kwargs)

        logger.info("Created backtest broker (using paper broker)")

        return broker

    def create_from_config(self, config: dict[str, Any]) -> IBroker:
        """
        Create broker from configuration dictionary.

        Args:
            config: Configuration dictionary with broker settings

        Returns:
            Configured broker instance

        Example config:
            {
                "type": "alpaca",
                "paper": true,
                "auto_connect": true,
                "api_key": "...",
                "secret_key": "..."
            }
        """
        # Use domain service to process configuration
        processed_config = self.config_service.process_broker_config(config)
        broker_type = processed_config.pop("type", None)
        return self.create_broker(broker_type, **processed_config)

    def get_default_config(self, broker_type: str) -> dict[str, Any]:
        """
        Get default configuration for a broker type.

        Args:
            broker_type: Type of broker

        Returns:
            Default configuration dictionary
        """
        # Delegate to domain service
        broker_enum = self.config_service.determine_broker_type(broker_type)
        config = self.config_service.get_default_config(broker_enum)
        return dict(config)
