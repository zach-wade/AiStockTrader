"""
Broker Factory - Creates appropriate broker instances based on configuration
"""

# Standard library imports
from decimal import Decimal
import logging
import os
from typing import Literal

# Local imports
from src.application.interfaces.broker import IBroker

from .alpaca_broker import AlpacaBroker
from .paper_broker import PaperBroker

logger = logging.getLogger(__name__)

BrokerType = Literal["alpaca", "paper", "backtest"]


class BrokerFactory:
    """
    Factory for creating broker instances.

    Supports multiple broker types and configurations.
    """

    @staticmethod
    def create_broker(broker_type: BrokerType | None = None, **kwargs) -> IBroker:
        """
        Create a broker instance based on type and configuration.

        Args:
            broker_type: Type of broker to create ("alpaca", "paper", "backtest")
                        If None, will check BROKER_TYPE environment variable
            **kwargs: Additional broker-specific configuration

        Returns:
            Configured broker instance

        Raises:
            ValueError: If broker type is not supported
        """
        # Determine broker type
        if broker_type is None:
            broker_type = os.getenv("BROKER_TYPE", "paper").lower()

        broker_type = broker_type.lower()

        logger.info(f"Creating {broker_type} broker")

        if broker_type == "alpaca":
            return BrokerFactory._create_alpaca_broker(**kwargs)
        elif broker_type == "paper":
            return BrokerFactory._create_paper_broker(**kwargs)
        elif broker_type == "backtest":
            return BrokerFactory._create_backtest_broker(**kwargs)
        else:
            raise ValueError(
                f"Unsupported broker type: {broker_type}. "
                f"Supported types: alpaca, paper, backtest"
            )

    @staticmethod
    def _create_alpaca_broker(**kwargs) -> AlpacaBroker:
        """Create Alpaca broker instance"""
        # Get configuration from environment or kwargs
        api_key = kwargs.get("api_key") or os.getenv("ALPACA_API_KEY")
        secret_key = kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY")

        # Determine if paper trading
        paper = kwargs.get("paper")
        if paper is None:
            # Default to paper trading unless explicitly set to live
            paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        base_url = kwargs.get("base_url") or os.getenv("ALPACA_BASE_URL")

        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            base_url=base_url,
        )

        # Auto-connect if requested
        if kwargs.get("auto_connect", True):
            broker.connect()

        return broker

    @staticmethod
    def _create_paper_broker(**kwargs) -> PaperBroker:
        """Create paper trading broker instance"""
        # Get configuration
        initial_capital = kwargs.get("initial_capital")
        if initial_capital is None:
            initial_capital = Decimal(os.getenv("PAPER_INITIAL_CAPITAL", "100000"))
        elif not isinstance(initial_capital, Decimal):
            initial_capital = Decimal(str(initial_capital))

        slippage_pct = kwargs.get("slippage_pct")
        if slippage_pct is None:
            slippage_pct = Decimal(os.getenv("PAPER_SLIPPAGE_PCT", "0.001"))
        elif not isinstance(slippage_pct, Decimal):
            slippage_pct = Decimal(str(slippage_pct))

        fill_delay = kwargs.get("fill_delay_seconds")
        if fill_delay is None:
            fill_delay = int(os.getenv("PAPER_FILL_DELAY", "1"))

        commission_per_share = kwargs.get("commission_per_share")
        if commission_per_share is None:
            commission_per_share = Decimal(os.getenv("PAPER_COMMISSION_PER_SHARE", "0.01"))
        elif not isinstance(commission_per_share, Decimal):
            commission_per_share = Decimal(str(commission_per_share))

        min_commission = kwargs.get("min_commission")
        if min_commission is None:
            min_commission = Decimal(os.getenv("PAPER_MIN_COMMISSION", "1.0"))
        elif not isinstance(min_commission, Decimal):
            min_commission = Decimal(str(min_commission))

        simulate_partial_fills = kwargs.get("simulate_partial_fills")
        if simulate_partial_fills is None:
            simulate_partial_fills = os.getenv("PAPER_PARTIAL_FILLS", "false").lower() == "true"

        broker = PaperBroker(
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            fill_delay_seconds=fill_delay,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            simulate_partial_fills=simulate_partial_fills,
        )

        # Auto-connect if requested
        if kwargs.get("auto_connect", True):
            broker.connect()

        return broker

    @staticmethod
    def _create_backtest_broker(**kwargs) -> PaperBroker:
        """
        Create backtesting broker instance.

        For now, this is similar to paper broker but with different defaults
        optimized for backtesting (no delays, minimal slippage).
        """
        # Override defaults for backtesting
        kwargs.setdefault("fill_delay_seconds", 0)  # No delay in backtesting
        kwargs.setdefault("slippage_pct", Decimal("0.0005"))  # Minimal slippage
        kwargs.setdefault("simulate_partial_fills", False)  # Simpler for backtesting

        broker = BrokerFactory._create_paper_broker(**kwargs)

        # Mark as backtest mode (could be used for special behavior)
        broker.portfolio.tags["mode"] = "backtest"

        return broker

    @staticmethod
    def create_from_config(config: dict) -> IBroker:
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
        broker_type = config.pop("type", None)
        return BrokerFactory.create_broker(broker_type, **config)

    @staticmethod
    def get_default_config(broker_type: BrokerType) -> dict:
        """
        Get default configuration for a broker type.

        Args:
            broker_type: Type of broker

        Returns:
            Default configuration dictionary
        """
        if broker_type == "alpaca":
            return {
                "type": "alpaca",
                "paper": True,
                "auto_connect": True,
                "api_key": None,  # Must be provided
                "secret_key": None,  # Must be provided
            }
        elif broker_type == "paper":
            return {
                "type": "paper",
                "auto_connect": True,
                "initial_capital": "100000",
                "slippage_pct": "0.001",
                "fill_delay_seconds": 1,
                "commission_per_share": "0.01",
                "min_commission": "1.0",
                "simulate_partial_fills": False,
            }
        elif broker_type == "backtest":
            return {
                "type": "backtest",
                "auto_connect": True,
                "initial_capital": "100000",
                "slippage_pct": "0.0005",
                "fill_delay_seconds": 0,
                "commission_per_share": "0.01",
                "min_commission": "1.0",
                "simulate_partial_fills": False,
            }
        else:
            raise ValueError(f"Unknown broker type: {broker_type}")
