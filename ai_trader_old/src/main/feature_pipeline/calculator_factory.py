"""
Feature Calculator Factory

Factory implementation for creating feature calculators.
This is the concrete implementation of ICalculatorFactory.
"""

# Standard library imports
import logging
from typing import Any

# Local imports
from main.interfaces.calculators import (
    CalculatorConfig,
    ICalculatorFactory,
    ISentimentCalculator,
    ITechnicalCalculator,
)

from .calculators.sentiment_adapter import SentimentCalculatorAdapter

# Import adapter implementations
from .calculators.technical_adapter import TechnicalCalculatorAdapter

logger = logging.getLogger(__name__)


class FeatureCalculatorFactory(ICalculatorFactory):
    """
    Factory for creating feature calculators.

    This factory creates calculator instances based on configuration
    and manages the registry of available calculators.
    """

    def __init__(self):
        """Initialize the calculator factory."""
        self._registry = {
            "technical": {
                "default": TechnicalCalculatorAdapter,
                "technical_indicators": TechnicalCalculatorAdapter,
            },
            "sentiment": {
                "default": SentimentCalculatorAdapter,
                "sentiment_features": SentimentCalculatorAdapter,
            },
        }

    def create_technical_calculator(
        self, config: dict[str, Any] | CalculatorConfig
    ) -> ITechnicalCalculator:
        """
        Create a technical indicator calculator.

        Args:
            config: Configuration for the calculator

        Returns:
            Technical calculator instance
        """
        if isinstance(config, CalculatorConfig):
            calculator_name = config.name
            calc_config = config.parameters
        else:
            calculator_name = config.get("name", "default")
            calc_config = config

        calculator_class = self._registry["technical"].get(
            calculator_name, self._registry["technical"]["default"]
        )

        logger.info(f"Creating technical calculator: {calculator_name}")
        return calculator_class(calc_config)

    def create_sentiment_calculator(
        self, config: dict[str, Any] | CalculatorConfig
    ) -> ISentimentCalculator:
        """
        Create a sentiment analysis calculator.

        Args:
            config: Configuration for the calculator

        Returns:
            Sentiment calculator instance
        """
        if isinstance(config, CalculatorConfig):
            calculator_name = config.name
            calc_config = config.parameters
        else:
            calculator_name = config.get("name", "default")
            calc_config = config

        calculator_class = self._registry["sentiment"].get(
            calculator_name, self._registry["sentiment"]["default"]
        )

        logger.info(f"Creating sentiment calculator: {calculator_name}")
        return calculator_class(calc_config)

    def get_available_calculators(self) -> dict[str, list[str]]:
        """
        Get list of available calculator types.

        Returns:
            Dictionary mapping calculator type to list of implementations
        """
        return {
            calc_type: list(implementations.keys())
            for calc_type, implementations in self._registry.items()
        }

    def register_calculator(
        self, calculator_type: str, calculator_class: type, name: str | None = None
    ) -> None:
        """
        Register a new calculator implementation.

        Args:
            calculator_type: Type of calculator (technical, sentiment, etc.)
            calculator_class: Class implementing the calculator interface
            name: Optional name for the calculator
        """
        if calculator_type not in self._registry:
            self._registry[calculator_type] = {}

        calc_name = name or calculator_class.__name__.lower()
        self._registry[calculator_type][calc_name] = calculator_class

        logger.info(
            f"Registered {calculator_type} calculator: {calc_name} "
            f"({calculator_class.__name__})"
        )


# Global factory instance
_global_factory = None


def get_calculator_factory() -> FeatureCalculatorFactory:
    """
    Get the global calculator factory instance.

    Returns:
        Global FeatureCalculatorFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = FeatureCalculatorFactory()
    return _global_factory
