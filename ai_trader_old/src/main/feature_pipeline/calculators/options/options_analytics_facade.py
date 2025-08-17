"""
Options Analytics Facade

Facade providing backward compatibility for the original OptionsAnalyticsCalculator
while delegating to specialized calculators for improved maintainability.

This facade implements the exact same interface as the original monolithic calculator
to ensure 100% backward compatibility for existing code.
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import pandas as pd

from .blackscholes_calculator import BlackScholesCalculator
from .greeks_calculator import GreeksCalculator
from .iv_calculator import ImpliedVolatilityCalculator
from .moneyness_calculator import MoneynessCalculator
from .options_config import OptionsConfig
from .putcall_calculator import PutCallAnalysisCalculator
from .sentiment_calculator import OptionsSentimentCalculator as SentimentCalculator
from .unusual_activity_calculator import UnusualActivityCalculator
from .volume_flow_calculator import VolumeFlowCalculator

logger = logging.getLogger(__name__)


class OptionsAnalyticsFacade:
    """
    Facade that maintains backward compatibility with the original OptionsAnalyticsCalculator
    while using the new modular architecture internally.
    """

    def __init__(self, config: dict | None = None):
        """Initialize the facade with all specialized calculators."""
        self.config = config or {}
        self.options_config = OptionsConfig(**self.config.get("options", {}))

        # Initialize all specialized calculators
        self._init_calculators()

        # Maintain original interface properties
        self.options_chain = None
        self.historical_iv = None

        logger.debug("Initialized OptionsAnalyticsFacade with 8 specialized calculators")

    def _init_calculators(self):
        """Initialize all specialized calculators."""
        try:
            calc_config = {"options": self.config.get("options", {})}

            self.volume_flow_calc = VolumeFlowCalculator(calc_config)
            self.putcall_calc = PutCallAnalysisCalculator(calc_config)
            self.iv_calc = ImpliedVolatilityCalculator(calc_config)
            self.greeks_calc = GreeksCalculator(calc_config)
            self.moneyness_calc = MoneynessCalculator(calc_config)
            self.unusual_activity_calc = UnusualActivityCalculator(calc_config)
            self.sentiment_calc = SentimentCalculator(calc_config)
            self.blackscholes_calc = BlackScholesCalculator(calc_config)

            # Store all calculators for easy iteration
            self.calculators = [
                self.volume_flow_calc,
                self.putcall_calc,
                self.iv_calc,
                self.greeks_calc,
                self.moneyness_calc,
                self.unusual_activity_calc,
                self.sentiment_calc,
                self.blackscholes_calc,
            ]

        except Exception as e:
            logger.error(f"Error initializing specialized calculators: {e}")
            raise

    def set_options_data(self, options_chain: pd.DataFrame):
        """
        Set options chain data for all calculators.
        Maintains original interface.
        """
        self.options_chain = options_chain

        # Propagate to all calculators
        for calc in self.calculators:
            try:
                calc.set_options_data(options_chain)
            except Exception as e:
                logger.warning(f"Error setting options data for {calc.__class__.__name__}: {e}")

    def set_historical_iv(self, historical_iv: pd.DataFrame):
        """
        Set historical implied volatility data for all calculators.
        Maintains original interface.
        """
        self.historical_iv = historical_iv

        # Propagate to calculators that use historical IV
        iv_calculators = [self.iv_calc, self.sentiment_calc, self.blackscholes_calc]
        for calc in iv_calculators:
            try:
                calc.set_historical_iv(historical_iv)
            except Exception as e:
                logger.warning(f"Error setting historical IV for {calc.__class__.__name__}: {e}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all options features using specialized calculators.
        Maintains exact same interface as original calculator.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all options features combined
        """
        try:
            if not self.validate_inputs(data):
                logger.error("Invalid input data for options calculation")
                return pd.DataFrame()

            # Initialize combined features DataFrame
            features = pd.DataFrame(index=data.index)

            # Execute all calculators and combine results
            for calc in self.calculators:
                try:
                    calc_features = calc.calculate(data)
                    if not calc_features.empty:
                        # Merge features (concat along columns)
                        features = pd.concat([features, calc_features], axis=1)
                        logger.debug(
                            f"{calc.__class__.__name__} calculated {len(calc_features.columns)} features"
                        )
                    else:
                        logger.warning(f"{calc.__class__.__name__} returned empty features")

                except Exception as e:
                    logger.error(f"Error in {calc.__class__.__name__}: {e}")
                    # Continue with other calculators even if one fails
                    continue

            logger.info(f"OptionsAnalyticsFacade calculated {len(features.columns)} total features")
            return features

        except Exception as e:
            logger.error(f"Error in OptionsAnalyticsFacade.calculate: {e}")
            return pd.DataFrame(index=data.index)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.
        Maintains original interface.
        """
        try:
            if data is None or data.empty:
                return False

            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_columns):
                return False

            # Check for sufficient data
            if len(data) < 1:
                return False

            # Check for valid price data
            if data["close"].iloc[-1] <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating inputs: {e}")
            return False

    def get_feature_names(self) -> list[str]:
        """
        Get all feature names from all calculators.
        Maintains original interface.
        """
        try:
            all_features = []

            for calc in self.calculators:
                try:
                    calc_features = calc.get_feature_names()
                    all_features.extend(calc_features)
                except Exception as e:
                    logger.warning(
                        f"Error getting feature names from {calc.__class__.__name__}: {e}"
                    )
                    continue

            return all_features

        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            return []

    def get_feature_count(self) -> int:
        """Get total number of features across all calculators."""
        return len(self.get_feature_names())

    def get_calculator_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all specialized calculators."""
        info = {}

        for calc in self.calculators:
            try:
                calc_name = calc.__class__.__name__
                feature_names = calc.get_feature_names()

                info[calc_name] = {
                    "feature_count": len(feature_names),
                    "feature_names": feature_names,
                    "description": (
                        calc.__doc__.split("\n")[0] if calc.__doc__ else "No description"
                    ),
                }

            except Exception as e:
                logger.warning(f"Error getting info for {calc.__class__.__name__}: {e}")
                continue

        return info

    # Additional methods to maintain backward compatibility with original interface

    # Legacy method removed - use calculate() instead

    # Legacy method removed - use calculate() instead

    # Legacy method removed - use calculate() instead

    def get_options_summary(self) -> dict[str, Any]:
        """Get summary of options data and configuration."""
        summary = {
            "config": {
                "min_volume": self.options_config.min_volume,
                "min_open_interest": self.options_config.min_open_interest,
                "expiry_windows": self.options_config.expiry_windows,
                "risk_free_rate": self.options_config.risk_free_rate,
                "unusual_volume_threshold": self.options_config.unusual_volume_threshold,
            },
            "data_status": {
                "has_options_chain": self.options_chain is not None
                and not self.options_chain.empty,
                "has_historical_iv": self.historical_iv is not None
                and not self.historical_iv.empty,
                "options_count": len(self.options_chain) if self.options_chain is not None else 0,
                "iv_history_length": (
                    len(self.historical_iv) if self.historical_iv is not None else 0
                ),
            },
            "calculators": self.get_calculator_info(),
            "total_features": self.get_feature_count(),
        }

        return summary


# For backward compatibility, create an alias with the original name
OptionsAnalyticsCalculator = OptionsAnalyticsFacade
