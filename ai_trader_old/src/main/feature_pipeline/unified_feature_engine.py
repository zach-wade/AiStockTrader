# File: feature_pipeline/unified_feature_engine.py (Final Code)

# Standard library imports
from datetime import datetime
import logging
from typing import Any

# Third-party imports
from omegaconf import DictConfig
import pandas as pd

from .calculators.cross_sectional import CrossSectionalCalculator
from .calculators.insider_analytics import InsiderAnalyticsCalculator
from .calculators.market_regime import MarketRegimeCalculator
from .calculators.microstructure import MicrostructureCalculator
from .calculators.news import NewsFeatureCalculator
from .calculators.options import OptionsAnalyticsFacade as OptionsAnalyticsCalculator
from .calculators.sector_analytics import SectorAnalyticsCalculator
from .calculators.sentiment_features import SentimentFeaturesCalculator
from .calculators.statistical import AdvancedStatisticalCalculator

# Import all calculator classes
from .calculators.technical_indicators import TechnicalIndicatorsCalculator

logger = logging.getLogger(__name__)


class UnifiedFeatureEngine:
    """
    A unified engine that manages all individual feature calculators and
    runs them to generate a complete feature set from raw data.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the engine and all registered feature calculators.
        """
        self.config = config
        # The engine creates and owns all the calculator instances
        self.calculators = self._load_calculators()
        logger.info(
            f"✅ UnifiedFeatureEngine initialized with {len(self.calculators)} calculators."
        )

    def _load_calculators(self) -> dict[str, Any]:
        """Dynamically loads calculator classes specified in the config."""
        calculators = {}

        # Load all available calculators
        try:
            calculators["technical"] = TechnicalIndicatorsCalculator(self.config)
            calculators["regime"] = MarketRegimeCalculator(self.config)
            calculators["cross_sectional"] = CrossSectionalCalculator(self.config)
            calculators["sentiment"] = SentimentFeaturesCalculator(self.config)
            calculators["news_features"] = NewsFeatureCalculator(self.config)
            calculators["microstructure"] = MicrostructureCalculator(self.config)
            calculators["options"] = OptionsAnalyticsCalculator(self.config)
            calculators["insider"] = InsiderAnalyticsCalculator(self.config)
            calculators["sector"] = SectorAnalyticsCalculator(self.config)
            calculators["statistical"] = AdvancedStatisticalCalculator(self.config)
        except Exception as e:
            logger.warning(f"Failed to load some calculators: {e}")
            # Load only essential calculators as fallback
            if "technical" not in calculators:
                calculators["technical"] = TechnicalIndicatorsCalculator(self.config)

        return calculators

    def calculate_for_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all registered feature calculators on a given DataFrame.

        Args:
            data: A DataFrame of raw market data (OHLCV).

        Returns:
            The original DataFrame with many new feature columns appended.
        """
        if data.empty:
            return data

        processed_df = data.copy()

        # Sequentially apply each calculator to the DataFrame
        for name, calculator in self.calculators.items():
            try:
                processed_df = calculator.calculate(processed_df)
            except Exception as e:
                logger.error(f"Error applying feature calculator '{name}': {e}", exc_info=True)

        return processed_df

    def calculate_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        calculators: list[str] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate features for the given data using specified calculators.
        This method matches the interface expected by BaseStrategy.

        Args:
            data: Input DataFrame with market data
            symbol: Symbol being processed
            calculators: List of calculator names to use (e.g., ['technical', 'regime'])
            use_cache: Whether to use cached results (not implemented yet)

        Returns:
            DataFrame with calculated features
        """
        if data.empty:
            logger.warning(f"Empty data provided for {symbol}")
            return data

        # If no specific calculators requested, use all
        if calculators is None:
            calculators_to_use = self.calculators
        else:
            # Filter to only requested calculators
            calculators_to_use = {k: v for k, v in self.calculators.items() if k in calculators}

        logger.debug(
            f"Calculating features for {symbol} using calculators: {list(calculators_to_use.keys())}"
        )

        processed_df = data.copy()

        # Apply each calculator
        for name, calculator in calculators_to_use.items():
            try:
                # Check if calculator has validate_input_data method for validation
                if hasattr(calculator, "validate_input_data") and callable(
                    calculator.validate_input_data
                ):
                    if not calculator.validate_input_data(processed_df):
                        logger.warning(f"Skipping {name} for {symbol} - validation failed")
                        continue
                # Fallback for older calculators that might still use validate_inputs as method
                elif hasattr(calculator, "validate_inputs") and callable(
                    calculator.validate_inputs
                ):
                    if not calculator.validate_inputs(processed_df):
                        logger.warning(f"Skipping {name} for {symbol} - validation failed")
                        continue

                # Use the standardized method if available
                if hasattr(calculator, "calculate_with_validation"):
                    features = calculator.calculate_with_validation(processed_df)
                else:
                    features = calculator.calculate(processed_df)

                # Merge features into the main dataframe
                if isinstance(features, pd.DataFrame) and not features.empty:
                    # Align indices before concatenation
                    features = features.reindex(processed_df.index)
                    # Add features as new columns
                    for col in features.columns:
                        if col not in processed_df.columns:
                            processed_df[col] = features[col]

                logger.debug(f"Successfully applied {name} calculator for {symbol}")

            except Exception as e:
                logger.error(f"Error applying {name} calculator for {symbol}: {e}", exc_info=True)

        # Add metadata
        processed_df.attrs["symbol"] = symbol
        processed_df.attrs["feature_timestamp"] = datetime.now()
        processed_df.attrs["calculators_used"] = list(calculators_to_use.keys())

        return processed_df
