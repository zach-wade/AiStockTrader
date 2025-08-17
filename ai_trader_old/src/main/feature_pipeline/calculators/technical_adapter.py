"""
Technical Calculator Adapter

Adapts the existing TechnicalIndicatorsCalculator to implement ITechnicalCalculator interface.
"""

# Standard library imports
from datetime import datetime
import logging
import time
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.calculators import CalculatorConfig, FeatureResult, ITechnicalCalculator

from .technical_indicators import TechnicalIndicatorsCalculator

logger = logging.getLogger(__name__)


class TechnicalCalculatorAdapter(ITechnicalCalculator):
    """
    Adapter that makes TechnicalIndicatorsCalculator implement ITechnicalCalculator.

    This allows the existing calculator to work with the new interface system
    without modifying the original implementation.
    """

    def __init__(self, config: dict[str, Any] | CalculatorConfig):
        """Initialize the adapter with a technical calculator."""
        if isinstance(config, CalculatorConfig):
            self._calculator = TechnicalIndicatorsCalculator(config.parameters)
            self._config = config
        else:
            self._calculator = TechnicalIndicatorsCalculator(config)
            self._config = CalculatorConfig(
                name="technical_indicators", enabled=True, parameters=config
            )

    def calculate(
        self,
        data: pd.DataFrame,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs,
    ) -> FeatureResult:
        """
        Calculate technical indicators from input data.

        Args:
            data: Input OHLCV data
            symbols: Optional list of symbols to calculate for
            start_date: Optional start date for calculation
            end_date: Optional end date for calculation
            **kwargs: Additional parameters

        Returns:
            FeatureResult with calculated indicators
        """
        start_time = time.time()
        errors = []

        try:
            # Filter data if date range specified
            if start_date or end_date:
                mask = pd.Series(True, index=data.index)
                if start_date and "timestamp" in data.columns:
                    mask &= data["timestamp"] >= start_date
                if end_date and "timestamp" in data.columns:
                    mask &= data["timestamp"] <= end_date
                data = data[mask]

            # Filter by symbols if specified
            if symbols and "symbol" in data.columns:
                data = data[data["symbol"].isin(symbols)]

            # Calculate features using the underlying calculator
            features_df = self._calculator.calculate(data, **kwargs)

            metadata = {
                "calculator": "technical_indicators",
                "config": self._config.parameters,
                "input_shape": data.shape,
                "output_shape": features_df.shape,
                "symbols_processed": (
                    symbols or data["symbol"].unique().tolist() if "symbol" in data.columns else []
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            errors.append(str(e))
            features_df = pd.DataFrame()
            metadata = {"error": str(e)}

        calculation_time = time.time() - start_time

        return FeatureResult(
            features=features_df,
            metadata=metadata,
            calculation_time=calculation_time,
            errors=errors,
        )

    def calculate_indicators(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
        indicator_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Args:
            price_data: OHLC price data
            volume_data: Optional volume data
            indicator_config: Configuration for specific indicators

        Returns:
            DataFrame with calculated indicators
        """
        # Combine price and volume data if separate
        if volume_data is not None:
            data = pd.concat([price_data, volume_data], axis=1)
        else:
            data = price_data

        # Use underlying calculator
        return self._calculator.calculate(data, **(indicator_config or {}))

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names this calculator produces.

        Returns:
            List of feature column names
        """
        # Get from underlying calculator if it has the method
        if hasattr(self._calculator, "get_feature_names"):
            return self._calculator.get_feature_names()

        # Otherwise return a default list based on typical indicators
        return [
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_10",
            "ema_20",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "atr",
            "obv",
            "vwap",
            "stoch_k",
            "stoch_d",
            "adx",
            "plus_di",
            "minus_di",
        ]

    def get_required_columns(self) -> list[str]:
        """
        Get list of required input columns.

        Returns:
            List of column names required in input data
        """
        return ["open", "high", "low", "close", "volume"]

    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns and format.

        Args:
            data: Input data to validate

        Returns:
            True if valid, raises exception if invalid
        """
        required = self.get_required_columns()
        missing = [col for col in required if col not in data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if data.empty:
            raise ValueError("Input data is empty")

        # Check for numeric data
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")

        return True
