"""
Volatility Indicators Calculator

Specialized calculator for volatility-based technical indicators including:
- ATR (Average True Range)
- Bollinger Bands
- Keltner Channels
- Historical Volatility
- Volatility ratios and derived metrics
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import talib

from .base_technical import BaseTechnicalCalculator


class VolatilityIndicatorsCalculator(BaseTechnicalCalculator):
    """Calculator for volatility indicators."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize volatility indicators calculator."""
        super().__init__(config)

        # ATR parameters
        self.atr_periods = self.config.get("atr_periods", [14, 20])

        # Bollinger Bands parameters
        self.bb_periods = self.config.get("bb_periods", [20, 30])
        self.bb_std_dev = self.config.get("bb_std_dev", 2.0)

        # Keltner Channels parameters
        self.kc_period = self.config.get("kc_period", 20)
        self.kc_atr_period = self.config.get("kc_atr_period", 10)
        self.kc_multiplier = self.config.get("kc_multiplier", 2.0)

        # Historical Volatility parameters
        self.hist_vol_periods = self.config.get("hist_vol_periods", [20, 60])
        self.annualization_factor = self.config.get(
            "annualization_factor", 252
        )  # trading days per year

    def get_feature_names(self) -> list[str]:
        """Return list of volatility indicator feature names."""
        feature_names = []

        # ATR indicators for multiple periods
        for period in self.atr_periods:
            feature_names.extend([f"atr_{period}", f"atr_{period}_pct"])

        # Bollinger Bands for multiple periods
        for period in self.bb_periods:
            feature_names.extend(
                [
                    f"bb_upper_{period}",
                    f"bb_middle_{period}",
                    f"bb_lower_{period}",
                    f"bb_width_{period}",
                    f"bb_position_{period}",
                    f"bb_squeeze_{period}",
                ]
            )

        # Keltner Channels
        feature_names.extend(["kc_upper", "kc_lower", "kc_position"])

        # Historical Volatility for multiple periods
        for period in self.hist_vol_periods:
            feature_names.append(f"hist_volatility_{period}")

        # Volatility ratio
        feature_names.append("volatility_ratio")

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility indicator features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_feature_dataframe(data.index)

            # Calculate volatility indicators for each symbol
            for symbol in data["symbol"].unique():
                mask = data["symbol"] == symbol
                symbol_data = data[mask].copy()

                if len(symbol_data) < 50:
                    continue

                # Calculate ATR indicators
                features = self._calculate_atr_indicators(features, symbol_data, mask)

                # Calculate Bollinger Bands
                features = self._calculate_bollinger_bands(features, symbol_data, mask)

                # Calculate Keltner Channels
                features = self._calculate_keltner_channels(features, symbol_data, mask)

                # Calculate Historical Volatility
                features = self._calculate_historical_volatility(features, symbol_data, mask)

                # Calculate Volatility Ratios
                features = self._calculate_volatility_ratios(features, symbol_data, mask)

            return features

        except Exception as e:
            print(f"Error calculating volatility indicators: {e}")
            return self.create_feature_dataframe(data.index)

    def _calculate_atr_indicators(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate ATR indicators."""
        # ATR (Average True Range)
        for period in self.atr_periods:
            atr = talib.ATR(
                symbol_data["high"].values,
                symbol_data["low"].values,
                symbol_data["close"].values,
                timeperiod=period,
            )
            features.loc[mask, f"atr_{period}"] = atr

            # ATR percentage
            features.loc[mask, f"atr_{period}_pct"] = (atr / symbol_data["close"] * 100).fillna(0)

        return features

    def _calculate_bollinger_bands(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators."""
        # Bollinger Bands
        for period in self.bb_periods:
            upper, middle, lower = talib.BBANDS(
                symbol_data["close"].values,
                timeperiod=period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev,
                matype=0,
            )
            features.loc[mask, f"bb_upper_{period}"] = upper
            features.loc[mask, f"bb_middle_{period}"] = middle
            features.loc[mask, f"bb_lower_{period}"] = lower

            # Bollinger Band width (normalized by middle band)
            bb_width = (upper - lower) / middle
            features.loc[mask, f"bb_width_{period}"] = bb_width

            # Price position in bands (0 = lower band, 1 = upper band)
            bb_position = (symbol_data["close"] - lower) / (upper - lower)
            features.loc[mask, f"bb_position_{period}"] = bb_position.fillna(0.5)

            # Band squeeze (when bands are narrow compared to historical norms)
            bb_width_series = pd.Series(bb_width, index=symbol_data.index)
            bb_squeeze = (
                bb_width_series < bb_width_series.rolling(100, min_periods=20).quantile(0.2)
            ).astype(int)
            features.loc[mask, f"bb_squeeze_{period}"] = bb_squeeze

        return features

    def _calculate_keltner_channels(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Keltner Channels indicators."""
        # Keltner Channels
        ema_middle = talib.EMA(symbol_data["close"].values, timeperiod=self.kc_period)
        atr_kc = talib.ATR(
            symbol_data["high"].values,
            symbol_data["low"].values,
            symbol_data["close"].values,
            timeperiod=self.kc_atr_period,
        )

        kc_upper = ema_middle + (self.kc_multiplier * atr_kc)
        kc_lower = ema_middle - (self.kc_multiplier * atr_kc)

        features.loc[mask, "kc_upper"] = kc_upper
        features.loc[mask, "kc_lower"] = kc_lower

        # Price position in Keltner Channels
        kc_position = (symbol_data["close"] - kc_lower) / (kc_upper - kc_lower)
        features.loc[mask, "kc_position"] = kc_position.fillna(0.5)

        return features

    def _calculate_historical_volatility(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Historical Volatility indicators."""
        # Calculate returns
        returns = symbol_data["close"].pct_change()

        # Historical Volatility for multiple periods
        for period in self.hist_vol_periods:
            # Annualized volatility (standard deviation of returns)
            hist_vol = returns.rolling(period, min_periods=10).std() * np.sqrt(
                self.annualization_factor
            )
            features.loc[mask, f"hist_volatility_{period}"] = hist_vol

        return features

    def _calculate_volatility_ratios(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate volatility ratios and derived metrics."""
        # Volatility ratio (short-term vs long-term volatility)
        if len(self.hist_vol_periods) >= 2:
            short_vol_col = f"hist_volatility_{min(self.hist_vol_periods)}"
            long_vol_col = f"hist_volatility_{max(self.hist_vol_periods)}"

            if short_vol_col in features.columns and long_vol_col in features.columns:
                volatility_ratio = (
                    features.loc[mask, short_vol_col] / features.loc[mask, long_vol_col]
                ).fillna(1.0)
                features.loc[mask, "volatility_ratio"] = volatility_ratio

        return features
