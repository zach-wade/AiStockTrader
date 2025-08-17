"""
Adaptive Indicators Calculator

Specialized calculator for adaptive technical indicators that adjust their parameters
based on market conditions including:
- KAMA (Kaufman Adaptive Moving Average)
- Adaptive RSI
- VMA (Variable Moving Average)
- FRAMA (Fractal Adaptive Moving Average)
- Market efficiency and volatility-based adaptations
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import talib

from .base_technical import BaseTechnicalCalculator


class AdaptiveIndicatorsCalculator(BaseTechnicalCalculator):
    """Calculator for adaptive indicators that adjust to market conditions."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize adaptive indicators calculator."""
        super().__init__(config)

        # KAMA parameters
        self.kama_period = self.config.get("kama_period", 10)
        self.kama_fast_ema = self.config.get("kama_fast_ema", 2)
        self.kama_slow_ema = self.config.get("kama_slow_ema", 30)

        # Adaptive RSI parameters
        self.adaptive_rsi_base_period = self.config.get("adaptive_rsi_base_period", 14)
        self.adaptive_rsi_volatility_window = self.config.get("adaptive_rsi_volatility_window", 50)

        # VMA parameters
        self.vma_base_period = self.config.get("vma_base_period", 20)
        self.vma_cmo_period = self.config.get("vma_cmo_period", 9)

        # FRAMA parameters
        self.frama_period = self.config.get("frama_period", 20)

        # General adaptive settings
        self.min_periods_for_adaptation = self.config.get("min_periods_for_adaptation", 100)

    def get_feature_names(self) -> list[str]:
        """Return list of adaptive indicator feature names."""
        feature_names = []

        # Adaptive indicators
        feature_names.extend(
            [
                "kama",
                "price_to_kama",
                "adaptive_rsi",
                "vma",
                "frama",
                "market_efficiency_ratio",
                "adaptive_momentum",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate adaptive indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with adaptive indicator features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_feature_dataframe(data.index)

            # Only calculate if adaptive mode is enabled
            if not self.adaptive_enabled:
                return features

            # Calculate adaptive indicators for each symbol
            for symbol in data["symbol"].unique():
                mask = data["symbol"] == symbol
                symbol_data = data[mask].copy()

                if len(symbol_data) < self.min_periods_for_adaptation:
                    continue

                # Calculate KAMA
                features = self._calculate_kama_indicators(features, symbol_data, mask)

                # Calculate Adaptive RSI
                features = self._calculate_adaptive_rsi(features, symbol_data, mask)

                # Calculate Variable Moving Average
                features = self._calculate_vma_indicators(features, symbol_data, mask)

                # Calculate Fractal Adaptive Moving Average
                features = self._calculate_frama_indicators(features, symbol_data, mask)

                # Calculate Market Efficiency indicators
                features = self._calculate_efficiency_indicators(features, symbol_data, mask)

            return features

        except Exception as e:
            print(f"Error calculating adaptive indicators: {e}")
            return self.create_feature_dataframe(data.index)

    def _calculate_kama_indicators(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate KAMA (Kaufman Adaptive Moving Average) indicators."""

        def calculate_kama(prices, period=10, fast_ema=2, slow_ema=30):
            """Calculate Kaufman Adaptive Moving Average."""
            direction = np.abs(prices.diff(period))
            volatility = np.abs(prices.diff()).rolling(period).sum()
            efficiency_ratio = direction / volatility

            fast_sc = 2 / (fast_ema + 1)
            slow_sc = 2 / (slow_ema + 1)

            smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

            kama = np.zeros_like(prices)
            kama[period] = prices.iloc[period]

            for i in range(period + 1, len(prices)):
                if not np.isnan(smoothing_constant.iloc[i]):
                    kama[i] = kama[i - 1] + smoothing_constant.iloc[i] * (
                        prices.iloc[i] - kama[i - 1]
                    )
                else:
                    kama[i] = kama[i - 1]

            return pd.Series(kama, index=prices.index)

        kama = calculate_kama(
            symbol_data["close"], self.kama_period, self.kama_fast_ema, self.kama_slow_ema
        )
        features.loc[mask, "kama"] = kama

        # Price to KAMA ratio
        price_to_kama = (symbol_data["close"] / kama).fillna(1.0)
        features.loc[mask, "price_to_kama"] = price_to_kama

        return features

    def _calculate_adaptive_rsi(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Adaptive RSI that adjusts period based on volatility."""
        # Calculate ATR for volatility measurement
        atr = talib.ATR(
            symbol_data["high"].values,
            symbol_data["low"].values,
            symbol_data["close"].values,
            timeperiod=14,
        )

        # Adjust RSI period based on volatility
        atr_series = pd.Series(atr, index=symbol_data.index)
        volatility_rank = atr_series.rolling(
            self.adaptive_rsi_volatility_window, min_periods=10
        ).rank(pct=True)

        # Use the most recent volatility rank for period adjustment
        if not volatility_rank.empty and not pd.isna(volatility_rank.iloc[-1]):
            adaptive_period = int(self.adaptive_rsi_base_period * (1 + volatility_rank.iloc[-1]))
            adaptive_period = max(7, min(adaptive_period, len(symbol_data) - 1))
        else:
            adaptive_period = self.adaptive_rsi_base_period

        adaptive_rsi = talib.RSI(symbol_data["close"].values, timeperiod=adaptive_period)
        features.loc[mask, "adaptive_rsi"] = adaptive_rsi

        return features

    def _calculate_vma_indicators(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Variable Moving Average indicators."""
        # Variable Moving Average using CMO for volatility index
        cmo = talib.CMO(symbol_data["close"].values, timeperiod=self.vma_cmo_period)
        volatility_index = np.abs(cmo) / 100

        vma = np.zeros_like(symbol_data["close"].values)
        vma[0] = symbol_data["close"].iloc[0]

        for i in range(1, len(vma)):
            if not np.isnan(volatility_index[i]) and volatility_index[i] > 0:
                alpha = 2 / (self.vma_base_period * volatility_index[i] + 1)
            else:
                alpha = 0.1  # Default alpha

            alpha = max(0.01, min(1.0, alpha))  # Clamp alpha
            vma[i] = alpha * symbol_data["close"].iloc[i] + (1 - alpha) * vma[i - 1]

        features.loc[mask, "vma"] = vma

        return features

    def _calculate_frama_indicators(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate Fractal Adaptive Moving Average indicators."""

        def calculate_frama(prices, period=20):
            """Calculate Fractal Adaptive Moving Average."""
            n1 = period // 2
            n2 = period

            frama = np.zeros_like(prices)
            frama[:n2] = prices[:n2].mean()

            for i in range(n2, len(prices)):
                # Calculate fractal dimension
                highest = prices[i - n2 : i].max()
                lowest = prices[i - n2 : i].min()

                n1_high = prices[i - n1 : i].max()
                n1_low = prices[i - n1 : i].min()
                n2_high = prices[i - n2 : i - n1].max()
                n2_low = prices[i - n2 : i - n1].min()

                if highest != lowest and n1_high != n1_low and n2_high != n2_low:
                    try:
                        d = (
                            np.log(np.abs(n1_high - n1_low) + np.abs(n2_high - n2_low))
                            - np.log(highest - lowest)
                        ) / np.log(2)
                    except (ValueError, ZeroDivisionError):
                        d = 1.5  # Default dimension
                else:
                    d = 1.5  # Default dimension

                # Calculate alpha with bounds checking
                alpha = np.exp(-4.6 * (d - 1))
                alpha = max(0.01, min(1, alpha))

                frama[i] = alpha * prices.iloc[i] + (1 - alpha) * frama[i - 1]

            return pd.Series(frama, index=prices.index)

        frama = calculate_frama(symbol_data["close"], self.frama_period)
        features.loc[mask, "frama"] = frama

        return features

    def _calculate_efficiency_indicators(
        self, features: pd.DataFrame, symbol_data: pd.DataFrame, mask: pd.Series
    ) -> pd.DataFrame:
        """Calculate market efficiency and adaptive momentum indicators."""
        # Market Efficiency Ratio (Kaufman's)
        close_prices = symbol_data["close"]
        efficiency_ratio = self.calculate_efficiency_ratio(close_prices, self.kama_period)
        features.loc[mask, "market_efficiency_ratio"] = efficiency_ratio

        # Adaptive Momentum based on efficiency
        price_change = close_prices.pct_change(10)
        adaptive_momentum = price_change * efficiency_ratio
        features.loc[mask, "adaptive_momentum"] = adaptive_momentum

        return features
