"""
Technical Indicators Calculator

Calculates traditional technical analysis indicators including:
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP)
- Trend indicators (ADX, Aroon)
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_calculator import BaseFeatureCalculator
from .helpers import (
    create_feature_dataframe,
    postprocess_features,
    preprocess_data,
    safe_divide,
    validate_ohlcv_data,
)

logger = get_logger(__name__)


@dataclass
class TechnicalConfig:
    """Configuration for technical indicators"""

    # Moving averages
    sma_periods: list[int] = None
    ema_periods: list[int] = None

    # Momentum
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14
    stoch_smooth: int = 3

    # Volatility
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Volume
    obv_ema: int = 20

    # Trend
    adx_period: int = 14
    aroon_period: int = 25

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [9, 12, 26, 50]


class TechnicalIndicatorsCalculator(BaseFeatureCalculator):
    """Calculator for technical analysis indicators"""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.tech_config = TechnicalConfig(**config.get("technical", {}) if config else {})

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicator features
        """
        try:
            # Validate inputs
            if not self.validate_input_data(data):
                logger.error("Invalid input data for technical indicators")
                return self._create_empty_features(
                    data.index if not data.empty else pd.DatetimeIndex([])
                )

            # Preprocess data
            processed_data = self.preprocess_data(data)
            if processed_data.empty:
                logger.error("Data preprocessing failed for technical indicators")
                return self._create_empty_features(
                    data.index if not data.empty else pd.DatetimeIndex([])
                )

            features = create_feature_dataframe(processed_data.index)
            feature_count = 0

            # Price-based features
            try:
                features = self._add_price_features(processed_data, features)
                feature_count += 8  # Approximate count
                logger.debug("Added price-based features")
            except Exception as e:
                logger.warning(f"Failed to add price features: {e}")

            # Moving averages
            try:
                features = self._add_moving_averages(processed_data, features)
                ma_count = (
                    len(self.tech_config.sma_periods) * 2
                    + len(self.tech_config.ema_periods) * 2
                    + 2
                )
                feature_count += ma_count
                logger.debug(f"Added {ma_count} moving average features")
            except Exception as e:
                logger.warning(f"Failed to add moving average features: {e}")

            # Momentum indicators
            try:
                features = self._add_momentum_indicators(processed_data, features)
                feature_count += 9  # RSI, MACD (3), Stochastic (2), ROC (3), Williams %R
                logger.debug("Added momentum indicator features")
            except Exception as e:
                logger.warning(f"Failed to add momentum features: {e}")

            # Volatility indicators
            try:
                features = self._add_volatility_indicators(processed_data, features)
                feature_count += 9  # BB (4), ATR (2), Volatility (3)
                logger.debug("Added volatility indicator features")
            except Exception as e:
                logger.warning(f"Failed to add volatility features: {e}")

            # Volume indicators
            if "volume" in processed_data.columns and processed_data["volume"].sum() > 0:
                try:
                    features = self._add_volume_indicators(processed_data, features)
                    feature_count += 7  # OBV (2), VWAP (2), Volume ROC, PVT, MFI
                    logger.debug("Added volume indicator features")
                except Exception as e:
                    logger.warning(f"Failed to add volume features: {e}")
            else:
                logger.debug("Skipping volume indicators - no volume data")

            # Trend indicators
            try:
                features = self._add_trend_indicators(processed_data, features)
                feature_count += 7  # ADX (3), Aroon (3), SAR
                logger.debug("Added trend indicator features")
            except Exception as e:
                logger.warning(f"Failed to add trend features: {e}")

            # Price patterns
            try:
                features = self._add_price_patterns(processed_data, features)
                feature_count += 11  # Patterns (5), Support/Resistance (6)
                logger.debug("Added price pattern features")
            except Exception as e:
                logger.warning(f"Failed to add price pattern features: {e}")

            # Apply technical-specific postprocessing
            features = self._postprocess_technical_features(features)

            # Apply general postprocessing
            features = postprocess_features(features)

            if features.empty:
                logger.error("All technical indicator calculations failed")
                return self._create_empty_features(
                    data.index if not data.empty else pd.DatetimeIndex([])
                )

            logger.info(f"Successfully calculated {len(features.columns)} technical indicators")

        except Exception as e:
            logger.error(f"Critical error in technical indicator calculation: {e}")
            return self._create_empty_features(
                data.index if not data.empty else pd.DatetimeIndex([])
            )

        return features

    def _create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """Create empty feature DataFrame with proper column names"""
        feature_names = self.get_feature_names()
        # Create dictionary of empty series for each feature
        features_dict = {name: pd.Series(0, index=index) for name in feature_names}
        return create_feature_dataframe(index, features_dict)

    def get_required_columns(self) -> list[str]:
        """Return list of required input columns"""
        return ["open", "high", "low", "close", "volume"]

    def get_feature_names(self) -> list[str]:
        """Return list of all technical indicator feature names"""
        feature_names = []

        # Price features
        feature_names.extend(
            [
                "high_low_ratio",
                "close_open_ratio",
                "daily_range",
                "daily_range_pct",
                "gap",
                "gap_pct",
                "returns",
                "log_returns",
            ]
        )

        # Moving averages
        for period in self.tech_config.sma_periods:
            feature_names.extend([f"sma_{period}", f"close_sma_{period}_ratio"])

        for period in self.tech_config.ema_periods:
            feature_names.extend([f"ema_{period}", f"close_ema_{period}_ratio"])

        feature_names.extend(["golden_cross", "death_cross"])

        # Momentum indicators
        feature_names.extend(
            [
                f"rsi_{self.tech_config.rsi_period}",
                "macd",
                "macd_signal",
                "macd_histogram",
                "stoch_k",
                "stoch_d",
                "williams_r",
            ]
        )

        for period in [5, 10, 20]:
            feature_names.append(f"roc_{period}")

        # Volatility indicators
        feature_names.extend(
            [
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_position",
                f"atr_{self.tech_config.atr_period}",
                "atr_pct",
            ]
        )

        for period in [10, 20, 30]:
            feature_names.append(f"volatility_{period}")

        # Volume indicators (if volume available)
        feature_names.extend(
            ["obv", "obv_ema", "vwap", "close_vwap_ratio", "volume_roc", "pvt", "mfi"]
        )

        # Trend indicators
        feature_names.extend(
            [
                "adx",
                "plus_di",
                "minus_di",
                "aroon_up",
                "aroon_down",
                "aroon_oscillator",
                "sar_signal",
            ]
        )

        # Price patterns
        feature_names.extend(
            ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]
        )

        # Support/Resistance
        for period in [20, 50]:
            feature_names.extend(
                [f"resistance_{period}", f"support_{period}", f"sr_position_{period}"]
            )

        return feature_names

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns (renamed to avoid conflict)"""
        is_valid, errors = validate_ohlcv_data(data, min_rows=200)
        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
        return is_valid

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for technical indicator calculation"""
        # Use the validation helper function for preprocessing
        processed = preprocess_data(
            data,
            handle_missing="forward_fill",
            remove_outliers=self.config.get("remove_outliers", False),
            outlier_threshold=self.config.get("outlier_threshold", 3.0),
        )
        return processed

    def _postprocess_technical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicator-specific postprocessing"""
        # Normalize RSI to proper bounds [0, 100]
        rsi_cols = [col for col in features.columns if col.startswith("rsi_")]
        for col in rsi_cols:
            features[col] = features[col].clip(0, 100)

        # Normalize Stochastic to proper bounds [0, 100]
        stoch_cols = [col for col in features.columns if col.startswith("stoch_")]
        for col in stoch_cols:
            features[col] = features[col].clip(0, 100)

        # Normalize Williams %R to proper bounds [-100, 0]
        if "williams_r" in features.columns:
            features["williams_r"] = features["williams_r"].clip(-100, 0)

        # Normalize Bollinger Band position to [0, 1] approximately
        if "bb_position" in features.columns:
            features["bb_position"] = features["bb_position"].clip(-0.5, 1.5)

        # Normalize MFI to proper bounds [0, 100]
        if "mfi" in features.columns:
            features["mfi"] = features["mfi"].clip(0, 100)

        # Normalize Aroon indicators to [0, 100]
        aroon_cols = [col for col in features.columns if col.startswith("aroon_")]
        for col in aroon_cols:
            if col != "aroon_oscillator":  # Oscillator can be [-100, 100]
                features[col] = features[col].clip(0, 100)

        # Clip extreme values for ratios
        ratio_cols = [col for col in features.columns if "ratio" in col]
        for col in ratio_cols:
            features[col] = features[col].clip(0.1, 10.0)

        # Fill binary indicators
        binary_cols = [
            "golden_cross",
            "death_cross",
            "doji",
            "hammer",
            "shooting_star",
            "bullish_engulfing",
            "bearish_engulfing",
        ]
        for col in binary_cols:
            if col in features.columns:
                features[col] = features[col].fillna(value=0)

        return features

    def _add_price_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features"""
        # Price ratios
        features["high_low_ratio"] = safe_divide(data["high"], data["low"])
        features["close_open_ratio"] = safe_divide(data["close"], data["open"])

        # Price ranges
        features["daily_range"] = data["high"] - data["low"]
        features["daily_range_pct"] = safe_divide(features["daily_range"], data["close"])

        # Gap features
        features["gap"] = data["open"] - data["close"].shift(1)
        features["gap_pct"] = safe_divide(features["gap"], data["close"].shift(1))

        # Returns
        features["returns"] = data["close"].pct_change()
        features["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        return features

    def _add_moving_averages(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        close = data["close"]

        # Simple Moving Averages
        for period in self.tech_config.sma_periods:
            features[f"sma_{period}"] = close.rolling(window=period).mean()
            features[f"close_sma_{period}_ratio"] = safe_divide(close, features[f"sma_{period}"])

        # Exponential Moving Averages
        for period in self.tech_config.ema_periods:
            features[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()
            features[f"close_ema_{period}_ratio"] = safe_divide(close, features[f"ema_{period}"])

        # Moving average convergence
        if "sma_50" in features.columns and "sma_200" in features.columns:
            features["golden_cross"] = (
                (features["sma_50"] > features["sma_200"])
                & (features["sma_50"].shift(1) <= features["sma_200"].shift(1))
            ).astype(int)

            features["death_cross"] = (
                (features["sma_50"] < features["sma_200"])
                & (features["sma_50"].shift(1) >= features["sma_200"].shift(1))
            ).astype(int)

        return features

    def _add_momentum_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # RSI
        rsi_period = self.tech_config.rsi_period
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = safe_divide(gain, loss)
        features[f"rsi_{rsi_period}"] = 100 - safe_divide(100, (1 + rs))

        # MACD
        ema_fast = close.ewm(span=self.tech_config.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.tech_config.macd_slow, adjust=False).mean()
        features["macd"] = ema_fast - ema_slow
        features["macd_signal"] = (
            features["macd"].ewm(span=self.tech_config.macd_signal, adjust=False).mean()
        )
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # Stochastic Oscillator
        period = self.tech_config.stoch_period
        smooth = self.tech_config.stoch_smooth

        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        features["stoch_k"] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        features["stoch_d"] = features["stoch_k"].rolling(window=smooth).mean()

        # Rate of Change
        for period in [5, 10, 20]:
            features[f"roc_{period}"] = close.pct_change(periods=period) * 100

        # Williams %R
        features["williams_r"] = -100 * ((highest_high - close) / (highest_high - lowest_low))

        return features

    def _add_volatility_indicators(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volatility indicators"""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Bollinger Bands
        period = self.tech_config.bb_period
        std_mult = self.tech_config.bb_std

        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        features["bb_upper"] = sma + (std * std_mult)
        features["bb_lower"] = sma - (std * std_mult)
        features["bb_width"] = features["bb_upper"] - features["bb_lower"]
        features["bb_position"] = safe_divide((close - features["bb_lower"]), features["bb_width"])

        # Average True Range
        period = self.tech_config.atr_period

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features[f"atr_{period}"] = true_range.rolling(window=period).mean()
        features["atr_pct"] = safe_divide(features[f"atr_{period}"], close)

        # Historical Volatility
        for period in [10, 20, 30]:
            features[f"volatility_{period}"] = close.pct_change().rolling(
                window=period
            ).std() * np.sqrt(252)

        return features

    def _add_volume_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        close = data["close"]
        volume = data["volume"]

        # On Balance Volume
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv"] = obv
        features["obv_ema"] = obv.ewm(span=self.tech_config.obv_ema, adjust=False).mean()

        # Volume Weighted Average Price
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        features["vwap"] = safe_divide((typical_price * volume).cumsum(), volume.cumsum())
        features["close_vwap_ratio"] = safe_divide(close, features["vwap"])

        # Volume Rate of Change
        features["volume_roc"] = volume.pct_change(periods=10)

        # Price Volume Trend
        features["pvt"] = (safe_divide(close.diff(), close.shift(1)) * volume).cumsum()

        # Money Flow Index
        period = 14
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        money_ratio = safe_divide(positive_sum, negative_sum)
        features["mfi"] = 100 - safe_divide(100, (1 + money_ratio))

        return features

    def _add_trend_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Average Directional Index (ADX)
        period = self.tech_config.adx_period

        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate True Range
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smooth the values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * safe_divide(plus_dm.rolling(window=period).mean(), atr)
        minus_di = 100 * safe_divide(minus_dm.rolling(window=period).mean(), atr)

        # Calculate ADX
        dx = 100 * safe_divide((plus_di - minus_di).abs(), (plus_di + minus_di))
        features["adx"] = dx.rolling(window=period).mean()
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di

        # Aroon Indicator
        period = self.tech_config.aroon_period

        high_period = high.rolling(window=period + 1).apply(lambda x: period - x.argmax(), raw=True)
        low_period = low.rolling(window=period + 1).apply(lambda x: period - x.argmin(), raw=True)

        features["aroon_up"] = safe_divide(100 * (period - high_period), period)
        features["aroon_down"] = safe_divide(100 * (period - low_period), period)
        features["aroon_oscillator"] = features["aroon_up"] - features["aroon_down"]

        # Parabolic SAR (simplified version)
        features["sar_signal"] = self._calculate_sar(high, low, close)

        return features

    def _add_price_patterns(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        open_price = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Candlestick patterns
        body = close - open_price
        body_abs = body.abs()
        upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low

        # Doji
        features["doji"] = (body_abs / (high - low) < 0.1).astype(int)

        # Hammer
        features["hammer"] = (
            (lower_shadow > 2 * body_abs) & (upper_shadow < body_abs * 0.1) & (body < 0)
        ).astype(int)

        # Shooting Star
        features["shooting_star"] = (
            (upper_shadow > 2 * body_abs) & (lower_shadow < body_abs * 0.1) & (body > 0)
        ).astype(int)

        # Engulfing patterns
        features["bullish_engulfing"] = (
            (body > 0)
            & (body.shift(1) < 0)
            & (open_price < close.shift(1))
            & (close > open_price.shift(1))
        ).astype(int)

        features["bearish_engulfing"] = (
            (body < 0)
            & (body.shift(1) > 0)
            & (open_price > close.shift(1))
            & (close < open_price.shift(1))
        ).astype(int)

        # Support and Resistance levels
        for period in [20, 50]:
            features[f"resistance_{period}"] = high.rolling(window=period).max()
            features[f"support_{period}"] = low.rolling(window=period).min()
            features[f"sr_position_{period}"] = safe_divide(
                (close - features[f"support_{period}"]),
                (features[f"resistance_{period}"] - features[f"support_{period}"]),
            )

        return features

    def _calculate_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        initial_af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """Calculate Parabolic SAR indicator"""
        # Simplified SAR calculation
        # In production, this would be more sophisticated
        trend = (close > close.shift(1)).astype(int)
        trend_changes = trend.diff() != 0

        # Return 1 for uptrend, -1 for downtrend
        sar_signal = trend * 2 - 1

        return sar_signal
