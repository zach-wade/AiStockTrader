"""
Trend Indicators Calculator

Calculates trend-following technical indicators that identify the direction
and strength of market trends across multiple timeframes.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_technical import BaseTechnicalCalculator
from ..helpers import create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TrendIndicatorsCalculator(BaseTechnicalCalculator):
    """
    Calculates trend-following indicators.

    Features include:
    - Moving averages (SMA, EMA, WMA, DEMA, TEMA)
    - ADX (Average Directional Index)
    - Aroon indicators
    - Parabolic SAR
    - Supertrend
    - Ichimoku Cloud
    - Price channels
    - Trend strength metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize trend indicators calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Trend-specific parameters
        self.ma_periods = config.get("ma_periods", [10, 20, 50, 100, 200])
        self.ma_types = config.get("ma_types", ["sma", "ema", "wma"])
        self.adx_period = config.get("adx_period", 14)
        self.aroon_period = config.get("aroon_period", 25)
        self.atr_multiplier = config.get("atr_multiplier", 3.0)

        # Ichimoku parameters
        self.ichimoku_params = config.get(
            "ichimoku_params", {"conversion": 9, "base": 26, "span_b": 52, "displacement": 26}
        )

        logger.info("Initialized TrendIndicatorsCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of trend indicator feature names."""
        features = []

        # Moving averages
        for ma_type in self.ma_types:
            for period in self.ma_periods:
                features.extend(
                    [
                        f"{ma_type}_{period}",
                        f"{ma_type}_{period}_slope",
                        f"price_to_{ma_type}_{period}_ratio",
                    ]
                )

        # Moving average crosses
        features.extend(["golden_cross", "death_cross", "ma_50_200_spread", "ma_alignment_score"])

        # ADX indicators
        features.extend(["adx", "plus_di", "minus_di", "adx_trend_strength", "di_crossover"])

        # Aroon indicators
        features.extend(["aroon_up", "aroon_down", "aroon_oscillator", "aroon_crossover"])

        # Parabolic SAR
        features.extend(["psar", "psar_trend", "psar_distance", "psar_reversal"])

        # Supertrend
        features.extend(["supertrend", "supertrend_direction", "supertrend_distance"])

        # Ichimoku Cloud
        features.extend(
            [
                "ichimoku_conversion",
                "ichimoku_base",
                "ichimoku_span_a",
                "ichimoku_span_b",
                "ichimoku_chikou",
                "ichimoku_cloud_thickness",
                "price_to_cloud_position",
            ]
        )

        # Price channels
        features.extend(
            [
                "donchian_upper",
                "donchian_lower",
                "donchian_middle",
                "channel_position",
                "channel_width",
            ]
        )

        # Trend strength
        features.extend(
            ["trend_strength", "trend_consistency", "trend_duration", "trend_composite_score"]
        )

        return features

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators from price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with trend indicators
        """
        try:
            # Validate input data
            validated_data = self.validate_ohlcv_data(data)
            if validated_data.empty:
                return self._create_empty_features(data.index)

            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)

            # Calculate moving averages
            ma_features = self._calculate_moving_averages(validated_data)
            features = pd.concat([features, ma_features], axis=1)

            # Calculate ADX indicators
            adx_features = self._calculate_adx_features(validated_data)
            features = pd.concat([features, adx_features], axis=1)

            # Calculate Aroon indicators
            aroon_features = self._calculate_aroon_features(validated_data)
            features = pd.concat([features, aroon_features], axis=1)

            # Calculate Parabolic SAR
            psar_features = self._calculate_psar_features(validated_data)
            features = pd.concat([features, psar_features], axis=1)

            # Calculate Supertrend
            supertrend_features = self._calculate_supertrend_features(validated_data)
            features = pd.concat([features, supertrend_features], axis=1)

            # Calculate Ichimoku Cloud
            ichimoku_features = self._calculate_ichimoku_features(validated_data)
            features = pd.concat([features, ichimoku_features], axis=1)

            # Calculate price channels
            channel_features = self._calculate_channel_features(validated_data)
            features = pd.concat([features, channel_features], axis=1)

            # Calculate trend strength metrics
            strength_features = self._calculate_trend_strength(features, validated_data)
            features = pd.concat([features, strength_features], axis=1)

            # Apply postprocessing
            features = self.postprocess_features(features)

            return features

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return self._create_empty_features(data.index)

    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages and related features."""
        features = pd.DataFrame(index=data.index)
        close = data["close"]

        # Store MAs for cross calculations
        ma_dict = {}

        for ma_type in self.ma_types:
            for period in self.ma_periods:
                # Calculate MA based on type
                if ma_type == "sma":
                    ma = self.calculate_sma(close, period)
                elif ma_type == "ema":
                    ma = self.calculate_ema(close, period)
                elif ma_type == "wma":
                    ma = self._calculate_wma(close, period)
                else:
                    continue

                features[f"{ma_type}_{period}"] = ma
                ma_dict[f"{ma_type}_{period}"] = ma

                # MA slope (rate of change)
                features[f"{ma_type}_{period}_slope"] = ma.pct_change(5)

                # Price to MA ratio
                features[f"price_to_{ma_type}_{period}_ratio"] = safe_divide(close, ma)

        # Golden cross and death cross (50/200 SMA)
        if "sma_50" in ma_dict and "sma_200" in ma_dict:
            cross_signal = self._detect_ma_cross(ma_dict["sma_50"], ma_dict["sma_200"])
            features["golden_cross"] = (cross_signal > 0).astype(int)
            features["death_cross"] = (cross_signal < 0).astype(int)
            features["ma_50_200_spread"] = safe_divide(
                ma_dict["sma_50"] - ma_dict["sma_200"], ma_dict["sma_200"]
            )

        # MA alignment score (how well MAs are aligned)
        features["ma_alignment_score"] = self._calculate_ma_alignment(ma_dict)

        return features

    def _calculate_adx_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX and directional indicators."""
        features = pd.DataFrame(index=data.index)

        # Calculate +DI and -DI
        plus_dm, minus_dm = self._calculate_directional_movement(data)
        tr = self._calculate_true_range(data)

        # Smooth using Wilder's method
        period = self.adx_period
        atr = self._wilders_smoothing(tr, period)
        plus_di = 100 * self._wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * self._wilders_smoothing(minus_dm, period) / atr

        features["plus_di"] = plus_di
        features["minus_di"] = minus_di

        # Calculate DX and ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * safe_divide(di_diff, di_sum)
        adx = self._wilders_smoothing(dx, period)

        features["adx"] = adx

        # ADX trend strength classification
        features["adx_trend_strength"] = self._classify_adx_strength(adx)

        # DI crossover signals
        features["di_crossover"] = self._detect_di_crossover(plus_di, minus_di)

        return features

    def _calculate_aroon_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Aroon indicators."""
        features = pd.DataFrame(index=data.index)
        period = self.aroon_period

        # Aroon Up: (period - periods since highest high) / period * 100
        high_period = (
            data["high"]
            .rolling(window=period + 1)
            .apply(lambda x: period - x.argmax() if len(x) == period + 1 else np.nan, raw=True)
        )
        features["aroon_up"] = (high_period / period) * 100

        # Aroon Down: (period - periods since lowest low) / period * 100
        low_period = (
            data["low"]
            .rolling(window=period + 1)
            .apply(lambda x: period - x.argmin() if len(x) == period + 1 else np.nan, raw=True)
        )
        features["aroon_down"] = (low_period / period) * 100

        # Aroon Oscillator
        features["aroon_oscillator"] = features["aroon_up"] - features["aroon_down"]

        # Aroon crossover
        features["aroon_crossover"] = self._detect_aroon_crossover(
            features["aroon_up"], features["aroon_down"]
        )

        return features

    def _calculate_psar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR features."""
        features = pd.DataFrame(index=data.index)

        # Calculate Parabolic SAR
        psar, trend = self._calculate_parabolic_sar(data)

        features["psar"] = psar
        features["psar_trend"] = trend

        # Distance from price to PSAR
        features["psar_distance"] = safe_divide(data["close"] - psar, data["close"])

        # PSAR reversal signal
        features["psar_reversal"] = trend.diff().fillna(0).abs()

        return features

    def _calculate_supertrend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        features = pd.DataFrame(index=data.index)

        # Calculate ATR
        atr = self._calculate_atr(data, self.adx_period)

        # Calculate basic bands
        hl_avg = (data["high"] + data["low"]) / 2
        upper_band = hl_avg + (self.atr_multiplier * atr)
        lower_band = hl_avg - (self.atr_multiplier * atr)

        # Calculate Supertrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)

        for i in range(1, len(data)):
            # Previous values
            prev_close = data["close"].iloc[i - 1]

            # Current values
            curr_close = data["close"].iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            # Determine trend
            if i == 1:
                if curr_close <= curr_upper:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = curr_upper
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = curr_lower
            else:
                # Previous direction
                prev_dir = direction.iloc[i - 1]

                if prev_dir == 1:
                    if curr_close <= curr_lower:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = curr_upper
                    else:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = max(curr_lower, supertrend.iloc[i - 1])
                elif curr_close >= curr_upper:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = curr_lower
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = min(curr_upper, supertrend.iloc[i - 1])

        features["supertrend"] = supertrend
        features["supertrend_direction"] = direction

        # Distance from price to Supertrend
        features["supertrend_distance"] = safe_divide(data["close"] - supertrend, data["close"])

        return features

    def _calculate_ichimoku_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators."""
        features = pd.DataFrame(index=data.index)
        params = self.ichimoku_params

        # Conversion Line (Tenkan-sen)
        conv_high = data["high"].rolling(window=params["conversion"]).max()
        conv_low = data["low"].rolling(window=params["conversion"]).min()
        features["ichimoku_conversion"] = (conv_high + conv_low) / 2

        # Base Line (Kijun-sen)
        base_high = data["high"].rolling(window=params["base"]).max()
        base_low = data["low"].rolling(window=params["base"]).min()
        features["ichimoku_base"] = (base_high + base_low) / 2

        # Leading Span A (Senkou Span A)
        features["ichimoku_span_a"] = (
            (features["ichimoku_conversion"] + features["ichimoku_base"]) / 2
        ).shift(params["displacement"])

        # Leading Span B (Senkou Span B)
        span_b_high = data["high"].rolling(window=params["span_b"]).max()
        span_b_low = data["low"].rolling(window=params["span_b"]).min()
        features["ichimoku_span_b"] = ((span_b_high + span_b_low) / 2).shift(params["displacement"])

        # Lagging Span (Chikou Span)
        features["ichimoku_chikou"] = data["close"].shift(-params["displacement"])

        # Cloud thickness
        features["ichimoku_cloud_thickness"] = abs(
            features["ichimoku_span_a"] - features["ichimoku_span_b"]
        )

        # Price position relative to cloud
        cloud_top = pd.concat(
            [features["ichimoku_span_a"], features["ichimoku_span_b"]], axis=1
        ).max(axis=1)

        cloud_bottom = pd.concat(
            [features["ichimoku_span_a"], features["ichimoku_span_b"]], axis=1
        ).min(axis=1)

        price_position = pd.Series(0, index=data.index)
        price_position[data["close"] > cloud_top] = 1  # Above cloud
        price_position[data["close"] < cloud_bottom] = -1  # Below cloud
        features["price_to_cloud_position"] = price_position

        return features

    def _calculate_channel_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price channel features."""
        features = pd.DataFrame(index=data.index)

        # Donchian Channel (20-period)
        period = 20
        features["donchian_upper"] = data["high"].rolling(window=period).max()
        features["donchian_lower"] = data["low"].rolling(window=period).min()
        features["donchian_middle"] = (features["donchian_upper"] + features["donchian_lower"]) / 2

        # Channel position (0 to 1)
        channel_range = features["donchian_upper"] - features["donchian_lower"]
        features["channel_position"] = safe_divide(
            data["close"] - features["donchian_lower"], channel_range
        )

        # Channel width (volatility measure)
        features["channel_width"] = safe_divide(channel_range, features["donchian_middle"])

        return features

    def _calculate_trend_strength(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite trend strength metrics."""
        strength_features = pd.DataFrame(index=data.index)

        # Trend strength based on ADX
        adx_strength = features.get("adx", 0) / 100

        # Trend strength based on MA alignment
        ma_strength = features.get("ma_alignment_score", 0)

        # Trend strength based on price position
        trend_signals = []

        # Check if price is above/below key MAs
        for period in [20, 50, 200]:
            ma_col = f"sma_{period}"
            if ma_col in features:
                signal = (data["close"] > features[ma_col]).astype(int) * 2 - 1
                trend_signals.append(signal)

        if trend_signals:
            price_trend = pd.concat(trend_signals, axis=1).mean(axis=1)
        else:
            price_trend = 0

        # Combined trend strength
        strength_features["trend_strength"] = (
            adx_strength * 0.4 + ma_strength * 0.3 + abs(price_trend) * 0.3
        )

        # Trend consistency (how stable the trend is)
        if "supertrend_direction" in features:
            direction_changes = features["supertrend_direction"].diff().abs()
            consistency = 1 - direction_changes.rolling(window=20).mean() / 2
            strength_features["trend_consistency"] = consistency
        else:
            strength_features["trend_consistency"] = 0.5

        # Trend duration (how long current trend has lasted)
        strength_features["trend_duration"] = self._calculate_trend_duration(
            features.get("supertrend_direction", pd.Series(0, index=data.index))
        )

        # Composite trend score
        strength_features["trend_composite_score"] = (
            strength_features["trend_strength"] * 0.5
            + strength_features["trend_consistency"] * 0.3
            + (strength_features["trend_duration"] / 100).clip(0, 1) * 0.2
        )

        return strength_features

    def _calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)

        def weighted_average(values):
            if len(values) < period:
                return np.nan
            return np.dot(values[-period:], weights) / weights.sum()

        return series.rolling(window=period).apply(weighted_average, raw=True)

    def _detect_ma_cross(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Detect MA crossovers."""
        diff = fast_ma - slow_ma
        diff_sign = np.sign(diff)
        cross = diff_sign.diff()

        # 1 for golden cross, -1 for death cross
        return cross

    def _calculate_ma_alignment(self, ma_dict: dict[str, pd.Series]) -> pd.Series:
        """Calculate how well MAs are aligned (trending)."""
        if not ma_dict:
            return pd.Series(0, index=next(iter(ma_dict.values())).index)

        # Get MAs sorted by period
        ma_items = []
        for key, ma in ma_dict.items():
            if "sma" in key:
                period = int(key.split("_")[1])
                ma_items.append((period, ma))

        ma_items.sort(key=lambda x: x[0])

        if len(ma_items) < 2:
            return pd.Series(0, index=next(iter(ma_dict.values())).index)

        # Check if MAs are in order (fast > slow for uptrend)
        alignment_scores = []

        for i in range(len(ma_items) - 1):
            period1, ma1 = ma_items[i]
            period2, ma2 = ma_items[i + 1]

            # Score based on correct ordering
            score = (ma1 > ma2).astype(float) * 2 - 1
            alignment_scores.append(score)

        # Average alignment score
        return pd.concat(alignment_scores, axis=1).mean(axis=1)

    def _calculate_directional_movement(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate +DM and -DM for ADX."""
        high_diff = data["high"].diff()
        low_diff = -data["low"].diff()

        plus_dm = pd.Series(0, index=data.index)
        minus_dm = pd.Series(0, index=data.index)

        # +DM
        plus_mask = (high_diff > low_diff) & (high_diff > 0)
        plus_dm[plus_mask] = high_diff[plus_mask]

        # -DM
        minus_mask = (low_diff > high_diff) & (low_diff > 0)
        minus_dm[minus_mask] = low_diff[minus_mask]

        return plus_dm, minus_dm

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr

    def _wilders_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """Apply Wilder's smoothing (used in ADX)."""
        # First value is SMA
        sma = series.rolling(window=period).mean()

        # Then apply Wilder's smoothing
        smooth = series.copy()
        smooth.iloc[:period] = sma.iloc[:period]

        for i in range(period, len(series)):
            if pd.notna(smooth.iloc[i - 1]):
                smooth.iloc[i] = (smooth.iloc[i - 1] * (period - 1) + series.iloc[i]) / period

        return smooth

    def _classify_adx_strength(self, adx: pd.Series) -> pd.Series:
        """Classify ADX strength."""
        strength = pd.Series(0, index=adx.index)

        strength[adx < 20] = 0  # No trend
        strength[(adx >= 20) & (adx < 25)] = 1  # Weak trend
        strength[(adx >= 25) & (adx < 50)] = 2  # Strong trend
        strength[adx >= 50] = 3  # Very strong trend

        return strength

    def _detect_di_crossover(self, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
        """Detect DI crossovers."""
        diff = plus_di - minus_di
        diff_sign = np.sign(diff)
        cross = diff_sign.diff()

        return cross

    def _detect_aroon_crossover(self, aroon_up: pd.Series, aroon_down: pd.Series) -> pd.Series:
        """Detect Aroon crossovers."""
        diff = aroon_up - aroon_down
        diff_sign = np.sign(diff)
        cross = diff_sign.diff()

        return cross

    def _calculate_parabolic_sar(
        self, data: pd.DataFrame, initial_af: float = 0.02, max_af: float = 0.2
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Parabolic SAR."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        psar = close.copy()
        trend = pd.Series(1, index=data.index)  # 1 for uptrend, -1 for downtrend
        af = pd.Series(initial_af, index=data.index)
        ep = high.copy()  # Extreme point

        for i in range(1, len(data)):
            # Previous values
            prev_psar = psar.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]
            prev_af = af.iloc[i - 1]
            prev_ep = ep.iloc[i - 1]

            # Calculate new PSAR
            if prev_trend == 1:  # Uptrend
                psar.iloc[i] = prev_psar + prev_af * (prev_ep - prev_psar)
                psar.iloc[i] = min(
                    psar.iloc[i], low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1]
                )

                if low.iloc[i] <= psar.iloc[i]:
                    # Reverse to downtrend
                    trend.iloc[i] = -1
                    psar.iloc[i] = prev_ep
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = initial_af
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(prev_af + initial_af, max_af)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
            else:  # Downtrend
                psar.iloc[i] = prev_psar - prev_af * (prev_psar - prev_ep)
                psar.iloc[i] = max(
                    psar.iloc[i], high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1]
                )

                if high.iloc[i] >= psar.iloc[i]:
                    # Reverse to uptrend
                    trend.iloc[i] = 1
                    psar.iloc[i] = prev_ep
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = initial_af
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(prev_af + initial_af, max_af)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af

        return psar, trend

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr = self._calculate_true_range(data)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_trend_duration(self, trend_direction: pd.Series) -> pd.Series:
        """Calculate how long current trend has lasted."""
        duration = pd.Series(0, index=trend_direction.index)

        current_trend = 0
        count = 0

        for i in range(len(trend_direction)):
            if trend_direction.iloc[i] != current_trend:
                current_trend = trend_direction.iloc[i]
                count = 1
            else:
                count += 1

            duration.iloc[i] = count

        return duration
