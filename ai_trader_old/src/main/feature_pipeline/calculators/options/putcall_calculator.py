"""
Put/Call Analysis Calculator

Specialized calculator for Put/Call ratio analysis and sentiment indicators including:
- Put/Call volume and open interest ratios
- Extremes detection and percentile analysis
- Smart money vs retail P/C analysis
- Trend and momentum analysis
- Moneyness-based P/C ratios
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

from .base_options import BaseOptionsCalculator
from ..helpers import create_rolling_features, safe_divide, safe_sqrt

logger = logging.getLogger(__name__)


class PutCallAnalysisCalculator(BaseOptionsCalculator):
    """Calculator for Put/Call ratio analysis and sentiment indicators."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Put/Call analysis calculator."""
        super().__init__(config)

        # P/C analysis specific parameters
        self.sentiment_config = self.options_config.get_sentiment_config()
        self.moneyness_config = self.options_config.get_moneyness_config()

        # Analysis windows
        self.short_window = 5
        self.long_window = 20
        self.percentile_window = 252  # 1 year for percentiles

        logger.debug(
            f"Initialized PutCallAnalysisCalculator with extremes: {self.sentiment_config['extreme_low']:.2f}-{self.sentiment_config['extreme_high']:.2f}"
        )

    def get_feature_names(self) -> list[str]:
        """Return list of Put/Call analysis feature names."""
        feature_names = [
            # Basic P/C ratios
            "pc_volume_ratio",
            "pc_oi_ratio",
            "pc_ratio_average",
            "pc_ratio_weighted",
            # Extremes and percentiles
            "pc_ratio_percentile",
            "pc_extreme_high",
            "pc_extreme_low",
            "pc_neutral_zone",
            # Moving averages and trends
            "pc_ratio_ma5",
            "pc_ratio_ma20",
            "pc_ratio_trend",
            "pc_ratio_momentum",
            "pc_ratio_acceleration",
            # Smart money indicators
            "smart_money_pc_ratio",
            "retail_pc_ratio",
            "institutional_bias",
            # Moneyness-based P/C ratios
            "itm_pc_ratio",
            "atm_pc_ratio",
            "otm_pc_ratio",
            "pc_moneyness_spread",
            # Advanced P/C metrics
            "pc_ratio_volatility",
            "pc_ratio_skewness",
            "pc_regime_indicator",
            "pc_sentiment_score",
        ]

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Put/Call analysis features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Put/Call analysis features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)

            processed_data = self.preprocess_data(data)

            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)

            # Create features DataFrame
            features = self.create_empty_features(data.index)

            # Calculate basic P/C ratios
            features = self._calculate_basic_pc_ratios(processed_data, features)

            # Calculate extremes and percentiles
            features = self._calculate_pc_extremes(processed_data, features)

            # Calculate moving averages and trends
            features = self._calculate_pc_trends(processed_data, features)

            # Calculate smart money indicators
            features = self._calculate_smart_money_indicators(processed_data, features)

            # Calculate moneyness-based ratios
            features = self._calculate_moneyness_pc_ratios(processed_data, features)

            # Calculate advanced metrics
            features = self._calculate_advanced_pc_metrics(processed_data, features)

            return features

        except Exception as e:
            logger.error(f"Error calculating Put/Call analysis features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_basic_pc_ratios(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate basic Put/Call ratios."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate P/C ratios from market conditions
                features = self._estimate_pc_ratios(data, features)
                return features

            # Filter valid options
            valid_options = self.options_chain[
                (self.options_chain["volume"] >= self.options_config.min_volume)
                | (self.options_chain["openInterest"] >= self.options_config.min_open_interest)
            ]

            if valid_options.empty:
                return features

            # Separate calls and puts
            calls = valid_options[valid_options["optionType"] == "call"]
            puts = valid_options[valid_options["optionType"] == "put"]

            # Volume-based P/C ratio
            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()

            pc_volume_ratio = safe_divide(put_volume, call_volume, default_value=1.0)
            features["pc_volume_ratio"] = pc_volume_ratio

            # Open interest-based P/C ratio
            call_oi = calls["openInterest"].sum()
            put_oi = puts["openInterest"].sum()

            pc_oi_ratio = safe_divide(put_oi, call_oi, default_value=1.0)
            features["pc_oi_ratio"] = pc_oi_ratio

            # Average P/C ratio
            pc_average = safe_divide((pc_volume_ratio + pc_oi_ratio), 2, default_value=0.0)
            features["pc_ratio_average"] = pc_average

            # Weighted P/C ratio (volume weighted)
            total_volume = call_volume + put_volume
            total_oi = call_oi + put_oi

            if total_volume > 0 and total_oi > 0:
                volume_weight = safe_divide(
                    total_volume, (total_volume + total_oi), default_value=0.5
                )
                oi_weight = safe_divide(total_oi, (total_volume + total_oi), default_value=0.5)

                pc_weighted = pc_volume_ratio * volume_weight + pc_oi_ratio * oi_weight
                features["pc_ratio_weighted"] = pc_weighted
            else:
                features["pc_ratio_weighted"] = pc_average

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic P/C ratios: {e}")
            return features

    def _estimate_pc_ratios(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate P/C ratios when options chain data is not available."""
        try:
            # Estimate based on market conditions
            if len(data) >= 20:
                # Calculate recent returns and volatility
                returns = data["close"].pct_change()
                recent_return = returns.iloc[-20:].mean()
                recent_vol = returns.iloc[-20:].std() * safe_sqrt(252)

                # Base P/C ratio around neutral
                base_pc = self.sentiment_config["neutral"]

                # Adjust for market conditions
                # Negative returns increase put buying
                return_adjustment = -recent_return * 2.0  # Scale factor

                # Higher volatility increases put buying
                vol_adjustment = (recent_vol - 0.2) * 0.5  # Adjust from 20% baseline

                estimated_pc = base_pc + return_adjustment + vol_adjustment

                # Clamp to reasonable range
                estimated_pc = np.clip(estimated_pc, 0.3, 3.0)

                features["pc_volume_ratio"] = estimated_pc
                features["pc_oi_ratio"] = estimated_pc * 1.1  # OI typically higher
                features["pc_ratio_average"] = estimated_pc
                features["pc_ratio_weighted"] = estimated_pc
            else:
                # Default neutral values
                neutral_pc = self.sentiment_config["neutral"]
                features["pc_volume_ratio"] = neutral_pc
                features["pc_oi_ratio"] = neutral_pc
                features["pc_ratio_average"] = neutral_pc
                features["pc_ratio_weighted"] = neutral_pc

            return features

        except Exception as e:
            logger.warning(f"Error estimating P/C ratios: {e}")
            return features

    def _calculate_pc_extremes(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate P/C extremes and percentile analysis."""
        try:
            if "pc_volume_ratio" not in features.columns:
                return features

            current_pc = features["pc_volume_ratio"].iloc[-1] if len(features) > 0 else 1.0

            # Extreme detection
            extreme_high = current_pc >= self.sentiment_config["extreme_high"]
            extreme_low = current_pc <= self.sentiment_config["extreme_low"]

            features["pc_extreme_high"] = int(extreme_high)
            features["pc_extreme_low"] = int(extreme_low)

            # Neutral zone detection
            neutral_zone = (
                current_pc >= self.sentiment_config["extreme_low"] * 1.1
                and current_pc <= self.sentiment_config["extreme_high"] * 0.9
            )
            features["pc_neutral_zone"] = int(neutral_zone)

            # Percentile calculation (if we have enough historical data)
            if len(features) >= self.percentile_window:
                pc_series = features["pc_volume_ratio"].fillna(self.sentiment_config["neutral"])

                # Calculate rolling percentile
                def calculate_percentile(series):
                    if len(series) < 10:
                        return 0.5  # Neutral percentile
                    current_value = series.iloc[-1]
                    historical_values = series.iloc[:-1]
                    percentile = safe_divide(
                        (historical_values < current_value).sum(),
                        len(historical_values),
                        default_value=0.5,
                    )
                    return percentile

                pc_percentile = pc_series.rolling(
                    window=min(self.percentile_window, len(pc_series)), min_periods=10
                ).apply(calculate_percentile)

                features["pc_ratio_percentile"] = pc_percentile
            else:
                # Default to neutral percentile
                features["pc_ratio_percentile"] = 0.5

            return features

        except Exception as e:
            logger.warning(f"Error calculating P/C extremes: {e}")
            return features

    def _calculate_pc_trends(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate P/C moving averages and trend analysis."""
        try:
            if "pc_volume_ratio" not in features.columns:
                return features

            pc_series = features["pc_volume_ratio"].fillna(self.sentiment_config["neutral"])

            # Create DataFrame for rolling features
            pc_df = pd.DataFrame({"pc_ratio": pc_series}, index=features.index)

            # Use create_rolling_features for moving averages
            rolling_features = create_rolling_features(
                pc_df,
                columns=["pc_ratio"],
                windows=[self.short_window, self.long_window],
                operations=["mean"],
                min_periods=1,
            )

            # Extract the specific features we need
            features["pc_ratio_ma5"] = rolling_features[
                f"pc_ratio_rolling_mean_{self.short_window}"
            ]
            features["pc_ratio_ma20"] = rolling_features[
                f"pc_ratio_rolling_mean_{self.long_window}"
            ]

            # Trend calculation (current vs MA)
            current_pc = pc_series.iloc[-1] if len(pc_series) > 0 else 1.0
            ma20_value = pc_ma20.iloc[-1] if len(pc_ma20) > 0 else 1.0

            pc_trend = safe_divide((current_pc - ma20_value), ma20_value, default_value=0.0)
            features["pc_ratio_trend"] = pc_trend

            # Momentum (rate of change)
            if len(pc_series) >= self.short_window:
                pc_momentum = pc_series.pct_change(self.short_window - 1)
                features["pc_ratio_momentum"] = pc_momentum
            else:
                features["pc_ratio_momentum"] = 0.0

            # Acceleration (second derivative)
            if len(pc_series) >= 3:
                pc_change = pc_series.diff()
                pc_acceleration = pc_change.diff()
                features["pc_ratio_acceleration"] = pc_acceleration
            else:
                features["pc_ratio_acceleration"] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Error calculating P/C trends: {e}")
            return features

    def _calculate_smart_money_indicators(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate smart money vs retail P/C indicators."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate smart money indicators
                if "pc_volume_ratio" in features.columns:
                    # Assume smart money has inverse sentiment to retail at extremes
                    retail_pc = features["pc_volume_ratio"].iloc[-1] if len(features) > 0 else 1.0

                    # Smart money typically contrarian at extremes
                    if retail_pc > self.sentiment_config["extreme_high"]:
                        smart_money_pc = retail_pc * 0.7  # Less bearish
                    elif retail_pc < self.sentiment_config["extreme_low"]:
                        smart_money_pc = retail_pc * 1.3  # Less bullish
                    else:
                        smart_money_pc = retail_pc * 0.9  # Slightly contrarian

                    features["smart_money_pc_ratio"] = smart_money_pc
                    features["retail_pc_ratio"] = retail_pc

                    # Institutional bias
                    institutional_bias = safe_divide(
                        (smart_money_pc - retail_pc), retail_pc, default_value=0.0
                    )
                    features["institutional_bias"] = institutional_bias

                return features

            # Classify options by size (proxy for smart money)
            large_trades = self.options_chain[
                self.options_chain["volume"] >= self.options_config.block_trade_size
            ]
            small_trades = self.options_chain[
                self.options_chain["volume"] < self.options_config.block_trade_size
            ]

            # Smart money P/C (large trades)
            if not large_trades.empty:
                large_calls = large_trades[large_trades["optionType"] == "call"]["volume"].sum()
                large_puts = large_trades[large_trades["optionType"] == "put"]["volume"].sum()
                smart_money_pc = safe_divide(large_puts, large_calls, default_value=1.0)
                features["smart_money_pc_ratio"] = smart_money_pc

            # Retail P/C (small trades)
            if not small_trades.empty:
                small_calls = small_trades[small_trades["optionType"] == "call"]["volume"].sum()
                small_puts = small_trades[small_trades["optionType"] == "put"]["volume"].sum()
                retail_pc = safe_divide(small_puts, small_calls, default_value=1.0)
                features["retail_pc_ratio"] = retail_pc

            # Institutional bias
            if "smart_money_pc_ratio" in features.columns and "retail_pc_ratio" in features.columns:
                smart_pc = features["smart_money_pc_ratio"].iloc[-1] if len(features) > 0 else 1.0
                retail_pc = features["retail_pc_ratio"].iloc[-1] if len(features) > 0 else 1.0

                institutional_bias = safe_divide(
                    (smart_pc - retail_pc), retail_pc, default_value=0.0
                )
                features["institutional_bias"] = institutional_bias

            return features

        except Exception as e:
            logger.warning(f"Error calculating smart money indicators: {e}")
            return features

    def _calculate_moneyness_pc_ratios(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate moneyness-based P/C ratios."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate moneyness P/C ratios
                base_pc = (
                    features.get("pc_volume_ratio", pd.Series([1.0])).iloc[-1]
                    if len(features) > 0
                    else 1.0
                )

                # ITM options typically have lower P/C (less speculative)
                features["itm_pc_ratio"] = base_pc * 0.8

                # ATM options have balanced P/C
                features["atm_pc_ratio"] = base_pc

                # OTM options typically have higher P/C (more speculative)
                features["otm_pc_ratio"] = base_pc * 1.2

                # Moneyness spread
                features["pc_moneyness_spread"] = (
                    features["otm_pc_ratio"] - features["itm_pc_ratio"]
                )

                return features

            current_price = data["close"].iloc[-1] if len(data) > 0 else 100

            # Classify options by moneyness using vectorized operations
            moneyness_series = self.options_chain.apply(
                lambda row: self.calculate_moneyness(
                    row["strike"], current_price, row["optionType"]
                ),
                axis=1,
            )

            # Create moneyness masks
            itm_mask = moneyness_series.isin(["deep_itm", "itm"])
            atm_mask = moneyness_series == "atm"
            otm_mask = moneyness_series.isin(["otm", "deep_otm"])

            # Calculate P/C ratios for each moneyness category
            for category, mask in [("itm", itm_mask), ("atm", atm_mask), ("otm", otm_mask)]:
                if mask.any():
                    options_subset = self.options_chain[mask]
                    calls = options_subset[options_subset["optionType"] == "call"]["volume"].sum()
                    puts = options_subset[options_subset["optionType"] == "put"]["volume"].sum()

                    pc_ratio = safe_divide(puts, calls, default_value=1.0)
                    features[f"{category}_pc_ratio"] = pc_ratio
                else:
                    features[f"{category}_pc_ratio"] = 1.0

            # Moneyness spread
            itm_pc = (
                features.get("itm_pc_ratio", pd.Series([1.0])).iloc[-1]
                if len(features) > 0
                else 1.0
            )
            otm_pc = (
                features.get("otm_pc_ratio", pd.Series([1.0])).iloc[-1]
                if len(features) > 0
                else 1.0
            )

            features["pc_moneyness_spread"] = otm_pc - itm_pc

            return features

        except Exception as e:
            logger.warning(f"Error calculating moneyness P/C ratios: {e}")
            return features

    def _calculate_advanced_pc_metrics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced P/C metrics."""
        try:
            if "pc_volume_ratio" not in features.columns:
                return features

            pc_series = features["pc_volume_ratio"].fillna(self.sentiment_config["neutral"])

            # Create DataFrame for rolling features
            pc_df = pd.DataFrame({"pc_ratio": pc_series}, index=features.index)

            # Use create_rolling_features for volatility calculation
            if len(pc_series) >= self.long_window:
                rolling_features = create_rolling_features(
                    pc_df,
                    columns=["pc_ratio"],
                    windows=[self.long_window],
                    operations=["std"],
                    min_periods=5,
                )
                features["pc_ratio_volatility"] = rolling_features[
                    f"pc_ratio_rolling_std_{self.long_window}"
                ]
            else:
                features["pc_ratio_volatility"] = 0.0

            # P/C ratio skewness
            if len(pc_series) >= self.long_window:

                def calculate_skewness(series):
                    if len(series) < 3:
                        return 0.0
                    return series.skew()

                pc_skewness = pc_series.rolling(window=self.long_window, min_periods=3).apply(
                    calculate_skewness
                )
                features["pc_ratio_skewness"] = pc_skewness
            else:
                features["pc_ratio_skewness"] = 0.0

            # P/C regime indicator
            current_pc = pc_series.iloc[-1] if len(pc_series) > 0 else 1.0

            if current_pc >= self.sentiment_config["extreme_high"]:
                regime = 1  # High fear regime
            elif current_pc <= self.sentiment_config["extreme_low"]:
                regime = -1  # High greed regime
            else:
                regime = 0  # Neutral regime

            features["pc_regime_indicator"] = regime

            # P/C sentiment score (-1 to 1, where -1 is very bullish, 1 is very bearish)
            # Normalize P/C ratio to sentiment score
            neutral_pc = self.sentiment_config["neutral"]
            max_pc = self.sentiment_config["extreme_high"]
            min_pc = self.sentiment_config["extreme_low"]

            if current_pc > neutral_pc:
                # Bearish territory
                sentiment_score = safe_divide(
                    (current_pc - neutral_pc), (max_pc - neutral_pc), default_value=0.0
                )
                sentiment_score = min(sentiment_score, 1.0)
            else:
                # Bullish territory
                sentiment_score = safe_divide(
                    (current_pc - neutral_pc), (neutral_pc - min_pc), default_value=0.0
                )
                sentiment_score = max(sentiment_score, -1.0)

            features["pc_sentiment_score"] = sentiment_score

            return features

        except Exception as e:
            logger.warning(f"Error calculating advanced P/C metrics: {e}")
            return features
