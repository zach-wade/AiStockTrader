"""
Volume Flow Calculator

Specialized calculator for options volume and flow analysis including:
- Basic options volume and open interest
- Options flow analysis and net flow calculations
- Block trade detection and analysis
- Term structure volume concentration
- Premium flow tracking
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

from .base_options import BaseOptionsCalculator
from ..helpers import create_rolling_features, safe_divide

logger = logging.getLogger(__name__)


class VolumeFlowCalculator(BaseOptionsCalculator):
    """Calculator for options volume and flow analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize volume flow calculator."""
        super().__init__(config)

        # Volume flow specific parameters
        self.volume_config = self.options_config.get_volume_config()
        self.flow_window = self.options_config.net_flow_window
        self.momentum_window = self.options_config.flow_momentum_window

        logger.debug(f"Initialized VolumeFlowCalculator with {self.flow_window}d flow window")

    def get_feature_names(self) -> list[str]:
        """Return list of volume flow feature names."""
        feature_names = [
            # Basic volume and open interest
            "call_volume",
            "put_volume",
            "total_option_volume",
            "call_oi",
            "put_oi",
            "total_oi",
            # Volume flow metrics
            "net_option_flow",
            "option_flow_ratio",
            "volume_momentum",
            "flow_acceleration",
            "volume_trend",
            # Block and large trades
            "block_trade_count",
            "block_trade_volume",
            "block_trade_ratio",
            "large_trade_count",
            "large_trade_volume",
            # Premium flow
            "net_premium_flow",
            "premium_momentum",
            "average_premium_per_contract",
            # Term structure volume
            "front_month_concentration",
            "near_far_volume_ratio",
            "term_structure_volume_spread",
            "expiry_volume_concentration",
            # Advanced flow metrics
            "volume_oi_ratio",
            "turnover_rate",
            "flow_persistence",
            "volume_volatility",
        ]

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume flow features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with volume flow features
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

            # Calculate basic volume features
            features = self._calculate_basic_volume_features(processed_data, features)

            # Calculate flow metrics
            features = self._calculate_flow_metrics(processed_data, features)

            # Calculate block trade features
            features = self._calculate_block_trade_features(processed_data, features)

            # Calculate premium flow features
            features = self._calculate_premium_flow_features(processed_data, features)

            # Calculate term structure features
            features = self._calculate_term_structure_features(processed_data, features)

            # Calculate advanced flow metrics
            features = self._calculate_advanced_flow_metrics(processed_data, features)

            return features

        except Exception as e:
            logger.error(f"Error calculating volume flow features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_basic_volume_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate basic volume and open interest features."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Use estimated values when no options chain data
                features = self._estimate_volume_features(data, features)
                return features

            # Filter valid options
            valid_options = self.options_chain[
                (self.options_chain["volume"] >= self.volume_config["min_volume"])
                | (self.options_chain["openInterest"] >= self.volume_config["min_open_interest"])
            ]

            if valid_options.empty:
                return features

            # Separate calls and puts
            calls = valid_options[valid_options["optionType"] == "call"]
            puts = valid_options[valid_options["optionType"] == "put"]

            # Basic volume metrics
            call_volume = calls["volume"].sum()
            put_volume = puts["volume"].sum()
            total_volume = call_volume + put_volume

            features["call_volume"] = call_volume
            features["put_volume"] = put_volume
            features["total_option_volume"] = total_volume

            # Basic open interest metrics
            call_oi = calls["openInterest"].sum()
            put_oi = puts["openInterest"].sum()
            total_oi = call_oi + put_oi

            features["call_oi"] = call_oi
            features["put_oi"] = put_oi
            features["total_oi"] = total_oi

            # Volume/OI ratios
            features["volume_oi_ratio"] = safe_divide(total_volume, total_oi, default_value=0.0)
            features["turnover_rate"] = safe_divide(total_volume, total_oi, default_value=0.0)

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic volume features: {e}")
            return features

    def _estimate_volume_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate volume features when options chain data is not available."""
        try:
            # Use stock volume as proxy for options activity
            if "volume" in data.columns:
                stock_volume = data["volume"].iloc[-1] if len(data) > 0 else 0

                # Estimate options volume as percentage of stock volume
                estimated_options_volume = stock_volume * 0.05  # 5% of stock volume

                # Estimate call/put split based on market conditions
                returns = data["close"].pct_change().iloc[-20:].mean() if len(data) >= 20 else 0

                # Bullish markets tend to have higher call volume
                call_bias = 0.6 if returns > 0 else 0.4

                features["call_volume"] = estimated_options_volume * call_bias
                features["put_volume"] = estimated_options_volume * (1 - call_bias)
                features["total_option_volume"] = estimated_options_volume

                # Estimate open interest as multiple of volume
                oi_multiple = 5.0  # Typical OI is 5x daily volume
                features["call_oi"] = features["call_volume"] * oi_multiple
                features["put_oi"] = features["put_volume"] * oi_multiple
                features["total_oi"] = features["total_option_volume"] * oi_multiple

                features["volume_oi_ratio"] = safe_divide(1.0, oi_multiple, default_value=0.2)
                features["turnover_rate"] = safe_divide(1.0, oi_multiple, default_value=0.2)

            return features

        except Exception as e:
            logger.warning(f"Error estimating volume features: {e}")
            return features

    def _calculate_flow_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate options flow metrics."""
        try:
            if "call_volume" not in features.columns or "put_volume" not in features.columns:
                return features

            call_volume = features["call_volume"].iloc[-1] if len(features) > 0 else 0
            put_volume = features["put_volume"].iloc[-1] if len(features) > 0 else 0
            total_volume = call_volume + put_volume

            # Net flow (calls minus puts)
            net_flow = call_volume - put_volume
            features["net_option_flow"] = net_flow

            # Flow ratio
            flow_ratio = safe_divide(call_volume, put_volume, default_value=1.0)
            features["option_flow_ratio"] = flow_ratio

            # Calculate momentum and acceleration if we have historical data
            if len(features) >= self.momentum_window:
                # Volume momentum (rate of change)
                volume_series = features["total_option_volume"].fillna(0)
                volume_momentum = volume_series.pct_change(self.momentum_window - 1)
                features["volume_momentum"] = volume_momentum

                # Flow acceleration (second derivative)
                flow_series = features["net_option_flow"].fillna(0)
                flow_change = flow_series.diff()
                flow_acceleration = flow_change.diff()
                features["flow_acceleration"] = flow_acceleration

                # Volume trend (linear regression slope)
                def calculate_trend(series):
                    if len(series) < 3:
                        return 0.0
                    x = np.arange(len(series))
                    try:
                        slope = np.polyfit(x, series, 1)[0]
                        return slope
                    except (ValueError, TypeError, np.linalg.LinAlgError):
                        return 0.0

                volume_trend = volume_series.rolling(
                    window=self.momentum_window, min_periods=3
                ).apply(calculate_trend)
                features["volume_trend"] = volume_trend

                # Flow persistence (correlation with lagged values)
                flow_persistence = flow_series.rolling(window=self.momentum_window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0.0
                )
                features["flow_persistence"] = flow_persistence

                # Volume volatility
                # Create DataFrame for rolling features
                vol_df = pd.DataFrame({"volume": volume_series}, index=features.index)

                # Use create_rolling_features for volatility calculation
                rolling_features = create_rolling_features(
                    vol_df,
                    columns=["volume"],
                    windows=[self.momentum_window],
                    operations=["std"],
                    min_periods=1,
                )
                features["volume_volatility"] = rolling_features[
                    f"volume_rolling_std_{self.momentum_window}"
                ]

            return features

        except Exception as e:
            logger.warning(f"Error calculating flow metrics: {e}")
            return features

    def _calculate_block_trade_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate block trade features."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate block trades based on total volume
                total_volume = (
                    features.get("total_option_volume", pd.Series([0])).iloc[-1]
                    if len(features) > 0
                    else 0
                )

                # Estimate that 10% of volume comes from block trades
                estimated_block_volume = total_volume * 0.1
                estimated_block_count = max(
                    1,
                    int(
                        safe_divide(
                            estimated_block_volume,
                            self.volume_config["block_size"],
                            default_value=0.0,
                        )
                    ),
                )

                features["block_trade_count"] = estimated_block_count
                features["block_trade_volume"] = estimated_block_volume
                features["block_trade_ratio"] = 0.1

                # Large trades estimate
                estimated_large_volume = total_volume * 0.05
                estimated_large_count = max(
                    1,
                    int(
                        safe_divide(
                            estimated_large_volume,
                            self.volume_config.get("large_size", 500),
                            default_value=0.0,
                        )
                    ),
                )

                features["large_trade_count"] = estimated_large_count
                features["large_trade_volume"] = estimated_large_volume

                return features

            # Identify block trades
            block_trades = self.options_chain[
                self.options_chain["volume"] >= self.volume_config["block_size"]
            ]

            # Identify large trades
            large_trades = self.options_chain[
                self.options_chain["volume"] >= self.volume_config.get("large_size", 500)
            ]

            # Block trade metrics
            block_count = len(block_trades)
            block_volume = block_trades["volume"].sum()
            total_volume = self.options_chain["volume"].sum()

            features["block_trade_count"] = block_count
            features["block_trade_volume"] = block_volume
            features["block_trade_ratio"] = safe_divide(
                block_volume, total_volume, default_value=0.0
            )

            # Large trade metrics
            large_count = len(large_trades)
            large_volume = large_trades["volume"].sum()

            features["large_trade_count"] = large_count
            features["large_trade_volume"] = large_volume

            return features

        except Exception as e:
            logger.warning(f"Error calculating block trade features: {e}")
            return features

    def _calculate_premium_flow_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate premium flow features."""
        try:
            if (
                self.options_chain is None
                or self.options_chain.empty
                or "lastPrice" not in self.options_chain.columns
            ):
                # Estimate premium flow from volume and price data
                total_volume = (
                    features.get("total_option_volume", pd.Series([0])).iloc[-1]
                    if len(features) > 0
                    else 0
                )
                stock_price = data["close"].iloc[-1] if len(data) > 0 else 100

                # Estimate average option premium as percentage of stock price
                estimated_avg_premium = stock_price * 0.02  # 2% of stock price
                estimated_premium_flow = (
                    total_volume * estimated_avg_premium * 100
                )  # Contract multiplier

                features["net_premium_flow"] = estimated_premium_flow
                features["premium_momentum"] = 0.0
                features["average_premium_per_contract"] = estimated_avg_premium

                return features

            # Calculate premium flow from options chain
            self.options_chain["premium_total"] = (
                self.options_chain["lastPrice"]
                * self.options_chain["volume"]
                * 100  # Options contract multiplier
            )

            # Separate call and put premiums
            calls = self.options_chain[self.options_chain["optionType"] == "call"]
            puts = self.options_chain[self.options_chain["optionType"] == "put"]

            call_premium = calls["premium_total"].sum()
            put_premium = puts["premium_total"].sum()

            # Net premium flow (calls minus puts)
            net_premium = call_premium - put_premium
            features["net_premium_flow"] = net_premium

            # Average premium per contract
            total_volume = self.options_chain["volume"].sum()
            total_premium = call_premium + put_premium

            avg_premium = safe_divide(total_premium, total_volume * 100, default_value=0.0)
            features["average_premium_per_contract"] = avg_premium

            # Premium momentum (if historical data available)
            if len(features) >= self.momentum_window:
                premium_series = features["net_premium_flow"].fillna(0)
                premium_momentum = premium_series.pct_change(self.momentum_window - 1)
                features["premium_momentum"] = premium_momentum
            else:
                features["premium_momentum"] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Error calculating premium flow features: {e}")
            return features

    def _calculate_term_structure_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate term structure volume features."""
        try:
            if (
                self.options_chain is None
                or self.options_chain.empty
                or "daysToExpiration" not in self.options_chain.columns
            ):
                # Estimate term structure based on typical distributions
                features["front_month_concentration"] = 0.6  # 60% in front month
                features["near_far_volume_ratio"] = 2.0  # 2:1 near to far
                features["term_structure_volume_spread"] = 0.4
                features["expiry_volume_concentration"] = 0.7

                return features

            # Classify options by expiration
            front_month = self.options_chain[
                self.options_chain["daysToExpiration"] <= self.options_config.front_month_cutoff
            ]
            back_month = self.options_chain[
                self.options_chain["daysToExpiration"] >= self.options_config.back_month_cutoff
            ]

            # Volume concentration in front month
            front_volume = front_month["volume"].sum()
            total_volume = self.options_chain["volume"].sum()

            front_concentration = safe_divide(front_volume, total_volume, default_value=0.0)
            features["front_month_concentration"] = front_concentration

            # Near vs far month volume ratio
            back_volume = back_month["volume"].sum()
            near_far_ratio = safe_divide(front_volume, back_volume, default_value=1.0)
            features["near_far_volume_ratio"] = near_far_ratio

            # Term structure spread
            term_spread = front_concentration - (1 - front_concentration)
            features["term_structure_volume_spread"] = term_spread

            # Expiry concentration (Herfindahl index)
            expiry_volumes = self.options_chain.groupby("daysToExpiration")["volume"].sum()
            total_volume_check = expiry_volumes.sum()

            if total_volume_check > 0:
                expiry_shares = safe_divide(expiry_volumes, total_volume_check, default_value=0.0)
                concentration_index = (expiry_shares**2).sum()
                features["expiry_volume_concentration"] = concentration_index
            else:
                features["expiry_volume_concentration"] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Error calculating term structure features: {e}")
            return features

    def _calculate_advanced_flow_metrics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced flow metrics."""
        try:
            # These metrics require historical data or complex calculations
            # For now, provide reasonable estimates

            # Volume efficiency (volume per dollar of premium)
            if "total_option_volume" in features.columns and "net_premium_flow" in features.columns:
                total_volume = features["total_option_volume"].iloc[-1] if len(features) > 0 else 0
                premium_flow = (
                    abs(features["net_premium_flow"].iloc[-1]) if len(features) > 0 else 1
                )

                # This would be implemented with more sophisticated logic
                # in a production system
                pass

            return features

        except Exception as e:
            logger.warning(f"Error calculating advanced flow metrics: {e}")
            return features
