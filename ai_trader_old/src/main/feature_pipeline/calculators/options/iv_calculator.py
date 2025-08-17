"""
Implied Volatility Calculator

Specialized calculator for implied volatility analysis including:
- IV rank and percentile analysis
- IV term structure calculation
- Volatility risk premium analysis
- IV momentum and acceleration
- Historical vs implied volatility comparison
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

from .base_options import BaseOptionsCalculator
from ..helpers import create_rolling_features, safe_divide, safe_sqrt

logger = logging.getLogger(__name__)


class ImpliedVolatilityCalculator(BaseOptionsCalculator):
    """Calculator for implied volatility analysis and term structure."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize implied volatility calculator."""
        super().__init__(config)

        # IV-specific parameters
        self.iv_windows = [20, 60, 252]  # Windows for IV analysis
        self.percentile_window = self.options_config.iv_percentile_window
        self.rank_window = self.options_config.iv_rank_window
        self.smoothing_window = self.options_config.iv_smoothing_window

        # Term structure parameters
        self.term_structure_expiries = self.options_config.expiry_windows
        self.min_term_expiry = self.options_config.iv_term_structure_min_expiry

        logger.debug(
            f"Initialized ImpliedVolatilityCalculator with {len(self.term_structure_expiries)} term structure points"
        )

    def get_feature_names(self) -> list[str]:
        """Return list of implied volatility feature names."""
        feature_names = []

        # Basic IV metrics
        feature_names.extend(
            [
                "iv_rank",
                "iv_percentile",
                "iv_momentum",
                "iv_acceleration",
                "iv_trend",
                "iv_volatility",
            ]
        )

        # Historical vs implied volatility
        feature_names.extend(
            [
                "iv_hv_ratio_20d",
                "iv_hv_ratio_60d",
                "iv_premium_20d",
                "iv_premium_60d",
                "vol_risk_premium",
                "volatility_efficiency",
            ]
        )

        # Term structure features
        feature_names.extend(
            ["iv_term_slope", "iv_term_curvature", "iv_term_level", "front_back_iv_spread"]
        )

        # IV for different expiration windows
        for expiry in self.term_structure_expiries:
            feature_names.append(f"iv_{expiry}d")

        # Advanced IV metrics
        feature_names.extend(
            [
                "iv_skewness",
                "iv_kurtosis",
                "iv_regime_indicator",
                "iv_stress_level",
                "iv_mean_reversion",
                "iv_persistence",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied volatility features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with IV features
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

            # Calculate basic IV metrics
            features = self._calculate_basic_iv_metrics(processed_data, features)

            # Calculate historical vs implied volatility
            features = self._calculate_iv_hv_comparison(processed_data, features)

            # Calculate term structure features
            features = self._calculate_term_structure_features(processed_data, features)

            # Calculate advanced IV metrics
            features = self._calculate_advanced_iv_metrics(processed_data, features)

            return features

        except Exception as e:
            logger.error(f"Error calculating IV features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_basic_iv_metrics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate basic implied volatility metrics."""
        try:
            # Get IV data
            iv_data = self._get_iv_data(data)

            if iv_data is None or iv_data.empty:
                logger.warning("No IV data available for analysis")
                return features

            # Ensure we have a Series
            if isinstance(iv_data, pd.DataFrame):
                iv_series = iv_data.iloc[:, 0]  # Use first column
            else:
                iv_series = iv_data

            # Basic IV level
            current_iv = iv_series.iloc[-1] if len(iv_series) > 0 else 0.2
            features["current_iv"] = current_iv

            # IV rank (percentile rank over specified window)
            if len(iv_series) >= self.rank_window:
                iv_rank = iv_series.rolling(window=self.rank_window, min_periods=20).rank(pct=True)
                features["iv_rank"] = iv_rank
            else:
                features["iv_rank"] = 0.5  # Neutral rank

            # IV percentile (using scipy for more accurate calculation)
            if len(iv_series) >= self.percentile_window:

                def calculate_percentile(series):
                    if len(series) < 10:
                        return 0.5
                    current_val = series.iloc[-1]
                    historical_vals = series.iloc[:-1]
                    return safe_divide(
                        stats.percentileofscore(historical_vals, current_val),
                        100,
                        default_value=0.5,
                    )

                iv_percentile = iv_series.rolling(
                    window=self.percentile_window, min_periods=20
                ).apply(calculate_percentile)
                features["iv_percentile"] = iv_percentile
            else:
                features["iv_percentile"] = 0.5  # Neutral percentile

            # IV momentum (rate of change)
            if len(iv_series) >= 5:
                iv_momentum = iv_series.pct_change(4)  # 5-day momentum
                features["iv_momentum"] = iv_momentum
            else:
                features["iv_momentum"] = 0.0

            # IV acceleration (second derivative)
            if len(iv_series) >= 6:
                iv_change = iv_series.diff()
                iv_acceleration = iv_change.diff()
                features["iv_acceleration"] = iv_acceleration
            else:
                features["iv_acceleration"] = 0.0

            # IV trend (linear regression slope)
            if len(iv_series) >= 10:

                def calculate_trend(series):
                    if len(series) < 3:
                        return 0.0
                    x = np.arange(len(series))
                    try:
                        slope = np.polyfit(x, series, 1)[0]
                        return slope
                    except (ValueError, TypeError, np.linalg.LinAlgError):
                        return 0.0

                iv_trend = iv_series.rolling(window=10, min_periods=3).apply(calculate_trend)
                features["iv_trend"] = iv_trend
            else:
                features["iv_trend"] = 0.0

            # IV volatility (volatility of volatility)
            if len(iv_series) >= 20:
                # Create DataFrame for rolling features
                iv_df = pd.DataFrame({"iv": iv_series}, index=features.index)

                # Use create_rolling_features for volatility calculation
                rolling_features = create_rolling_features(
                    iv_df, columns=["iv"], windows=[20], operations=["std"], min_periods=5
                )
                features["iv_volatility"] = rolling_features["iv_rolling_std_20"]
            else:
                features["iv_volatility"] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic IV metrics: {e}")
            return features

    def _get_iv_data(self, data: pd.DataFrame) -> pd.Series | None:
        """Get implied volatility data from available sources."""
        try:
            # Priority 1: Use historical IV data if provided
            if self.historical_iv is not None and not self.historical_iv.empty:
                return (
                    self.historical_iv.iloc[:, 0]
                    if isinstance(self.historical_iv, pd.DataFrame)
                    else self.historical_iv
                )

            # Priority 2: Calculate from options chain if available
            if self.options_chain is not None and not self.options_chain.empty:
                if "impliedVolatility" in self.options_chain.columns:
                    # Use ATM options for representative IV
                    current_price = data["close"].iloc[-1] if len(data) > 0 else 100

                    # Filter for near-the-money options
                    atm_options = self.options_chain[
                        (self.options_chain["strike"] >= current_price * 0.95)
                        & (self.options_chain["strike"] <= current_price * 1.05)
                    ]

                    if not atm_options.empty:
                        # Volume-weighted average IV
                        weights = atm_options["volume"] + atm_options["openInterest"]
                        if weights.sum() > 0:
                            weighted_iv = safe_divide(
                                (atm_options["impliedVolatility"] * weights).sum(),
                                weights.sum(),
                                default_value=0.2,
                            )
                            return pd.Series([weighted_iv], index=[data.index[-1]])

            # Priority 3: Estimate from historical volatility
            estimated_iv = self.estimate_implied_volatility(data["close"], window=20)
            return estimated_iv

        except Exception as e:
            logger.warning(f"Error getting IV data: {e}")
            return None

    def _calculate_iv_hv_comparison(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate implied vs historical volatility comparisons."""
        try:
            # Calculate historical volatility for different windows
            returns = data["close"].pct_change()

            for window in [20, 60]:
                hist_vol = returns.rolling(window=window).std() * safe_sqrt(252)

                # Get corresponding IV (estimate if needed)
                iv_data = self._get_iv_data(data)

                if iv_data is not None:
                    # Current IV value
                    current_iv = (
                        iv_data.iloc[-1]
                        if len(iv_data) > 0
                        else self.options_config.default_volatility
                    )

                    # IV/HV ratio
                    iv_hv_ratio = safe_divide(current_iv, hist_vol.iloc[-1], default_value=1.0)
                    features[f"iv_hv_ratio_{window}d"] = iv_hv_ratio

                    # IV premium (absolute difference)
                    iv_premium = current_iv - hist_vol.iloc[-1]
                    features[f"iv_premium_{window}d"] = iv_premium

                else:
                    # Use default ratios when IV unavailable
                    features[f"iv_hv_ratio_{window}d"] = 1.2  # Typical IV premium
                    features[f"iv_premium_{window}d"] = (
                        hist_vol.iloc[-1] * 0.2 if len(hist_vol) > 0 else 0.04
                    )

            # Volatility risk premium (difference between implied and realized)
            if len(returns) >= 20:
                realized_vol = returns.iloc[-20:].std() * safe_sqrt(252)
                iv_data = self._get_iv_data(data)

                if iv_data is not None:
                    current_iv = (
                        iv_data.iloc[-1]
                        if len(iv_data) > 0
                        else self.options_config.default_volatility
                    )
                    vol_risk_premium = current_iv - realized_vol
                    features["vol_risk_premium"] = vol_risk_premium
                else:
                    features["vol_risk_premium"] = 0.0

            # Volatility efficiency (how well IV predicts future realized vol)
            # This would require forward-looking calculation in production
            features["volatility_efficiency"] = 0.0  # Placeholder

            return features

        except Exception as e:
            logger.warning(f"Error calculating IV/HV comparison: {e}")
            return features

    def _calculate_term_structure_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate IV term structure features."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate term structure based on typical patterns
                features = self._estimate_term_structure(data, features)
                return features

            # Calculate IV for each expiration
            term_structure_ivs = self._calculate_iv_term_structure()

            if not term_structure_ivs:
                return features

            # Store individual expiry IVs
            for expiry, iv_value in term_structure_ivs.items():
                features[f"iv_{expiry}d"] = iv_value

            # Calculate term structure statistics
            sorted_expiries = sorted(term_structure_ivs.keys())
            iv_values = [term_structure_ivs[exp] for exp in sorted_expiries]

            if len(iv_values) >= 2:
                # Term structure slope (front to back)
                front_iv = iv_values[0]
                back_iv = iv_values[-1]

                iv_term_slope = safe_divide((back_iv - front_iv), front_iv, default_value=0.0)
                features["iv_term_slope"] = iv_term_slope

                # Front-back spread
                features["front_back_iv_spread"] = back_iv - front_iv

                # Term structure level (average IV)
                features["iv_term_level"] = np.mean(iv_values)

            if len(iv_values) >= 3:
                # Term structure curvature (second derivative)
                # Using simple finite difference approximation
                front_slope = iv_values[1] - iv_values[0]
                back_slope = iv_values[2] - iv_values[1]
                curvature = back_slope - front_slope
                features["iv_term_curvature"] = curvature

            return features

        except Exception as e:
            logger.warning(f"Error calculating term structure features: {e}")
            return features

    def _calculate_iv_term_structure(self) -> dict[int, float]:
        """Calculate IV term structure from options chain."""
        try:
            term_structure = {}

            if "impliedVolatility" not in self.options_chain.columns:
                return term_structure

            current_price = self.options_chain.get("underlyingPrice", 100)  # Fallback price

            # Group by expiration and calculate representative IV
            for expiry in self.term_structure_expiries:
                # Find options near this expiration
                mask = (self.options_chain["daysToExpiration"] >= expiry - 3) & (
                    self.options_chain["daysToExpiration"] <= expiry + 3
                )

                expiry_options = self.options_chain[mask]

                if expiry_options.empty:
                    continue

                # Filter for ATM options
                atm_options = expiry_options[
                    (expiry_options["strike"] >= current_price * 0.95)
                    & (expiry_options["strike"] <= current_price * 1.05)
                ]

                if not atm_options.empty:
                    # Volume-weighted IV
                    weights = atm_options["volume"] + atm_options["openInterest"]
                    if weights.sum() > 0:
                        weighted_iv = safe_divide(
                            (atm_options["impliedVolatility"] * weights).sum(),
                            weights.sum(),
                            default_value=0.2,
                        )
                        term_structure[expiry] = weighted_iv
                    else:
                        # Simple average if no volume data
                        term_structure[expiry] = atm_options["impliedVolatility"].mean()

            return term_structure

        except Exception as e:
            logger.warning(f"Error calculating IV term structure: {e}")
            return {}

    def _estimate_term_structure(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate term structure when options chain unavailable."""
        try:
            # Calculate base IV from historical volatility
            base_iv = (
                self.estimate_implied_volatility(data["close"], window=20).iloc[-1]
                if len(data) > 0
                else 0.2
            )

            # Typical term structure pattern (contango most common)
            for i, expiry in enumerate(self.term_structure_expiries):
                # Front months typically have lower IV
                if expiry <= 30:
                    iv_factor = 0.9  # 10% discount for front month
                elif expiry <= 60:
                    iv_factor = 1.0  # Par for 2-month
                else:
                    iv_factor = 1.1  # 10% premium for longer dated

                estimated_iv = base_iv * iv_factor
                features[f"iv_{expiry}d"] = estimated_iv

            # Estimate term structure statistics
            features["iv_term_slope"] = 0.1  # Typical contango
            features["iv_term_level"] = base_iv
            features["iv_term_curvature"] = 0.0  # Neutral curvature
            features["front_back_iv_spread"] = base_iv * 0.2  # 20% spread

            return features

        except Exception as e:
            logger.warning(f"Error estimating term structure: {e}")
            return features

    def _calculate_advanced_iv_metrics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced IV metrics."""
        try:
            iv_data = self._get_iv_data(data)

            if iv_data is None or len(iv_data) < 20:
                # Set default values for advanced metrics
                features["iv_skewness"] = 0.0
                features["iv_kurtosis"] = 3.0  # Normal distribution
                features["iv_regime_indicator"] = 0  # Neutral regime
                features["iv_stress_level"] = 0.0
                features["iv_mean_reversion"] = 0.0
                features["iv_persistence"] = 0.0
                return features

            iv_series = iv_data if isinstance(iv_data, pd.Series) else iv_data.iloc[:, 0]

            # IV distribution characteristics
            if len(iv_series) >= 60:
                # Rolling skewness
                def calculate_skewness(series):
                    if len(series) < 3:
                        return 0.0
                    return series.skew()

                iv_skewness = iv_series.rolling(window=60, min_periods=10).apply(calculate_skewness)
                features["iv_skewness"] = iv_skewness

                # Rolling kurtosis
                def calculate_kurtosis(series):
                    if len(series) < 4:
                        return 3.0
                    return series.kurtosis()

                iv_kurtosis = iv_series.rolling(window=60, min_periods=10).apply(calculate_kurtosis)
                features["iv_kurtosis"] = iv_kurtosis

            # IV regime classification
            current_iv = iv_series.iloc[-1]

            if len(iv_series) >= self.rank_window:
                iv_percentile = safe_divide(
                    (iv_series.iloc[-self.rank_window :] < current_iv).sum(),
                    self.rank_window,
                    default_value=0.5,
                )

                if iv_percentile >= 0.8:
                    regime = 1  # High IV regime
                elif iv_percentile <= 0.2:
                    regime = -1  # Low IV regime
                else:
                    regime = 0  # Normal regime

                features["iv_regime_indicator"] = regime

                # Stress level (based on IV percentile and momentum)
                iv_momentum = (
                    features.get("iv_momentum", pd.Series([0.0])).iloc[-1]
                    if len(features) > 0
                    else 0.0
                )
                stress_level = iv_percentile + abs(iv_momentum) * 0.5
                features["iv_stress_level"] = min(stress_level, 1.0)

            # Mean reversion tendency
            if len(iv_series) >= 30:
                # Calculate autocorrelation at lag 1
                def calculate_autocorr(series):
                    if len(series) < 10:
                        return 0.0
                    try:
                        return series.autocorr(lag=1)
                    except (ValueError, TypeError):
                        return 0.0

                iv_persistence = iv_series.rolling(window=30, min_periods=10).apply(
                    calculate_autocorr
                )
                features["iv_persistence"] = iv_persistence

                # Mean reversion (negative of persistence)
                features["iv_mean_reversion"] = -iv_persistence

            return features

        except Exception as e:
            logger.warning(f"Error calculating advanced IV metrics: {e}")
            return features
