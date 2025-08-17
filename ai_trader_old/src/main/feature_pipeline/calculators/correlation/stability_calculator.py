"""
Correlation Stability Calculator

Analyzes the stability and persistence of correlations over time,
detecting regime changes and structural breaks in correlation patterns.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from main.utils.core import AsyncCircuitBreaker, get_logger

from .base_correlation import BaseCorrelationCalculator
from ..helpers import (
    calculate_entropy,
    calculate_hurst_exponent,
    create_feature_dataframe,
    safe_divide,
)

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class StabilityAnalysisCalculator(BaseCorrelationCalculator):
    """
    Calculates correlation stability metrics and regime change indicators.

    Features include:
    - Correlation stability metrics
    - Regime change detection
    - Structural break identification
    - Correlation persistence measures
    - Dynamic correlation confidence intervals
    - Correlation forecast accuracy
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize stability analysis calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Stability-specific parameters
        self.stability_windows = config.get("stability_windows", [20, 50, 100])
        self.breakpoint_threshold = config.get("breakpoint_threshold", 2.0)
        self.regime_min_length = config.get("regime_min_length", 20)

        # Circuit breaker for computationally intensive operations
        self.circuit_breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=30)

        logger.info("Initialized StabilityAnalysisCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of stability feature names."""
        features = []

        # Basic stability metrics
        for window in self.stability_windows:
            features.extend(
                [
                    f"corr_stability_{window}",
                    f"corr_mean_{window}",
                    f"corr_std_{window}",
                    f"corr_cv_{window}",
                    f"corr_range_{window}",
                    f"corr_iqr_{window}",
                    f"corr_mad_{window}",
                ]
            )

        # Persistence measures
        features.extend(
            [
                "corr_hurst_exponent",
                "corr_half_life",
                "corr_mean_reversion_speed",
                "corr_autocorrelation_1",
                "corr_autocorrelation_5",
                "corr_autocorrelation_10",
            ]
        )

        # Regime detection
        features.extend(
            [
                "regime_duration",
                "regime_count",
                "current_regime_stability",
                "regime_transition_probability",
                "regime_entropy",
            ]
        )

        # Structural breaks
        features.extend(
            [
                "structural_break_indicator",
                "time_since_break",
                "break_magnitude",
                "break_confidence",
            ]
        )

        # Confidence intervals
        for confidence in [0.90, 0.95]:
            features.extend(
                [
                    f"corr_ci_lower_{int(confidence*100)}",
                    f"corr_ci_upper_{int(confidence*100)}",
                    f"corr_ci_width_{int(confidence*100)}",
                ]
            )

        # Stability forecasts
        features.extend(
            [
                "expected_corr_1d",
                "expected_corr_5d",
                "corr_forecast_error",
                "corr_forecast_confidence",
            ]
        )

        # Cross-correlation stability
        features.extend(
            ["cross_corr_stability", "lead_lag_stability", "correlation_network_stability"]
        )

        return features

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation stability features.

        Args:
            data: Input DataFrame with price data

        Returns:
            DataFrame with stability features
        """
        try:
            # Initialize features DataFrame
            features = create_feature_dataframe(data.index)

            # Calculate returns
            returns = self.calculate_returns(data["close"])

            # If we have multi-asset data, calculate pairwise stability
            if self.asset_data:
                # Get benchmark correlation if available
                if self.benchmark_data is not None:
                    benchmark_returns = self.calculate_returns(self.benchmark_data["close"])
                    base_correlation = returns.rolling(window=20, min_periods=10).corr(
                        benchmark_returns
                    )
                else:
                    # Use first available asset as reference
                    first_asset = list(self.asset_data.keys())[0]
                    first_returns = self.calculate_returns(self.asset_data[first_asset]["close"])
                    base_correlation = returns.rolling(window=20, min_periods=10).corr(
                        first_returns
                    )
            else:
                # Self-correlation stability (autocorrelation)
                base_correlation = returns.rolling(window=20, min_periods=10).apply(
                    lambda x: x.autocorr(lag=1), raw=True
                )

            # Calculate basic stability metrics
            stability_features = self._calculate_stability_metrics(base_correlation)
            features = pd.concat([features, stability_features], axis=1)

            # Calculate persistence measures
            persistence_features = self._calculate_persistence_metrics(base_correlation)
            features = pd.concat([features, persistence_features], axis=1)

            # Detect regimes
            regime_features = self._detect_correlation_regimes(base_correlation)
            features = pd.concat([features, regime_features], axis=1)

            # Detect structural breaks
            break_features = self._detect_structural_breaks(base_correlation)
            features = pd.concat([features, break_features], axis=1)

            # Calculate confidence intervals
            ci_features = self._calculate_confidence_intervals(base_correlation)
            features = pd.concat([features, ci_features], axis=1)

            # Generate stability forecasts
            forecast_features = self._generate_stability_forecasts(base_correlation)
            features = pd.concat([features, forecast_features], axis=1)

            # Multi-asset stability if available
            if len(self.asset_data) > 1:
                multi_features = self._calculate_multi_asset_stability()
                features = pd.concat([features, multi_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating stability features: {e}")
            return self._create_empty_features(data.index)

    def _calculate_stability_metrics(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Calculate basic correlation stability metrics."""
        features = pd.DataFrame(index=correlation_series.index)

        for window in self.stability_windows:
            # Rolling statistics
            rolling_mean = correlation_series.rolling(window=window, min_periods=window // 2).mean()

            rolling_std = correlation_series.rolling(window=window, min_periods=window // 2).std()

            # Stability score (inverse of coefficient of variation)
            cv = safe_divide(rolling_std, rolling_mean.abs(), default_value=1.0)
            features[f"corr_stability_{window}"] = 1 / (1 + cv)

            # Basic statistics
            features[f"corr_mean_{window}"] = rolling_mean
            features[f"corr_std_{window}"] = rolling_std
            features[f"corr_cv_{window}"] = cv

            # Range and IQR
            features[f"corr_range_{window}"] = correlation_series.rolling(
                window=window, min_periods=window // 2
            ).apply(lambda x: x.max() - x.min(), raw=True)

            features[f"corr_iqr_{window}"] = correlation_series.rolling(
                window=window, min_periods=window // 2
            ).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True)

            # Median absolute deviation
            features[f"corr_mad_{window}"] = correlation_series.rolling(
                window=window, min_periods=window // 2
            ).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

        return features

    def _calculate_persistence_metrics(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Calculate correlation persistence and mean reversion metrics."""
        features = pd.DataFrame(index=correlation_series.index)

        # Hurst exponent (long-term memory)
        features["corr_hurst_exponent"] = correlation_series.rolling(
            window=100, min_periods=50
        ).apply(lambda x: calculate_hurst_exponent(pd.Series(x)), raw=True)

        # Half-life of correlation shocks
        features["corr_half_life"] = correlation_series.rolling(window=50, min_periods=25).apply(
            self._calculate_half_life, raw=True
        )

        # Mean reversion speed
        features["corr_mean_reversion_speed"] = correlation_series.rolling(
            window=50, min_periods=25
        ).apply(self._calculate_mean_reversion_speed, raw=True)

        # Autocorrelations
        for lag in [1, 5, 10]:
            features[f"corr_autocorrelation_{lag}"] = correlation_series.rolling(
                window=50, min_periods=25
            ).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=True)

        return features

    def _detect_correlation_regimes(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Detect and characterize correlation regimes."""
        features = pd.DataFrame(index=correlation_series.index)

        # Simple regime detection using rolling mean and std
        rolling_mean = correlation_series.rolling(window=20, min_periods=10).mean()
        rolling_std = correlation_series.rolling(window=20, min_periods=10).std()

        # Define regimes based on mean +/- std bands
        upper_band = rolling_mean + rolling_std
        lower_band = rolling_mean - rolling_std

        # Regime classification
        regime = pd.Series(index=correlation_series.index, dtype=int)
        regime[correlation_series > upper_band] = 1  # High correlation regime
        regime[correlation_series < lower_band] = -1  # Low correlation regime
        regime.fillna(0, inplace=True)  # Normal regime

        # Regime duration
        regime_changes = regime.diff().fillna(0) != 0
        regime_groups = regime_changes.cumsum()

        features["regime_duration"] = regime_groups.groupby(regime_groups).transform("count")
        features["regime_count"] = regime_changes.rolling(window=100).sum()

        # Current regime stability
        features["current_regime_stability"] = (
            correlation_series.rolling(window=20, min_periods=10).std().rolling(window=20).mean()
        )

        # Regime transition probability
        features["regime_transition_probability"] = regime_changes.rolling(
            window=100, min_periods=50
        ).mean()

        # Regime entropy
        features["regime_entropy"] = regime.rolling(window=50, min_periods=25).apply(
            lambda x: calculate_entropy(pd.Series(x), bins=3), raw=True
        )

        return features

    def _detect_structural_breaks(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Detect structural breaks in correlation series."""
        features = pd.DataFrame(index=correlation_series.index)

        # CUSUM test for structural breaks
        cusum = self._calculate_cusum(correlation_series)

        # Detect breaks when CUSUM exceeds threshold
        threshold = self.breakpoint_threshold * correlation_series.std()
        breaks = (cusum.abs() > threshold).astype(int)

        features["structural_break_indicator"] = breaks

        # Time since last break
        break_times = correlation_series.index[breaks == 1]
        features["time_since_break"] = 0

        for i, idx in enumerate(correlation_series.index):
            if len(break_times) > 0:
                # Find most recent break before current time
                recent_breaks = break_times[break_times <= idx]
                if len(recent_breaks) > 0:
                    last_break = recent_breaks[-1]
                    # Calculate business days since break
                    features.loc[idx, "time_since_break"] = len(pd.bdate_range(last_break, idx))

        # Break magnitude
        features["break_magnitude"] = cusum.abs()

        # Break confidence (based on statistical significance)
        features["break_confidence"] = self._calculate_break_confidence(correlation_series, cusum)

        return features

    def _calculate_confidence_intervals(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Calculate confidence intervals for correlation stability."""
        features = pd.DataFrame(index=correlation_series.index)

        # Rolling statistics for CI calculation
        window = 50
        rolling_mean = correlation_series.rolling(window=window, min_periods=window // 2).mean()
        rolling_std = correlation_series.rolling(window=window, min_periods=window // 2).std()

        # Calculate confidence intervals
        for confidence in [0.90, 0.95]:
            z_score = stats.norm.ppf((1 + confidence) / 2)

            features[f"corr_ci_lower_{int(confidence*100)}"] = rolling_mean - z_score * rolling_std
            features[f"corr_ci_upper_{int(confidence*100)}"] = rolling_mean + z_score * rolling_std
            features[f"corr_ci_width_{int(confidence*100)}"] = 2 * z_score * rolling_std

        return features

    def _generate_stability_forecasts(self, correlation_series: pd.Series) -> pd.DataFrame:
        """Generate correlation stability forecasts."""
        features = pd.DataFrame(index=correlation_series.index)

        # Simple AR(1) forecast
        window = 50

        # 1-day forecast
        features["expected_corr_1d"] = correlation_series.rolling(
            window=window, min_periods=window // 2
        ).apply(lambda x: self._ar1_forecast(x, 1), raw=True)

        # 5-day forecast
        features["expected_corr_5d"] = correlation_series.rolling(
            window=window, min_periods=window // 2
        ).apply(lambda x: self._ar1_forecast(x, 5), raw=True)

        # Forecast error (using previous forecasts)
        features["corr_forecast_error"] = (
            correlation_series - features["expected_corr_1d"].shift(1)
        ).abs()

        # Forecast confidence (based on recent forecast accuracy)
        features["corr_forecast_confidence"] = 1 / (
            1 + features["corr_forecast_error"].rolling(window=20).mean()
        )

        return features

    def _calculate_multi_asset_stability(self) -> pd.DataFrame:
        """Calculate stability metrics for multi-asset correlations."""
        # Prepare returns for all assets
        returns_df = self.prepare_multi_asset_returns()

        if returns_df.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=returns_df.index)

        # Average pairwise correlation stability
        n_assets = len(returns_df.columns)
        stability_scores = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset1 = returns_df.columns[i]
                asset2 = returns_df.columns[j]

                # Calculate rolling correlation
                corr = (
                    returns_df[asset1].rolling(window=20, min_periods=10).corr(returns_df[asset2])
                )

                # Calculate stability
                stability = 1 / (1 + corr.rolling(window=20).std())
                stability_scores.append(stability)

        if stability_scores:
            features["cross_corr_stability"] = pd.concat(stability_scores, axis=1).mean(axis=1)

        # Lead-lag stability
        features["lead_lag_stability"] = self._calculate_lead_lag_stability(returns_df)

        # Correlation network stability (eigenvalue-based)
        features["correlation_network_stability"] = self._calculate_network_stability(returns_df)

        return features

    def _calculate_half_life(self, x: np.ndarray) -> float:
        """Calculate half-life of mean reversion."""
        try:
            if len(x) < 10:
                return np.nan

            # Fit AR(1) model
            y = x[1:]
            X = x[:-1]

            # Add small constant to avoid division by zero
            phi = np.corrcoef(X, y)[0, 1]

            if phi <= 0 or phi >= 1:
                return np.nan

            # Half-life = -ln(2) / ln(phi)
            half_life = -np.log(2) / np.log(phi)

            return np.clip(half_life, 0, 100)  # Cap at 100 periods

        except Exception:
            return np.nan

    def _calculate_mean_reversion_speed(self, x: np.ndarray) -> float:
        """Calculate speed of mean reversion."""
        try:
            if len(x) < 10:
                return 0.0

            # Calculate deviations from mean
            mean = np.mean(x)
            deviations = x - mean

            # Calculate how quickly deviations decay
            autocorr = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]

            # Speed = 1 - autocorrelation
            speed = 1 - abs(autocorr)

            return speed

        except Exception:
            return 0.0

    def _calculate_cusum(self, series: pd.Series) -> pd.Series:
        """Calculate CUSUM statistic for break detection."""
        mean = series.mean()
        cusum = (series - mean).cumsum()
        return cusum

    def _calculate_break_confidence(self, series: pd.Series, cusum: pd.Series) -> pd.Series:
        """Calculate confidence level for structural breaks."""
        # Normalize CUSUM by standard deviation
        normalized_cusum = cusum / (series.std() * np.sqrt(len(series)))

        # Convert to probability using normal CDF
        confidence = 2 * (1 - stats.norm.cdf(normalized_cusum.abs()))

        return confidence

    def _ar1_forecast(self, x: np.ndarray, horizon: int) -> float:
        """Simple AR(1) forecast."""
        try:
            if len(x) < 10:
                return x[-1]

            # Fit AR(1)
            phi = np.corrcoef(x[:-1], x[1:])[0, 1]
            mean = np.mean(x)

            # Multi-step forecast
            forecast = x[-1]
            for _ in range(horizon):
                forecast = mean + phi * (forecast - mean)

            return forecast

        except Exception:
            return x[-1] if len(x) > 0 else 0.0

    def _calculate_lead_lag_stability(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate stability of lead-lag relationships."""
        # Simplified version - check correlation at different lags
        stability_scores = []

        for lag in [-2, -1, 0, 1, 2]:
            lagged_corr = returns_df.corr().shift(lag)
            stability = 1 / (1 + lagged_corr.std().mean())
            stability_scores.append(stability)

        return pd.Series(np.mean(stability_scores), index=returns_df.index)

    def _calculate_network_stability(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate correlation network stability using eigenvalues."""
        window = 50
        stability = pd.Series(index=returns_df.index, dtype=float)

        for i in range(window, len(returns_df)):
            window_data = returns_df.iloc[i - window : i]

            # Calculate correlation matrix
            corr_matrix = window_data.corr()

            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(corr_matrix)

            # Stability metric: inverse of eigenvalue spread
            spread = np.std(eigenvalues)
            stability.iloc[i] = 1 / (1 + spread)

        return stability
