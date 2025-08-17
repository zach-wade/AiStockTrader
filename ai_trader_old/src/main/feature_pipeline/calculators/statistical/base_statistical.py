"""
Base Statistical Calculator

Base class for statistical feature calculators, providing shared utilities
and common statistical computations.
"""

# Standard library imports
from abc import abstractmethod
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .statistical_config import StatisticalConfig
from ..base_calculator import BaseFeatureCalculator
from ..helpers import (
    calculate_entropy,
    calculate_hurst_exponent,
    calculate_moving_average,
    calculate_rolling_quantile,
    calculate_rolling_std,
    fit_distribution,
    remove_outliers,
    safe_divide,
)

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BaseStatisticalCalculator(BaseFeatureCalculator):
    """
    Base class for statistical feature calculators.

    Provides:
    - Common statistical calculations
    - Distribution analysis utilities
    - Time series statistics
    - Moment calculations
    - Entropy and complexity measures
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize base statistical calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Initialize statistical configuration
        stat_config = config.get("statistical", {}) if config else {}
        self.stat_config = StatisticalConfig(**stat_config)

        # Common parameters
        self.lookback_periods = self.stat_config.lookback_periods
        self.distribution_types = self.stat_config.distribution_types
        self.entropy_bins = self.stat_config.entropy_bins

        logger.debug(f"Initialized {self.name} with statistical config")

    def get_required_columns(self) -> list[str]:
        """Get required columns for statistical calculations."""
        return ["symbol", "timestamp", "close"]

    def calculate_returns(self, prices: pd.Series, return_type: str = "simple") -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            return_type: Type of returns ('simple', 'log')

        Returns:
            Returns series
        """
        if return_type == "simple":
            returns = prices.pct_change()
        elif return_type == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return type: {return_type}")

        # Handle outliers if configured
        if self.stat_config.remove_outliers:
            returns = remove_outliers(
                returns,
                method=self.stat_config.outlier_method,
                threshold=self.stat_config.outlier_threshold,
            )

        return returns

    def calculate_basic_statistics(self, series: pd.Series, prefix: str = "") -> dict[str, float]:
        """
        Calculate basic statistical measures.

        Args:
            series: Input series
            prefix: Prefix for feature names

        Returns:
            Dictionary of statistics
        """
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {}

        stats = {
            f"{prefix}mean": clean_series.mean(),
            f"{prefix}std": clean_series.std(),
            f"{prefix}variance": clean_series.var(),
            f"{prefix}skewness": clean_series.skew(),
            f"{prefix}kurtosis": clean_series.kurtosis(),
            f"{prefix}min": clean_series.min(),
            f"{prefix}max": clean_series.max(),
            f"{prefix}range": clean_series.max() - clean_series.min(),
            f"{prefix}median": clean_series.median(),
            f"{prefix}mad": (clean_series - clean_series.median()).abs().median(),
        }

        # Add percentiles
        for p in [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]:
            stats[f"{prefix}percentile_{int(p*100)}"] = clean_series.quantile(p)

        # Add coefficient of variation
        if stats[f"{prefix}mean"] != 0:
            stats[f"{prefix}cv"] = stats[f"{prefix}std"] / abs(stats[f"{prefix}mean"])
        else:
            stats[f"{prefix}cv"] = 0

        return stats

    def calculate_rolling_statistics(
        self,
        series: pd.Series,
        windows: list[int] | None = None,
        operations: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for multiple windows.

        Args:
            series: Input series
            windows: List of window sizes
            operations: List of operations to apply

        Returns:
            DataFrame with rolling statistics
        """
        if windows is None:
            windows = self.lookback_periods

        if operations is None:
            operations = ["mean", "std", "min", "max", "median"]

        features = pd.DataFrame(index=series.index)

        for window in windows:
            if "mean" in operations:
                features[f"rolling_mean_{window}"] = calculate_moving_average(series, window)

            if "std" in operations:
                features[f"rolling_std_{window}"] = calculate_rolling_std(series, window)

            if "min" in operations:
                features[f"rolling_min_{window}"] = series.rolling(
                    window=window, min_periods=1
                ).min()

            if "max" in operations:
                features[f"rolling_max_{window}"] = series.rolling(
                    window=window, min_periods=1
                ).max()

            if "median" in operations:
                features[f"rolling_median_{window}"] = series.rolling(
                    window=window, min_periods=1
                ).median()

            if "skew" in operations:
                features[f"rolling_skew_{window}"] = series.rolling(
                    window=window, min_periods=max(3, window // 2)
                ).skew()

            if "kurt" in operations:
                features[f"rolling_kurt_{window}"] = series.rolling(
                    window=window, min_periods=max(4, window // 2)
                ).kurt()

        return features

    def calculate_distribution_features(
        self, series: pd.Series, window: int | None = None
    ) -> pd.DataFrame:
        """
        Calculate distribution-based features.

        Args:
            series: Input series
            window: Rolling window size (None for full series)

        Returns:
            DataFrame with distribution features
        """
        features = pd.DataFrame(index=series.index)

        if window is None:
            # Full series distribution
            for dist_type in self.distribution_types:
                dist_params = fit_distribution(series, dist_type)

                if "error" not in dist_params:
                    # Add distribution parameters as features
                    for param_name, param_value in dist_params.get("params", {}).items():
                        features[f"{dist_type}_{param_name}"] = param_value

                    # Add goodness of fit
                    features[f"{dist_type}_ks_stat"] = dist_params.get("ks_statistic", np.nan)
                    features[f"{dist_type}_p_value"] = dist_params.get("p_value", np.nan)
        else:
            # Rolling distribution parameters
            def fit_rolling_dist(x, dist_type):
                if len(x) < 10:
                    return np.nan

                dist_params = fit_distribution(pd.Series(x), dist_type)
                if "error" not in dist_params:
                    # Return first parameter (e.g., mean for normal)
                    params = dist_params.get("params", {})
                    return list(params.values())[0] if params else np.nan
                return np.nan

            for dist_type in self.distribution_types:
                features[f"rolling_{dist_type}_param_{window}"] = series.rolling(
                    window=window, min_periods=10
                ).apply(lambda x: fit_rolling_dist(x, dist_type), raw=True)

        return features

    def calculate_entropy_features(
        self, series: pd.Series, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Calculate entropy-based features.

        Args:
            series: Input series
            windows: List of window sizes for rolling entropy

        Returns:
            DataFrame with entropy features
        """
        features = pd.DataFrame(index=series.index)

        if windows is None:
            windows = [20, 50, 100]

        # Shannon entropy
        for window in windows:
            features[f"shannon_entropy_{window}"] = series.rolling(
                window=window, min_periods=window // 2
            ).apply(lambda x: calculate_entropy(pd.Series(x), bins=self.entropy_bins), raw=True)

        # Normalized entropy
        for window in windows:
            features[f"normalized_entropy_{window}"] = series.rolling(
                window=window, min_periods=window // 2
            ).apply(
                lambda x: calculate_entropy(
                    pd.Series(x), bins=self.entropy_bins, method="normalized"
                ),
                raw=True,
            )

        return features

    def calculate_autocorrelation_features(
        self, series: pd.Series, lags: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Calculate autocorrelation features.

        Args:
            series: Input series
            lags: List of lag values

        Returns:
            DataFrame with autocorrelation features
        """
        features = pd.DataFrame(index=series.index)

        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        for lag in lags:
            # Simple autocorrelation
            features[f"autocorr_lag_{lag}"] = series.rolling(
                window=max(50, lag * 2), min_periods=lag + 10
            ).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=True)

            # Absolute autocorrelation (for volatility clustering)
            abs_series = series.abs()
            features[f"abs_autocorr_lag_{lag}"] = abs_series.rolling(
                window=max(50, lag * 2), min_periods=lag + 10
            ).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=True)

        return features

    def calculate_moment_features(
        self, series: pd.Series, windows: list[int] | None = None, max_moment: int = 4
    ) -> pd.DataFrame:
        """
        Calculate statistical moments.

        Args:
            series: Input series
            windows: List of window sizes
            max_moment: Maximum moment order

        Returns:
            DataFrame with moment features
        """
        features = pd.DataFrame(index=series.index)

        if windows is None:
            windows = self.lookback_periods

        for window in windows:
            for moment in range(1, max_moment + 1):
                if moment == 1:
                    # First moment (mean)
                    features[f"moment_1_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).mean()
                elif moment == 2:
                    # Second moment (variance)
                    features[f"moment_2_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).var()
                else:
                    # Higher moments
                    features[f"moment_{moment}_{window}"] = series.rolling(
                        window=window, min_periods=max(moment, window // 2)
                    ).apply(lambda x: ((x - x.mean()) ** moment).mean(), raw=True)

            # Standardized moments
            if window >= 4:
                # Skewness (standardized 3rd moment)
                features[f"std_moment_3_{window}"] = series.rolling(
                    window=window, min_periods=max(3, window // 2)
                ).skew()

                # Kurtosis (standardized 4th moment)
                features[f"std_moment_4_{window}"] = series.rolling(
                    window=window, min_periods=max(4, window // 2)
                ).kurt()

        return features

    def calculate_tail_features(
        self,
        series: pd.Series,
        windows: list[int] | None = None,
        tail_percentiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate tail distribution features.

        Args:
            series: Input series
            windows: List of window sizes
            tail_percentiles: Percentiles defining tails

        Returns:
            DataFrame with tail features
        """
        features = pd.DataFrame(index=series.index)

        if windows is None:
            windows = [20, 50, 100]

        if tail_percentiles is None:
            tail_percentiles = [0.05, 0.10]

        for window in windows:
            for percentile in tail_percentiles:
                # Lower tail
                lower_q = calculate_rolling_quantile(series, window, percentile)
                features[f"lower_tail_{int(percentile*100)}pct_{window}"] = lower_q

                # Upper tail
                upper_q = calculate_rolling_quantile(series, window, 1 - percentile)
                features[f"upper_tail_{int((1-percentile)*100)}pct_{window}"] = upper_q

                # Tail ratio
                features[f"tail_ratio_{int(percentile*100)}pct_{window}"] = safe_divide(
                    upper_q.abs(), lower_q.abs(), default_value=1.0
                )

        return features

    def calculate_complexity_features(
        self, series: pd.Series, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Calculate complexity and predictability features.

        Args:
            series: Input series
            windows: List of window sizes

        Returns:
            DataFrame with complexity features
        """
        features = pd.DataFrame(index=series.index)

        if windows is None:
            windows = [50, 100, 200]

        # Hurst exponent (long-term memory)
        for window in windows:
            if window >= 50:  # Minimum required for reliable Hurst
                features[f"hurst_exponent_{window}"] = series.rolling(
                    window=window, min_periods=50
                ).apply(lambda x: calculate_hurst_exponent(pd.Series(x)), raw=True)

        # Approximate entropy (regularity measure)
        for window in windows:
            features[f"approx_entropy_{window}"] = series.rolling(
                window=window, min_periods=window // 2
            ).apply(lambda x: self._calculate_approx_entropy(x), raw=True)

        return features

    def _calculate_approx_entropy(
        self, series: np.ndarray, m: int = 2, r_multiplier: float = 0.2
    ) -> float:
        """
        Calculate approximate entropy.

        Args:
            series: Input series
            m: Pattern length
            r_multiplier: Tolerance as fraction of std

        Returns:
            Approximate entropy value
        """
        try:
            N = len(series)
            if m + 1 > N:
                return 0.0

            # Set tolerance
            r = r_multiplier * np.std(series)

            def _maxdist(xi, xj, m):
                """Maximum distance between patterns."""
                return max([abs(ua - va) for ua, va in zip(xi, xj)])

            def _phi(m):
                """Calculate phi(m)."""
                patterns = [[series[j] for j in range(i, i + m)] for i in range(N - m + 1)]
                C = [0] * (N - m + 1)

                for i in range(N - m + 1):
                    template = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template, patterns[j], m) <= r:
                            C[i] += 1

                phi = sum([np.log(c / (N - m + 1)) for c in C]) / (N - m + 1)
                return phi

            return _phi(m) - _phi(m + 1)

        except Exception:
            return 0.0

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features - to be implemented by subclasses."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get feature names - to be implemented by subclasses."""
        pass
