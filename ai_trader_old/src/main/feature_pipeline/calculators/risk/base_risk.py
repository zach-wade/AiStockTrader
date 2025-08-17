"""
Base Risk Calculator

Base class for all risk metric calculators providing:
- Common risk calculation utilities
- Data validation and preprocessing
- Statistical helper methods
- Return handling and normalization
- Risk metric result structures
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from main.utils.core import get_logger

from .risk_config import RiskConfig
from ..base_calculator import BaseFeatureCalculator
from ..helpers import create_feature_dataframe

logger = get_logger(__name__)


class BaseRiskCalculator(BaseFeatureCalculator, ABC):
    """Base class for all risk metric calculators."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize base risk calculator."""
        super().__init__(config)

        # Initialize risk configuration
        risk_config = config.get("risk", {}) if config else {}
        self.risk_config = RiskConfig(**risk_config)

        # Cache enabled from config
        self._cache_enabled = self.risk_config.enable_caching

        logger.debug(f"Initialized {self.__class__.__name__} with risk config")

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for risk calculations.

        Args:
            data: DataFrame with price/return data

        Returns:
            bool: True if data is valid
        """
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided")
                return False

            # Check for required columns
            if "close" not in data.columns:
                logger.warning("Missing 'close' column in data")
                return False

            # Check for sufficient data
            if len(data) < self.risk_config.var_min_observations:
                logger.warning(
                    f"Insufficient data: {len(data)} < {self.risk_config.var_min_observations}"
                )
                return False

            # Check for valid price data
            if data["close"].iloc[-1] <= 0:
                logger.warning("Invalid price data (non-positive values)")
                return False

            # Check for missing values
            if data["close"].isna().any():
                logger.warning("Missing values in price data")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False

    def calculate_returns(self, data: pd.DataFrame, return_type: str = "simple") -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            data: DataFrame with price data
            return_type: Type of returns ('simple', 'log')

        Returns:
            pd.Series: Returns series
        """
        try:
            prices = data["close"]

            if return_type == "simple":
                returns = prices.pct_change().dropna()
            elif return_type == "log":
                returns = np.log(prices / prices.shift(1)).dropna()
            else:
                raise ValueError(f"Unknown return type: {return_type}")

            # Remove outliers if configured
            if self.risk_config.remove_outliers:
                returns = self._remove_outliers(returns, self.risk_config.outlier_threshold)

            # Apply return limits if configured
            returns = returns.clip(
                lower=self.risk_config.min_return_threshold,
                upper=self.risk_config.max_return_threshold,
            )

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(dtype=float)

    def _remove_outliers(self, returns: pd.Series, threshold: float) -> pd.Series:
        """Remove outliers from returns series."""
        try:
            mean = returns.mean()
            std = returns.std()

            # Remove observations beyond threshold standard deviations
            mask = np.abs(returns - mean) <= threshold * std
            filtered_returns = returns[mask]

            removed_count = len(returns) - len(filtered_returns)
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} outliers from {len(returns)} observations")

            return filtered_returns

        except Exception as e:
            logger.warning(f"Error removing outliers: {e}")
            return returns

    def calculate_basic_statistics(self, returns: pd.Series) -> dict[str, float]:
        """
        Calculate basic statistical properties of returns.

        Args:
            returns: Returns series

        Returns:
            Dict: Basic statistics
        """
        try:
            if len(returns) == 0:
                return {}

            stats_dict = {
                "mean": returns.mean(),
                "std": returns.std(),
                "variance": returns.var(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
                "min": returns.min(),
                "max": returns.max(),
                "count": len(returns),
            }

            # Add percentiles
            percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            for p in percentiles:
                stats_dict[f"percentile_{int(p*100)}"] = returns.quantile(p)

            return stats_dict

        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            return {}

    def calculate_rolling_statistics(self, returns: pd.Series, window: int) -> pd.DataFrame:
        """
        Calculate rolling statistics.

        Args:
            returns: Returns series
            window: Rolling window size

        Returns:
            pd.DataFrame: Rolling statistics
        """
        try:
            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for rolling statistics: {len(returns)} < {window}"
                )
                return pd.DataFrame()

            rolling_stats = pd.DataFrame(index=returns.index)

            # Basic rolling statistics
            rolling_stats["mean"] = returns.rolling(window=window).mean()
            rolling_stats["std"] = returns.rolling(window=window).std()
            rolling_stats["variance"] = returns.rolling(window=window).var()
            rolling_stats["skewness"] = returns.rolling(window=window).skew()
            rolling_stats["kurtosis"] = returns.rolling(window=window).kurt()

            # Rolling percentiles
            rolling_stats["percentile_5"] = returns.rolling(window=window).quantile(0.05)
            rolling_stats["percentile_95"] = returns.rolling(window=window).quantile(0.95)

            return rolling_stats

        except Exception as e:
            logger.error(f"Error calculating rolling statistics: {e}")
            return pd.DataFrame()

    def calculate_exponential_statistics(
        self, returns: pd.Series, decay: float = 0.94
    ) -> pd.Series:
        """
        Calculate exponentially weighted statistics.

        Args:
            returns: Returns series
            decay: Decay factor (lambda)

        Returns:
            pd.Series: Exponentially weighted statistics
        """
        try:
            if len(returns) == 0:
                return pd.Series(dtype=float)

            # EWMA variance (RiskMetrics approach)
            ewma_var = returns.ewm(alpha=1 - decay, adjust=False).var()

            return ewma_var

        except Exception as e:
            logger.error(f"Error calculating exponential statistics: {e}")
            return pd.Series(dtype=float)

    def fit_distribution(self, returns: pd.Series, distribution: str = "normal") -> dict[str, Any]:
        """
        Fit statistical distribution to returns.

        Args:
            returns: Returns series
            distribution: Distribution type ('normal', 't', 'skewed_t')

        Returns:
            Dict: Distribution parameters and fit statistics
        """
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for distribution fitting")
                return {}

            if distribution == "normal":
                # Normal distribution
                mu, sigma = stats.norm.fit(returns)

                # Goodness of fit test
                ks_stat, ks_pvalue = stats.kstest(returns, "norm", args=(mu, sigma))

                return {
                    "distribution": "normal",
                    "parameters": {"mu": mu, "sigma": sigma},
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "aic": len(returns) * np.log(2 * np.pi * sigma**2) + len(returns),
                    "bic": len(returns) * np.log(2 * np.pi * sigma**2)
                    + len(returns)
                    + 2 * np.log(len(returns)),
                }

            elif distribution == "t":
                # Student's t-distribution
                df, loc, scale = stats.t.fit(returns)

                ks_stat, ks_pvalue = stats.kstest(returns, "t", args=(df, loc, scale))

                return {
                    "distribution": "t",
                    "parameters": {"df": df, "loc": loc, "scale": scale},
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "aic": -2 * np.sum(stats.t.logpdf(returns, df, loc, scale)) + 2 * 3,
                    "bic": -2 * np.sum(stats.t.logpdf(returns, df, loc, scale))
                    + 3 * np.log(len(returns)),
                }

            else:
                logger.warning(f"Unknown distribution: {distribution}")
                return {}

        except Exception as e:
            logger.error(f"Error fitting distribution: {e}")
            return {}

    def calculate_correlation_matrix(
        self, returns_data: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple return series.

        Args:
            returns_data: DataFrame with multiple return series
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            if returns_data.empty:
                return pd.DataFrame()

            correlation_matrix = returns_data.corr(method=method)

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    # Removed duplicate safe_divide - using centralized helper instead

    def create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """
        Create empty features DataFrame with NaN values.

        Args:
            index: Index for the DataFrame

        Returns:
            pd.DataFrame: Empty features DataFrame
        """
        feature_names = self.get_feature_names()
        return create_feature_dataframe(index, feature_names, fill_value=np.nan)

    def annualize_metric(
        self, metric: float | pd.Series, periods_per_year: float = 252.0
    ) -> float | pd.Series:
        """
        Annualize a metric (return, volatility, etc.).

        Args:
            metric: Metric to annualize
            periods_per_year: Number of periods per year

        Returns:
            Annualized metric
        """
        try:
            return metric * np.sqrt(periods_per_year)
        except Exception as e:
            logger.warning(f"Error annualizing metric: {e}")
            return metric

    def check_stationarity(self, returns: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
        """
        Check stationarity of returns series using Augmented Dickey-Fuller test.

        Args:
            returns: Returns series
            alpha: Significance level

        Returns:
            Dict: Stationarity test results
        """
        try:
            # Third-party imports
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(returns.dropna())

            return {
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < alpha,
                "test_type": "Augmented Dickey-Fuller",
            }

        except ImportError:
            logger.warning("statsmodels not available for stationarity testing")
            return {}
        except Exception as e:
            logger.error(f"Error checking stationarity: {e}")
            return {}

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names calculated by this calculator."""
        pass

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk metrics features."""
        pass
