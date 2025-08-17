"""
VaR Calculator

Specialized calculator for Value at Risk (VaR) computations including:
- Historical VaR (non-parametric)
- Parametric VaR (normal, t-distribution)
- Monte Carlo VaR simulation
- Extreme Value Theory VaR
- Cornish-Fisher VaR (higher moments)
- Expected Shortfall (CVaR)
- VaR backtesting and validation
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

from .base_risk import BaseRiskCalculator
from .risk_config import VaRMethod

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Comprehensive VaR calculation result."""

    # Basic VaR metrics
    var_value: float  # VaR in portfolio currency
    var_percentage: float  # VaR as percentage of portfolio
    confidence_level: float  # Confidence level (e.g., 0.95)
    time_horizon: int  # Time horizon in days
    method: VaRMethod  # Calculation method used

    # Extended metrics
    expected_shortfall: float = 0.0  # CVaR/ES
    worst_case_loss: float = 0.0  # Maximum historical loss
    best_case_gain: float = 0.0  # Maximum historical gain

    # Distribution characteristics
    mean_return: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Method-specific details
    historical_observations: int | None = None
    distribution_params: dict[str, float] = field(default_factory=dict)
    num_simulations: int | None = None

    # Model validation
    exceptions: int | None = None  # VaR breaches in backtest
    exception_rate: float | None = None  # Actual vs expected breach rate
    kupiec_test_pvalue: float | None = None  # Kupiec test p-value

    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)

    def is_breach(self, actual_loss: float) -> bool:
        """Check if actual loss exceeds VaR threshold."""
        return actual_loss > self.var_value

    def breach_severity(self, actual_loss: float) -> float:
        """Calculate breach severity as multiple of VaR."""
        if self.var_value == 0:
            return 0.0
        return actual_loss / self.var_value


class VaRCalculator(BaseRiskCalculator):
    """Calculator for Value at Risk (VaR) and Expected Shortfall metrics."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize VaR calculator."""
        super().__init__(config)

        # VaR-specific configuration
        self.var_config = self.risk_config.get_var_config()
        self.confidence_levels = self.var_config["confidence_levels"]
        self.time_horizons = self.var_config["time_horizons"]
        self.lookback_window = self.var_config["lookback_window"]
        self.default_method = self.var_config["default_method"]

        logger.debug(
            f"Initialized VaRCalculator with {len(self.confidence_levels)} confidence levels"
        )

    def get_required_columns(self) -> list[str]:
        """
        Get list of required input columns for VaR calculations.

        Returns:
            List of required column names
        """
        # VaR calculations typically need price or returns data
        return ["close"]

    def get_feature_names(self) -> list[str]:
        """Return list of VaR feature names."""
        feature_names = []

        # Basic VaR metrics for different confidence levels and horizons
        for confidence in self.confidence_levels:
            conf_str = f"{int(confidence*100)}"
            for horizon in self.time_horizons:
                horizon_str = f"{horizon}d"

                # VaR values
                feature_names.extend(
                    [
                        f"var_{conf_str}_{horizon_str}",
                        f"var_pct_{conf_str}_{horizon_str}",
                        f"expected_shortfall_{conf_str}_{horizon_str}",
                    ]
                )

        # Method-specific VaR metrics
        feature_names.extend(
            [
                "var_historical_95_1d",
                "var_parametric_95_1d",
                "var_monte_carlo_95_1d",
                "var_extreme_value_95_1d",
            ]
        )

        # VaR statistics and validation
        feature_names.extend(
            [
                "var_mean_return",
                "var_volatility",
                "var_skewness",
                "var_kurtosis",
                "var_worst_case_loss",
                "var_best_case_gain",
                "var_exception_rate",
                "var_kupiec_test_pvalue",
                "var_model_accuracy",
            ]
        )

        # VaR ratios and comparisons
        feature_names.extend(
            [
                "var_parametric_vs_historical",
                "var_monte_carlo_vs_historical",
                "var_extreme_value_vs_historical",
                "expected_shortfall_ratio",
                "var_efficiency_ratio",
            ]
        )

        # Rolling VaR metrics
        feature_names.extend(
            [
                "var_rolling_mean",
                "var_rolling_std",
                "var_rolling_trend",
                "var_rolling_volatility",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VaR features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with VaR features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)

            # Calculate returns
            returns = self.calculate_returns(data)

            if len(returns) < self.var_config["min_observations"]:
                logger.warning("Insufficient data for VaR calculation")
                return self.create_empty_features(data.index)

            # Create features DataFrame
            features = self.create_empty_features(data.index)

            # Calculate VaR for all methods and parameters
            features = self._calculate_basic_var_metrics(returns, features)
            features = self._calculate_method_specific_var(returns, features)
            features = self._calculate_var_statistics(returns, features)
            features = self._calculate_var_validation(returns, features)
            features = self._calculate_var_ratios(returns, features)
            features = self._calculate_rolling_var(returns, features)

            return features

        except Exception as e:
            logger.error(f"Error calculating VaR features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_basic_var_metrics(
        self, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate basic VaR metrics for all confidence levels and horizons."""
        try:
            portfolio_value = 1000000  # Assume $1M portfolio for calculation

            for confidence in self.confidence_levels:
                conf_str = f"{int(confidence*100)}"

                for horizon in self.time_horizons:
                    horizon_str = f"{horizon}d"

                    # Calculate VaR using default method
                    var_result = self._calculate_var_single(
                        returns, confidence, horizon, self.default_method
                    )

                    # Store basic metrics
                    features[f"var_{conf_str}_{horizon_str}"] = var_result.var_value
                    features[f"var_pct_{conf_str}_{horizon_str}"] = var_result.var_percentage
                    features[f"expected_shortfall_{conf_str}_{horizon_str}"] = (
                        var_result.expected_shortfall
                    )

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic VaR metrics: {e}")
            return features

    def _calculate_method_specific_var(
        self, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate VaR using different methods for comparison."""
        try:
            confidence = 0.95
            horizon = 1

            # Calculate VaR using each method
            methods = [
                VaRMethod.HISTORICAL,
                VaRMethod.PARAMETRIC,
                VaRMethod.MONTE_CARLO,
                VaRMethod.EXTREME_VALUE,
            ]

            var_results = {}
            for method in methods:
                try:
                    var_result = self._calculate_var_single(returns, confidence, horizon, method)
                    var_results[method] = var_result

                    # Store method-specific results
                    method_name = method.value
                    features[f"var_{method_name}_95_1d"] = var_result.var_value

                except Exception as e:
                    logger.warning(f"Error calculating VaR for method {method}: {e}")
                    continue

            return features

        except Exception as e:
            logger.warning(f"Error calculating method-specific VaR: {e}")
            return features

    def _calculate_var_single(
        self, returns: pd.Series, confidence: float, horizon: int, method: VaRMethod
    ) -> VaRResult:
        """Calculate VaR using a single method."""
        try:
            # Adjust returns for time horizon
            if horizon > 1:
                scaled_returns = returns * np.sqrt(horizon)
            else:
                scaled_returns = returns

            # Calculate basic statistics
            mean_return = returns.mean()
            volatility = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            # Calculate VaR based on method
            if method == VaRMethod.HISTORICAL:
                var_value = self._historical_var(scaled_returns, confidence)
                expected_shortfall = self._historical_expected_shortfall(scaled_returns, confidence)

            elif method == VaRMethod.PARAMETRIC:
                var_value = self._parametric_var(scaled_returns, confidence)
                expected_shortfall = self._parametric_expected_shortfall(scaled_returns, confidence)

            elif method == VaRMethod.MONTE_CARLO:
                var_value = self._monte_carlo_var(scaled_returns, confidence)
                expected_shortfall = self._monte_carlo_expected_shortfall(
                    scaled_returns, confidence
                )

            elif method == VaRMethod.EXTREME_VALUE:
                var_value = self._extreme_value_var(scaled_returns, confidence)
                expected_shortfall = self._extreme_value_expected_shortfall(
                    scaled_returns, confidence
                )

            else:
                raise ValueError(f"Unknown VaR method: {method}")

            # Create result object
            result = VaRResult(
                var_value=abs(var_value),
                var_percentage=abs(var_value),  # Assuming unit portfolio
                confidence_level=confidence,
                time_horizon=horizon,
                method=method,
                expected_shortfall=abs(expected_shortfall),
                worst_case_loss=abs(returns.min()),
                best_case_gain=returns.max(),
                mean_return=mean_return,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                historical_observations=len(returns),
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating single VaR: {e}")
            # Return default result
            return VaRResult(
                var_value=0.0,
                var_percentage=0.0,
                confidence_level=confidence,
                time_horizon=horizon,
                method=method,
            )

    def _historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical VaR (non-parametric)."""
        try:
            if len(returns) == 0:
                return 0.0

            # Calculate empirical quantile
            alpha = 1 - confidence
            var_value = returns.quantile(alpha)

            return var_value

        except Exception as e:
            logger.warning(f"Error calculating historical VaR: {e}")
            return 0.0

    def _parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        try:
            if len(returns) == 0:
                return 0.0

            # Fit normal distribution
            mu = returns.mean()
            sigma = returns.std()

            # Calculate VaR using inverse normal
            alpha = 1 - confidence
            var_value = stats.norm.ppf(alpha, mu, sigma)

            return var_value

        except Exception as e:
            logger.warning(f"Error calculating parametric VaR: {e}")
            return 0.0

    def _monte_carlo_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Monte Carlo VaR."""
        try:
            if len(returns) == 0:
                return 0.0

            # Fit distribution to returns
            mu = returns.mean()
            sigma = returns.std()

            # Generate random scenarios
            np.random.seed(self.var_config["mc_random_seed"])
            scenarios = secure_numpy_normal(mu, sigma, self.var_config["mc_simulations"])

            # Calculate VaR from simulated scenarios
            alpha = 1 - confidence
            var_value = np.percentile(scenarios, alpha * 100)

            return var_value

        except Exception as e:
            logger.warning(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0

    def _extreme_value_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate VaR using Extreme Value Theory."""
        try:
            if len(returns) < 100:  # Need sufficient data for EVT
                return self._historical_var(returns, confidence)

            # Use threshold approach (POT method)
            threshold_percentile = self.var_config["evt_threshold_percentile"]
            threshold = returns.quantile(threshold_percentile)

            # Get exceedances
            exceedances = returns[returns > threshold] - threshold

            if len(exceedances) < 10:  # Need enough exceedances
                return self._historical_var(returns, confidence)

            # Fit Generalized Pareto Distribution to exceedances
            # Simplified approach - use exponential distribution
            scale = exceedances.mean()

            # Calculate VaR
            n = len(returns)
            nu = len(exceedances)

            # EVT VaR formula
            alpha = 1 - confidence
            if alpha < nu / n:
                var_value = threshold + scale * np.log((n * alpha) / nu)
            else:
                var_value = self._historical_var(returns, confidence)

            return var_value

        except Exception as e:
            logger.warning(f"Error calculating extreme value VaR: {e}")
            return self._historical_var(returns, confidence)

    def _historical_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical Expected Shortfall (CVaR)."""
        try:
            if len(returns) == 0:
                return 0.0

            alpha = 1 - confidence
            var_value = returns.quantile(alpha)

            # Expected shortfall is mean of returns below VaR
            tail_returns = returns[returns <= var_value]

            if len(tail_returns) == 0:
                return var_value

            expected_shortfall = tail_returns.mean()

            return expected_shortfall

        except Exception as e:
            logger.warning(f"Error calculating historical expected shortfall: {e}")
            return 0.0

    def _parametric_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric Expected Shortfall."""
        try:
            if len(returns) == 0:
                return 0.0

            mu = returns.mean()
            sigma = returns.std()

            alpha = 1 - confidence
            var_value = stats.norm.ppf(alpha, mu, sigma)

            # Expected shortfall for normal distribution
            expected_shortfall = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

            return expected_shortfall

        except Exception as e:
            logger.warning(f"Error calculating parametric expected shortfall: {e}")
            return 0.0

    def _monte_carlo_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Monte Carlo Expected Shortfall."""
        try:
            if len(returns) == 0:
                return 0.0

            mu = returns.mean()
            sigma = returns.std()

            # Generate scenarios
            np.random.seed(self.var_config["mc_random_seed"])
            scenarios = secure_numpy_normal(mu, sigma, self.var_config["mc_simulations"])

            # Calculate Expected Shortfall
            alpha = 1 - confidence
            var_value = np.percentile(scenarios, alpha * 100)

            tail_scenarios = scenarios[scenarios <= var_value]
            if len(tail_scenarios) == 0:
                return var_value

            expected_shortfall = tail_scenarios.mean()

            return expected_shortfall

        except Exception as e:
            logger.warning(f"Error calculating Monte Carlo expected shortfall: {e}")
            return 0.0

    def _extreme_value_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Expected Shortfall using Extreme Value Theory."""
        try:
            # For simplicity, use historical method for EVT ES
            return self._historical_expected_shortfall(returns, confidence)

        except Exception as e:
            logger.warning(f"Error calculating extreme value expected shortfall: {e}")
            return 0.0

    def _calculate_var_statistics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate VaR-related statistics."""
        try:
            # Basic return statistics
            features["var_mean_return"] = returns.mean()
            features["var_volatility"] = returns.std()
            features["var_skewness"] = returns.skew()
            features["var_kurtosis"] = returns.kurtosis()
            features["var_worst_case_loss"] = abs(returns.min())
            features["var_best_case_gain"] = returns.max()

            return features

        except Exception as e:
            logger.warning(f"Error calculating VaR statistics: {e}")
            return features

    def _calculate_var_validation(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate VaR model validation metrics."""
        try:
            # Simple backtesting for 95% VaR
            confidence = 0.95
            var_values = []

            # Calculate rolling VaR
            window = min(60, len(returns) // 2)  # Use 60-day window or half the data

            for i in range(window, len(returns)):
                window_returns = returns.iloc[i - window : i]
                var_value = self._historical_var(window_returns, confidence)
                var_values.append(var_value)

            if len(var_values) > 0:
                var_series = pd.Series(var_values, index=returns.index[window:])
                actual_returns = returns.iloc[window:]

                # Count exceptions (breaches)
                exceptions = (actual_returns < var_series).sum()
                exception_rate = exceptions / len(actual_returns)

                # Expected exception rate
                expected_rate = 1 - confidence

                # Kupiec test (simplified)
                if exceptions > 0:
                    kupiec_lr = -2 * np.log(
                        (expected_rate**exceptions)
                        * ((1 - expected_rate) ** (len(actual_returns) - exceptions))
                    )
                    kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_lr, df=1)
                else:
                    kupiec_pvalue = 1.0

                features["var_exception_rate"] = exception_rate
                features["var_kupiec_test_pvalue"] = kupiec_pvalue
                features["var_model_accuracy"] = 1 - abs(exception_rate - expected_rate)

            return features

        except Exception as e:
            logger.warning(f"Error calculating VaR validation: {e}")
            return features

    def _calculate_var_ratios(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate VaR method comparison ratios."""
        try:
            confidence = 0.95
            horizon = 1

            # Calculate VaR using different methods
            hist_var = self._historical_var(returns, confidence)
            param_var = self._parametric_var(returns, confidence)
            mc_var = self._monte_carlo_var(returns, confidence)
            evt_var = self._extreme_value_var(returns, confidence)

            # Calculate ratios
            if hist_var != 0:
                features["var_parametric_vs_historical"] = param_var / hist_var
                features["var_monte_carlo_vs_historical"] = mc_var / hist_var
                features["var_extreme_value_vs_historical"] = evt_var / hist_var

            # Expected shortfall ratio
            hist_es = self._historical_expected_shortfall(returns, confidence)
            if hist_var != 0:
                features["expected_shortfall_ratio"] = hist_es / hist_var

            # VaR efficiency ratio (VaR / volatility)
            volatility = returns.std()
            if volatility != 0:
                features["var_efficiency_ratio"] = abs(hist_var) / volatility

            return features

        except Exception as e:
            logger.warning(f"Error calculating VaR ratios: {e}")
            return features

    def _calculate_rolling_var(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling VaR metrics."""
        try:
            window = min(60, len(returns) // 2)

            if len(returns) < window:
                return features

            # Calculate rolling VaR
            rolling_var = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i - window : i]
                var_value = self._historical_var(window_returns, 0.95)
                rolling_var.append(abs(var_value))

            if len(rolling_var) > 0:
                rolling_var_series = pd.Series(rolling_var)

                features["var_rolling_mean"] = rolling_var_series.mean()
                features["var_rolling_std"] = rolling_var_series.std()

                # Calculate trend
                if len(rolling_var_series) > 1:
                    x = np.arange(len(rolling_var_series))
                    slope = np.polyfit(x, rolling_var_series, 1)[0]
                    features["var_rolling_trend"] = slope

                # Rolling volatility of VaR
                features["var_rolling_volatility"] = rolling_var_series.std()

            return features

        except Exception as e:
            logger.warning(f"Error calculating rolling VaR: {e}")
            return features
