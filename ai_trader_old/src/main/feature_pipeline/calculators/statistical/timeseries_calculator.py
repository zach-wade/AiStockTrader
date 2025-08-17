"""
Time Series Analysis Calculator

Specialized calculator for time series statistical tests and analysis including:
- Stationarity tests (ADF, KPSS, Phillips-Perron)
- Change point detection (CUSUM, Bayesian methods)
- Serial correlation tests (Ljung-Box, Portmanteau)
- Long memory detection (R/S analysis)
- Regime detection and structural breaks
- Unit root tests with multiple specifications
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from .base_statistical import BaseStatisticalCalculator

logger = logging.getLogger(__name__)


class TimeseriesCalculator(BaseStatisticalCalculator):
    """Calculator for time series analysis and tests."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize time series calculator."""
        super().__init__(config)

        # Stationarity test parameters
        self.stationarity_windows = self.config.get("stationarity_windows", [60, 120])
        self.adf_maxlag = self.config.get("adf_maxlag", None)
        self.adf_regression = self.stat_config.adf_regression
        self.kpss_regression = self.config.get("kpss_regression", "ct")  # 'c' or 'ct'

        # Serial correlation test parameters
        self.ljungbox_lags = self.stat_config.timeseries_test_lags
        self.ljungbox_window = self.config.get("ljungbox_window", 60)
        self.portmanteau_window = self.config.get("portmanteau_window", 60)

        # Long memory test parameters
        self.rs_window = self.config.get("rs_window", 120)
        self.variance_ratio_lags = self.config.get("variance_ratio_lags", [2, 4, 8, 16])

        # Change point detection parameters
        self.cusum_window = self.config.get("cusum_window", 100)
        self.changepoint_min_segment = self.stat_config.changepoint_min_size
        self.changepoint_penalty = self.stat_config.changepoint_penalty

        # Regime detection parameters
        self.regime_states = self.config.get("regime_states", 2)
        self.threshold_quantiles = self.config.get("threshold_quantiles", [0.25, 0.75])

    def get_feature_names(self) -> list[str]:
        """Return list of time series feature names."""
        feature_names = []

        # Stationarity tests for different windows
        for window in self.stationarity_windows:
            feature_names.extend(
                [
                    f"adf_pvalue_{window}",
                    f"adf_statistic_{window}",
                    f"kpss_pvalue_{window}",
                    f"kpss_statistic_{window}",
                    f"pp_pvalue_{window}",  # Phillips-Perron
                ]
            )

        # Serial correlation tests
        feature_names.extend(
            [
                "ljungbox_pvalue",
                "ljungbox_statistic",
                "portmanteau_stat",
                "autocorr_1",
                "autocorr_5",
                "autocorr_10",
            ]
        )

        # Long memory tests
        feature_names.extend(
            [
                "rs_statistic",
                "hurst_rs",
                "variance_ratio_2",
                "variance_ratio_4",
                "variance_ratio_8",
                "variance_ratio_16",
                "long_memory_test",
            ]
        )

        # Change point detection
        feature_names.extend(
            [
                "cusum_stat",
                "changepoint_prob",
                "structural_break_test",
                "regime_probability",
                "regime_persistence",
                "regime_volatility",
            ]
        )

        # Unit root tests
        feature_names.extend(
            [
                "unit_root_prob",
                "trend_strength",
                "seasonal_strength",
                "residual_autocorr",
                "heteroskedasticity_test",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time series analysis features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with time series features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)

            # Calculate price series and returns
            close_prices = data["close"]
            returns = self.calculate_returns(close_prices)

            # Calculate stationarity tests
            features = self._calculate_stationarity_tests(close_prices, returns, features)

            # Calculate serial correlation tests
            features = self._calculate_serial_correlation_tests(returns, features)

            # Calculate long memory tests
            features = self._calculate_long_memory_tests(returns, features)

            # Calculate change point detection
            features = self._calculate_changepoint_features(close_prices, returns, features)

            # Calculate regime detection
            features = self._calculate_regime_features(returns, features)

            # Calculate additional unit root and diagnostic tests
            features = self._calculate_diagnostic_tests(close_prices, returns, features)

            return features

        except Exception as e:
            logger.error(f"Error calculating time series features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_stationarity_tests(
        self, prices: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate stationarity tests (ADF, KPSS, Phillips-Perron)."""

        for window in self.stationarity_windows:
            # Augmented Dickey-Fuller test
            def adf_test_func(x):
                return self._adf_test(x)

            adf_results = self.rolling_apply_safe(prices, window, adf_test_func)
            features[f"adf_pvalue_{window}"] = adf_results.apply(
                lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan
            )
            features[f"adf_statistic_{window}"] = adf_results.apply(
                lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else np.nan
            )

            # KPSS test
            def kpss_test_func(x):
                return self._kpss_test(x)

            kpss_results = self.rolling_apply_safe(prices, window, kpss_test_func)
            features[f"kpss_pvalue_{window}"] = kpss_results.apply(
                lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan
            )
            features[f"kpss_statistic_{window}"] = kpss_results.apply(
                lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else np.nan
            )

            # Phillips-Perron test
            def pp_test_func(x):
                return self._phillips_perron_test(x)

            features[f"pp_pvalue_{window}"] = self.rolling_apply_safe(prices, window, pp_test_func)

        return features

    def _calculate_serial_correlation_tests(
        self, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate serial correlation tests."""

        # Ljung-Box test
        def ljungbox_func(x):
            return self._ljungbox_test(x)

        ljungbox_results = self.rolling_apply_safe(returns, self.ljungbox_window, ljungbox_func)
        features["ljungbox_pvalue"] = ljungbox_results.apply(
            lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan
        )
        features["ljungbox_statistic"] = ljungbox_results.apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else np.nan
        )

        # Portmanteau test
        def portmanteau_func(x):
            return self._portmanteau_test(x)

        features["portmanteau_stat"] = self.rolling_apply_safe(
            returns, self.portmanteau_window, portmanteau_func
        )

        # Individual autocorrelations
        def autocorr_func(x, lag):
            if len(x) < lag + 5:
                return np.nan
            try:
                return pd.Series(x).autocorr(lag=lag)
            except (ValueError, RuntimeWarning):
                return np.nan

        features["autocorr_1"] = self.rolling_apply_safe(returns, 30, lambda x: autocorr_func(x, 1))
        features["autocorr_5"] = self.rolling_apply_safe(returns, 30, lambda x: autocorr_func(x, 5))
        features["autocorr_10"] = self.rolling_apply_safe(
            returns, 40, lambda x: autocorr_func(x, 10)
        )

        return features

    def _calculate_long_memory_tests(
        self, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate long memory and persistence tests."""

        # R/S statistic
        def rs_func(x):
            return self._calculate_rs_statistic(x)

        features["rs_statistic"] = self.rolling_apply_safe(returns, self.rs_window, rs_func)

        # Hurst exponent from R/S analysis
        def hurst_rs_func(x):
            return self._hurst_from_rs(x)

        features["hurst_rs"] = self.rolling_apply_safe(returns, self.rs_window, hurst_rs_func)

        # Variance ratio tests
        for lag in self.variance_ratio_lags:

            def variance_ratio_func(x, k=lag):
                return self._variance_ratio_test(x, k)

            features[f"variance_ratio_{lag}"] = self.rolling_apply_safe(
                returns, max(60, lag * 10), variance_ratio_func
            )

        # Long memory test (GPH estimator)
        def long_memory_func(x):
            return self._gph_long_memory_test(x)

        features["long_memory_test"] = self.rolling_apply_safe(returns, 100, long_memory_func)

        return features

    def _calculate_changepoint_features(
        self, prices: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate change point detection features."""

        # CUSUM test
        def cusum_func(x):
            return self._cusum_test(x)

        features["cusum_stat"] = self.rolling_apply_safe(returns, self.cusum_window, cusum_func)

        # Bayesian changepoint probability
        def changepoint_prob_func(x):
            return self._bayesian_changepoint_probability(x)

        features["changepoint_prob"] = self.rolling_apply_safe(
            returns, self.cusum_window, changepoint_prob_func
        )

        # Structural break test
        def structural_break_func(x):
            return self._structural_break_test(x)

        features["structural_break_test"] = self.rolling_apply_safe(
            prices, self.cusum_window, structural_break_func
        )

        return features

    def _calculate_regime_features(
        self, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate regime detection features."""

        # Regime probability (threshold model)
        def regime_prob_func(x):
            return self._threshold_regime_probability(x)

        features["regime_probability"] = self.rolling_apply_safe(returns, 60, regime_prob_func)

        # Regime persistence
        def regime_persistence_func(x):
            return self._calculate_regime_persistence(x)

        features["regime_persistence"] = self.rolling_apply_safe(
            returns, 60, regime_persistence_func
        )

        # Regime volatility
        def regime_volatility_func(x):
            return self._calculate_regime_volatility(x)

        features["regime_volatility"] = self.rolling_apply_safe(returns, 60, regime_volatility_func)

        return features

    def _calculate_diagnostic_tests(
        self, prices: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate additional diagnostic tests."""

        # Unit root probability (composite test)
        def unit_root_func(x):
            return self._composite_unit_root_test(x)

        features["unit_root_prob"] = self.rolling_apply_safe(prices, 60, unit_root_func)

        # Trend strength
        def trend_strength_func(x):
            return self._calculate_trend_strength(x)

        features["trend_strength"] = self.rolling_apply_safe(prices, 60, trend_strength_func)

        # Seasonal strength (simplified)
        def seasonal_strength_func(x):
            return self._calculate_seasonal_strength(x)

        features["seasonal_strength"] = self.rolling_apply_safe(returns, 60, seasonal_strength_func)

        # Residual autocorrelation from AR(1) fit
        def residual_autocorr_func(x):
            return self._ar1_residual_autocorr(x)

        features["residual_autocorr"] = self.rolling_apply_safe(returns, 40, residual_autocorr_func)

        # Heteroskedasticity test (Engle's ARCH test)
        def arch_test_func(x):
            return self._engle_arch_test(x)

        features["heteroskedasticity_test"] = self.rolling_apply_safe(returns, 60, arch_test_func)

        return features

    # Helper methods for statistical tests

    def _adf_test(self, data: np.ndarray) -> tuple[float, float]:
        """Perform Augmented Dickey-Fuller test."""
        if len(data) < 20:
            return (np.nan, np.nan)

        try:
            result = adfuller(
                data, maxlag=self.adf_maxlag, regression=self.adf_regression, autolag="AIC"
            )
            return (result[0], result[1])  # statistic, p-value
        except (ValueError, RuntimeWarning):
            return (np.nan, np.nan)

    def _kpss_test(self, data: np.ndarray) -> tuple[float, float]:
        """Perform KPSS test."""
        if len(data) < 20:
            return (np.nan, np.nan)

        try:
            result = kpss(data, regression=self.kpss_regression, nlags="auto")
            return (result[0], result[1])  # statistic, p-value
        except (ValueError, RuntimeWarning):
            return (np.nan, np.nan)

    def _phillips_perron_test(self, data: np.ndarray) -> float:
        """Perform Phillips-Perron test (simplified)."""
        if len(data) < 20:
            return np.nan

        try:
            # Simplified PP test using ADF with fixed lag
            result = adfuller(data, maxlag=1, regression=self.adf_regression)
            return result[1]  # p-value
        except (ValueError, RuntimeWarning):
            return np.nan

    def _ljungbox_test(self, data: np.ndarray) -> tuple[float, float]:
        """Perform Ljung-Box test."""
        if len(data) < self.ljungbox_lags + 5:
            return (np.nan, np.nan)

        try:
            result = acorr_ljungbox(data, lags=self.ljungbox_lags, return_df=False)
            # Return the test statistic and p-value for the last lag
            return (result[0][-1], result[1][-1])
        except (ValueError, RuntimeWarning):
            return (np.nan, np.nan)

    def _portmanteau_test(self, data: np.ndarray) -> float:
        """Calculate Portmanteau test statistic."""
        if len(data) < 20:
            return np.nan

        try:
            n = len(data)
            max_lag = min(10, n // 4)

            # Calculate autocorrelations
            autocorrs = []
            for lag in range(1, max_lag + 1):
                if lag < n:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)

            if not autocorrs:
                return np.nan

            # Ljung-Box statistic
            lb_stat = n * (n + 2) * sum(corr**2 / (n - i - 1) for i, corr in enumerate(autocorrs))

            return lb_stat

        except (ValueError, RuntimeWarning):
            return np.nan

    def _calculate_rs_statistic(self, data: np.ndarray) -> float:
        """Calculate rescaled range statistic."""
        if len(data) < 10:
            return np.nan

        try:
            # Demean the series
            mean_val = np.mean(data)
            demeaned = data - mean_val

            # Calculate cumulative sum
            cumsum = np.cumsum(demeaned)

            # Calculate range
            R = np.max(cumsum) - np.min(cumsum)

            # Calculate standard deviation
            S = np.std(data)

            # R/S statistic
            if self.numerical_tolerance < S:
                return R / S
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _hurst_from_rs(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent from R/S analysis."""
        if len(data) < 20:
            return np.nan

        try:
            # Calculate R/S for different subsample sizes
            sizes = [int(len(data) / k) for k in [8, 4, 2, 1] if len(data) / k >= 10]
            rs_values = []

            for size in sizes:
                if size < 10:
                    continue

                # Use first 'size' points
                subset = data[:size]
                rs = self._calculate_rs_statistic(subset)
                if not np.isnan(rs) and rs > 0:
                    rs_values.append(rs)

            if len(rs_values) >= 2 and len(sizes) >= 2:
                # Fit power law: R/S ~ n^H
                log_sizes = np.log(sizes[: len(rs_values)])
                log_rs = np.log(rs_values)

                hurst = np.polyfit(log_sizes, log_rs, 1)[0]
                return hurst

            return 0.5  # Default for random walk

        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan

    def _variance_ratio_test(self, data: np.ndarray, k: int) -> float:
        """Calculate variance ratio test statistic."""
        if len(data) < k * 10:
            return np.nan

        try:
            n = len(data)

            # Calculate variance of k-period returns
            k_returns = []
            for i in range(0, n - k + 1, k):
                k_return = np.sum(data[i : i + k])
                k_returns.append(k_return)

            if len(k_returns) < 3:
                return np.nan

            var_k = np.var(k_returns, ddof=1)
            var_1 = np.var(data, ddof=1)

            # Variance ratio
            if var_1 > self.numerical_tolerance:
                vr = var_k / (k * var_1)
                return vr
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _gph_long_memory_test(self, data: np.ndarray) -> float:
        """GPH (Geweke-Porter-Hudak) test for long memory."""
        if len(data) < 50:
            return np.nan

        try:
            # Calculate periodogram
            n = len(data)
            freqs = np.fft.fftfreq(n)[1 : n // 2]  # Exclude zero frequency
            fft_vals = np.fft.fft(data - np.mean(data))[1 : n // 2]
            periodogram = np.abs(fft_vals) ** 2 / n

            # Use low frequencies for GPH regression
            m = int(n**0.5)  # Number of low frequencies
            if m < 5:
                return np.nan

            log_freqs = np.log(freqs[:m])
            log_periodogram = np.log(periodogram[:m])

            # GPH regression: log(I(λ)) = c - d*log(4*sin²(λ/2))
            log_sin = np.log(4 * np.sin(freqs[:m] * np.pi) ** 2)

            # Estimate d (long memory parameter)
            d_estimate = -np.polyfit(log_sin, log_periodogram, 1)[0]

            return d_estimate

        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan

    def _cusum_test(self, data: np.ndarray) -> float:
        """Calculate CUSUM test statistic."""
        if len(data) < 20:
            return np.nan

        try:
            # Standardize data
            standardized = (data - np.mean(data)) / (np.std(data) + self.numerical_tolerance)

            # CUSUM statistic
            cusum = np.cumsum(standardized)
            cusum_stat = np.max(np.abs(cusum)) / np.sqrt(len(data))

            return cusum_stat

        except (ValueError, RuntimeWarning):
            return np.nan

    def _bayesian_changepoint_probability(self, data: np.ndarray) -> float:
        """Estimate probability of changepoint using Bayesian approach."""
        if len(data) < 20:
            return np.nan

        try:
            n = len(data)

            # Test each point as potential changepoint
            log_probs = []

            for i in range(5, n - 5):  # Avoid endpoints
                # Calculate likelihood under no change
                var_full = np.var(data)
                if var_full < self.numerical_tolerance:
                    continue

                ll_no_change = -n / 2 * np.log(var_full)

                # Calculate likelihood under change
                var1 = np.var(data[:i])
                var2 = np.var(data[i:])

                if var1 < self.numerical_tolerance or var2 < self.numerical_tolerance:
                    continue

                ll_change = -i / 2 * np.log(var1) - (n - i) / 2 * np.log(var2)

                # Bayes factor
                log_bf = ll_change - ll_no_change
                log_probs.append(log_bf)

            if not log_probs:
                return np.nan

            # Maximum log Bayes factor as changepoint evidence
            max_log_bf = np.max(log_probs)

            # Convert to probability scale
            prob = 1 / (1 + np.exp(-max_log_bf))

            return prob

        except (ValueError, RuntimeWarning):
            return np.nan

    def _structural_break_test(self, data: np.ndarray) -> float:
        """Test for structural breaks in time series."""
        if len(data) < 30:
            return np.nan

        try:
            n = len(data)

            # Fit linear trend to full series
            t = np.arange(n)
            full_coeffs = np.polyfit(t, data, 1)
            full_residuals = data - np.polyval(full_coeffs, t)
            full_ssr = np.sum(full_residuals**2)

            # Test breakpoint in middle third of series
            start_test = n // 3
            end_test = 2 * n // 3

            min_ssr_sum = np.inf

            for break_point in range(start_test, end_test):
                # Fit separate trends before and after break
                t1 = np.arange(break_point)
                t2 = np.arange(break_point, n)

                if len(t1) < 5 or len(t2) < 5:
                    continue

                coeffs1 = np.polyfit(t1, data[:break_point], 1)
                coeffs2 = np.polyfit(t2 - break_point, data[break_point:], 1)

                residuals1 = data[:break_point] - np.polyval(coeffs1, t1)
                residuals2 = data[break_point:] - np.polyval(coeffs2, t2 - break_point)

                ssr_sum = np.sum(residuals1**2) + np.sum(residuals2**2)
                min_ssr_sum = min(min_ssr_sum, ssr_sum)

            # F-statistic for structural break
            if min_ssr_sum < full_ssr and full_ssr > self.numerical_tolerance:
                f_stat = ((full_ssr - min_ssr_sum) / 2) / (min_ssr_sum / (n - 4))
                return f_stat
            else:
                return np.nan

        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan

    def _threshold_regime_probability(self, data: np.ndarray) -> float:
        """Calculate regime probability using threshold model."""
        if len(data) < 20:
            return np.nan

        try:
            # Define high and low volatility regimes
            rolling_vol = pd.Series(data).rolling(5).std()
            vol_threshold = np.percentile(rolling_vol.dropna(), 75)

            # Probability of being in high volatility regime
            high_vol_periods = rolling_vol > vol_threshold
            prob_high_vol = np.mean(high_vol_periods)

            return prob_high_vol

        except (ValueError, RuntimeWarning):
            return np.nan

    def _calculate_regime_persistence(self, data: np.ndarray) -> float:
        """Calculate regime persistence."""
        if len(data) < 20:
            return np.nan

        try:
            # Define regimes based on quantiles
            q25, q75 = np.percentile(data, [25, 75])

            regimes = np.where(data < q25, -1, np.where(data > q75, 1, 0))

            # Calculate persistence (probability of staying in same regime)
            transitions = []
            for i in range(1, len(regimes)):
                if regimes[i - 1] != 0 and regimes[i] != 0:
                    transitions.append(regimes[i] == regimes[i - 1])

            if transitions:
                return np.mean(transitions)
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _calculate_regime_volatility(self, data: np.ndarray) -> float:
        """Calculate volatility difference between regimes."""
        if len(data) < 20:
            return np.nan

        try:
            # Define high and low regimes
            median_val = np.median(data)

            high_regime = data[data > median_val]
            low_regime = data[data <= median_val]

            if len(high_regime) > 2 and len(low_regime) > 2:
                vol_high = np.std(high_regime)
                vol_low = np.std(low_regime)

                # Ratio of volatilities
                if vol_low > self.numerical_tolerance:
                    return vol_high / vol_low
                else:
                    return np.nan
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _composite_unit_root_test(self, data: np.ndarray) -> float:
        """Composite unit root test combining multiple tests."""
        if len(data) < 20:
            return np.nan

        try:
            # Combine ADF and KPSS tests
            adf_stat, adf_pval = self._adf_test(data)
            kpss_stat, kpss_pval = self._kpss_test(data)

            # Unit root probability from ADF (higher p-value = more likely unit root)
            adf_prob = adf_pval if not np.isnan(adf_pval) else 0.5

            # Stationarity probability from KPSS (higher p-value = more likely stationary)
            kpss_prob = 1 - kpss_pval if not np.isnan(kpss_pval) else 0.5

            # Combine probabilities (equal weight)
            unit_root_prob = (adf_prob + kpss_prob) / 2

            return unit_root_prob

        except (ValueError, RuntimeWarning):
            return np.nan

    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(data) < 10:
            return np.nan

        try:
            # Linear trend coefficient
            t = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(t, data)

            # Trend strength as R-squared
            return r_value**2

        except (ValueError, RuntimeWarning):
            return np.nan

    def _calculate_seasonal_strength(self, data: np.ndarray) -> float:
        """Calculate seasonal strength (simplified)."""
        if len(data) < 14:
            return np.nan

        try:
            # Weekly seasonality test (simplified)
            if len(data) >= 14:
                # Compare variance within weeks vs between weeks
                week_groups = [data[i::7] for i in range(7) if len(data[i::7]) > 1]

                if len(week_groups) >= 3:
                    within_var = np.mean([np.var(group) for group in week_groups if len(group) > 1])
                    between_var = np.var([np.mean(group) for group in week_groups])

                    total_var = np.var(data)

                    if total_var > self.numerical_tolerance:
                        seasonal_strength = between_var / total_var
                        return seasonal_strength

            return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _ar1_residual_autocorr(self, data: np.ndarray) -> float:
        """Calculate residual autocorrelation from AR(1) model."""
        if len(data) < 10:
            return np.nan

        try:
            # Fit AR(1) model
            X = data[:-1]
            y = data[1:]

            if len(X) < 3 or np.var(X) < self.numerical_tolerance:
                return np.nan

            # Linear regression
            slope = np.cov(X, y)[0, 1] / np.var(X)
            intercept = np.mean(y) - slope * np.mean(X)

            # Calculate residuals
            fitted = slope * X + intercept
            residuals = y - fitted

            # Autocorrelation of residuals
            if len(residuals) > 2:
                residual_autocorr = pd.Series(residuals).autocorr(lag=1)
                return residual_autocorr if not np.isnan(residual_autocorr) else 0.0

            return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _engle_arch_test(self, data: np.ndarray) -> float:
        """Engle's ARCH test for heteroskedasticity."""
        if len(data) < 20:
            return np.nan

        try:
            # Calculate squared residuals
            squared_data = data**2

            # Test for autocorrelation in squared returns
            lag_1_corr = pd.Series(squared_data).autocorr(lag=1)

            if not np.isnan(lag_1_corr):
                # Simple ARCH test statistic
                n = len(data)
                lm_stat = n * lag_1_corr**2
                return lm_stat

            return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan
