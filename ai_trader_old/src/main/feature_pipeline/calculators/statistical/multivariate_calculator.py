"""
Multivariate Calculator

Specialized calculator for multivariate statistical analysis including:
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Extreme Value Theory (EVT) analysis
- Wavelet decomposition features
- Multivariate statistical measures
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

try:
    # Third-party imports
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

from .base_statistical import BaseStatisticalCalculator

logger = logging.getLogger(__name__)


class MultivariateCalculator(BaseStatisticalCalculator):
    """Calculator for multivariate statistical analysis and component decomposition."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize multivariate calculator."""
        super().__init__(config)

        # Component analysis parameters
        self.n_components = self.stat_config.n_components
        self.pca_components = self.config.get("pca_components", 5)
        self.ica_components = self.config.get("ica_components", 3)

        # Extreme value analysis parameters
        self.evt_windows = self.config.get("evt_windows", [60, 252])
        self.evt_quantiles = self.config.get("evt_quantiles", [0.95, 0.99])
        self.hill_estimator_fraction = self.config.get("hill_estimator_fraction", 0.1)

        # Wavelet analysis parameters
        self.wavelet_family = self.stat_config.wavelet_family
        self.wavelet_levels = self.stat_config.wavelet_levels
        self.wavelet_window = self.config.get("wavelet_window", 60)

        # Minimum data requirements
        self.min_data_for_components = self.config.get("min_data_for_components", 50)

    def get_feature_names(self) -> list[str]:
        """Return list of multivariate feature names."""
        feature_names = []

        # Principal components
        for i in range(1, self.pca_components + 1):
            feature_names.append(f"pc_{i}")
        feature_names.append("pca_explained_var")

        # Independent components
        for i in range(1, self.ica_components + 1):
            feature_names.append(f"ic_{i}")

        # Extreme value features for different windows
        for window in self.evt_windows:
            feature_names.extend(
                [
                    f"evt_location_{window}",
                    f"evt_scale_{window}",
                    f"var_95_{window}",
                    f"es_95_{window}",
                    f"tail_index_{window}",
                ]
            )

        # Peak over threshold features
        feature_names.extend(["pot_exceedances", "pot_mean_excess"])

        # Wavelet features
        if HAS_PYWT:
            feature_names.append("wavelet_approx_energy")
            for i in range(1, self.wavelet_levels + 1):
                feature_names.append(f"wavelet_detail_{i}_energy")
            feature_names.append("wavelet_entropy")

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multivariate statistical features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with multivariate features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)

            # Calculate component analysis features
            features = self._calculate_component_features(data, features)

            # Calculate extreme value features
            features = self._calculate_extreme_value_features(data, features)

            # Calculate wavelet features
            if HAS_PYWT:
                features = self._calculate_wavelet_features(data, features)
            else:
                features = self._add_placeholder_wavelet_features(features)

            return features

        except Exception as e:
            logger.error(f"Error calculating multivariate features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_component_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate PCA and ICA component features."""
        returns = self.calculate_returns(data["close"]).dropna()

        if len(returns) < self.min_data_for_components:
            logger.warning("Insufficient data for component analysis")
            return features

        # Create lagged return matrix for component analysis
        return_matrix = self._create_lagged_matrix(returns, self.n_components)

        if len(return_matrix) > self.n_components * 2:
            # Principal Component Analysis
            features = self._calculate_pca_features(return_matrix, features, data.index)

            # Independent Component Analysis
            features = self._calculate_ica_features(return_matrix, features, data.index)

        return features

    def _calculate_pca_features(
        self, return_matrix: pd.DataFrame, features: pd.DataFrame, original_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate PCA features."""
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(return_matrix)

            # Fit PCA
            n_components = min(self.pca_components, return_matrix.shape[1])
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled_returns)

            # Add principal components to features
            for i in range(components.shape[1]):
                pc_series = pd.Series(components[:, i], index=return_matrix.index)
                features[f"pc_{i+1}"] = pc_series.reindex(original_index).fillna(method="ffill")

            # Add explained variance
            total_explained_var = sum(pca.explained_variance_ratio_)
            features["pca_explained_var"] = total_explained_var

            logger.debug(
                f"PCA: {n_components} components explain {total_explained_var:.3f} of variance"
            )

        except Exception as e:
            logger.warning(f"Error in PCA calculation: {e}")
            # Fill with NaN if PCA fails
            for i in range(1, self.pca_components + 1):
                features[f"pc_{i}"] = np.nan
            features["pca_explained_var"] = np.nan

        return features

    def _calculate_ica_features(
        self, return_matrix: pd.DataFrame, features: pd.DataFrame, original_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate ICA features."""
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_returns = scaler.fit_transform(return_matrix)

            # Fit ICA
            n_components = min(self.ica_components, return_matrix.shape[1])
            ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
            sources = ica.fit_transform(scaled_returns)

            # Add independent components to features
            for i in range(sources.shape[1]):
                ic_series = pd.Series(sources[:, i], index=return_matrix.index)
                features[f"ic_{i+1}"] = ic_series.reindex(original_index).fillna(method="ffill")

            logger.debug(f"ICA: Extracted {n_components} independent components")

        except Exception as e:
            logger.warning(f"Error in ICA calculation: {e}")
            # Fill with NaN if ICA fails
            for i in range(1, self.ica_components + 1):
                features[f"ic_{i}"] = np.nan

        return features

    def _calculate_extreme_value_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate extreme value theory features."""
        returns = self.calculate_returns(data["close"])

        for window in self.evt_windows:
            # EVT distribution parameters
            def evt_location_func(x):
                return self._fit_evt(x)[0]

            def evt_scale_func(x):
                return self._fit_evt(x)[1]

            features[f"evt_location_{window}"] = self.rolling_apply_safe(
                returns, window, evt_location_func
            )
            features[f"evt_scale_{window}"] = self.rolling_apply_safe(
                returns, window, evt_scale_func
            )

            # Value at Risk (parametric)
            evt_location = features[f"evt_location_{window}"]
            evt_scale = features[f"evt_scale_{window}"]
            features[f"var_95_{window}"] = evt_location - evt_scale * np.log(-np.log(0.95))

            # Expected Shortfall
            def expected_shortfall_func(x):
                return self._calculate_expected_shortfall(x, alpha=0.95)

            features[f"es_95_{window}"] = self.rolling_apply_safe(
                returns, window, expected_shortfall_func
            )

            # Tail index (Hill estimator)
            def hill_estimator_func(x):
                return self._hill_estimator(x)

            features[f"tail_index_{window}"] = self.rolling_apply_safe(
                returns, window, hill_estimator_func
            )

        # Peak over threshold analysis
        threshold_window = max(self.evt_windows)  # Use largest window for threshold
        threshold = returns.rolling(threshold_window).quantile(0.95)

        # Number of exceedances in rolling window
        exceedances = (returns > threshold).astype(int)
        features["pot_exceedances"] = exceedances.rolling(20).sum()

        # Mean excess over threshold
        excess_returns = returns - threshold
        features["pot_mean_excess"] = excess_returns[excess_returns > 0].rolling(20).mean()

        return features

    def _calculate_wavelet_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate wavelet decomposition features."""
        close_prices = data["close"]

        try:
            # Perform wavelet decomposition on price series
            coeffs = pywt.wavedec(
                close_prices.values, self.wavelet_family, level=self.wavelet_levels
            )

            # Energy at approximation level
            approx_energy = coeffs[0] ** 2
            approx_series = pd.Series(approx_energy, index=close_prices.index[: len(approx_energy)])
            features["wavelet_approx_energy"] = approx_series.reindex(close_prices.index).fillna(
                method="ffill"
            )

            # Energy at detail levels
            for i, detail_coeffs in enumerate(coeffs[1:], 1):
                detail_energy = detail_coeffs**2
                detail_series = pd.Series(
                    detail_energy, index=close_prices.index[: len(detail_energy)]
                )
                features[f"wavelet_detail_{i}_energy"] = detail_series.reindex(
                    close_prices.index
                ).fillna(method="ffill")

            # Wavelet entropy
            def wavelet_entropy_func(x):
                return self._calculate_wavelet_entropy(x)

            features["wavelet_entropy"] = self.rolling_apply_safe(
                close_prices, self.wavelet_window, wavelet_entropy_func
            )

            logger.debug(f"Calculated wavelet features with {self.wavelet_levels} levels")

        except Exception as e:
            logger.warning(f"Error in wavelet decomposition: {e}")
            features = self._add_placeholder_wavelet_features(features)

        return features

    def _add_placeholder_wavelet_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder wavelet features when PyWavelets is not available."""
        features["wavelet_approx_energy"] = 0.0
        for i in range(1, self.wavelet_levels + 1):
            features[f"wavelet_detail_{i}_energy"] = 0.0
        features["wavelet_entropy"] = 0.0

        logger.debug("Added placeholder wavelet features (PyWavelets not available)")
        return features

    def _create_lagged_matrix(self, series: pd.Series, n_lags: int) -> pd.DataFrame:
        """Create matrix with lagged versions of the series."""
        lagged_matrix = pd.DataFrame()

        for lag in range(n_lags):
            lagged_matrix[f"lag_{lag}"] = series.shift(lag)

        return lagged_matrix.dropna()

    def _fit_evt(self, data: np.ndarray) -> tuple[float, float]:
        """Fit extreme value distribution using simplified Gumbel approach."""
        if len(data) < 20:
            return (np.nan, np.nan)

        try:
            # Use maximum likelihood estimation for Gumbel distribution
            sorted_data = np.sort(data)
            top_10_pct = sorted_data[int(0.9 * len(sorted_data)) :]

            if len(top_10_pct) > 2:
                # Gumbel distribution parameters
                location = np.mean(top_10_pct) - 0.5772 * np.std(top_10_pct) * np.sqrt(6) / np.pi
                scale = np.std(top_10_pct) * np.sqrt(6) / np.pi
                return (location, scale)
            else:
                return (np.nan, np.nan)

        except (ValueError, RuntimeWarning):
            return (np.nan, np.nan)

    def _calculate_expected_shortfall(self, data: np.ndarray, alpha: float) -> float:
        """Calculate expected shortfall (Conditional Value at Risk)."""
        if len(data) < 20:
            return np.nan

        try:
            var = np.percentile(data, (1 - alpha) * 100)
            tail_losses = data[data <= var]

            if len(tail_losses) > 0:
                return np.mean(tail_losses)
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan

    def _hill_estimator(self, data: np.ndarray) -> float:
        """Calculate Hill estimator for tail index."""
        if len(data) < 50:
            return np.nan

        try:
            # Use absolute values and take top fraction
            abs_data = np.abs(data)
            sorted_data = np.sort(abs_data)[::-1]  # Descending order
            k = max(2, int(self.hill_estimator_fraction * len(sorted_data)))

            if k < len(sorted_data):
                # Hill estimator
                threshold = sorted_data[k]
                exceedances = sorted_data[:k]

                if threshold > self.numerical_tolerance:
                    log_ratios = np.log(exceedances / threshold)
                    alpha = k / np.sum(log_ratios)
                    return alpha
                else:
                    return np.nan
            else:
                return np.nan

        except (ValueError, RuntimeWarning, ZeroDivisionError):
            return np.nan

    def _calculate_wavelet_entropy(self, data: np.ndarray) -> float:
        """Calculate wavelet entropy."""
        if len(data) < 16:
            return np.nan

        if not HAS_PYWT:
            return 0.0

        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(data, self.wavelet_family, level=min(3, int(np.log2(len(data)))))

            # Calculate energy at each level
            energies = []
            for coeff in coeffs:
                energy = np.sum(coeff**2)
                energies.append(energy)

            total_energy = sum(energies)

            if total_energy > self.numerical_tolerance:
                # Normalize to probabilities
                probs = np.array(energies) / total_energy

                # Calculate Shannon entropy
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                return entropy
            else:
                return np.nan

        except (ValueError, RuntimeWarning):
            return np.nan
