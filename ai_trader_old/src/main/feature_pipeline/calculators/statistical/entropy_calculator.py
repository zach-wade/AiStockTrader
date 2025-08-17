"""
Entropy Calculator

Calculates various entropy measures for time series data,
capturing complexity, predictability, and information content.
"""

# Standard library imports
from itertools import permutations
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_statistical import BaseStatisticalCalculator
from .statistical_config import EntropyConfig
from ..helpers import calculate_entropy as basic_entropy
from ..helpers import create_feature_dataframe

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EntropyCalculator(BaseStatisticalCalculator):
    """
    Calculates entropy-based features for time series analysis.

    Features include:
    - Shannon entropy
    - Renyi entropy
    - Tsallis entropy
    - Approximate entropy
    - Sample entropy
    - Permutation entropy
    - Multiscale entropy
    - Spectral entropy
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize entropy calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Initialize entropy configuration
        entropy_config = config.get("entropy", {}) if config else {}
        self.entropy_config = EntropyConfig(**entropy_config)

        # Feature name cache
        self._feature_names_cache = None

        logger.info("Initialized EntropyCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of entropy feature names."""
        if self._feature_names_cache is not None:
            return self._feature_names_cache

        features = []

        # Shannon entropy features
        if self.entropy_config.shannon:
            for window in self.lookback_periods:
                features.extend(
                    [
                        f"shannon_entropy_{window}",
                        f"normalized_shannon_entropy_{window}",
                        f"entropy_rate_{window}",
                    ]
                )

        # Renyi entropy
        if self.entropy_config.renyi:
            for window in self.lookback_periods:
                features.append(f"renyi_entropy_{window}")

        # Tsallis entropy
        if self.entropy_config.tsallis:
            for window in self.lookback_periods:
                features.append(f"tsallis_entropy_{window}")

        # Approximate entropy
        if self.entropy_config.approximate:
            for window in [50, 100, 200]:  # Limited windows for computational efficiency
                features.append(f"approximate_entropy_{window}")

        # Sample entropy
        if self.entropy_config.sample:
            for window in [50, 100, 200]:
                features.append(f"sample_entropy_{window}")

        # Permutation entropy
        if self.entropy_config.permutation:
            for window in self.lookback_periods:
                features.extend(
                    [f"permutation_entropy_{window}", f"weighted_permutation_entropy_{window}"]
                )

        # Multiscale entropy
        if self.entropy_config.multiscale:
            for scale in self.entropy_config.multiscale_factors:
                features.append(f"multiscale_entropy_scale_{scale}")

        # Spectral entropy
        for window in [50, 100, 200]:
            features.append(f"spectral_entropy_{window}")

        # Differential entropy
        for window in self.lookback_periods:
            features.append(f"differential_entropy_{window}")

        # Cross-entropy features
        features.extend(
            [
                "relative_entropy_vs_normal",
                "relative_entropy_vs_uniform",
                "kullback_leibler_divergence",
            ]
        )

        self._feature_names_cache = features
        return features

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate entropy features.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with entropy features
        """
        try:
            # Initialize features DataFrame
            features = create_feature_dataframe(data.index)

            # Calculate returns for entropy analysis
            returns = self.calculate_returns(data["close"])

            # Shannon entropy
            if self.entropy_config.shannon:
                shannon_features = self._calculate_shannon_entropy(returns)
                features = pd.concat([features, shannon_features], axis=1)

            # Renyi entropy
            if self.entropy_config.renyi:
                renyi_features = self._calculate_renyi_entropy(
                    returns, alpha=self.entropy_config.renyi_alpha
                )
                features = pd.concat([features, renyi_features], axis=1)

            # Tsallis entropy
            if self.entropy_config.tsallis:
                tsallis_features = self._calculate_tsallis_entropy(
                    returns, q=self.entropy_config.tsallis_q
                )
                features = pd.concat([features, tsallis_features], axis=1)

            # Approximate entropy
            if self.entropy_config.approximate:
                approx_features = self._calculate_approximate_entropy(returns)
                features = pd.concat([features, approx_features], axis=1)

            # Sample entropy
            if self.entropy_config.sample:
                sample_features = self._calculate_sample_entropy(returns)
                features = pd.concat([features, sample_features], axis=1)

            # Permutation entropy
            if self.entropy_config.permutation:
                perm_features = self._calculate_permutation_entropy(returns)
                features = pd.concat([features, perm_features], axis=1)

            # Multiscale entropy
            if self.entropy_config.multiscale:
                multiscale_features = self._calculate_multiscale_entropy(returns)
                features = pd.concat([features, multiscale_features], axis=1)

            # Spectral entropy
            spectral_features = self._calculate_spectral_entropy(returns)
            features = pd.concat([features, spectral_features], axis=1)

            # Differential entropy
            diff_features = self._calculate_differential_entropy(returns)
            features = pd.concat([features, diff_features], axis=1)

            # Cross-entropy measures
            cross_features = self._calculate_cross_entropy(returns)
            features = pd.concat([features, cross_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating entropy features: {e}")
            return self._create_empty_features(data.index)

    def _calculate_shannon_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate Shannon entropy features."""
        features = pd.DataFrame(index=returns.index)

        for window in self.lookback_periods:
            # Basic Shannon entropy
            features[f"shannon_entropy_{window}"] = returns.rolling(
                window=window, min_periods=self.stat_config.get_min_periods(window)
            ).apply(lambda x: basic_entropy(pd.Series(x), bins=self.entropy_config.bins), raw=True)

            # Normalized Shannon entropy
            features[f"normalized_shannon_entropy_{window}"] = returns.rolling(
                window=window, min_periods=self.stat_config.get_min_periods(window)
            ).apply(
                lambda x: basic_entropy(
                    pd.Series(x), bins=self.entropy_config.bins, method="normalized"
                ),
                raw=True,
            )

            # Entropy rate (change in entropy)
            if window > 1:
                features[f"entropy_rate_{window}"] = features[f"shannon_entropy_{window}"].diff()

        return features

    def _calculate_renyi_entropy(self, returns: pd.Series, alpha: float = 2.0) -> pd.DataFrame:
        """Calculate Renyi entropy."""
        features = pd.DataFrame(index=returns.index)

        def renyi_entropy(x, alpha, bins):
            """Calculate Renyi entropy for a given alpha."""
            # Discretize the data
            hist, _ = np.histogram(x, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities

            if alpha == 1:
                # Limit case: Shannon entropy
                return -np.sum(probs * np.log(probs))
            else:
                return (1 / (1 - alpha)) * np.log(np.sum(probs**alpha))

        for window in self.lookback_periods:
            features[f"renyi_entropy_{window}"] = returns.rolling(
                window=window, min_periods=self.stat_config.get_min_periods(window)
            ).apply(lambda x: renyi_entropy(x, alpha, self.entropy_config.bins), raw=True)

        return features

    def _calculate_tsallis_entropy(self, returns: pd.Series, q: float = 2.0) -> pd.DataFrame:
        """Calculate Tsallis entropy."""
        features = pd.DataFrame(index=returns.index)

        def tsallis_entropy(x, q, bins):
            """Calculate Tsallis entropy for a given q."""
            # Discretize the data
            hist, _ = np.histogram(x, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities

            if q == 1:
                # Limit case: Shannon entropy
                return -np.sum(probs * np.log(probs))
            else:
                return (1 / (q - 1)) * (1 - np.sum(probs**q))

        for window in self.lookback_periods:
            features[f"tsallis_entropy_{window}"] = returns.rolling(
                window=window, min_periods=self.stat_config.get_min_periods(window)
            ).apply(lambda x: tsallis_entropy(x, q, self.entropy_config.bins), raw=True)

        return features

    def _calculate_approximate_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate approximate entropy."""
        features = pd.DataFrame(index=returns.index)

        # Use limited windows for computational efficiency
        windows = [w for w in [50, 100, 200] if w in self.lookback_periods]

        for window in windows:
            features[f"approximate_entropy_{window}"] = returns.rolling(
                window=window, min_periods=window
            ).apply(
                lambda x: self._approx_entropy(
                    x, m=self.entropy_config.approx_m, r=self.entropy_config.approx_r
                ),
                raw=True,
            )

        return features

    def _approx_entropy(self, series: np.ndarray, m: int, r: float) -> float:
        """
        Calculate approximate entropy for a series.

        Args:
            series: Input series
            m: Pattern length
            r: Tolerance (as fraction of std)

        Returns:
            Approximate entropy value
        """
        try:
            N = len(series)
            if m + 1 > N:
                return 0.0

            # Scale tolerance by standard deviation
            tolerance = r * np.std(series)

            def _maxdist(xi, xj, m):
                """Maximum distance between patterns."""
                return max([abs(float(a) - float(b)) for a, b in zip(xi, xj)])

            def _phi(m):
                """Calculate phi(m)."""
                # Create patterns
                patterns = np.array([series[i : i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)

                # Count pattern matches
                for i in range(N - m + 1):
                    for j in range(N - m + 1):
                        if _maxdist(patterns[i], patterns[j], m) <= tolerance:
                            C[i] += 1

                # Calculate phi
                phi = 0
                for c in C:
                    if c > 0:
                        phi += np.log(c / (N - m + 1))

                return phi / (N - m + 1)

            return _phi(m) - _phi(m + 1)

        except Exception as e:
            logger.warning(f"Error in approximate entropy calculation: {e}")
            return 0.0

    def _calculate_sample_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate sample entropy."""
        features = pd.DataFrame(index=returns.index)

        # Use limited windows for computational efficiency
        windows = [w for w in [50, 100, 200] if w in self.lookback_periods]

        for window in windows:
            features[f"sample_entropy_{window}"] = returns.rolling(
                window=window, min_periods=window
            ).apply(
                lambda x: self._sample_entropy(
                    x, m=self.entropy_config.sample_m, r=self.entropy_config.sample_r
                ),
                raw=True,
            )

        return features

    def _sample_entropy(self, series: np.ndarray, m: int, r: float) -> float:
        """
        Calculate sample entropy for a series.

        Args:
            series: Input series
            m: Pattern length
            r: Tolerance (as fraction of std)

        Returns:
            Sample entropy value
        """
        try:
            N = len(series)
            if m + 1 > N:
                return 0.0

            # Scale tolerance by standard deviation
            tolerance = r * np.std(series)

            # Count pattern matches for length m
            B = 0  # matches for length m
            A = 0  # matches for length m+1

            # Create patterns
            for i in range(N - m):
                for j in range(i + 1, N - m):
                    # Check m-length pattern match
                    if max([abs(series[i + k] - series[j + k]) for k in range(m)]) <= tolerance:
                        B += 1

                        # Check m+1-length pattern match
                        if i < N - m - 1 and j < N - m - 1:
                            if abs(series[i + m] - series[j + m]) <= tolerance:
                                A += 1

            # Calculate sample entropy
            if B == 0:
                return 0.0

            return -np.log(A / B) if A > 0 else -np.log(1 / (N * (N - 1)))

        except Exception as e:
            logger.warning(f"Error in sample entropy calculation: {e}")
            return 0.0

    def _calculate_permutation_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate permutation entropy."""
        features = pd.DataFrame(index=returns.index)

        order = self.entropy_config.permutation_order

        for window in self.lookback_periods:
            if window >= order:
                # Standard permutation entropy
                features[f"permutation_entropy_{window}"] = returns.rolling(
                    window=window, min_periods=window
                ).apply(lambda x: self._permutation_entropy(x, order), raw=True)

                # Weighted permutation entropy
                features[f"weighted_permutation_entropy_{window}"] = returns.rolling(
                    window=window, min_periods=window
                ).apply(lambda x: self._weighted_permutation_entropy(x, order), raw=True)

        return features

    def _permutation_entropy(self, series: np.ndarray, order: int) -> float:
        """
        Calculate permutation entropy.

        Args:
            series: Input series
            order: Permutation order

        Returns:
            Permutation entropy value
        """
        try:
            n = len(series)
            if n < order:
                return 0.0

            # Get all possible permutations
            perms = list(permutations(range(order)))
            perm_counts = {perm: 0 for perm in perms}

            # Count occurrences of each permutation pattern
            for i in range(n - order + 1):
                # Get order statistics (ranks)
                segment = series[i : i + order]
                sorted_indices = np.argsort(segment)
                perm = tuple(sorted_indices)

                if perm in perm_counts:
                    perm_counts[perm] += 1

            # Calculate probabilities
            total = sum(perm_counts.values())
            if total == 0:
                return 0.0

            probs = np.array([count / total for count in perm_counts.values() if count > 0])

            # Calculate entropy
            return -np.sum(probs * np.log(probs))

        except Exception as e:
            logger.warning(f"Error in permutation entropy calculation: {e}")
            return 0.0

    def _weighted_permutation_entropy(self, series: np.ndarray, order: int) -> float:
        """
        Calculate weighted permutation entropy.

        Args:
            series: Input series
            order: Permutation order

        Returns:
            Weighted permutation entropy value
        """
        try:
            n = len(series)
            if n < order:
                return 0.0

            # Calculate weights based on relative magnitudes
            weights = []
            perms = []

            for i in range(n - order + 1):
                segment = series[i : i + order]
                sorted_indices = np.argsort(segment)
                perm = tuple(sorted_indices)
                perms.append(perm)

                # Weight by variance of the segment
                weight = np.var(segment)
                weights.append(weight)

            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones_like(weights) / len(weights)

            # Count weighted occurrences
            unique_perms = list(set(perms))
            weighted_probs = []

            for perm in unique_perms:
                prob = sum(weights[i] for i, p in enumerate(perms) if p == perm)
                if prob > 0:
                    weighted_probs.append(prob)

            weighted_probs = np.array(weighted_probs)

            # Calculate weighted entropy
            return -np.sum(weighted_probs * np.log(weighted_probs))

        except Exception as e:
            logger.warning(f"Error in weighted permutation entropy calculation: {e}")
            return 0.0

    def _calculate_multiscale_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate multiscale entropy."""
        features = pd.DataFrame(index=returns.index)

        # Calculate for each scale factor
        for scale in self.entropy_config.multiscale_factors:
            # Coarse-grain the time series
            coarse_series = self._coarse_grain(returns, scale)

            # Calculate sample entropy at this scale
            features[f"multiscale_entropy_scale_{scale}"] = self._calculate_sample_entropy(
                coarse_series
            ).iloc[
                :, 0
            ]  # Take first column (smallest window)

        return features

    def _coarse_grain(self, series: pd.Series, scale: int) -> pd.Series:
        """
        Coarse-grain time series for multiscale entropy.

        Args:
            series: Input series
            scale: Scale factor

        Returns:
            Coarse-grained series
        """
        n = len(series)
        coarse_length = n // scale

        coarse_series = pd.Series(index=series.index[:coarse_length])

        for i in range(coarse_length):
            coarse_series.iloc[i] = series.iloc[i * scale : (i + 1) * scale].mean()

        return coarse_series

    def _calculate_spectral_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate spectral entropy using FFT."""
        features = pd.DataFrame(index=returns.index)

        windows = [w for w in [50, 100, 200] if w <= len(returns)]

        for window in windows:
            features[f"spectral_entropy_{window}"] = returns.rolling(
                window=window, min_periods=window
            ).apply(lambda x: self._spectral_entropy(x), raw=True)

        return features

    def _spectral_entropy(self, series: np.ndarray) -> float:
        """
        Calculate spectral entropy using power spectral density.

        Args:
            series: Input series

        Returns:
            Spectral entropy value
        """
        try:
            # Remove mean
            series = series - np.mean(series)

            # Calculate power spectral density
            fft = np.fft.fft(series)
            psd = np.abs(fft) ** 2

            # Normalize to get probability distribution
            psd = psd[: len(psd) // 2]  # Take positive frequencies only
            psd_norm = psd / psd.sum()

            # Remove zero values
            psd_norm = psd_norm[psd_norm > 0]

            # Calculate entropy
            return -np.sum(psd_norm * np.log(psd_norm))

        except Exception as e:
            logger.warning(f"Error in spectral entropy calculation: {e}")
            return 0.0

    def _calculate_differential_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate differential entropy (continuous entropy)."""
        features = pd.DataFrame(index=returns.index)

        for window in self.lookback_periods:
            features[f"differential_entropy_{window}"] = returns.rolling(
                window=window, min_periods=self.stat_config.get_min_periods(window)
            ).apply(lambda x: self._differential_entropy(x), raw=True)

        return features

    def _differential_entropy(self, series: np.ndarray) -> float:
        """
        Calculate differential entropy assuming Gaussian distribution.

        Args:
            series: Input series

        Returns:
            Differential entropy value
        """
        try:
            # For Gaussian distribution: h(X) = 0.5 * log(2 * pi * e * variance)
            variance = np.var(series)
            if variance <= 0:
                return 0.0

            return 0.5 * np.log(2 * np.pi * np.e * variance)

        except Exception as e:
            logger.warning(f"Error in differential entropy calculation: {e}")
            return 0.0

    def _calculate_cross_entropy(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate cross-entropy measures."""
        features = pd.DataFrame(index=returns.index)

        # Calculate relative entropy vs theoretical distributions
        clean_returns = returns.dropna()

        if len(clean_returns) > 30:
            # Relative entropy vs normal distribution
            features["relative_entropy_vs_normal"] = self._relative_entropy_vs_normal(clean_returns)

            # Relative entropy vs uniform distribution
            features["relative_entropy_vs_uniform"] = self._relative_entropy_vs_uniform(
                clean_returns
            )

            # KL divergence (using rolling window)
            features["kullback_leibler_divergence"] = self._calculate_kl_divergence(returns)

        return features

    def _relative_entropy_vs_normal(self, returns: pd.Series) -> float:
        """Calculate relative entropy vs normal distribution."""
        try:
            # Fit normal distribution
            mu, sigma = returns.mean(), returns.std()

            # Discretize returns
            hist, bin_edges = np.histogram(returns, bins=self.entropy_config.bins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]

            # Calculate probabilities
            p = hist * bin_width  # Empirical distribution
            p = p[p > 0]  # Remove zeros

            # Calculate theoretical normal probabilities
            # Third-party imports
            from scipy.stats import norm

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            q = norm.pdf(bin_centers, mu, sigma) * bin_width
            q = q[hist > 0]  # Match dimensions
            q = q / q.sum()  # Normalize
            p = p / p.sum()

            # Calculate KL divergence
            return np.sum(p * np.log(p / q))

        except Exception as e:
            logger.warning(f"Error in relative entropy vs normal: {e}")
            return np.nan

    def _relative_entropy_vs_uniform(self, returns: pd.Series) -> float:
        """Calculate relative entropy vs uniform distribution."""
        try:
            # Discretize returns
            hist, _ = np.histogram(returns, bins=self.entropy_config.bins)
            p = hist / hist.sum()
            p = p[p > 0]  # Remove zeros

            # Uniform distribution
            q = np.ones(len(p)) / len(p)

            # Calculate KL divergence
            return np.sum(p * np.log(p / q))

        except Exception as e:
            logger.warning(f"Error in relative entropy vs uniform: {e}")
            return np.nan

    def _calculate_kl_divergence(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling KL divergence."""
        window = 100  # Fixed window for KL divergence

        def kl_div(x):
            """Calculate KL divergence between first and second half."""
            if len(x) < 20:
                return np.nan

            mid = len(x) // 2
            first_half = x[:mid]
            second_half = x[mid:]

            # Discretize
            bins = np.linspace(x.min(), x.max(), self.entropy_config.bins)
            hist1, _ = np.histogram(first_half, bins=bins)
            hist2, _ = np.histogram(second_half, bins=bins)

            # Normalize
            p = hist1 / hist1.sum()
            q = hist2 / hist2.sum()

            # Add small constant to avoid log(0)
            epsilon = 1e-10
            p = p + epsilon
            q = q + epsilon

            # Calculate KL divergence
            return np.sum(p * np.log(p / q))

        return returns.rolling(window=window, min_periods=20).apply(kl_div, raw=True)
