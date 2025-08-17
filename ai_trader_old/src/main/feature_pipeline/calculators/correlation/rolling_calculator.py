"""
Rolling Correlation Calculator

Specialized calculator for rolling correlation analysis including:
- Benchmark correlations across multiple time windows
- Rolling correlation statistics (mean, std, trends)
- Exponentially weighted correlations
- Cross-sectional correlation analysis
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

from .base_correlation import BaseCorrelationCalculator

logger = logging.getLogger(__name__)


class RollingCorrelationCalculator(BaseCorrelationCalculator):
    """Calculator for rolling correlation analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize rolling correlation calculator."""
        super().__init__(config)

        # Rolling-specific parameters
        self.rolling_config = self.correlation_config.get_window_config("rolling")
        self.decay_halflife = self.rolling_config.get("decay_halflife", 20)

        logger.debug(
            f"Initialized RollingCorrelationCalculator with {len(self.benchmark_symbols)} benchmarks"
        )

    def get_feature_names(self) -> list[str]:
        """Return list of rolling correlation feature names."""
        feature_names = []

        # Rolling correlations with benchmarks for each window
        for benchmark in self.benchmark_symbols:
            for window in self.correlation_windows:
                feature_names.extend(
                    [
                        f"corr_{benchmark}_{window}d",
                        f"corr_{benchmark}_{window}d_std",
                        f"corr_{benchmark}_{window}d_trend",
                    ]
                )

        # Exponentially weighted correlations
        for benchmark in self.benchmark_symbols:
            feature_names.extend([f"ewm_corr_{benchmark}", f"ewm_corr_{benchmark}_change"])

        # Cross-sectional correlation features
        feature_names.extend(
            [
                "avg_benchmark_corr_20d",
                "avg_benchmark_corr_60d",
                "avg_benchmark_corr_120d",
                "max_benchmark_corr_20d",
                "min_benchmark_corr_20d",
                "corr_dispersion_20d",
                "corr_stability_score",
                "dominant_benchmark_corr",
            ]
        )

        # Correlation dynamics
        feature_names.extend(
            [
                "corr_momentum_20d",
                "corr_acceleration_20d",
                "corr_mean_reversion_20d",
                "corr_volatility_20d",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling correlation features.

        Args:
            data: DataFrame with symbol, timestamp, close columns

        Returns:
            DataFrame with rolling correlation features
        """
        try:
            # Validate and preprocess data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)

            processed_data = self.preprocess_data(data)

            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)

            # Create features DataFrame
            unique_timestamps = processed_data["timestamp"].unique()
            features = self.create_empty_features(pd.Index(unique_timestamps))

            # Calculate benchmark correlations
            features = self._calculate_benchmark_correlations(processed_data, features)

            # Calculate exponentially weighted correlations
            features = self._calculate_ewm_correlations(processed_data, features)

            # Calculate cross-sectional correlation features
            features = self._calculate_cross_sectional_features(processed_data, features)

            # Calculate correlation dynamics
            features = self._calculate_correlation_dynamics(processed_data, features)

            # Reshape to match original data index if needed
            if len(features) != len(data):
                features = self._align_features_with_data(features, data)

            return features

        except Exception as e:
            logger.error(f"Error calculating rolling correlation features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_benchmark_correlations(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate rolling correlations with benchmark symbols."""
        try:
            # Pivot returns for correlation calculation
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                logger.warning("Unable to pivot returns data")
                return features

            # Calculate correlations for each benchmark and window
            for benchmark in self.benchmark_symbols:
                if benchmark not in returns_pivot.columns:
                    logger.warning(f"Benchmark {benchmark} not found in data")
                    continue

                benchmark_returns = returns_pivot[benchmark]

                for window in self.correlation_windows:
                    # Calculate rolling correlations for all symbols
                    for symbol in returns_pivot.columns:
                        if symbol == benchmark:
                            continue

                        symbol_returns = returns_pivot[symbol]

                        # Rolling correlation
                        rolling_corr = self.calculate_rolling_correlation(
                            symbol_returns, benchmark_returns, window
                        )

                        # Rolling correlation standard deviation
                        corr_std = rolling_corr.rolling(window=window // 2, min_periods=5).std()

                        # Correlation trend (slope of correlation over time)
                        corr_trend = self._calculate_correlation_trend(rolling_corr, window // 2)

                        # Store features by timestamp
                        for timestamp in rolling_corr.index:
                            if timestamp in features.index:
                                # For now, store the last symbol's correlation (in practice, you'd aggregate)
                                features.loc[timestamp, f"corr_{benchmark}_{window}d"] = (
                                    rolling_corr.loc[timestamp]
                                )
                                features.loc[timestamp, f"corr_{benchmark}_{window}d_std"] = (
                                    corr_std.loc[timestamp]
                                )
                                features.loc[timestamp, f"corr_{benchmark}_{window}d_trend"] = (
                                    corr_trend.loc[timestamp]
                                )

            return features

        except Exception as e:
            logger.warning(f"Error calculating benchmark correlations: {e}")
            return features

    def _calculate_ewm_correlations(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate exponentially weighted correlations."""
        try:
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                return features

            for benchmark in self.benchmark_symbols:
                if benchmark not in returns_pivot.columns:
                    continue

                benchmark_returns = returns_pivot[benchmark]

                # Calculate EWM correlations for all symbols
                ewm_correlations = []
                for symbol in returns_pivot.columns:
                    if symbol == benchmark:
                        continue

                    symbol_returns = returns_pivot[symbol]
                    ewm_corr = self.calculate_ewm_correlation(
                        symbol_returns, benchmark_returns, self.decay_halflife
                    )
                    ewm_correlations.append(ewm_corr)

                if ewm_correlations:
                    # Average EWM correlation across symbols
                    avg_ewm_corr = pd.concat(ewm_correlations, axis=1).mean(axis=1)
                    ewm_change = avg_ewm_corr.diff()

                    # Store in features
                    for timestamp in avg_ewm_corr.index:
                        if timestamp in features.index:
                            features.loc[timestamp, f"ewm_corr_{benchmark}"] = avg_ewm_corr.loc[
                                timestamp
                            ]
                            features.loc[timestamp, f"ewm_corr_{benchmark}_change"] = (
                                ewm_change.loc[timestamp]
                            )

            return features

        except Exception as e:
            logger.warning(f"Error calculating EWM correlations: {e}")
            return features

    def _calculate_cross_sectional_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate cross-sectional correlation features."""
        try:
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                return features

            # Calculate correlation matrix for each timestamp with rolling window
            for window in [20, 60, 120]:
                rolling_corr_matrices = []

                for i in range(window, len(returns_pivot)):
                    window_data = returns_pivot.iloc[i - window : i]
                    corr_matrix = window_data.corr()

                    # Extract upper triangle (excluding diagonal)
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    ).stack()

                    timestamp = returns_pivot.index[i]

                    if timestamp in features.index:
                        # Average correlation
                        features.loc[timestamp, f"avg_benchmark_corr_{window}d"] = (
                            upper_triangle.mean()
                        )

                        if window == 20:  # Only calculate these for 20d window
                            # Max and min correlations
                            features.loc[timestamp, f"max_benchmark_corr_{window}d"] = (
                                upper_triangle.max()
                            )
                            features.loc[timestamp, f"min_benchmark_corr_{window}d"] = (
                                upper_triangle.min()
                            )

                            # Correlation dispersion
                            features.loc[timestamp, f"corr_dispersion_{window}d"] = (
                                upper_triangle.std()
                            )

                            # Correlation stability
                            if len(rolling_corr_matrices) > 10:
                                recent_matrices = rolling_corr_matrices[-10:]
                                stability = self._calculate_matrix_stability(recent_matrices)
                                features.loc[timestamp, "corr_stability_score"] = stability

                            # Dominant benchmark correlation
                            if len(self.benchmark_symbols) > 0:
                                benchmark_corrs = []
                                for benchmark in self.benchmark_symbols:
                                    if benchmark in corr_matrix.columns:
                                        benchmark_avg_corr = (
                                            corr_matrix[benchmark].drop(benchmark).abs().mean()
                                        )
                                        benchmark_corrs.append(benchmark_avg_corr)

                                if benchmark_corrs:
                                    features.loc[timestamp, "dominant_benchmark_corr"] = max(
                                        benchmark_corrs
                                    )

                    rolling_corr_matrices.append(corr_matrix)

            return features

        except Exception as e:
            logger.warning(f"Error calculating cross-sectional features: {e}")
            return features

    def _calculate_correlation_dynamics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate correlation momentum and dynamics features."""
        try:
            # Use the 20d average benchmark correlation for dynamics
            if "avg_benchmark_corr_20d" not in features.columns:
                return features

            corr_series = features["avg_benchmark_corr_20d"]

            # Correlation momentum (rate of change)
            corr_momentum = corr_series.diff(periods=5)
            features["corr_momentum_20d"] = corr_momentum

            # Correlation acceleration (change in momentum)
            corr_acceleration = corr_momentum.diff(periods=5)
            features["corr_acceleration_20d"] = corr_acceleration

            # Mean reversion indicator
            rolling_mean = corr_series.rolling(window=60, min_periods=20).mean()
            mean_reversion = (corr_series - rolling_mean) / (
                rolling_mean + self.numerical_tolerance
            )
            features["corr_mean_reversion_20d"] = mean_reversion

            # Correlation volatility
            corr_volatility = corr_series.rolling(window=20, min_periods=10).std()
            features["corr_volatility_20d"] = corr_volatility

            return features

        except Exception as e:
            logger.warning(f"Error calculating correlation dynamics: {e}")
            return features

    def _calculate_correlation_trend(self, correlation_series: pd.Series, window: int) -> pd.Series:
        """Calculate trend (slope) of correlation series."""

        def trend_function(values):
            if len(values) < 3:
                return 0.0
            try:
                x = np.arange(len(values))
                y = values.values
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except (ValueError, TypeError, np.linalg.LinAlgError):
                return 0.0

        return correlation_series.rolling(window=window, min_periods=3).apply(trend_function)

    def _calculate_matrix_stability(self, correlation_matrices: list[pd.DataFrame]) -> float:
        """Calculate stability of correlation matrices over time."""
        try:
            if len(correlation_matrices) < 2:
                return 1.0

            # Calculate pairwise differences between consecutive matrices
            stability_scores = []

            for i in range(1, len(correlation_matrices)):
                prev_matrix = correlation_matrices[i - 1]
                curr_matrix = correlation_matrices[i]

                # Ensure matrices have same structure
                common_symbols = prev_matrix.index.intersection(curr_matrix.index)
                if len(common_symbols) < 2:
                    continue

                prev_common = prev_matrix.loc[common_symbols, common_symbols]
                curr_common = curr_matrix.loc[common_symbols, common_symbols]

                # Calculate Frobenius norm of difference
                diff_norm = np.linalg.norm(prev_common.values - curr_common.values, "fro")
                stability_scores.append(diff_norm)

            if not stability_scores:
                return 1.0

            # Stability is inverse of average change
            avg_change = np.mean(stability_scores)
            stability = 1.0 / (1.0 + avg_change)

            return stability

        except Exception as e:
            logger.warning(f"Error calculating matrix stability: {e}")
            return 1.0

    def _align_features_with_data(
        self, features: pd.DataFrame, original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align features DataFrame with original data structure."""
        try:
            # Create a mapping from timestamp to features
            if "timestamp" in original_data.columns:
                # Expand features to match all rows in original data
                expanded_features = original_data[["timestamp"]].merge(
                    features.reset_index().rename(columns={"index": "timestamp"}),
                    on="timestamp",
                    how="left",
                )

                # Set index to match original data
                expanded_features = expanded_features.drop("timestamp", axis=1)
                expanded_features.index = original_data.index

                return expanded_features
            else:
                # If no timestamp column, try to align by index
                return features.reindex(original_data.index, fill_value=0.0)

        except Exception as e:
            logger.warning(f"Error aligning features with data: {e}")
            return features
