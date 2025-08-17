"""
Lead-Lag Calculator

Specialized calculator for temporal relationship analysis including:
- Lead-lag correlation analysis at multiple horizons
- Optimal lead/lag detection and timing
- Cross-asset timing relationships
- Momentum and reversal timing patterns
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

from .base_correlation import BaseCorrelationCalculator

logger = logging.getLogger(__name__)


class LeadLagCalculator(BaseCorrelationCalculator):
    """Calculator for lead-lag relationship analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize lead-lag calculator."""
        super().__init__(config)

        # Lead-lag specific parameters
        self.leadlag_config = self.correlation_config.get_window_config("lead_lag")
        self.lead_lag_periods = self.leadlag_config.get("lead_lag_periods", [-5, -3, -1, 1, 3, 5])
        self.max_lead_lag = self.leadlag_config.get("max_lead_lag", 10)

        logger.debug(f"Initialized LeadLagCalculator with {len(self.lead_lag_periods)} periods")

    def get_feature_names(self) -> list[str]:
        """Return list of lead-lag feature names."""
        feature_names = []

        # Lead-lag correlations for specific periods
        for period in self.lead_lag_periods:
            if period < 0:
                feature_names.append(f"lead_corr_{abs(period)}d")
            else:
                feature_names.append(f"lag_corr_{period}d")

        # Optimal lead-lag features
        feature_names.extend(
            [
                "max_lead_corr",
                "optimal_lead",
                "max_lag_corr",
                "optimal_lag",
                "lead_lag_asymmetry",
                "optimal_timing_strength",
            ]
        )

        # Cross-asset timing features
        feature_names.extend(
            [
                "sector_lead_strength",
                "sector_lag_strength",
                "size_lead_correlation",
                "momentum_lead_indicator",
                "reversal_lag_indicator",
            ]
        )

        # Timing dynamics
        feature_names.extend(
            [
                "timing_consistency_score",
                "lead_lag_stability_20d",
                "timing_regime_indicator",
                "cross_correlation_peak",
                "timing_signal_quality",
            ]
        )

        # Advanced timing features
        feature_names.extend(
            [
                "bidirectional_timing_score",
                "timing_seasonality_score",
                "lead_lag_concentration",
                "temporal_diversification",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lead-lag relationship features.

        Args:
            data: DataFrame with symbol, timestamp, close columns

        Returns:
            DataFrame with lead-lag features
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

            # Calculate basic lead-lag correlations
            features = self._calculate_basic_leadlag(processed_data, features)

            # Calculate optimal timing features
            features = self._calculate_optimal_timing(processed_data, features)

            # Calculate cross-asset timing
            features = self._calculate_cross_asset_timing(processed_data, features)

            # Calculate timing dynamics
            features = self._calculate_timing_dynamics(processed_data, features)

            # Calculate advanced timing features
            features = self._calculate_advanced_timing(processed_data, features)

            # Align features with original data
            if len(features) != len(data):
                features = self._align_features_with_data(features, data)

            return features

        except Exception as e:
            logger.error(f"Error calculating lead-lag features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_basic_leadlag(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic lead-lag correlations."""
        try:
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                return features

            # Get market proxy for lead-lag analysis
            market_returns = self.get_market_proxy(data)

            if market_returns.empty:
                return features

            # Calculate lead-lag correlations for each specified period
            lead_lag_correlations = {}

            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:  # Skip market proxy
                    continue

                symbol_returns = returns_pivot[symbol]

                # Calculate lead-lag correlations
                symbol_leadlag = self.calculate_lead_lag_correlation(
                    symbol_returns, market_returns, self.max_lead_lag
                )

                for period in self.lead_lag_periods:
                    if period in symbol_leadlag:
                        if period not in lead_lag_correlations:
                            lead_lag_correlations[period] = []
                        lead_lag_correlations[period].append(symbol_leadlag[period])

            # Aggregate lead-lag correlations across symbols
            for period in self.lead_lag_periods:
                if period in lead_lag_correlations:
                    avg_correlation = np.mean(lead_lag_correlations[period])

                    if period < 0:
                        feature_name = f"lead_corr_{abs(period)}d"
                    else:
                        feature_name = f"lag_corr_{period}d"

                    features[feature_name] = avg_correlation

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic lead-lag: {e}")
            return features

    def _calculate_optimal_timing(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimal lead-lag timing features."""
        try:
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                return features

            market_returns = self.get_market_proxy(data)

            if market_returns.empty:
                return features

            # Find optimal leads and lags for each symbol
            optimal_leads = []
            optimal_lags = []
            max_lead_corrs = []
            max_lag_corrs = []

            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:
                    continue

                symbol_returns = returns_pivot[symbol]

                # Calculate full lead-lag spectrum
                leadlag_corrs = self.calculate_lead_lag_correlation(
                    symbol_returns, market_returns, self.max_lead_lag
                )

                # Find optimal lead (negative periods)
                lead_corrs = {k: v for k, v in leadlag_corrs.items() if k < 0}
                if lead_corrs:
                    optimal_lead = min(lead_corrs.keys(), key=lambda x: -abs(lead_corrs[x]))
                    max_lead_corr = abs(lead_corrs[optimal_lead])

                    optimal_leads.append(abs(optimal_lead))
                    max_lead_corrs.append(max_lead_corr)

                # Find optimal lag (positive periods)
                lag_corrs = {k: v for k, v in leadlag_corrs.items() if k > 0}
                if lag_corrs:
                    optimal_lag = min(lag_corrs.keys(), key=lambda x: -abs(lag_corrs[x]))
                    max_lag_corr = abs(lag_corrs[optimal_lag])

                    optimal_lags.append(optimal_lag)
                    max_lag_corrs.append(max_lag_corr)

            # Aggregate optimal timing features
            if optimal_leads:
                features["optimal_lead"] = np.median(optimal_leads)
                features["max_lead_corr"] = np.mean(max_lead_corrs)

            if optimal_lags:
                features["optimal_lag"] = np.median(optimal_lags)
                features["max_lag_corr"] = np.mean(max_lag_corrs)

            # Lead-lag asymmetry
            if max_lead_corrs and max_lag_corrs:
                lead_strength = np.mean(max_lead_corrs)
                lag_strength = np.mean(max_lag_corrs)
                asymmetry = lead_strength - lag_strength
                features["lead_lag_asymmetry"] = asymmetry

                # Optimal timing strength
                timing_strength = max(lead_strength, lag_strength)
                features["optimal_timing_strength"] = timing_strength

            return features

        except Exception as e:
            logger.warning(f"Error calculating optimal timing: {e}")
            return features

    def _calculate_cross_asset_timing(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate cross-asset timing relationships."""
        try:
            returns_pivot = self.pivot_returns_data(data)

            if returns_pivot.empty:
                return features

            # Sector lead-lag analysis
            sector_leads = []
            sector_lags = []

            for sector, symbols in self.correlation_config.sector_symbols.items():
                sector_returns = []

                for symbol in symbols:
                    if symbol in returns_pivot.columns:
                        sector_returns.append(returns_pivot[symbol])

                if len(sector_returns) >= 2:
                    # Calculate average sector return
                    sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)

                    # Calculate lead-lag with overall market
                    market_returns = self.get_market_proxy(data)

                    if not market_returns.empty:
                        leadlag_corrs = self.calculate_lead_lag_correlation(
                            sector_avg, market_returns, 5
                        )

                        # Find strongest lead and lag
                        lead_corrs = {k: abs(v) for k, v in leadlag_corrs.items() if k < 0}
                        lag_corrs = {k: abs(v) for k, v in leadlag_corrs.items() if k > 0}

                        if lead_corrs:
                            max_lead = max(lead_corrs.values())
                            sector_leads.append(max_lead)

                        if lag_corrs:
                            max_lag = max(lag_corrs.values())
                            sector_lags.append(max_lag)

            # Aggregate sector timing
            if sector_leads:
                features["sector_lead_strength"] = np.mean(sector_leads)

            if sector_lags:
                features["sector_lag_strength"] = np.mean(sector_lags)

            # Size-based lead-lag analysis
            size_lead_corrs = []

            for size_category, symbols in self.correlation_config.size_symbols.items():
                size_returns = []

                for symbol in symbols:
                    if symbol in returns_pivot.columns:
                        size_returns.append(returns_pivot[symbol])

                if len(size_returns) >= 1:
                    size_avg = pd.concat(size_returns, axis=1).mean(axis=1)
                    market_returns = self.get_market_proxy(data)

                    if not market_returns.empty:
                        # Calculate 1-day lead correlation
                        lead_corr = self.safe_correlation(size_avg.shift(1), market_returns)
                        size_lead_corrs.append(abs(lead_corr))

            if size_lead_corrs:
                features["size_lead_correlation"] = np.mean(size_lead_corrs)

            # Momentum and reversal timing indicators
            momentum_leads = []
            reversal_lags = []

            # Simplified momentum/reversal detection
            market_returns = self.get_market_proxy(data)

            if not market_returns.empty:
                # Momentum periods (consecutive positive returns)
                momentum_mask = (market_returns > 0) & (market_returns.shift(1) > 0)

                # Reversal periods (direction changes)
                reversal_mask = (market_returns > 0) & (market_returns.shift(1) < 0)
                reversal_mask |= (market_returns < 0) & (market_returns.shift(1) > 0)

                for symbol in returns_pivot.columns:
                    if symbol in market_returns.index:
                        continue

                    symbol_returns = returns_pivot[symbol]

                    # Momentum lead correlation
                    if momentum_mask.any():
                        momentum_periods = symbol_returns[momentum_mask]
                        market_momentum = market_returns[momentum_mask]

                        if len(momentum_periods) > 5:
                            momentum_lead = self.safe_correlation(
                                momentum_periods.shift(1), market_momentum
                            )
                            momentum_leads.append(abs(momentum_lead))

                    # Reversal lag correlation
                    if reversal_mask.any():
                        reversal_periods = symbol_returns[reversal_mask]
                        market_reversal = market_returns[reversal_mask]

                        if len(reversal_periods) > 5:
                            reversal_lag = self.safe_correlation(
                                reversal_periods, market_reversal.shift(1)
                            )
                            reversal_lags.append(abs(reversal_lag))

            if momentum_leads:
                features["momentum_lead_indicator"] = np.mean(momentum_leads)

            if reversal_lags:
                features["reversal_lag_indicator"] = np.mean(reversal_lags)

            return features

        except Exception as e:
            logger.warning(f"Error calculating cross-asset timing: {e}")
            return features

    def _calculate_timing_dynamics(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate timing consistency and dynamics."""
        try:
            # This would be calculated using rolling windows in practice
            # For now, using simplified static calculations

            # Timing consistency score
            lead_features = [col for col in features.columns if "lead_corr" in col]
            lag_features = [col for col in features.columns if "lag_corr" in col]

            if lead_features:
                lead_values = [
                    features[col].iloc[0] if len(features) > 0 else 0.0 for col in lead_features
                ]
                lead_consistency = 1.0 - np.std(lead_values) if lead_values else 0.0
                features["timing_consistency_score"] = lead_consistency

            # Lead-lag stability (would be rolling in practice)
            if "optimal_timing_strength" in features.columns:
                # Simplified stability score
                timing_strength = (
                    features["optimal_timing_strength"].iloc[0] if len(features) > 0 else 0.0
                )
                stability_score = min(timing_strength * 2, 1.0)  # Simplified
                features["lead_lag_stability_20d"] = stability_score

            # Timing regime indicator
            if "lead_lag_asymmetry" in features.columns:
                asymmetry = features["lead_lag_asymmetry"].iloc[0] if len(features) > 0 else 0.0

                if asymmetry > 0.1:
                    regime = 1  # Lead regime
                elif asymmetry < -0.1:
                    regime = -1  # Lag regime
                else:
                    regime = 0  # Balanced regime

                features["timing_regime_indicator"] = regime

            # Cross-correlation peak
            all_correlations = []
            for col in features.columns:
                if "corr" in col and col.endswith("d"):
                    val = features[col].iloc[0] if len(features) > 0 else 0.0
                    all_correlations.append(abs(val))

            if all_correlations:
                features["cross_correlation_peak"] = max(all_correlations)

            # Timing signal quality
            if (
                "optimal_timing_strength" in features.columns
                and "timing_consistency_score" in features.columns
            ):
                strength = features["optimal_timing_strength"].iloc[0] if len(features) > 0 else 0.0
                consistency = (
                    features["timing_consistency_score"].iloc[0] if len(features) > 0 else 0.0
                )
                signal_quality = (strength + consistency) / 2.0
                features["timing_signal_quality"] = signal_quality

            return features

        except Exception as e:
            logger.warning(f"Error calculating timing dynamics: {e}")
            return features

    def _calculate_advanced_timing(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced timing features."""
        try:
            # Bidirectional timing score
            if "max_lead_corr" in features.columns and "max_lag_corr" in features.columns:
                lead_strength = features["max_lead_corr"].iloc[0] if len(features) > 0 else 0.0
                lag_strength = features["max_lag_corr"].iloc[0] if len(features) > 0 else 0.0

                # Bidirectional strength
                bidirectional_score = min(lead_strength, lag_strength)
                features["bidirectional_timing_score"] = bidirectional_score

            # Timing seasonality (simplified)
            returns_pivot = self.pivot_returns_data(data)

            if not returns_pivot.empty:
                # Day of week effect on timing
                if "timestamp" in data.columns:
                    data_with_dow = data.copy()
                    data_with_dow["day_of_week"] = pd.to_datetime(
                        data_with_dow["timestamp"]
                    ).dt.dayofweek

                    # Calculate average timing strength by day of week
                    dow_timing = []
                    for dow in range(5):  # Monday to Friday
                        dow_data = data_with_dow[data_with_dow["day_of_week"] == dow]
                        if len(dow_data) > 10:
                            # Simplified timing strength for this day
                            dow_timing.append(len(dow_data) / len(data_with_dow))

                    if dow_timing:
                        seasonality_score = np.std(dow_timing)
                        features["timing_seasonality_score"] = seasonality_score

            # Lead-lag concentration
            all_leadlag_corrs = []
            for col in features.columns:
                if ("lead_corr" in col or "lag_corr" in col) and col.endswith("d"):
                    val = features[col].iloc[0] if len(features) > 0 else 0.0
                    all_leadlag_corrs.append(abs(val))

            if all_leadlag_corrs:
                # Concentration using Herfindahl index
                total_corr = sum(all_leadlag_corrs)
                if total_corr > 0:
                    proportions = [c / total_corr for c in all_leadlag_corrs]
                    concentration = sum(p**2 for p in proportions)
                    features["lead_lag_concentration"] = concentration

                # Temporal diversification (inverse of concentration)
                diversification = (
                    1.0 - concentration if "lead_lag_concentration" in features.columns else 0.0
                )
                features["temporal_diversification"] = diversification

            return features

        except Exception as e:
            logger.warning(f"Error calculating advanced timing: {e}")
            return features

    def _align_features_with_data(
        self, features: pd.DataFrame, original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align features DataFrame with original data structure."""
        try:
            if "timestamp" in original_data.columns:
                expanded_features = original_data[["timestamp"]].merge(
                    features.reset_index().rename(columns={"index": "timestamp"}),
                    on="timestamp",
                    how="left",
                )

                expanded_features = expanded_features.drop("timestamp", axis=1)
                expanded_features.index = original_data.index
                expanded_features = expanded_features.fillna(0.0)

                return expanded_features
            else:
                return features.reindex(original_data.index, fill_value=0.0)

        except Exception as e:
            logger.warning(f"Error aligning features with data: {e}")
            return features
