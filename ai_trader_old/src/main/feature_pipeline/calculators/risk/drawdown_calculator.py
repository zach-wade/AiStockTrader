"""
Drawdown Calculator

Specialized calculator for drawdown analysis including:
- Maximum drawdown and recovery analysis
- Drawdown duration and frequency
- Underwater curve analysis
- Drawdown distribution statistics
- Recovery time analysis
- Calmar ratio and drawdown-adjusted returns
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_risk import BaseRiskCalculator

logger = get_logger(__name__)


class DrawdownCalculator(BaseRiskCalculator):
    """Calculator for drawdown analysis and recovery metrics."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize drawdown calculator."""
        super().__init__(config)

        # Drawdown-specific configuration
        self.drawdown_config = self.risk_config.get_drawdown_config()
        self.lookback_window = self.drawdown_config["lookback_window"]
        self.min_observations = self.drawdown_config["min_observations"]
        self.recovery_threshold = self.drawdown_config["recovery_threshold"]

        logger.debug(f"Initialized DrawdownCalculator with {self.lookback_window}d lookback")

    def get_required_columns(self) -> list[str]:
        """Get list of required input columns."""
        return ["close"]

    def get_feature_names(self) -> list[str]:
        """Return list of drawdown feature names."""
        feature_names = [
            # Basic drawdown metrics
            "max_drawdown",
            "current_drawdown",
            "max_drawdown_duration",
            "current_drawdown_duration",
            "recovery_time",
            "time_underwater",
            "underwater_ratio",
            # Drawdown distribution
            "avg_drawdown",
            "drawdown_frequency",
            "drawdown_volatility",
            "drawdown_skewness",
            "drawdown_kurtosis",
            "drawdown_95_percentile",
            "drawdown_99_percentile",
            # Recovery analysis
            "avg_recovery_time",
            "recovery_factor",
            "recovery_efficiency",
            "max_recovery_time",
            "recovery_volatility",
            "partial_recovery_ratio",
            # Rolling drawdown metrics
            "max_drawdown_1m",
            "max_drawdown_3m",
            "max_drawdown_6m",
            "max_drawdown_1y",
            "drawdown_trend",
            "drawdown_acceleration",
            # Drawdown-adjusted performance
            "calmar_ratio",
            "sterling_ratio",
            "burke_ratio",
            "pain_index",
            "ulcer_index",
            "martin_ratio",
            # Drawdown risk metrics
            "drawdown_at_risk",
            "conditional_drawdown",
            "drawdown_beta",
            "drawdown_tracking_error",
            "drawdown_information_ratio",
            # Advanced drawdown metrics
            "drawdown_clustering",
            "drawdown_persistence",
            "drawdown_severity_index",
            "drawdown_surprise_index",
            "drawdown_stress_test",
        ]

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with drawdown features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)

            # Calculate returns and cumulative returns
            returns = self.calculate_returns(data)

            if len(returns) < self.min_observations:
                logger.warning("Insufficient data for drawdown calculation")
                return self.create_empty_features(data.index)

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()

            # Create features DataFrame
            features = self.create_empty_features(data.index)

            # Calculate drawdown metrics
            features = self._calculate_basic_drawdown_metrics(cumulative_returns, features)
            features = self._calculate_drawdown_distribution(cumulative_returns, features)
            features = self._calculate_recovery_analysis(cumulative_returns, returns, features)
            features = self._calculate_rolling_drawdown_metrics(cumulative_returns, features)
            features = self._calculate_drawdown_adjusted_performance(
                cumulative_returns, returns, features
            )
            features = self._calculate_drawdown_risk_metrics(cumulative_returns, returns, features)
            features = self._calculate_advanced_drawdown_metrics(
                cumulative_returns, returns, features
            )

            return features

        except Exception as e:
            logger.error(f"Error calculating drawdown features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_drawdown_series(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        try:
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            return drawdown

        except Exception as e:
            logger.error(f"Error calculating drawdown series: {e}")
            return pd.Series(dtype=float)

    def _calculate_basic_drawdown_metrics(
        self, cumulative_returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate basic drawdown metrics."""
        try:
            # Calculate drawdown series
            drawdown = self._calculate_drawdown_series(cumulative_returns)

            if len(drawdown) == 0:
                return features

            # Maximum drawdown
            max_drawdown = drawdown.min()
            features["max_drawdown"] = abs(max_drawdown)

            # Current drawdown
            current_drawdown = drawdown.iloc[-1]
            features["current_drawdown"] = abs(current_drawdown)

            # Drawdown duration analysis
            underwater_periods = self._get_underwater_periods(drawdown)

            if underwater_periods:
                # Maximum drawdown duration
                max_duration = max(period["duration"] for period in underwater_periods)
                features["max_drawdown_duration"] = max_duration

                # Average recovery time
                recovery_times = [
                    period["recovery_time"]
                    for period in underwater_periods
                    if period["recovery_time"] is not None
                ]
                if recovery_times:
                    features["recovery_time"] = np.mean(recovery_times)

                # Time underwater
                total_underwater_periods = sum(period["duration"] for period in underwater_periods)
                features["time_underwater"] = total_underwater_periods
                features["underwater_ratio"] = total_underwater_periods / len(drawdown)

            # Current drawdown duration
            if current_drawdown < -0.001:  # Currently in drawdown
                current_duration = self._get_current_drawdown_duration(drawdown)
                features["current_drawdown_duration"] = current_duration
            else:
                features["current_drawdown_duration"] = 0

            return features

        except Exception as e:
            logger.warning(f"Error calculating basic drawdown metrics: {e}")
            return features

    def _get_underwater_periods(self, drawdown: pd.Series) -> list[dict]:
        """Get underwater periods (drawdown cycles)."""
        try:
            periods = []
            in_drawdown = False
            start_idx = None
            peak_idx = None
            min_drawdown = 0

            for i, dd in enumerate(drawdown):
                if dd < -0.001 and not in_drawdown:  # Start of drawdown
                    in_drawdown = True
                    start_idx = i
                    peak_idx = i
                    min_drawdown = dd
                elif dd < min_drawdown and in_drawdown:  # Deeper drawdown
                    min_drawdown = dd
                    peak_idx = i
                elif dd >= -0.001 and in_drawdown:  # Recovery
                    in_drawdown = False

                    # Calculate recovery time
                    recovery_time = i - peak_idx if peak_idx is not None else None

                    periods.append(
                        {
                            "start": start_idx,
                            "peak": peak_idx,
                            "end": i,
                            "duration": i - start_idx,
                            "max_drawdown": abs(min_drawdown),
                            "recovery_time": recovery_time,
                        }
                    )

            # Handle case where we end in drawdown
            if in_drawdown and start_idx is not None:
                periods.append(
                    {
                        "start": start_idx,
                        "peak": peak_idx,
                        "end": len(drawdown) - 1,
                        "duration": len(drawdown) - 1 - start_idx,
                        "max_drawdown": abs(min_drawdown),
                        "recovery_time": None,  # Still in drawdown
                    }
                )

            return periods

        except Exception as e:
            logger.warning(f"Error getting underwater periods: {e}")
            return []

    def _get_current_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Get current drawdown duration."""
        try:
            duration = 0
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown.iloc[i] < -0.001:
                    duration += 1
                else:
                    break
            return duration

        except Exception as e:
            logger.warning(f"Error getting current drawdown duration: {e}")
            return 0

    def _calculate_drawdown_distribution(
        self, cumulative_returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate drawdown distribution statistics."""
        try:
            drawdown = self._calculate_drawdown_series(cumulative_returns)

            if len(drawdown) == 0:
                return features

            # Filter out zero drawdowns for distribution analysis
            non_zero_drawdowns = drawdown[drawdown < -0.001]

            if len(non_zero_drawdowns) > 0:
                # Distribution statistics
                features["avg_drawdown"] = abs(non_zero_drawdowns.mean())
                features["drawdown_frequency"] = len(non_zero_drawdowns) / len(drawdown)
                features["drawdown_volatility"] = non_zero_drawdowns.std()
                features["drawdown_skewness"] = non_zero_drawdowns.skew()
                features["drawdown_kurtosis"] = non_zero_drawdowns.kurtosis()

                # Percentiles
                features["drawdown_95_percentile"] = abs(
                    non_zero_drawdowns.quantile(0.05)
                )  # 5th percentile
                features["drawdown_99_percentile"] = abs(
                    non_zero_drawdowns.quantile(0.01)
                )  # 1st percentile

            return features

        except Exception as e:
            logger.warning(f"Error calculating drawdown distribution: {e}")
            return features

    def _calculate_recovery_analysis(
        self, cumulative_returns: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate recovery analysis metrics."""
        try:
            drawdown = self._calculate_drawdown_series(cumulative_returns)
            underwater_periods = self._get_underwater_periods(drawdown)

            if not underwater_periods:
                return features

            # Recovery time analysis
            recovery_times = [
                period["recovery_time"]
                for period in underwater_periods
                if period["recovery_time"] is not None
            ]

            if recovery_times:
                features["avg_recovery_time"] = np.mean(recovery_times)
                features["max_recovery_time"] = max(recovery_times)
                features["recovery_volatility"] = np.std(recovery_times)

            # Recovery factor (gain needed to recover from max drawdown)
            max_drawdown = features.get("max_drawdown", 0)
            if max_drawdown > 0:
                recovery_factor = max_drawdown / (1 - max_drawdown)
                features["recovery_factor"] = recovery_factor

            # Recovery efficiency (average return during recovery periods)
            recovery_returns = []
            for period in underwater_periods:
                if period["recovery_time"] is not None:
                    start_recovery = period["peak"]
                    end_recovery = period["end"]
                    if start_recovery < len(returns) and end_recovery < len(returns):
                        period_returns = returns.iloc[start_recovery:end_recovery]
                        if len(period_returns) > 0:
                            recovery_returns.extend(period_returns.tolist())

            if recovery_returns:
                features["recovery_efficiency"] = np.mean(recovery_returns)

            # Partial recovery ratio
            completed_recoveries = len(
                [p for p in underwater_periods if p["recovery_time"] is not None]
            )
            total_drawdown_periods = len(underwater_periods)
            if total_drawdown_periods > 0:
                features["partial_recovery_ratio"] = completed_recoveries / total_drawdown_periods

            return features

        except Exception as e:
            logger.warning(f"Error calculating recovery analysis: {e}")
            return features

    def _calculate_rolling_drawdown_metrics(
        self, cumulative_returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate rolling drawdown metrics."""
        try:
            # Different time windows (trading days)
            windows = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}

            for period_name, window in windows.items():
                if len(cumulative_returns) >= window:
                    # Calculate rolling maximum drawdown
                    rolling_max_dd = []

                    for i in range(window, len(cumulative_returns) + 1):
                        period_returns = cumulative_returns.iloc[i - window : i]
                        period_drawdown = self._calculate_drawdown_series(period_returns)
                        if len(period_drawdown) > 0:
                            rolling_max_dd.append(abs(period_drawdown.min()))

                    if rolling_max_dd:
                        features[f"max_drawdown_{period_name}"] = rolling_max_dd[-1]

            # Drawdown trend and acceleration
            if len(cumulative_returns) >= 60:
                recent_drawdown = self._calculate_drawdown_series(cumulative_returns.iloc[-60:])
                if len(recent_drawdown) > 10:
                    # Calculate trend
                    x = np.arange(len(recent_drawdown))
                    try:
                        trend = np.polyfit(x, recent_drawdown, 1)[0]
                        features["drawdown_trend"] = trend

                        # Calculate acceleration (second derivative)
                        if len(recent_drawdown) > 20:
                            acceleration = np.polyfit(x, recent_drawdown, 2)[0]
                            features["drawdown_acceleration"] = acceleration
                    except (ValueError, TypeError, np.linalg.LinAlgError):
                        features["drawdown_trend"] = 0.0
                        features["drawdown_acceleration"] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Error calculating rolling drawdown metrics: {e}")
            return features

    def _calculate_drawdown_adjusted_performance(
        self, cumulative_returns: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate drawdown-adjusted performance metrics."""
        try:
            # Annualized return
            total_return = cumulative_returns.iloc[-1] - 1
            years = len(returns) / 252
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0

            max_drawdown = features.get("max_drawdown", 0)

            # Calmar ratio (annualized return / max drawdown)
            if max_drawdown > 0:
                features["calmar_ratio"] = annualized_return / max_drawdown

            # Sterling ratio (modified Calmar with average of worst drawdowns)
            drawdown = self._calculate_drawdown_series(cumulative_returns)
            if len(drawdown) > 0:
                # Average of worst 3 drawdowns
                worst_drawdowns = drawdown.nsmallest(min(3, len(drawdown)))
                if len(worst_drawdowns) > 0:
                    avg_worst_drawdown = abs(worst_drawdowns.mean())
                    if avg_worst_drawdown > 0:
                        features["sterling_ratio"] = annualized_return / avg_worst_drawdown

            # Burke ratio (return / square root of sum of squared drawdowns)
            if len(drawdown) > 0:
                sum_squared_drawdowns = (drawdown**2).sum()
                if sum_squared_drawdowns > 0:
                    features["burke_ratio"] = annualized_return / np.sqrt(sum_squared_drawdowns)

            # Pain index (average drawdown)
            if len(drawdown) > 0:
                pain_index = abs(drawdown.mean())
                features["pain_index"] = pain_index

            # Ulcer index (square root of average squared drawdown)
            if len(drawdown) > 0:
                ulcer_index = np.sqrt((drawdown**2).mean())
                features["ulcer_index"] = ulcer_index

                # Martin ratio (return / ulcer index)
                if ulcer_index > 0:
                    features["martin_ratio"] = annualized_return / ulcer_index

            return features

        except Exception as e:
            logger.warning(f"Error calculating drawdown-adjusted performance: {e}")
            return features

    def _calculate_drawdown_risk_metrics(
        self, cumulative_returns: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate drawdown risk metrics."""
        try:
            drawdown = self._calculate_drawdown_series(cumulative_returns)

            if len(drawdown) == 0:
                return features

            # Drawdown at Risk (DaR) - similar to VaR but for drawdowns
            confidence_level = 0.95
            if len(drawdown) > 10:
                alpha = 1 - confidence_level
                dar = abs(drawdown.quantile(alpha))
                features["drawdown_at_risk"] = dar

            # Conditional Drawdown (CDD) - expected drawdown beyond DaR
            dar = features.get("drawdown_at_risk", 0)
            if dar > 0:
                extreme_drawdowns = drawdown[drawdown <= -dar]
                if len(extreme_drawdowns) > 0:
                    features["conditional_drawdown"] = abs(extreme_drawdowns.mean())

            # Drawdown beta (correlation with market-like benchmark)
            # For simplicity, assume market has similar characteristics
            if len(drawdown) > 30:
                # Create synthetic market drawdown
                market_returns = returns + secure_numpy_normal(0, 0.01, len(returns))  # Add noise
                market_cumulative = (1 + market_returns).cumprod()
                market_drawdown = self._calculate_drawdown_series(market_cumulative)

                if len(market_drawdown) == len(drawdown):
                    correlation = drawdown.corr(market_drawdown)
                    if not np.isnan(correlation):
                        # Beta = corr * (std_asset / std_market)
                        beta = correlation * (drawdown.std() / market_drawdown.std())
                        features["drawdown_beta"] = beta

            # Drawdown tracking error
            if len(drawdown) > 20:
                # Use rolling average as "benchmark"
                benchmark_drawdown = drawdown.rolling(window=20).mean()
                tracking_error = (drawdown - benchmark_drawdown).std()
                features["drawdown_tracking_error"] = tracking_error

                # Information ratio
                excess_drawdown = drawdown - benchmark_drawdown
                if tracking_error > 0:
                    features["drawdown_information_ratio"] = excess_drawdown.mean() / tracking_error

            return features

        except Exception as e:
            logger.warning(f"Error calculating drawdown risk metrics: {e}")
            return features

    def _calculate_advanced_drawdown_metrics(
        self, cumulative_returns: pd.Series, returns: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced drawdown metrics."""
        try:
            drawdown = self._calculate_drawdown_series(cumulative_returns)

            if len(drawdown) == 0:
                return features

            # Drawdown clustering (autocorrelation of drawdown states)
            drawdown_states = (drawdown < -0.01).astype(int)  # Binary: in drawdown or not
            if len(drawdown_states) > 10:
                try:
                    clustering = drawdown_states.autocorr(lag=1)
                    features["drawdown_clustering"] = clustering if not np.isnan(clustering) else 0
                except (ValueError, TypeError):
                    features["drawdown_clustering"] = 0

            # Drawdown persistence
            if len(drawdown) > 20:
                # Calculate autocorrelation of drawdown series
                try:
                    persistence = drawdown.autocorr(lag=1)
                    features["drawdown_persistence"] = (
                        persistence if not np.isnan(persistence) else 0
                    )
                except (ValueError, TypeError):
                    features["drawdown_persistence"] = 0

            # Drawdown severity index
            underwater_periods = self._get_underwater_periods(drawdown)
            if underwater_periods:
                # Weighted average of drawdown severity and duration
                severity_scores = []
                for period in underwater_periods:
                    severity = period["max_drawdown"] * np.sqrt(period["duration"])
                    severity_scores.append(severity)

                if severity_scores:
                    features["drawdown_severity_index"] = np.mean(severity_scores)

            # Drawdown surprise index (how often drawdowns exceed expectations)
            if len(drawdown) > 60:
                # Calculate rolling VaR-like measure for drawdowns
                rolling_expected_dd = []
                for i in range(30, len(drawdown)):
                    window_drawdown = drawdown.iloc[i - 30 : i]
                    expected_dd = window_drawdown.quantile(0.05)  # 5th percentile
                    rolling_expected_dd.append(expected_dd)

                if rolling_expected_dd and len(rolling_expected_dd) > 0:
                    actual_dd = drawdown.iloc[30:]
                    surprises = (
                        actual_dd < pd.Series(rolling_expected_dd, index=actual_dd.index)
                    ).sum()
                    features["drawdown_surprise_index"] = surprises / len(actual_dd)

            # Drawdown stress test (maximum potential drawdown based on volatility)
            if len(returns) > 60:
                volatility = returns.std()
                # Estimate maximum drawdown based on volatility (rough approximation)
                stress_dd = volatility * np.sqrt(252) * 2  # 2 standard deviations annualized
                features["drawdown_stress_test"] = stress_dd

            return features

        except Exception as e:
            logger.warning(f"Error calculating advanced drawdown metrics: {e}")
            return features
