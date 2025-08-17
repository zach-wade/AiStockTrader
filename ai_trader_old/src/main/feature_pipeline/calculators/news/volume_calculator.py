"""
News Volume Calculator

Analyzes news flow volume, velocity, and coverage patterns to identify
information flow intensity and potential market-moving news clusters.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

from .base_news import BaseNewsCalculator
from ..helpers import calculate_entropy, create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class VolumeCalculator(BaseNewsCalculator):
    """
    Calculates news volume and flow-based features.

    Features include:
    - News count and frequency metrics
    - News velocity (rate of change)
    - News acceleration
    - Coverage breadth and depth
    - News clustering and burst detection
    - Source diversity over time
    - Information flow patterns
    - News lifecycle metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize volume calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Volume-specific configuration
        self.volume_windows = config.get("volume_windows", [1, 6, 24, 72, 168])
        self.burst_threshold = config.get("burst_threshold", 2.0)  # Standard deviations
        self.cluster_time_threshold = config.get("cluster_time_threshold", 30)  # minutes

        # Rolling window sizes for dynamics
        self.velocity_windows = config.get("velocity_windows", [1, 6, 24])
        self.baseline_window = config.get("baseline_window", 168)  # 1 week baseline

        logger.info("Initialized VolumeCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of volume feature names."""
        features = []

        # Basic volume metrics for each window
        for window in self.volume_windows:
            window_suffix = self._get_window_suffix(window)
            features.extend(
                [
                    f"news_count_{window_suffix}",
                    f"news_frequency_{window_suffix}",
                    f"unique_sources_{window_suffix}",
                    f"avg_articles_per_source_{window_suffix}",
                    f"news_density_{window_suffix}",
                ]
            )

        # News velocity and acceleration
        for window in self.velocity_windows:
            window_suffix = self._get_window_suffix(window)
            features.extend(
                [
                    f"news_velocity_{window_suffix}",
                    f"news_acceleration_{window_suffix}",
                    f"velocity_ratio_{window_suffix}",
                ]
            )

        # Burst and clustering metrics
        features.extend(
            [
                "news_burst_indicator",
                "burst_intensity",
                "cluster_count",
                "avg_cluster_size",
                "max_cluster_size",
                "clustering_coefficient",
                "time_since_last_burst",
            ]
        )

        # Coverage metrics
        features.extend(
            [
                "coverage_breadth",
                "coverage_depth",
                "source_entropy",
                "source_concentration",
                "exclusive_coverage_ratio",
                "redundancy_score",
            ]
        )

        # Flow patterns
        features.extend(
            [
                "news_momentum",
                "flow_stability",
                "periodicity_score",
                "news_lifecycle_stage",
                "information_decay_rate",
            ]
        )

        # Composite scores
        features.extend(
            [
                "news_intensity_score",
                "information_flow_score",
                "coverage_quality_score",
                "news_saturation_index",
            ]
        )

        return features

    def calculate(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume features from news data.

        Args:
            news_data: DataFrame with news articles

        Returns:
            DataFrame with volume features
        """
        try:
            # Validate and prepare data
            validated_data = self.validate_and_prepare_data(news_data)
            if validated_data.empty:
                return self._create_empty_features(news_data.index)

            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)

            # Calculate basic volume metrics
            for window in self.volume_windows:
                window_features = self._calculate_window_volumes(validated_data, window)
                features = pd.concat([features, window_features], axis=1)

            # Calculate velocity and acceleration
            velocity_features = self._calculate_velocity_metrics(validated_data)
            features = pd.concat([features, velocity_features], axis=1)

            # Calculate burst and clustering metrics
            burst_features = self._calculate_burst_metrics(validated_data)
            features = pd.concat([features, burst_features], axis=1)

            # Calculate coverage metrics
            coverage_features = self._calculate_coverage_metrics(validated_data)
            features = pd.concat([features, coverage_features], axis=1)

            # Calculate flow patterns
            flow_features = self._calculate_flow_patterns(validated_data)
            features = pd.concat([features, flow_features], axis=1)

            # Calculate composite scores
            composite_features = self._calculate_composite_scores(features, validated_data)
            features = pd.concat([features, composite_features], axis=1)

            return features

        except Exception as e:
            logger.error(f"Error calculating volume features: {e}")
            return self._create_empty_features(news_data.index)

    def _calculate_window_volumes(self, news_data: pd.DataFrame, window_hours: int) -> pd.DataFrame:
        """Calculate volume metrics for a specific time window."""
        features = pd.DataFrame(index=news_data.index)
        window_suffix = self._get_window_suffix(window_hours)

        # Filter data within window
        window_data = self.filter_by_time_window(news_data, window_hours)

        if not window_data.empty:
            # Group by timestamp for calculations
            grouped = window_data.groupby(window_data.index)

            # News count
            features[f"news_count_{window_suffix}"] = grouped.size()

            # News frequency (articles per hour)
            features[f"news_frequency_{window_suffix}"] = safe_divide(
                features[f"news_count_{window_suffix}"], window_hours, default_value=0.0
            )

            # Unique sources
            features[f"unique_sources_{window_suffix}"] = grouped["source"].nunique()

            # Average articles per source
            features[f"avg_articles_per_source_{window_suffix}"] = safe_divide(
                features[f"news_count_{window_suffix}"], features[f"unique_sources_{window_suffix}"]
            )

            # News density (relative to baseline)
            baseline_count = self._get_baseline_count(news_data)
            features[f"news_density_{window_suffix}"] = safe_divide(
                features[f"news_count_{window_suffix}"], baseline_count
            )
        else:
            # Fill with zeros
            for col in [
                f"news_count_{window_suffix}",
                f"news_frequency_{window_suffix}",
                f"unique_sources_{window_suffix}",
                f"avg_articles_per_source_{window_suffix}",
                f"news_density_{window_suffix}",
            ]:
                features[col] = 0

        return features

    def _calculate_velocity_metrics(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate news velocity and acceleration metrics."""
        features = pd.DataFrame(index=news_data.index)

        # Calculate counts for different windows
        counts = {}
        for window in self.velocity_windows + [w * 2 for w in self.velocity_windows]:
            window_data = self.filter_by_time_window(news_data, window)
            counts[window] = window_data.groupby(window_data.index).size()

        # Calculate velocity for each window
        for window in self.velocity_windows:
            window_suffix = self._get_window_suffix(window)

            # Current period count
            current_count = counts[window]
            # Previous period count
            prev_count = counts[window * 2] - current_count
            prev_count = prev_count.clip(lower=0)

            # Velocity (rate of change)
            features[f"news_velocity_{window_suffix}"] = current_count - prev_count

            # Velocity ratio
            features[f"velocity_ratio_{window_suffix}"] = safe_divide(
                current_count, prev_count + 1  # Add 1 to avoid division by zero
            )

            # Acceleration (change in velocity)
            if window > 1:
                smaller_window = window // 2
                if smaller_window in self.velocity_windows:
                    features[f"news_acceleration_{window_suffix}"] = (
                        features[f"news_velocity_{self._get_window_suffix(smaller_window)}"]
                        - features[f"news_velocity_{window_suffix}"]
                    )
                else:
                    features[f"news_acceleration_{window_suffix}"] = 0
            else:
                features[f"news_acceleration_{window_suffix}"] = 0

        return features

    def _calculate_burst_metrics(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate news burst and clustering metrics."""
        features = pd.DataFrame(index=news_data.index)

        # Calculate baseline statistics
        baseline_data = self.filter_by_time_window(news_data, self.baseline_window)
        if not baseline_data.empty:
            hourly_counts = baseline_data.groupby(pd.Grouper(freq="1H")).size()
            baseline_mean = hourly_counts.mean()
            baseline_std = hourly_counts.std()
        else:
            baseline_mean = 1
            baseline_std = 1

        # Current hour count
        current_count = news_data.groupby(news_data.index).size()

        # Burst detection
        z_score = safe_divide(current_count - baseline_mean, baseline_std)
        features["news_burst_indicator"] = (z_score > self.burst_threshold).astype(int)
        features["burst_intensity"] = z_score.clip(lower=0)

        # Clustering analysis
        clusters = self._detect_news_clusters(news_data)
        features["cluster_count"] = clusters["count"]
        features["avg_cluster_size"] = clusters["avg_size"]
        features["max_cluster_size"] = clusters["max_size"]
        features["clustering_coefficient"] = clusters["coefficient"]

        # Time since last burst
        features["time_since_last_burst"] = self._calculate_time_since_burst(
            features["news_burst_indicator"]
        )

        return features

    def _calculate_coverage_metrics(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate news coverage breadth and depth metrics."""
        features = pd.DataFrame(index=news_data.index)

        # Use 24h window for coverage metrics
        window_data = self.filter_by_time_window(news_data, 24)

        if not window_data.empty:
            grouped = window_data.groupby(window_data.index)

            # Coverage breadth (number of unique sources)
            total_sources = window_data["source"].nunique()
            features["coverage_breadth"] = grouped["source"].nunique() / max(total_sources, 1)

            # Coverage depth (articles per topic)
            # Using headline similarity as proxy for topic
            features["coverage_depth"] = self._calculate_coverage_depth(window_data)

            # Source entropy
            source_counts = window_data["source"].value_counts()
            features["source_entropy"] = calculate_entropy(
                source_counts.values / source_counts.sum()
            )

            # Source concentration (Herfindahl index)
            source_shares = source_counts / source_counts.sum()
            features["source_concentration"] = (source_shares**2).sum()

            # Exclusive coverage ratio
            features["exclusive_coverage_ratio"] = self._calculate_exclusive_ratio(window_data)

            # Redundancy score
            features["redundancy_score"] = self._calculate_redundancy_score(window_data)
        else:
            # Fill with defaults
            features["coverage_breadth"] = 0
            features["coverage_depth"] = 0
            features["source_entropy"] = 0
            features["source_concentration"] = 1
            features["exclusive_coverage_ratio"] = 0
            features["redundancy_score"] = 0

        return features

    def _calculate_flow_patterns(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate news flow patterns and dynamics."""
        features = pd.DataFrame(index=news_data.index)

        # News momentum (weighted sum of recent changes)
        momentum = 0
        weights = [0.5, 0.3, 0.2]

        for i, window in enumerate(self.velocity_windows[:3]):
            window_suffix = self._get_window_suffix(window)
            velocity = features.get(f"news_velocity_{window_suffix}", 0)
            momentum += velocity * weights[i] if i < len(weights) else 0

        features["news_momentum"] = momentum

        # Flow stability (inverse of velocity variance)
        velocity_values = []
        for window in self.velocity_windows:
            window_suffix = self._get_window_suffix(window)
            vel = features.get(f"news_velocity_{window_suffix}", 0)
            velocity_values.append(vel)

        if len(velocity_values) > 1:
            velocity_std = np.std(velocity_values)
            features["flow_stability"] = 1 / (1 + velocity_std)
        else:
            features["flow_stability"] = 1

        # Periodicity score
        features["periodicity_score"] = self._calculate_periodicity(news_data)

        # News lifecycle stage
        features["news_lifecycle_stage"] = self._determine_lifecycle_stage(news_data, features)

        # Information decay rate
        features["information_decay_rate"] = self._calculate_decay_rate(news_data)

        return features

    def _calculate_composite_scores(
        self, features: pd.DataFrame, news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite volume scores."""
        composite = pd.DataFrame(index=features.index)

        # News intensity score
        intensity_components = [
            features.get("news_density_1h", 0) * 0.4,
            features.get("burst_intensity", 0) * 0.3,
            features.get("news_velocity_1h", 0) * 0.2,
            features.get("cluster_count", 0) * 0.1,
        ]
        composite["news_intensity_score"] = sum(intensity_components)

        # Information flow score
        flow_components = [
            features.get("news_momentum", 0) * 0.3,
            features.get("flow_stability", 0) * 0.3,
            features.get("coverage_breadth", 0) * 0.2,
            features.get("source_entropy", 0) * 0.2,
        ]
        composite["information_flow_score"] = sum(flow_components)

        # Coverage quality score
        quality_components = [
            features.get("coverage_depth", 0) * 0.3,
            features.get("exclusive_coverage_ratio", 0) * 0.3,
            (1 - features.get("redundancy_score", 0)) * 0.2,
            features.get("source_entropy", 0) * 0.2,
        ]
        composite["coverage_quality_score"] = sum(quality_components)

        # News saturation index
        saturation_components = [
            features.get("news_count_24h", 0) / 100,  # Normalize
            features.get("unique_sources_24h", 0) / 20,  # Normalize
            features.get("clustering_coefficient", 0),
            features.get("redundancy_score", 0),
        ]
        composite["news_saturation_index"] = np.mean(saturation_components)

        return composite

    def _get_baseline_count(self, news_data: pd.DataFrame) -> float:
        """Get baseline news count for normalization."""
        baseline_data = self.filter_by_time_window(news_data, self.baseline_window)
        if baseline_data.empty:
            return 1.0

        hourly_counts = baseline_data.groupby(pd.Grouper(freq="1H")).size()
        return hourly_counts.mean() if len(hourly_counts) > 0 else 1.0

    def _detect_news_clusters(self, news_data: pd.DataFrame) -> dict[str, float]:
        """Detect temporal clusters in news flow."""
        if news_data.empty:
            return {"count": 0, "avg_size": 0, "max_size": 0, "coefficient": 0}

        # Sort by timestamp
        sorted_data = news_data.sort_index()

        # Find clusters (articles within threshold minutes of each other)
        clusters = []
        current_cluster = []

        for i, (timestamp, _) in enumerate(sorted_data.iterrows()):
            if i == 0:
                current_cluster = [timestamp]
            else:
                time_diff = (timestamp - current_cluster[-1]).total_seconds() / 60

                if time_diff <= self.cluster_time_threshold:
                    current_cluster.append(timestamp)
                else:
                    if len(current_cluster) > 1:
                        clusters.append(len(current_cluster))
                    current_cluster = [timestamp]

        # Don't forget the last cluster
        if len(current_cluster) > 1:
            clusters.append(len(current_cluster))

        if clusters:
            return {
                "count": len(clusters),
                "avg_size": np.mean(clusters),
                "max_size": max(clusters),
                "coefficient": len(clusters) / len(news_data),
            }
        else:
            return {"count": 0, "avg_size": 0, "max_size": 0, "coefficient": 0}

    def _calculate_time_since_burst(self, burst_indicator: pd.Series) -> pd.Series:
        """Calculate time since last news burst."""
        time_since = pd.Series(index=burst_indicator.index, dtype=float)

        last_burst_time = None
        for timestamp, is_burst in burst_indicator.items():
            if is_burst:
                last_burst_time = timestamp
                time_since[timestamp] = 0
            elif last_burst_time is not None:
                hours_diff = (timestamp - last_burst_time).total_seconds() / 3600
                time_since[timestamp] = hours_diff
            else:
                time_since[timestamp] = np.inf

        return time_since.replace(np.inf, 168)  # Cap at 1 week

    def _calculate_coverage_depth(self, window_data: pd.DataFrame) -> pd.Series:
        """Calculate depth of coverage (articles per topic)."""
        # Simplified: use word overlap as topic similarity
        # In practice, you'd use more sophisticated topic modeling

        headlines = window_data["headline"].str.lower()

        # Count significant words (>4 chars) overlap
        depth_scores = []
        for idx in window_data.index.unique():
            idx_headlines = headlines[headlines.index == idx]

            if len(idx_headlines) > 1:
                # Calculate average overlap
                overlaps = []
                for i in range(len(idx_headlines)):
                    for j in range(i + 1, len(idx_headlines)):
                        words1 = set(idx_headlines.iloc[i].split())
                        words2 = set(idx_headlines.iloc[j].split())

                        # Filter significant words
                        words1 = {w for w in words1 if len(w) > 4}
                        words2 = {w for w in words2 if len(w) > 4}

                        if words1 and words2:
                            overlap = safe_divide(
                                len(words1 & words2),
                                min(len(words1), len(words2)),
                                default_value=0.0,
                            )
                            overlaps.append(overlap)

                depth = np.mean(overlaps) if overlaps else 0
            else:
                depth = 0

            depth_scores.append(depth)

        return pd.Series(depth_scores, index=window_data.index.unique()).reindex(
            window_data.index, method="ffill"
        )

    def _calculate_exclusive_ratio(self, window_data: pd.DataFrame) -> pd.Series:
        """Calculate ratio of exclusive (single-source) coverage."""
        # Group by approximate topic (using headline similarity)
        # Simplified implementation

        grouped = window_data.groupby(window_data.index)

        def calc_exclusive(group):
            if len(group) == 0:
                return 0

            # Count unique sources per "topic" (simplified)
            source_counts = group["source"].value_counts()
            exclusive_topics = (source_counts == 1).sum()

            return safe_divide(exclusive_topics, len(source_counts), default_value=0.0)

        return grouped.apply(calc_exclusive)

    def _calculate_redundancy_score(self, window_data: pd.DataFrame) -> pd.Series:
        """Calculate news redundancy score."""
        grouped = window_data.groupby(window_data.index)

        def calc_redundancy(group):
            if len(group) <= 1:
                return 0

            # Check headline similarity
            headlines = group["headline"].str.lower()

            # Count very similar headlines
            redundant_count = 0
            for i in range(len(headlines)):
                for j in range(i + 1, len(headlines)):
                    # Simple word overlap check
                    words1 = set(headlines.iloc[i].split())
                    words2 = set(headlines.iloc[j].split())

                    if len(words1) > 3 and len(words2) > 3:
                        overlap = safe_divide(
                            len(words1 & words2), min(len(words1), len(words2)), default_value=0.0
                        )
                        if overlap > 0.7:  # 70% similarity threshold
                            redundant_count += 1

            return safe_divide(
                redundant_count, (len(group) * (len(group) - 1) / 2), default_value=0.0
            )

        return grouped.apply(calc_redundancy)

    def _calculate_periodicity(self, news_data: pd.DataFrame) -> float:
        """Calculate periodicity score of news flow."""
        # Use FFT to detect periodic patterns
        # Simplified: check for daily patterns

        hourly_counts = news_data.groupby(pd.Grouper(freq="1H")).size()

        if len(hourly_counts) < 48:  # Need at least 2 days
            return 0

        # Check 24-hour periodicity
        daily_correlation = []
        for offset in range(24):
            shifted = hourly_counts.shift(offset)
            corr = hourly_counts.corr(shifted)
            daily_correlation.append(corr)

        # Peak at 24h indicates daily pattern
        if len(daily_correlation) >= 24:
            periodicity = daily_correlation[23] if not np.isnan(daily_correlation[23]) else 0
            return max(0, periodicity)

        return 0

    def _determine_lifecycle_stage(
        self, news_data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.Series:
        """Determine news lifecycle stage (emerging, peak, declining, dormant)."""
        # Based on velocity and acceleration
        velocity = features.get("news_velocity_6h", 0)
        acceleration = features.get("news_acceleration_6h", 0)

        stages = pd.Series(index=features.index, dtype=int)

        # Define stages
        # 0: Dormant, 1: Emerging, 2: Peak, 3: Declining
        conditions = [
            (velocity > 0) & (acceleration > 0),  # Emerging
            (velocity > 0) & (acceleration <= 0),  # Peak
            (velocity <= 0) & (acceleration < 0),  # Declining
            (velocity <= 0) & (acceleration >= 0),  # Dormant
        ]

        for i, condition in enumerate(conditions):
            if i == 0:
                stages[condition] = 1  # Emerging
            elif i == 1:
                stages[condition] = 2  # Peak
            elif i == 2:
                stages[condition] = 3  # Declining
            else:
                stages[condition] = 0  # Dormant

        return stages

    def _calculate_decay_rate(self, news_data: pd.DataFrame) -> float:
        """Calculate information decay rate."""
        # Measure how quickly news volume decreases after a peak

        hourly_counts = news_data.groupby(pd.Grouper(freq="1H")).size()

        if len(hourly_counts) < 24:
            return 0

        # Find peaks
        peaks = []
        for i in range(1, len(hourly_counts) - 1):
            if (
                hourly_counts.iloc[i] > hourly_counts.iloc[i - 1]
                and hourly_counts.iloc[i] > hourly_counts.iloc[i + 1]
            ):
                peaks.append(i)

        if not peaks:
            return 0

        # Calculate average decay after peaks
        decay_rates = []
        for peak_idx in peaks:
            if peak_idx + 6 < len(hourly_counts):  # Look 6 hours ahead
                peak_value = hourly_counts.iloc[peak_idx]
                future_value = hourly_counts.iloc[peak_idx + 6]

                if peak_value > 0:
                    decay = safe_divide(peak_value - future_value, peak_value, default_value=0.0)
                    decay_rates.append(max(0, decay))

        return np.mean(decay_rates) if decay_rates else 0
