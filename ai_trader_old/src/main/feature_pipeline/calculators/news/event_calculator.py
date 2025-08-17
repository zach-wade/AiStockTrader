"""
News Event Calculator

Specialized calculator for news event detection and analysis including:
- Event categorization (earnings, mergers, regulatory, product launches)
- Event intensity and significance scoring
- Event clustering and timeline analysis
- Anomaly detection in news patterns
- Breaking news identification
- Event impact measurement
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

from .base_news import BaseNewsCalculator
from ..helpers import create_rolling_features

logger = logging.getLogger(__name__)


class NewsEventCalculator(BaseNewsCalculator):
    """Calculator for news event detection and analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize news event calculator."""
        super().__init__(config)

        # Event-specific parameters
        self.event_threshold = self.news_config.event_threshold
        self.min_articles_for_event = self.news_config.event_min_articles
        self.event_time_cluster = self.news_config.event_time_cluster
        self.anomaly_window = self.news_config.anomaly_detection_window

        # Event category definitions
        self.event_categories = {
            "earnings": [
                "earnings",
                "revenue",
                "eps",
                "guidance",
                "quarter",
                "beat",
                "miss",
                "forecast",
                "outlook",
            ],
            "merger": [
                "merger",
                "acquisition",
                "acquire",
                "buyout",
                "takeover",
                "deal",
                "consolidation",
            ],
            "regulatory": [
                "sec",
                "investigation",
                "probe",
                "lawsuit",
                "fine",
                "penalty",
                "regulation",
                "compliance",
                "audit",
            ],
            "product": [
                "launch",
                "release",
                "announce",
                "unveil",
                "introduce",
                "new product",
                "innovation",
            ],
            "management": [
                "ceo",
                "cfo",
                "resign",
                "appoint",
                "hire",
                "fire",
                "management change",
                "executive",
            ],
            "analyst": [
                "upgrade",
                "downgrade",
                "initiate",
                "reiterate",
                "price target",
                "analyst",
                "rating",
            ],
            "financial": ["dividend", "split", "buyback", "debt", "financing", "ipo", "offering"],
            "crisis": [
                "investigation",
                "scandal",
                "fraud",
                "bankruptcy",
                "default",
                "crisis",
                "emergency",
            ],
        }

    def get_feature_names(self) -> list[str]:
        """Return list of news event feature names."""
        feature_names = []

        # Event category features
        for category in self.event_categories.keys():
            feature_names.extend(
                [
                    f"news_event_{category}_count_24h",
                    f"news_event_{category}_score_24h",
                    f"news_event_{category}_intensity",
                    f"news_event_{category}_trend_72h",
                ]
            )

        # Composite event features
        feature_names.extend(
            [
                "news_event_total_intensity",
                "news_event_diversity_24h",
                "news_event_spike_24h",
                "news_event_cluster_size",
                "news_event_significance_score",
                "news_breaking_news_count_1h",
                "news_breaking_news_count_6h",
                "news_breaking_news_intensity",
                "news_anomaly_score_24h",
                "news_event_momentum_24h",
                "news_event_persistence_72h",
                "news_event_recency_weight",
                "news_major_event_indicator",
                "news_event_sentiment_correlation",
                "news_event_volume_correlation",
            ]
        )

        return feature_names

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate news event features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with news event features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)

            if self.news_data is None or self.news_data.empty:
                logger.warning("No news data available for event calculation")
                return features

            # Calculate event category features
            features = self._calculate_event_category_features(data, features)

            # Calculate breaking news features
            features = self._calculate_breaking_news_features(data, features)

            # Calculate event composite features
            features = self._calculate_event_composite_features(features)

            # Calculate anomaly detection features
            features = self._calculate_anomaly_features(data, features)

            # Calculate event dynamics features
            features = self._calculate_event_dynamics_features(features)

            return features

        except Exception as e:
            logger.error(f"Error calculating news event features: {e}")
            return self.create_empty_features(data.index)

    def _calculate_event_category_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate features for each event category."""

        for category, keywords in self.event_categories.items():

            # 24h event features
            def event_count_24h_func(timestamp):
                return self._count_event_articles(timestamp, keywords, 24)

            def event_score_24h_func(timestamp):
                return self._calculate_event_score(timestamp, keywords, 24)

            def event_intensity_func(timestamp):
                return self._calculate_event_intensity(timestamp, keywords, 24)

            # Calculate for all timestamps
            event_counts = [event_count_24h_func(ts) for ts in data.index]
            event_scores = [event_score_24h_func(ts) for ts in data.index]
            event_intensities = [event_intensity_func(ts) for ts in data.index]

            features[f"news_event_{category}_count_24h"] = event_counts
            features[f"news_event_{category}_score_24h"] = event_scores
            features[f"news_event_{category}_intensity"] = event_intensities

            # 72h trend
            score_series = pd.Series(event_scores, index=data.index)
            features[f"news_event_{category}_trend_72h"] = self._calculate_trend(
                score_series, window=72
            )

        return features

    def _calculate_breaking_news_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate breaking news detection features."""

        # Breaking news count (1h and 6h)
        def breaking_count_1h_func(timestamp):
            return self._count_breaking_news(timestamp, 1)

        def breaking_count_6h_func(timestamp):
            return self._count_breaking_news(timestamp, 6)

        def breaking_intensity_func(timestamp):
            return self._calculate_breaking_intensity(timestamp, 6)

        breaking_1h = [breaking_count_1h_func(ts) for ts in data.index]
        breaking_6h = [breaking_count_6h_func(ts) for ts in data.index]
        breaking_intensities = [breaking_intensity_func(ts) for ts in data.index]

        features["news_breaking_news_count_1h"] = breaking_1h
        features["news_breaking_news_count_6h"] = breaking_6h
        features["news_breaking_news_intensity"] = breaking_intensities

        return features

    def _calculate_event_composite_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite event features."""

        # Total event intensity (sum across all categories)
        intensity_cols = [
            col
            for col in features.columns
            if "event_" in col and "_intensity" in col and "breaking" not in col
        ]
        if intensity_cols:
            features["news_event_total_intensity"] = features[intensity_cols].sum(axis=1)
        else:
            features["news_event_total_intensity"] = 0.0

        # Event diversity (how many different event types)
        score_cols = [col for col in features.columns if "event_" in col and "_score_24h" in col]
        if score_cols:
            # Count number of active event categories (score > threshold)
            threshold = 0.1
            active_events = (features[score_cols] > threshold).sum(axis=1)
            features["news_event_diversity_24h"] = active_events
        else:
            features["news_event_diversity_24h"] = 0.0

        # Event spike detection
        if "news_event_total_intensity" in features.columns:
            # Spike when intensity is above mean + 2*std over rolling window
            # Create DataFrame for rolling features
            intensity_df = pd.DataFrame(
                {"intensity": features["news_event_total_intensity"]}, index=features.index
            )
            rolling_features = create_rolling_features(
                intensity_df,
                columns=["intensity"],
                windows=[168],
                operations=["mean", "std"],
                min_periods=24,
            )
            rolling_mean = rolling_features["intensity_rolling_mean_168"]
            rolling_std = rolling_features["intensity_rolling_std_168"]

            spike_threshold = rolling_mean + self.event_threshold * rolling_std
            features["news_event_spike_24h"] = (
                features["news_event_total_intensity"] > spike_threshold
            ).astype(int)
        else:
            features["news_event_spike_24h"] = 0

        # Event cluster size (events happening close in time)
        def event_cluster_func(timestamp):
            return self._calculate_event_cluster_size(timestamp, self.event_time_cluster)

        cluster_sizes = [event_cluster_func(ts) for ts in features.index]
        features["news_event_cluster_size"] = cluster_sizes

        # Event significance score (composite measure)
        if intensity_cols:
            # Combine intensity, diversity, and clustering
            intensity_norm = features["news_event_total_intensity"] / (
                features["news_event_total_intensity"].max() + 1e-10
            )
            diversity_norm = features["news_event_diversity_24h"] / (
                features["news_event_diversity_24h"].max() + 1e-10
            )
            cluster_norm = features["news_event_cluster_size"] / (
                features["news_event_cluster_size"].max() + 1e-10
            )

            features["news_event_significance_score"] = (
                intensity_norm + diversity_norm + cluster_norm
            ) / 3.0
        else:
            features["news_event_significance_score"] = 0.0

        return features

    def _calculate_anomaly_features(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate anomaly detection features."""

        def anomaly_score_func(timestamp):
            return self._calculate_anomaly_score(timestamp, self.anomaly_window)

        anomaly_scores = [anomaly_score_func(ts) for ts in data.index]
        features["news_anomaly_score_24h"] = anomaly_scores

        return features

    def _calculate_event_dynamics_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate event momentum and dynamics features."""

        # Event momentum (rate of change in event intensity)
        if "news_event_total_intensity" in features.columns:
            features["news_event_momentum_24h"] = features["news_event_total_intensity"].diff(
                periods=24
            )
        else:
            features["news_event_momentum_24h"] = 0.0

        # Event persistence (how long events stay active)
        def event_persistence_func(values):
            if len(values) < 3:
                return 0.0
            # Measure how many consecutive periods have above-average activity
            threshold = values.mean()
            consecutive_runs = []
            current_run = 0

            for val in values:
                if val > threshold:
                    current_run += 1
                else:
                    if current_run > 0:
                        consecutive_runs.append(current_run)
                    current_run = 0

            if current_run > 0:
                consecutive_runs.append(current_run)

            return max(consecutive_runs) if consecutive_runs else 0.0

        if "news_event_total_intensity" in features.columns:
            # Use custom implementation for persistence calculation
            persistence_values = pd.Series(index=features.index, dtype=float)
            intensity_series = features["news_event_total_intensity"]

            for i in range(len(intensity_series)):
                if i < 2:  # min_periods=3
                    persistence_values.iloc[i] = 0.0
                else:
                    window_start = max(0, i - 71)  # 72 window
                    window_data = intensity_series.iloc[window_start : i + 1]
                    if len(window_data) >= 3:
                        persistence_values.iloc[i] = event_persistence_func(window_data.values)
                    else:
                        persistence_values.iloc[i] = 0.0

            features["news_event_persistence_72h"] = persistence_values
        else:
            features["news_event_persistence_72h"] = 0.0

        # Event recency weight (recent events weighted more heavily)
        def recency_weight_func(timestamp):
            return self._calculate_event_recency_weight(timestamp, 24)

        recency_weights = [recency_weight_func(ts) for ts in features.index]
        features["news_event_recency_weight"] = recency_weights

        # Major event indicator (multiple criteria)
        if all(
            col in features.columns
            for col in [
                "news_event_significance_score",
                "news_event_spike_24h",
                "news_breaking_news_intensity",
            ]
        ):
            major_event_conditions = (
                (features["news_event_significance_score"] > 0.7)
                | (features["news_event_spike_24h"] == 1)
                | (features["news_breaking_news_intensity"] > 0.5)
            )
            features["news_major_event_indicator"] = major_event_conditions.astype(int)
        else:
            features["news_major_event_indicator"] = 0

        # Event-sentiment correlation
        def event_sentiment_corr_func(timestamp):
            return self._calculate_event_sentiment_correlation(timestamp, 24)

        correlations = [event_sentiment_corr_func(ts) for ts in features.index]
        features["news_event_sentiment_correlation"] = correlations

        # Event-volume correlation
        def event_volume_corr_func(timestamp):
            return self._calculate_event_volume_correlation(timestamp, 24)

        volume_correlations = [event_volume_corr_func(ts) for ts in features.index]
        features["news_event_volume_correlation"] = volume_correlations

        return features

    def _count_event_articles(
        self, timestamp: pd.Timestamp, keywords: list[str], window_hours: int
    ) -> int:
        """Count articles containing event keywords."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "text" not in window_news.columns:
                return 0

            event_count = 0
            for text in window_news["text"]:
                if pd.isna(text):
                    continue

                text_lower = str(text).lower()
                if any(keyword.lower() in text_lower for keyword in keywords):
                    event_count += 1

            return event_count

        except Exception as e:
            logger.warning(f"Error counting event articles: {e}")
            return 0

    def _calculate_event_score(
        self, timestamp: pd.Timestamp, keywords: list[str], window_hours: int
    ) -> float:
        """Calculate weighted event score."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "text" not in window_news.columns:
                return 0.0

            event_weights = []
            total_weight = 0.0

            for idx, text in enumerate(window_news["text"]):
                if pd.isna(text):
                    continue

                text_lower = str(text).lower()
                if any(keyword.lower() in text_lower for keyword in keywords):
                    # Get article weight
                    if "weight" in window_news.columns:
                        weight = window_news.iloc[idx]["weight"]
                    else:
                        weight = 1.0

                    event_weights.append(weight)

                total_weight += 1.0

            if total_weight > 0:
                return sum(event_weights) / total_weight
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating event score: {e}")
            return 0.0

    def _calculate_event_intensity(
        self, timestamp: pd.Timestamp, keywords: list[str], window_hours: int
    ) -> float:
        """Calculate event intensity (score weighted by sentiment magnitude)."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "text" not in window_news.columns:
                return 0.0

            intensity_scores = []

            for text in window_news["text"]:
                if pd.isna(text):
                    continue

                text_lower = str(text).lower()
                if any(keyword.lower() in text_lower for keyword in keywords):
                    # Get sentiment magnitude
                    sentiment = self.calculate_sentiment(text)
                    magnitude = abs(sentiment["polarity"]) * sentiment["subjectivity"]
                    intensity_scores.append(magnitude)

            return np.mean(intensity_scores) if intensity_scores else 0.0

        except Exception as e:
            logger.warning(f"Error calculating event intensity: {e}")
            return 0.0

    def _count_breaking_news(self, timestamp: pd.Timestamp, window_hours: int) -> int:
        """Count breaking news articles."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "text" not in window_news.columns:
                return 0

            breaking_count = 0
            for text in window_news["text"]:
                if pd.isna(text):
                    continue

                if self.news_config.is_breaking_news(str(text), window_hours):
                    breaking_count += 1

            return breaking_count

        except Exception as e:
            logger.warning(f"Error counting breaking news: {e}")
            return 0

    def _calculate_breaking_intensity(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate breaking news intensity."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "text" not in window_news.columns:
                return 0.0

            intensities = []
            total_articles = 0

            for text in window_news["text"]:
                if pd.isna(text):
                    continue

                total_articles += 1

                if self.news_config.is_breaking_news(str(text), window_hours):
                    # Breaking news intensity based on recency and sentiment
                    sentiment = self.calculate_sentiment(text)
                    magnitude = abs(sentiment["polarity"]) + sentiment["subjectivity"]
                    intensities.append(magnitude)

            if total_articles > 0:
                return sum(intensities) / total_articles
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating breaking intensity: {e}")
            return 0.0

    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend for a time series."""

        def trend_func(values):
            if len(values) < 3:
                return 0.0
            try:
                x = np.arange(len(values))
                y = values.values
                trend = np.polyfit(x, y, 1)[0]  # Slope
                return trend
            except (ValueError, TypeError, np.linalg.LinAlgError):
                return 0.0

        # Use custom implementation for trend calculation
        trend_values = pd.Series(index=series.index, dtype=float)

        for i in range(len(series)):
            if i < 2:  # min_periods=3
                trend_values.iloc[i] = 0.0
            else:
                window_start = max(0, i - window + 1)
                window_data = series.iloc[window_start : i + 1]
                if len(window_data) >= 3:
                    trend_values.iloc[i] = trend_func(window_data.values)
                else:
                    trend_values.iloc[i] = 0.0

        return trend_values

    def _calculate_event_cluster_size(self, timestamp: pd.Timestamp, cluster_hours: int) -> float:
        """Calculate size of event cluster around timestamp."""
        try:
            # Get news in clustering window
            cluster_news = self.get_news_in_window(timestamp, cluster_hours)

            if cluster_news.empty or "text" not in cluster_news.columns:
                return 0.0

            # Count articles with any event keywords
            all_keywords = []
            for keywords in self.event_categories.values():
                all_keywords.extend(keywords)

            event_articles = 0
            for text in cluster_news["text"]:
                if pd.isna(text):
                    continue

                text_lower = str(text).lower()
                if any(keyword.lower() in text_lower for keyword in all_keywords):
                    event_articles += 1

            # Normalize by time window
            return event_articles / cluster_hours if cluster_hours > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating event cluster size: {e}")
            return 0.0

    def _calculate_anomaly_score(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate anomaly score based on unusual news patterns."""
        try:
            # Get current and historical windows
            current_news = self.get_news_in_window(timestamp, 1)  # Last hour
            historical_news = self.get_news_in_window(timestamp, window_hours)

            if historical_news.empty:
                return 0.0

            # Current activity level
            current_activity = len(current_news)

            # Historical baseline (hourly average)
            historical_activity = len(historical_news) / window_hours

            # Calculate z-score
            if historical_activity > 0:
                # Simple anomaly score based on activity deviation
                anomaly_score = abs(current_activity - historical_activity) / (
                    historical_activity + 1
                )
                return min(anomaly_score, 5.0)  # Cap at 5.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating anomaly score: {e}")
            return 0.0

    def _calculate_event_recency_weight(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate recency-weighted event score."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or "timestamp" not in window_news.columns:
                return 0.0

            total_weight = 0.0
            total_articles = 0

            # Filter out rows with missing text
            valid_articles = window_news[window_news["text"].notna()]

            if not valid_articles.empty:
                # Calculate time weights vectorized
                hours_ago = (timestamp - valid_articles["timestamp"]).dt.total_seconds() / 3600
                time_weights = hours_ago.apply(self.news_config.calculate_time_decay_weight)

                total_weight = time_weights.sum()
                total_articles = len(valid_articles)

            return total_weight / total_articles if total_articles > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating event recency weight: {e}")
            return 0.0

    def _calculate_event_sentiment_correlation(
        self, timestamp: pd.Timestamp, window_hours: int
    ) -> float:
        """Calculate correlation between event presence and sentiment."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty or len(window_news) < 3 or "text" not in window_news.columns:
                return 0.0

            event_scores = []
            sentiments = []

            all_keywords = []
            for keywords in self.event_categories.values():
                all_keywords.extend(keywords)

            for text in window_news["text"]:
                if pd.isna(text):
                    continue

                # Calculate event score for article
                text_lower = str(text).lower()
                event_matches = sum(1 for keyword in all_keywords if keyword.lower() in text_lower)
                event_score = event_matches / len(all_keywords) if all_keywords else 0.0

                # Calculate sentiment
                sentiment = self.calculate_sentiment(text)

                event_scores.append(event_score)
                sentiments.append(sentiment["polarity"])

            if len(event_scores) >= 3:
                correlation = np.corrcoef(event_scores, sentiments)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating event-sentiment correlation: {e}")
            return 0.0

    def _calculate_event_volume_correlation(
        self, timestamp: pd.Timestamp, window_hours: int
    ) -> float:
        """Calculate correlation between event intensity and news volume."""
        try:
            # This is a simplified version - in practice would correlate with actual volume data
            window_news = self.get_news_in_window(timestamp, window_hours)

            if window_news.empty:
                return 0.0

            # Count events and total volume as proxy
            all_keywords = []
            for keywords in self.event_categories.values():
                all_keywords.extend(keywords)

            event_articles = 0
            total_articles = len(window_news)

            for text in window_news.get("text", []):
                if pd.isna(text):
                    continue

                text_lower = str(text).lower()
                if any(keyword.lower() in text_lower for keyword in all_keywords):
                    event_articles += 1

            # Simple ratio as correlation proxy
            return event_articles / total_articles if total_articles > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating event-volume correlation: {e}")
            return 0.0
