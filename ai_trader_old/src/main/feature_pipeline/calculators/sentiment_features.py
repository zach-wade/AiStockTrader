"""
Sentiment Analysis Features Calculator

Calculates sentiment-based features from various sources including:
- News sentiment analysis
- Social media sentiment (Reddit, Twitter mentions)
- Options sentiment (put/call ratios, skew)
- Analyst ratings and target prices
- Insider trading sentiment
- Retail vs institutional sentiment
"""

# Standard library imports
from dataclasses import dataclass
from datetime import timedelta
import logging

# Third-party imports
import numpy as np
import pandas as pd
from textblob import TextBlob

from .base_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuration for sentiment calculations"""

    # Sentiment windows
    sentiment_windows: list[int] = None

    # News parameters
    news_lookback_days: int = 7
    news_decay_factor: float = 0.9

    # Social media parameters
    social_lookback_hours: int = 24
    mention_threshold: int = 10

    # Options parameters
    options_expiry_days: list[int] = None
    skew_percentiles: list[float] = None

    # Analyst parameters
    analyst_weight_days: int = 90

    def __post_init__(self):
        if self.sentiment_windows is None:
            self.sentiment_windows = [1, 3, 7, 14]
        if self.options_expiry_days is None:
            self.options_expiry_days = [7, 30, 60]
        if self.skew_percentiles is None:
            self.skew_percentiles = [25, 50, 75]


class SentimentFeaturesCalculator(BaseFeatureCalculator):
    """Calculator for sentiment-based features"""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.sentiment_config = SentimentConfig(**config.get("sentiment", {}) if config else {})
        self.news_data = None
        self.social_data = None
        self.options_data = None
        self.analyst_data = None
        self.sentiment_repository = None  # Optional database integration

    def set_news_data(self, news_data: pd.DataFrame):
        """Set news data for sentiment analysis"""
        self.news_data = news_data

    def set_social_data(self, social_data: pd.DataFrame):
        """Set social media data"""
        self.social_data = social_data

    def set_options_data(self, options_data: pd.DataFrame):
        """Set options data"""
        self.options_data = options_data

    def set_analyst_data(self, analyst_data: pd.DataFrame):
        """Set analyst ratings data"""
        self.analyst_data = analyst_data

    def set_sentiment_repository(self, sentiment_repository):
        """Set sentiment repository for database integration"""
        self.sentiment_repository = sentiment_repository

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with sentiment features
        """
        if not self.validate_inputs(data):
            logger.error("Invalid input data for sentiment calculation")
            return pd.DataFrame()

        features = pd.DataFrame(index=data.index)

        try:
            # News sentiment
            if self.news_data is not None:
                features = self._add_news_sentiment(data, features)

            # Social media sentiment
            if self.social_data is not None:
                features = self._add_social_sentiment(data, features)

            # Options sentiment
            if self.options_data is not None:
                features = self._add_options_sentiment(data, features)
            else:
                # Try to fetch basic options data
                features = self._add_basic_options_sentiment(data, features)

            # Analyst sentiment
            if self.analyst_data is not None:
                features = self._add_analyst_sentiment(data, features)

            # Price-based sentiment
            features = self._add_price_sentiment(data, features)

            # Database-sourced sentiment (if repository available)
            if self.sentiment_repository:
                features = self._add_database_sentiment(data, features)

            # Composite sentiment scores
            features = self._add_composite_sentiment(features)

            # Sentiment momentum
            features = self._add_sentiment_momentum(features)

            logger.info(f"Calculated {len(features.columns)} sentiment features")

        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")

        return features

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required = ["close"]
        missing = [col for col in required if col not in data.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        return True

    def get_required_columns(self) -> list[str]:
        """Return list of required input columns"""
        return ["open", "high", "low", "close", "volume"]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this calculator produces"""
        feature_names = []

        # News sentiment features
        for window in self.sentiment_config.sentiment_windows:
            feature_names.extend([f"news_sentiment_{window}d", f"news_volume_{window}d"])

        feature_names.append("news_sentiment_std")
        feature_names.append("news_keyword_sentiment")

        # Social media sentiment features
        for window in [1, 6, 12, 24]:  # Hours
            feature_names.extend([f"social_sentiment_{window}h", f"social_volume_{window}h"])

        feature_names.extend(
            [
                "mention_velocity",
                "mention_acceleration",
                "is_viral",
                "wsb_mentions",
                "wsb_trending",
                "reddit_upvote_ratio",
            ]
        )

        # Options sentiment features
        feature_names.extend(
            [
                "put_call_ratio",
                "put_call_ratio_ma",
                "abnormal_put_volume",
                "abnormal_call_volume",
                "iv_skew",
                "iv_skew_percentile",
                "net_options_flow",
                "options_flow_ma",
                "iv_term_slope",
                "backwardation",
            ]
        )

        # Options term structure features
        for days in self.sentiment_config.options_expiry_days:
            feature_names.append(f"iv_term_{days}d")

        # Basic options sentiment (when detailed data unavailable)
        feature_names.extend(
            ["implied_move_1w", "implied_move_1m", "vol_term_structure", "fear_gauge"]
        )

        # Analyst sentiment features
        feature_names.extend(
            [
                "analyst_consensus",
                "analyst_dispersion",
                "analyst_count",
                "upgrades",
                "downgrades",
                "net_rating_change",
                "avg_price_target",
                "price_target_upside",
                "price_target_dispersion",
                "targets_above_price",
            ]
        )

        # Price-based sentiment features
        for period in [5, 10, 20]:
            feature_names.extend([f"price_momentum_{period}d", f"momentum_percentile_{period}d"])

        feature_names.extend(
            [
                "near_20d_high",
                "near_20d_low",
                "near_52w_high",
                "near_52w_low",
                "volume_sentiment",
                "high_volume_up",
                "high_volume_down",
                "accumulation_distribution",
                "ad_divergence",
                "above_sma_20",
                "above_sma_50",
                "above_sma_200",
                "ma_breadth",
            ]
        )

        # Composite sentiment features
        feature_names.extend(
            [
                "composite_sentiment",
                "sentiment_dispersion",
                "sentiment_skew",
                "bullish_sentiment",
                "bearish_sentiment",
                "neutral_sentiment",
                "sentiment_agreement",
                "strong_consensus",
            ]
        )

        # Sentiment momentum features
        feature_names.extend(
            [
                "sentiment_momentum",
                "sentiment_acceleration",
                "sentiment_reversal_up",
                "sentiment_reversal_down",
            ]
        )

        # Database sentiment features
        feature_names.extend(
            [
                "db_sentiment_latest",
                "db_sentiment_mean_24h",
                "db_sentiment_std_24h",
                "db_sentiment_count_24h",
                "db_sentiment_mean_7d",
                "db_sentiment_std_7d",
                "db_sentiment_count_7d",
                "db_sentiment_source_count",
                "db_sentiment_momentum",
            ]
        )

        # Dynamic momentum features (these get created for each sentiment column)
        base_sentiment_cols = ["news_sentiment_1d", "social_sentiment_1h", "composite_sentiment"]
        for col in base_sentiment_cols:
            feature_names.append(f"{col}_momentum")

        return feature_names

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data before calculation"""
        data = super().preprocess_data(data)

        # Ensure we have volume column (needed for price sentiment)
        if "volume" not in data.columns:
            data["volume"] = 1  # Default volume if not available
            logger.debug("No volume data available, using default value")

        # Validate and prepare external sentiment data sources
        if self.news_data is not None:
            self.news_data = self._validate_news_data(self.news_data)

        if self.social_data is not None:
            self.social_data = self._validate_social_data(self.social_data)

        if self.options_data is not None:
            self.options_data = self._validate_options_data(self.options_data)

        if self.analyst_data is not None:
            self.analyst_data = self._validate_analyst_data(self.analyst_data)

        return data

    def postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Postprocess calculated features"""
        features = super().postprocess_features(features)

        if features.empty:
            return features

        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values based on feature type

        # Sentiment score features: fill with 0 (neutral sentiment)
        sentiment_columns = [
            col
            for col in features.columns
            if any(keyword in col for keyword in ["sentiment", "momentum"])
        ]
        features[sentiment_columns] = features[sentiment_columns].fillna(0)

        # Ratio features: fill with 1 (neutral ratio) for ratios, 0 for others
        ratio_columns = [
            col
            for col in features.columns
            if any(keyword in col for keyword in ["ratio", "upside", "dispersion"])
        ]
        features[ratio_columns] = features[ratio_columns].fillna(1)

        # Count and volume features: fill with 0
        count_columns = [
            col
            for col in features.columns
            if any(
                keyword in col
                for keyword in ["volume", "count", "mentions", "upgrades", "downgrades"]
            )
        ]
        features[count_columns] = features[count_columns].fillna(0)

        # Binary features: fill with 0 and convert to int
        binary_columns = [
            col
            for col in features.columns
            if any(
                keyword in col
                for keyword in [
                    "is_",
                    "near_",
                    "above_",
                    "trending",
                    "bullish",
                    "bearish",
                    "neutral",
                ]
            )
        ]
        for col in binary_columns:
            if col in features.columns:
                features[col] = features[col].fillna(0).astype(int)

        # Clip extreme values for better stability

        # Sentiment scores: clip to reasonable range
        sentiment_cols = [
            col for col in features.columns if "sentiment" in col and "momentum" not in col
        ]
        for col in sentiment_cols:
            if col in features.columns:
                features[col] = features[col].clip(-3, 3)

        # Momentum features: clip to prevent extreme values
        momentum_cols = [
            col for col in features.columns if "momentum" in col or "acceleration" in col
        ]
        for col in momentum_cols:
            if col in features.columns:
                features[col] = features[col].clip(-5, 5)

        # Ratio features: clip to reasonable bounds
        for col in ratio_columns:
            if col in features.columns:
                if "ratio" in col:
                    features[col] = features[col].clip(0, 10)  # Ratios should be positive
                else:
                    features[col] = features[col].clip(-5, 5)

        # Percentile features: ensure [0, 1] range
        percentile_columns = [col for col in features.columns if "percentile" in col]
        for col in percentile_columns:
            if col in features.columns:
                features[col] = features[col].clip(0, 1)

        logger.info(f"Postprocessed {len(features.columns)} sentiment features")
        return features

    def _add_news_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add news-based sentiment features"""
        # Filter news data to relevant time period
        end_date = data.index[-1]
        start_date = end_date - timedelta(days=self.sentiment_config.news_lookback_days)

        relevant_news = self.news_data[
            (self.news_data["timestamp"] >= start_date) & (self.news_data["timestamp"] <= end_date)
        ]

        if relevant_news.empty:
            return features

        # Calculate sentiment scores
        for window in self.sentiment_config.sentiment_windows:
            window_scores = []

            for idx in data.index:
                window_end = idx
                window_start = idx - timedelta(days=window)

                window_news = relevant_news[
                    (relevant_news["timestamp"] >= window_start)
                    & (relevant_news["timestamp"] <= window_end)
                ]

                if not window_news.empty:
                    # Calculate weighted sentiment
                    time_weights = self._calculate_time_decay_weights(
                        window_news["timestamp"],
                        window_end,
                        self.sentiment_config.news_decay_factor,
                    )

                    if "sentiment_score" in window_news.columns:
                        weighted_sentiment = (
                            window_news["sentiment_score"] * time_weights
                        ).sum() / time_weights.sum()
                    else:
                        # Calculate sentiment from text
                        sentiments = window_news["text"].apply(self._calculate_text_sentiment)
                        weighted_sentiment = (sentiments * time_weights).sum() / time_weights.sum()

                    window_scores.append(weighted_sentiment)
                else:
                    window_scores.append(0)

            features[f"news_sentiment_{window}d"] = window_scores

            # News volume
            news_counts = []
            for idx in data.index:
                window_end = idx
                window_start = idx - timedelta(days=window)
                count = len(
                    relevant_news[
                        (relevant_news["timestamp"] >= window_start)
                        & (relevant_news["timestamp"] <= window_end)
                    ]
                )
                news_counts.append(count)

            features[f"news_volume_{window}d"] = news_counts

        # News sentiment volatility
        features["news_sentiment_std"] = features[
            [col for col in features.columns if "news_sentiment" in col]
        ].std(axis=1)

        # Headline keyword analysis
        features = self._add_keyword_sentiment(relevant_news, data, features)

        return features

    def _add_social_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add social media sentiment features"""
        # Similar structure to news sentiment but with hourly granularity
        for window in [1, 6, 12, 24]:  # Hours
            window_sentiments = []
            window_volumes = []

            for idx in data.index:
                window_end = idx
                window_start = idx - timedelta(hours=window)

                window_social = self.social_data[
                    (self.social_data["timestamp"] >= window_start)
                    & (self.social_data["timestamp"] <= window_end)
                ]

                if not window_social.empty:
                    # Sentiment score
                    if "sentiment" in window_social.columns:
                        avg_sentiment = window_social["sentiment"].mean()
                    else:
                        sentiments = window_social["text"].apply(self._calculate_text_sentiment)
                        avg_sentiment = sentiments.mean()

                    window_sentiments.append(avg_sentiment)
                    window_volumes.append(len(window_social))
                else:
                    window_sentiments.append(0)
                    window_volumes.append(0)

            features[f"social_sentiment_{window}h"] = window_sentiments
            features[f"social_volume_{window}h"] = window_volumes

        # Reddit-specific features
        if "source" in self.social_data.columns:
            reddit_data = self.social_data[self.social_data["source"] == "reddit"]
            if not reddit_data.empty:
                features = self._add_reddit_features(reddit_data, data, features)

        # Mention velocity
        features["mention_velocity"] = features["social_volume_1h"].diff()
        features["mention_acceleration"] = features["mention_velocity"].diff()

        # Viral threshold
        features["is_viral"] = (
            features["social_volume_24h"] > self.sentiment_config.mention_threshold
        ).astype(int)

        return features

    def _add_options_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add options-based sentiment features"""
        # Put-Call Ratio
        if "put_volume" in self.options_data.columns and "call_volume" in self.options_data.columns:
            features["put_call_ratio"] = self.options_data["put_volume"] / (
                self.options_data["call_volume"] + 1
            )
            features["put_call_ratio_ma"] = features["put_call_ratio"].rolling(5).mean()

            # Abnormal put/call volume
            put_vol_mean = self.options_data["put_volume"].rolling(20).mean()
            call_vol_mean = self.options_data["call_volume"].rolling(20).mean()
            features["abnormal_put_volume"] = self.options_data["put_volume"] / (put_vol_mean + 1)
            features["abnormal_call_volume"] = self.options_data["call_volume"] / (
                call_vol_mean + 1
            )

        # Implied Volatility Skew
        if "iv_skew" in self.options_data.columns:
            features["iv_skew"] = self.options_data["iv_skew"]
            features["iv_skew_percentile"] = self.options_data["iv_skew"].rolling(60).rank(pct=True)

        # Options flow
        if (
            "call_premium" in self.options_data.columns
            and "put_premium" in self.options_data.columns
        ):
            features["net_options_flow"] = (
                self.options_data["call_premium"] - self.options_data["put_premium"]
            )
            features["options_flow_ma"] = features["net_options_flow"].rolling(5).mean()

        # Term structure sentiment
        for days in self.sentiment_config.options_expiry_days:
            if f"iv_{days}d" in self.options_data.columns:
                features[f"iv_term_{days}d"] = self.options_data[f"iv_{days}d"]

        # Calculate term structure slope
        if "iv_term_7d" in features.columns and "iv_term_30d" in features.columns:
            features["iv_term_slope"] = features["iv_term_30d"] - features["iv_term_7d"]
            features["backwardation"] = (features["iv_term_slope"] < 0).astype(int)

        return features

    def _add_basic_options_sentiment(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add basic options sentiment without detailed options data"""
        # This is a simplified version using available data
        close = data["close"]
        returns = close.pct_change()

        # Implied move (simplified using historical volatility as proxy)
        features["implied_move_1w"] = returns.rolling(5).std() * np.sqrt(5)
        features["implied_move_1m"] = returns.rolling(20).std() * np.sqrt(20)

        # Volatility term structure proxy
        features["vol_term_structure"] = features["implied_move_1m"] / features["implied_move_1w"]

        # Fear gauge (high short-term vol relative to long-term)
        features["fear_gauge"] = (
            features["implied_move_1w"] / features["implied_move_1m"].rolling(20).mean()
        )

        return features

    def _add_analyst_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add analyst-based sentiment features"""
        # Current consensus
        if "rating" in self.analyst_data.columns:
            latest_ratings = self.analyst_data.groupby("analyst")["rating"].last()
            features["analyst_consensus"] = latest_ratings.mean()
            features["analyst_dispersion"] = latest_ratings.std()
            features["analyst_count"] = len(latest_ratings)

        # Rating changes
        if "rating" in self.analyst_data.columns and "date" in self.analyst_data.columns:
            # Recent upgrades/downgrades
            recent_changes = self.analyst_data[
                self.analyst_data["date"] >= data.index[-1] - timedelta(days=30)
            ]

            if not recent_changes.empty:
                rating_changes = recent_changes.groupby("analyst")["rating"].diff()
                features["upgrades"] = (rating_changes > 0).sum()
                features["downgrades"] = (rating_changes < 0).sum()
                features["net_rating_change"] = rating_changes.sum()

        # Price targets
        if "price_target" in self.analyst_data.columns:
            current_price = data["close"].iloc[-1]
            latest_targets = self.analyst_data.groupby("analyst")["price_target"].last()

            features["avg_price_target"] = latest_targets.mean()
            features["price_target_upside"] = (
                features["avg_price_target"] - current_price
            ) / current_price
            features["price_target_dispersion"] = latest_targets.std() / latest_targets.mean()

            # Percentage of targets above current price
            features["targets_above_price"] = (latest_targets > current_price).mean()

        return features

    def _add_price_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add price action-based sentiment features"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data.get("volume", pd.Series(1, index=data.index))

        # Momentum sentiment
        for period in [5, 10, 20]:
            returns = close.pct_change(periods=period)
            features[f"price_momentum_{period}d"] = returns
            features[f"momentum_percentile_{period}d"] = returns.rolling(60).rank(pct=True)

        # New highs/lows sentiment
        rolling_max_20 = high.rolling(20).max()
        rolling_min_20 = low.rolling(20).min()
        rolling_max_52w = high.rolling(252).max()
        rolling_min_52w = low.rolling(252).min()

        features["near_20d_high"] = (close > rolling_max_20 * 0.97).astype(int)
        features["near_20d_low"] = (close < rolling_min_20 * 1.03).astype(int)
        features["near_52w_high"] = (close > rolling_max_52w * 0.95).astype(int)
        features["near_52w_low"] = (close < rolling_min_52w * 1.05).astype(int)

        # Volume sentiment
        avg_volume = volume.rolling(20).mean()
        features["volume_sentiment"] = volume / avg_volume
        features["high_volume_up"] = (
            (volume > avg_volume * 1.5) & (close > close.shift(1))
        ).astype(int)
        features["high_volume_down"] = (
            (volume > avg_volume * 1.5) & (close < close.shift(1))
        ).astype(int)

        # Accumulation/Distribution sentiment
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        features["accumulation_distribution"] = money_flow_volume.cumsum()
        features["ad_divergence"] = features["accumulation_distribution"].pct_change(
            20
        ) - close.pct_change(20)

        # Breadth sentiment (using price levels)
        features["above_sma_20"] = (close > close.rolling(20).mean()).astype(int)
        features["above_sma_50"] = (close > close.rolling(50).mean()).astype(int)
        features["above_sma_200"] = (close > close.rolling(200).mean()).astype(int)
        features["ma_breadth"] = (
            features["above_sma_20"] + features["above_sma_50"] + features["above_sma_200"]
        ) / 3

        return features

    def _add_composite_sentiment(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create composite sentiment scores"""
        sentiment_cols = []

        # Collect all sentiment columns
        for col in features.columns:
            if any(keyword in col for keyword in ["sentiment", "momentum", "fear", "consensus"]):
                sentiment_cols.append(col)

        if sentiment_cols:
            # Normalize sentiment scores
            normalized_sentiments = pd.DataFrame()
            for col in sentiment_cols:
                if features[col].std() > 0:
                    normalized_sentiments[col] = (features[col] - features[col].mean()) / features[
                        col
                    ].std()

            # Composite scores
            if not normalized_sentiments.empty:
                features["composite_sentiment"] = normalized_sentiments.mean(axis=1)
                features["sentiment_dispersion"] = normalized_sentiments.std(axis=1)
                features["sentiment_skew"] = normalized_sentiments.skew(axis=1)

                # Sentiment regime
                features["bullish_sentiment"] = (features["composite_sentiment"] > 1).astype(int)
                features["bearish_sentiment"] = (features["composite_sentiment"] < -1).astype(int)
                features["neutral_sentiment"] = (
                    (features["composite_sentiment"] >= -1) & (features["composite_sentiment"] <= 1)
                ).astype(int)

        # Sentiment agreement score
        direction_cols = []
        for col in features.columns:
            if col.endswith("_sentiment") or "momentum" in col:
                direction_cols.append((features[col] > 0).astype(int))

        if direction_cols:
            direction_df = pd.concat(direction_cols, axis=1)
            features["sentiment_agreement"] = direction_df.mean(axis=1)
            features["strong_consensus"] = (
                (features["sentiment_agreement"] > 0.8) | (features["sentiment_agreement"] < 0.2)
            ).astype(int)

        return features

    def _add_sentiment_momentum(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment momentum and acceleration features"""
        if "composite_sentiment" in features.columns:
            # Sentiment momentum
            features["sentiment_momentum"] = features["composite_sentiment"].diff(5)
            features["sentiment_acceleration"] = features["sentiment_momentum"].diff(5)

            # Sentiment reversals
            features["sentiment_reversal_up"] = (
                (features["composite_sentiment"].shift(1) < -1)
                & (features["composite_sentiment"] > -0.5)
            ).astype(int)

            features["sentiment_reversal_down"] = (
                (features["composite_sentiment"].shift(1) > 1)
                & (features["composite_sentiment"] < 0.5)
            ).astype(int)

        # Individual sentiment momentum
        for col in features.columns:
            if "sentiment" in col and not any(
                x in col for x in ["momentum", "acceleration", "reversal"]
            ):
                features[f"{col}_momentum"] = features[col].diff(3)

        return features

    def _add_keyword_sentiment(
        self, news_data: pd.DataFrame, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze specific keywords in news"""
        # Define sentiment keywords
        positive_keywords = [
            "upgrade",
            "beat",
            "exceed",
            "strong",
            "buy",
            "growth",
            "record",
            "breakthrough",
        ]
        negative_keywords = [
            "downgrade",
            "miss",
            "weak",
            "sell",
            "decline",
            "concern",
            "risk",
            "lawsuit",
        ]

        keyword_scores = []

        for idx in data.index:
            day_news = news_data[news_data["timestamp"].dt.date == idx.date()]

            if not day_news.empty and "text" in day_news.columns:
                all_text = " ".join(day_news["text"].str.lower())

                positive_count = sum(keyword in all_text for keyword in positive_keywords)
                negative_count = sum(keyword in all_text for keyword in negative_keywords)

                if positive_count + negative_count > 0:
                    keyword_score = (positive_count - negative_count) / (
                        positive_count + negative_count
                    )
                else:
                    keyword_score = 0

                keyword_scores.append(keyword_score)
            else:
                keyword_scores.append(0)

        features["news_keyword_sentiment"] = keyword_scores

        return features

    def _add_reddit_features(
        self, reddit_data: pd.DataFrame, data: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add Reddit-specific sentiment features"""
        # WSB mention detection
        if "subreddit" in reddit_data.columns:
            wsb_data = reddit_data[
                reddit_data["subreddit"].str.contains("wallstreetbets", case=False, na=False)
            ]

            wsb_mentions = []
            for idx in data.index:
                day_mentions = len(wsb_data[wsb_data["timestamp"].dt.date == idx.date()])
                wsb_mentions.append(day_mentions)

            features["wsb_mentions"] = wsb_mentions
            features["wsb_trending"] = (
                features["wsb_mentions"] > features["wsb_mentions"].rolling(7).mean() * 2
            ).astype(int)

        # Upvote ratio as sentiment proxy
        if "upvotes" in reddit_data.columns and "downvotes" in reddit_data.columns:
            upvote_ratios = []
            for idx in data.index:
                day_data = reddit_data[reddit_data["timestamp"].dt.date == idx.date()]
                if not day_data.empty:
                    total_upvotes = day_data["upvotes"].sum()
                    total_downvotes = day_data["downvotes"].sum()
                    ratio = total_upvotes / (total_upvotes + total_downvotes + 1)
                    upvote_ratios.append(ratio)
                else:
                    upvote_ratios.append(0.5)

            features["reddit_upvote_ratio"] = upvote_ratios

        return features

    def _add_database_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features from database repository"""
        try:
            # Standard library imports
            import asyncio

            # Extract symbol from data if available (fallback to generic approach)
            symbol = getattr(data, "symbol", "UNKNOWN")
            if symbol == "UNKNOWN" and hasattr(data.index, "name"):
                # Try to extract symbol from index name or other metadata
                symbol = "AAPL"  # Default for testing

            # Calculate time range for sentiment data
            start_date = (
                data.index[0] if len(data.index) > 0 else pd.Timestamp.now() - pd.Timedelta(days=7)
            )
            end_date = data.index[-1] if len(data.index) > 0 else pd.Timestamp.now()

            # Get sentiment aggregates from database
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get latest sentiment scores by source
                latest_sentiment = loop.run_until_complete(
                    self.sentiment_repository.get_latest_sentiment([symbol])
                )

                # Get sentiment aggregates for different time windows
                aggregates_24h = loop.run_until_complete(
                    self.sentiment_repository.calculate_sentiment_aggregates(
                        symbol, window_hours=24
                    )
                )

                aggregates_7d = loop.run_until_complete(
                    self.sentiment_repository.calculate_sentiment_aggregates(
                        symbol, window_hours=168
                    )  # 7 days
                )

                # Get sentiment by source for diversity metrics
                sentiment_by_source = loop.run_until_complete(
                    self.sentiment_repository.get_sentiment_by_source(symbol, hours_back=24)
                )

                # Create sentiment features
                for idx in data.index:
                    features.loc[idx, "db_sentiment_latest"] = latest_sentiment.get(symbol, 0.0)
                    features.loc[idx, "db_sentiment_mean_24h"] = aggregates_24h.get(
                        "mean_sentiment", 0.0
                    )
                    features.loc[idx, "db_sentiment_std_24h"] = aggregates_24h.get(
                        "std_sentiment", 0.0
                    )
                    features.loc[idx, "db_sentiment_count_24h"] = aggregates_24h.get("count", 0)
                    features.loc[idx, "db_sentiment_mean_7d"] = aggregates_7d.get(
                        "mean_sentiment", 0.0
                    )
                    features.loc[idx, "db_sentiment_std_7d"] = aggregates_7d.get(
                        "std_sentiment", 0.0
                    )
                    features.loc[idx, "db_sentiment_count_7d"] = aggregates_7d.get("count", 0)

                    # Source diversity metrics
                    features.loc[idx, "db_sentiment_source_count"] = len(sentiment_by_source)

                    # Calculate sentiment momentum (current vs historical)
                    current_sentiment = latest_sentiment.get(symbol, 0.0)
                    historical_sentiment = aggregates_7d.get("mean_sentiment", 0.0)
                    features.loc[idx, "db_sentiment_momentum"] = (
                        current_sentiment - historical_sentiment
                    )

                logger.info(f"Added database sentiment features for symbol {symbol}")

            finally:
                loop.close()

        except Exception as e:
            logger.warning(f"Could not load database sentiment data: {e}")
            # Add default values if database access fails
            for idx in data.index:
                features.loc[idx, "db_sentiment_latest"] = 0.0
                features.loc[idx, "db_sentiment_mean_24h"] = 0.0
                features.loc[idx, "db_sentiment_std_24h"] = 0.0
                features.loc[idx, "db_sentiment_count_24h"] = 0
                features.loc[idx, "db_sentiment_mean_7d"] = 0.0
                features.loc[idx, "db_sentiment_std_7d"] = 0.0
                features.loc[idx, "db_sentiment_count_7d"] = 0
                features.loc[idx, "db_sentiment_source_count"] = 0
                features.loc[idx, "db_sentiment_momentum"] = 0.0

        return features

    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text"""
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Sentiment analysis failed for text: {e}")
            return 0.0

    def _calculate_time_decay_weights(
        self, timestamps: pd.Series, reference_time: pd.Timestamp, decay_factor: float
    ) -> pd.Series:
        """Calculate time-based decay weights"""
        time_diffs = (reference_time - timestamps).dt.total_seconds() / 86400  # Days
        weights = decay_factor**time_diffs
        return weights / weights.sum()

    def _validate_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean news data"""
        if news_data.empty:
            return news_data

        # Ensure timestamp column exists and is datetime
        if "timestamp" not in news_data.columns:
            if "date" in news_data.columns:
                news_data = news_data.rename(columns={"date": "timestamp"})
            else:
                logger.warning("News data missing timestamp column")
                return pd.DataFrame()

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(news_data["timestamp"]):
            news_data["timestamp"] = pd.to_datetime(news_data["timestamp"])

        # Ensure text column exists
        if "text" not in news_data.columns:
            if "headline" in news_data.columns:
                news_data = news_data.rename(columns={"headline": "text"})
            elif "title" in news_data.columns:
                news_data = news_data.rename(columns={"title": "text"})
            else:
                logger.warning("News data missing text/headline column")
                return pd.DataFrame()

        # Remove rows with empty text
        news_data = news_data.dropna(subset=["text"])

        logger.debug(f"Validated news data: {len(news_data)} records")
        return news_data

    def _validate_social_data(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean social media data"""
        if social_data.empty:
            return social_data

        # Similar validation to news data
        if "timestamp" not in social_data.columns:
            logger.warning("Social data missing timestamp column")
            return pd.DataFrame()

        if not pd.api.types.is_datetime64_any_dtype(social_data["timestamp"]):
            social_data["timestamp"] = pd.to_datetime(social_data["timestamp"])

        if "text" not in social_data.columns:
            logger.warning("Social data missing text column")
            return pd.DataFrame()

        social_data = social_data.dropna(subset=["text"])

        logger.debug(f"Validated social data: {len(social_data)} records")
        return social_data

    def _validate_options_data(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean options data"""
        if options_data.empty:
            return options_data

        # Ensure we have basic options columns
        required_cols = ["timestamp"]
        missing = [col for col in required_cols if col not in options_data.columns]

        if missing:
            logger.warning(f"Options data missing required columns: {missing}")
            return pd.DataFrame()

        if not pd.api.types.is_datetime64_any_dtype(options_data["timestamp"]):
            options_data["timestamp"] = pd.to_datetime(options_data["timestamp"])

        logger.debug(f"Validated options data: {len(options_data)} records")
        return options_data

    def _validate_analyst_data(self, analyst_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean analyst data"""
        if analyst_data.empty:
            return analyst_data

        # Ensure required columns
        if "date" not in analyst_data.columns and "timestamp" not in analyst_data.columns:
            logger.warning("Analyst data missing date/timestamp column")
            return pd.DataFrame()

        # Standardize date column
        if "date" in analyst_data.columns and "timestamp" not in analyst_data.columns:
            analyst_data = analyst_data.rename(columns={"date": "timestamp"})

        if not pd.api.types.is_datetime64_any_dtype(analyst_data["timestamp"]):
            analyst_data["timestamp"] = pd.to_datetime(analyst_data["timestamp"])

        logger.debug(f"Validated analyst data: {len(analyst_data)} records")
        return analyst_data
