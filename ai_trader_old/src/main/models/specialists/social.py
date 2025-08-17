"""
Social Media Catalyst Specialist

Predicts outcomes for social media-driven catalysts by analyzing
social sentiment patterns, volume spikes, and community engagement.
"""

# Standard library imports
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np

from .base import BaseCatalystSpecialist
from . import CatalystPrediction

logger = logging.getLogger(__name__)


class SocialSpecialist(BaseCatalystSpecialist):
    """
    Specialist for predicting outcomes of social media catalysts.

    Analyzes patterns in:
    - Social sentiment scores and changes
    - Post volume and engagement metrics
    - Community growth and participation
    - Influencer activity and reach
    - Correlation with price movements
    """

    def __init__(self, config: Any):
        """Initialize social specialist with configuration."""
        super().__init__(config, "social")

        # Social-specific thresholds
        self.sentiment_threshold = self.specialist_config.get("sentiment_threshold", 0.7)
        self.volume_spike_threshold = self.specialist_config.get("volume_spike_threshold", 3.0)
        self.engagement_threshold = self.specialist_config.get("engagement_threshold", 2.0)

        # Platform weights
        self.platform_weights = self.specialist_config.get(
            "platform_weights", {"reddit": 0.4, "twitter": 0.3, "stocktwits": 0.2, "discord": 0.1}
        )

        # Feature importance for social catalysts
        self.feature_importance = {
            "sentiment_score": 0.25,
            "sentiment_delta": 0.20,
            "volume_ratio": 0.20,
            "engagement_score": 0.15,
            "influencer_activity": 0.10,
            "community_growth": 0.10,
        }

    def extract_specialist_features(self, catalyst_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract social media specific features from catalyst data.

        Args:
            catalyst_features: Raw catalyst features including social metrics

        Returns:
            Dictionary of social-specific features
        """
        try:
            features = {}

            # Core sentiment features
            features["sentiment_score"] = catalyst_features.get("social_sentiment_score", 0.0)
            features["sentiment_delta"] = catalyst_features.get("social_sentiment_delta_24h", 0.0)
            features["sentiment_consistency"] = catalyst_features.get("sentiment_consistency", 0.5)

            # Volume and activity features
            features["post_volume_ratio"] = catalyst_features.get("social_post_volume_ratio", 1.0)
            features["unique_users_ratio"] = catalyst_features.get("unique_users_ratio", 1.0)
            features["engagement_rate"] = catalyst_features.get("engagement_rate", 0.0)

            # Platform-specific features
            features["reddit_activity"] = catalyst_features.get("reddit_activity_score", 0.0)
            features["twitter_mentions"] = catalyst_features.get("twitter_mention_ratio", 0.0)
            features["stocktwits_volume"] = catalyst_features.get("stocktwits_volume_ratio", 1.0)

            # Influencer and viral features
            features["influencer_score"] = catalyst_features.get("influencer_activity_score", 0.0)
            features["viral_coefficient"] = catalyst_features.get("viral_coefficient", 0.0)
            features["reach_expansion"] = catalyst_features.get("reach_expansion_rate", 0.0)

            # Community metrics
            features["community_growth"] = catalyst_features.get("community_growth_rate", 0.0)
            features["new_user_ratio"] = catalyst_features.get("new_user_ratio", 0.0)
            features["holder_sentiment"] = catalyst_features.get("holder_sentiment", 0.5)

            # Quality indicators
            features["spam_ratio"] = catalyst_features.get("spam_ratio", 0.0)
            features["bot_activity"] = catalyst_features.get("bot_activity_score", 0.0)
            features["content_quality"] = catalyst_features.get("content_quality_score", 0.5)

            # Time-based features
            features["momentum_1h"] = catalyst_features.get("social_momentum_1h", 0.0)
            features["momentum_24h"] = catalyst_features.get("social_momentum_24h", 0.0)
            features["time_since_spike"] = catalyst_features.get("hours_since_social_spike", 24.0)

            # Normalize time features
            features["time_decay"] = np.exp(-features["time_since_spike"] / 24.0)

            # Calculate composite scores
            features["platform_weighted_sentiment"] = self._calculate_platform_weighted_score(
                catalyst_features
            )
            features["engagement_quality"] = self._calculate_engagement_quality(features)
            features["catalyst_strength"] = self._calculate_social_catalyst_strength(features)

            return features

        except Exception as e:
            logger.error(f"Error extracting social features: {e}")
            return {}

    def get_minimum_catalyst_strength(self) -> float:
        """Get minimum catalyst strength for social signals."""
        return self.specialist_config.get("min_catalyst_strength", 2.5)

    def _calculate_platform_weighted_score(self, catalyst_features: Dict[str, Any]) -> float:
        """
        Calculate sentiment score weighted by platform importance.

        Different platforms have different signal quality and impact.
        """
        weighted_score = 0.0
        total_weight = 0.0

        platform_sentiments = {
            "reddit": catalyst_features.get("reddit_sentiment", 0.5),
            "twitter": catalyst_features.get("twitter_sentiment", 0.5),
            "stocktwits": catalyst_features.get("stocktwits_sentiment", 0.5),
            "discord": catalyst_features.get("discord_sentiment", 0.5),
        }

        for platform, sentiment in platform_sentiments.items():
            if sentiment != 0.5:  # Only count platforms with actual data
                weight = self.platform_weights.get(platform, 0.1)
                weighted_score += sentiment * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight

        return 0.5  # Neutral if no data

    def _calculate_engagement_quality(self, features: Dict[str, float]) -> float:
        """
        Calculate quality-adjusted engagement score.

        High engagement with low quality (spam/bots) is penalized.
        """
        base_engagement = features.get("engagement_rate", 0.0)

        # Quality adjustments
        spam_penalty = 1.0 - features.get("spam_ratio", 0.0)
        bot_penalty = 1.0 - (features.get("bot_activity", 0.0) * 0.5)  # Less severe penalty
        quality_bonus = features.get("content_quality", 0.5) * 0.5 + 0.5

        # Influencer bonus
        influencer_bonus = 1.0 + (features.get("influencer_score", 0.0) * 0.2)

        adjusted_engagement = (
            base_engagement * spam_penalty * bot_penalty * quality_bonus * influencer_bonus
        )

        return min(adjusted_engagement, 1.0)

    def _calculate_social_catalyst_strength(self, features: Dict[str, float]) -> float:
        """
        Calculate overall strength of social catalyst.

        Combines multiple factors to assess catalyst potential.
        """
        # Sentiment strength
        sentiment_strength = abs(features.get("sentiment_score", 0.5) - 0.5) * 2
        sentiment_momentum = abs(features.get("sentiment_delta", 0.0))

        # Volume strength
        volume_spike = max(0, features.get("post_volume_ratio", 1.0) - 1.0)
        user_growth = features.get("unique_users_ratio", 1.0) - 1.0

        # Engagement strength
        engagement_quality = features.get("engagement_quality", 0.0)
        viral_factor = features.get("viral_coefficient", 0.0)

        # Time decay
        time_factor = features.get("time_decay", 0.5)

        # Weighted combination
        catalyst_strength = (
            sentiment_strength * 0.25
            + sentiment_momentum * 0.20
            + volume_spike * 0.20
            + user_growth * 0.15
            + engagement_quality * 0.10
            + viral_factor * 0.10
        ) * time_factor

        # Scale to typical range (0-5)
        return catalyst_strength * 5.0

    async def predict(self, catalyst_features: Dict[str, Any]) -> Optional[CatalystPrediction]:
        """
        Generate prediction for social media catalyst.

        Overrides base class to add social-specific logic.
        """
        # Extract features
        features = self.extract_specialist_features(catalyst_features)
        if not features:
            return None

        # Check catalyst strength
        catalyst_strength = features.get("catalyst_strength", 0.0)
        if catalyst_strength < self.get_minimum_catalyst_strength():
            logger.debug(f"Social catalyst too weak: {catalyst_strength:.2f}")
            return None

        # Determine sentiment direction
        sentiment = features.get("platform_weighted_sentiment", 0.5)
        is_bullish = sentiment > self.sentiment_threshold
        is_bearish = sentiment < (1 - self.sentiment_threshold)

        if not (is_bullish or is_bearish):
            logger.debug("Social sentiment not strong enough for signal")
            return None

        # Calculate prediction confidence
        base_confidence = self._calculate_base_confidence(features)

        # Apply model if trained
        if self.is_trained and self.model:
            try:
                model_confidence = await self._get_model_prediction(features)
                # Blend rule-based and model confidence
                final_confidence = base_confidence * 0.4 + model_confidence * 0.6
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                final_confidence = base_confidence
        else:
            final_confidence = base_confidence

        # Generate prediction
        return CatalystPrediction(
            catalyst_type="social",
            symbol=catalyst_features.get("symbol", "UNKNOWN"),
            direction="bullish" if is_bullish else "bearish",
            confidence=final_confidence,
            expected_magnitude=self._estimate_magnitude(features),
            time_horizon="short",  # Social catalysts typically short-term
            metadata={
                "sentiment_score": sentiment,
                "catalyst_strength": catalyst_strength,
                "engagement_quality": features.get("engagement_quality", 0.0),
                "platforms_active": self._get_active_platforms(catalyst_features),
                "key_factors": self._identify_key_factors(features),
            },
        )

    def _calculate_base_confidence(self, features: Dict[str, float]) -> float:
        """
        Calculate rule-based confidence for social catalyst.
        """
        confidence_factors = []

        # Sentiment consistency
        consistency = features.get("sentiment_consistency", 0.5)
        if consistency > 0.7:
            confidence_factors.append(0.9)
        elif consistency > 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Volume spike significance
        volume_ratio = features.get("post_volume_ratio", 1.0)
        if volume_ratio > self.volume_spike_threshold:
            confidence_factors.append(0.85)
        elif volume_ratio > 2.0:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Engagement quality
        engagement = features.get("engagement_quality", 0.0)
        if engagement > 0.7:
            confidence_factors.append(0.9)
        elif engagement > 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)

        # Influencer involvement
        if features.get("influencer_score", 0.0) > 0.5:
            confidence_factors.append(0.8)

        # Community growth
        if features.get("community_growth", 0.0) > 0.1:
            confidence_factors.append(0.75)

        # Average confidence
        if confidence_factors:
            base_confidence = np.mean(confidence_factors)
        else:
            base_confidence = 0.5

        # Apply time decay
        time_decay = features.get("time_decay", 1.0)
        return base_confidence * (0.7 + 0.3 * time_decay)

    def _estimate_magnitude(self, features: Dict[str, float]) -> float:
        """
        Estimate expected price movement magnitude.
        """
        # Base magnitude on catalyst strength
        catalyst_strength = features.get("catalyst_strength", 0.0)

        # Social catalysts typically have moderate impact
        if catalyst_strength > 4.0:
            base_magnitude = 0.10  # 10% move
        elif catalyst_strength > 3.0:
            base_magnitude = 0.07  # 7% move
        elif catalyst_strength > 2.5:
            base_magnitude = 0.05  # 5% move
        else:
            base_magnitude = 0.03  # 3% move

        # Adjust for viral potential
        viral_factor = features.get("viral_coefficient", 0.0)
        if viral_factor > 0.7:
            base_magnitude *= 1.3
        elif viral_factor > 0.5:
            base_magnitude *= 1.1

        # Cap at reasonable levels
        return min(base_magnitude, 0.20)  # Max 20% expected move

    def _get_active_platforms(self, catalyst_features: Dict[str, Any]) -> List[str]:
        """Identify which platforms show significant activity."""
        active_platforms = []

        platform_activity = {
            "reddit": catalyst_features.get("reddit_activity_score", 0.0),
            "twitter": catalyst_features.get("twitter_mention_ratio", 0.0),
            "stocktwits": catalyst_features.get("stocktwits_volume_ratio", 0.0),
            "discord": catalyst_features.get("discord_activity_score", 0.0),
        }

        for platform, activity in platform_activity.items():
            if activity > 1.5:  # 50% above normal
                active_platforms.append(platform)

        return active_platforms

    def _identify_key_factors(self, features: Dict[str, float]) -> List[str]:
        """Identify the key factors driving the prediction."""
        factors = []

        # Check each factor
        if features.get("sentiment_score", 0.5) > 0.8:
            factors.append("extremely_positive_sentiment")
        elif features.get("sentiment_score", 0.5) < 0.2:
            factors.append("extremely_negative_sentiment")

        if features.get("post_volume_ratio", 1.0) > 3.0:
            factors.append("massive_volume_spike")

        if features.get("influencer_score", 0.0) > 0.7:
            factors.append("high_influencer_activity")

        if features.get("viral_coefficient", 0.0) > 0.6:
            factors.append("viral_spread_detected")

        if features.get("community_growth", 0.0) > 0.2:
            factors.append("rapid_community_growth")

        if features.get("engagement_quality", 0.0) > 0.8:
            factors.append("high_quality_engagement")

        return factors

    async def _get_model_prediction(self, features: Dict[str, float]) -> float:
        """Get prediction from trained model if available."""
        if not self.model:
            return 0.5

        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return 0.5

            # Get prediction
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(feature_vector.reshape(1, -1))
                return probabilities[0, 1]  # Probability of positive outcome
            else:
                prediction = self.model.predict(feature_vector.reshape(1, -1))
                return float(prediction[0])

        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return 0.5
