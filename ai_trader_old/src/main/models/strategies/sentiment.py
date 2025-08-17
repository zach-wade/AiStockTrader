"""
The single, authoritative sentiment trading strategy.
This strategy relies on pre-computed features from the UnifiedFeatureEngine.
It contains NO direct data collection, API calls, or database queries.
This is the gold-standard template for feature-based strategies.
"""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

# Import from your established common strategy definitions
from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class FinalSentimentStrategy(BaseStrategy):
    """
    Combines social media and news sentiment with technical confirmation to
    generate trading signals based on pre-computed features.
    """

    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        # The constructor correctly accepts the feature_engine dependency.
        super().__init__(config, feature_engine)
        self.name = "final_sentiment"

        # Load parameters from the unified config
        strategy_conf = self.config.get("strategies", {}).get(self.name, {})
        self.sentiment_threshold = strategy_conf.get("sentiment_threshold", 0.3)
        self.volume_spike_threshold = strategy_conf.get("volume_spike_threshold", 2.0)

    def get_required_feature_sets(self) -> List[str]:
        """Specifies which features this strategy needs from the FeatureEngine."""
        return ["technical", "sentiment_features", "news_features"]

    # REFACTOR: Updated signature to be async and accept current_position.
    async def generate_signals(
        self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]
    ) -> List[Signal]:
        """
        Generates trading signals based on pre-computed features.
        This is pure alpha logic, with no data processing.
        """
        if features.empty:
            return []

        latest_features = features.iloc[-1]

        # --- Core Alpha Logic ---
        # Consume pre-computed features.
        social_sentiment = latest_features.get("social_sentiment_score", 0)
        news_sentiment = latest_features.get("news_sentiment_24h", 0)

        # Create a blended sentiment score
        blended_sentiment = (social_sentiment * 0.6) + (news_sentiment * 0.4)

        # Get confirming features
        volume_ratio = latest_features.get("volume_ratio", 1.0)
        rsi = latest_features.get("rsi_14", 50.0)

        # --- Decision Making ---

        # Bullish Case
        if (
            blended_sentiment > self.sentiment_threshold
            and volume_ratio > self.volume_spike_threshold
        ):
            # Only generate a buy signal if we are not already long
            if not current_position or current_position.get("direction") != "long":
                confidence = min(blended_sentiment * 1.2, 1.0)
                if rsi > 70:  # Reduce confidence if technically overbought
                    confidence *= 0.75
                return [Signal(symbol=symbol, direction="buy", confidence=confidence)]

        # Bearish Case
        elif (
            blended_sentiment < -self.sentiment_threshold
            and volume_ratio > self.volume_spike_threshold
        ):
            # Only generate a sell signal if we are not already short
            if not current_position or current_position.get("direction") != "short":
                confidence = min(abs(blended_sentiment) * 1.2, 1.0)
                if rsi < 30:  # Reduce confidence if technically oversold
                    confidence *= 0.75
                return [Signal(symbol=symbol, direction="sell", confidence=confidence)]

        # Exit Case: If we have a position but sentiment has neutralized
        if current_position and abs(blended_sentiment) < 0.1:
            return [Signal(symbol=symbol, direction="close", confidence=1.0)]

        return []

    # REFACTOR: Overriding the base class method for custom sizing is a valid pattern.
    def _get_position_size(self, symbol: str, signal: Signal, features: pd.DataFrame) -> float:
        """Calculates position size based on signal confidence and risk factors."""
        strategy_conf = self.config.get("strategies", {}).get(self.name, {})
        base_size = strategy_conf.get("base_position_size", 0.02)

        size = base_size * signal.confidence

        # Adjust for volatility
        volatility = features.iloc[-1].get("volatility_20d", 0.02)  # Use 20-day vol
        if volatility > 0.04:  # If daily volatility is > 4%
            size *= 0.7  # Reduce size for highly volatile assets

        return size
