# File: tests/unit/test_enhanced_sentiment_strategy.py

# Standard library imports
from unittest.mock import MagicMock

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
# The class we are testing
from main.models.strategies.sentiment import FinalSentimentStrategy

# --- Test Fixtures ---

# Database adapter fixture removed - strategy now uses only features


@pytest.fixture
def mock_feature_engine():
    """Mocks the UnifiedFeatureEngine."""
    engine = MagicMock()

    # Configure it to return a DataFrame with some technical features
    def get_features(*args, **kwargs):
        dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=50))
        return pd.DataFrame(
            {
                "rsi_14": np.full(50, 50.0),  # Neutral RSI by default
                "volume_ratio": np.full(50, 1.2),  # Normal volume by default
            },
            index=dates,
        )

    engine.calculate_features = MagicMock(side_effect=get_features)
    return engine


@pytest.fixture
def strategy_config():
    """Provides a sample configuration for the strategy."""
    return {
        "strategies": {
            "enhanced_sentiment": {
                "sentiment_threshold": 0.3,
                "volume_spike_threshold": 1.5,
                "min_posts": 20,
                "lookback_hours": 24,
                "meme_stock_multiplier": 1.5,
                "base_position_size": 0.02,
                "max_position_size": 0.05,
            }
        }
    }


@pytest.fixture
def sentiment_strategy(strategy_config, mock_feature_engine):
    """Initializes the FinalSentimentStrategy with mocked dependencies."""
    return FinalSentimentStrategy(config=strategy_config, feature_engine=mock_feature_engine)


# --- Test Cases ---


@pytest.mark.asyncio
async def test_generate_strong_buy_signal(sentiment_strategy, mock_feature_engine):
    """
    Tests that a strong buy signal is generated when sentiment features are highly bullish
    and meet all criteria.
    """
    # Arrange
    # Configure mock feature engine to return strong bullish sentiment features
    bullish_features = pd.DataFrame(
        {
            "social_sentiment_score": [0.8],  # Strong bullish social sentiment
            "news_sentiment_24h": [0.7],  # Strong bullish news sentiment
            "volume_ratio": [2.5],  # High volume spike
            "rsi_14": [45.0],  # Not overbought
        }
    )
    mock_feature_engine.calculate_features.return_value = bullish_features

    # Act
    signals = await sentiment_strategy.generate_signals("GME", bullish_features, None)

    # Assert
    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "GME"
    assert signal.direction == "buy"
    assert signal.confidence > 0.5  # Should be a strong buy signal
    # Blended sentiment = 0.8 * 0.6 + 0.7 * 0.4 = 0.76, confidence = min(0.76 * 1.2, 1.0) = 0.91
    assert np.isclose(signal.confidence, 0.91, atol=0.01)


@pytest.mark.asyncio
async def test_no_signal_due_to_low_volume(sentiment_strategy, mock_feature_engine):
    """
    Tests that no signal is generated if volume is below the threshold despite good sentiment.
    """
    # Arrange
    # Configure mock feature engine to return good sentiment but low volume
    low_volume_features = pd.DataFrame(
        {
            "social_sentiment_score": [0.8],  # Strong sentiment
            "news_sentiment_24h": [0.7],  # Strong sentiment
            "volume_ratio": [1.2],  # Below threshold (2.0)
            "rsi_14": [45.0],
        }
    )
    mock_feature_engine.calculate_features.return_value = low_volume_features

    # Act
    signals = await sentiment_strategy.generate_signals("LOWVOL", low_volume_features, None)

    # Assert
    # No signal should be generated due to insufficient volume
    assert len(signals) == 0


@pytest.mark.asyncio
async def test_signal_reduced_by_conflicting_technicals(sentiment_strategy, mock_feature_engine):
    """
    Tests that a bullish sentiment signal is dampened if technicals are overbought.
    """
    # Arrange
    # Strong sentiment but overbought technicals
    overbought_features = pd.DataFrame(
        {
            "social_sentiment_score": [0.8],  # Strong bullish sentiment
            "news_sentiment_24h": [0.7],  # Strong bullish sentiment
            "volume_ratio": [2.5],  # Good volume
            "rsi_14": [85.0],  # Overbought - should reduce confidence
        }
    )
    mock_feature_engine.calculate_features.return_value = overbought_features

    # Act
    signals = await sentiment_strategy.generate_signals("PEAK", overbought_features, None)

    # Assert
    assert len(signals) == 1
    signal = signals[0]
    assert signal.direction == "buy"
    # Base confidence would be min(0.76 * 1.2, 1.0) = 0.91
    # With RSI > 70, it gets reduced by 0.75: 0.91 * 0.75 = 0.6825
    assert signal.confidence < 0.91  # Should be reduced from base confidence
    assert np.isclose(signal.confidence, 0.6825, atol=0.01)


def test_position_sizing_adjusts_for_volatility(sentiment_strategy):
    """
    Tests that the position size is reduced for high-volatility assets.
    """
    # Arrange
    # Local imports
    from main.models.strategies.base_strategy import Signal

    strong_signal = Signal(symbol="VOL_STOCK", direction="buy", confidence=0.9)

    # Case 1: Low volatility
    low_vol_features = pd.DataFrame({"volatility_20d": [0.01]})  # 1% daily vol

    # Act
    size_low_vol = sentiment_strategy._get_position_size(
        "VOL_STOCK", strong_signal, low_vol_features
    )

    # Case 2: High volatility
    high_vol_features = pd.DataFrame({"volatility_20d": [0.05]})  # 5% daily vol

    # Act
    size_high_vol = sentiment_strategy._get_position_size(
        "VOL_STOCK", strong_signal, high_vol_features
    )

    # Assert
    assert size_high_vol < size_low_vol
    # Base size = 0.02 * 0.9 = 0.018
    # High volatility (>4%) reduces size by 0.7: 0.018 * 0.7 = 0.0126
    # Low volatility keeps base size: 0.018
    assert np.isclose(size_low_vol, 0.018, atol=0.001)
    assert np.isclose(size_high_vol, 0.0126, atol=0.001)
