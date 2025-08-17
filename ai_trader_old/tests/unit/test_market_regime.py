# File: tests/unit/test_market_regime.py

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
# The class and enum we are testing
from main.feature_pipeline.calculators.market_regime import MarketRegime, MarketRegimeDetector

# --- Test Fixtures ---


@pytest.fixture(scope="module")
def regime_detector():
    """Provides an instance of the MarketRegimeDetector with a standard lookback."""
    return MarketRegimeDetector(lookback_period=50)


@pytest.fixture(scope="module")
def trending_up_data() -> pd.DataFrame:
    """Creates a DataFrame with a clear, consistent upward trend."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))
    # A nearly straight line with minor noise
    prices = np.linspace(100, 150, 100) + secure_numpy_normal(0, 0.1, 100)
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture(scope="module")
def trending_down_data() -> pd.DataFrame:
    """Creates a DataFrame with a clear downward trend."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))
    prices = np.linspace(150, 100, 100) + secure_numpy_normal(0, 0.1, 100)
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture(scope="module")
def ranging_data() -> pd.DataFrame:
    """Creates a DataFrame that oscillates around a mean (ranging/mean-reverting)."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))
    # Sine wave to simulate ranging behavior
    prices = 100 + 5 * np.sin(np.linspace(0, 20, 100))
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture(scope="module")
def high_volatility_data() -> pd.DataFrame:
    """Creates a DataFrame with high volatility but no clear trend."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))
    # High standard deviation noise
    returns = secure_numpy_normal(0, 0.04, 99)  # 4% daily volatility
    prices = 100 * (1 + returns).cumprod()
    return pd.DataFrame({"close": np.insert(prices, 0, 100)}, index=dates)


# --- Test Cases ---


def test_detect_trending_up_regime(regime_detector, trending_up_data):
    """
    Tests that a clear upward trend is classified correctly.
    """
    # Act
    regime, confidence = regime_detector.detect_regime(trending_up_data)

    # Assert
    assert regime == MarketRegime.TRENDING_UP
    assert confidence > 0.7, "Confidence in a clear trend should be high"


def test_detect_trending_down_regime(regime_detector, trending_down_data):
    """
    Tests that a clear downward trend is classified correctly.
    """
    # Act
    regime, confidence = regime_detector.detect_regime(trending_down_data)

    # Assert
    assert regime == MarketRegime.TRENDING_DOWN
    assert confidence > 0.7, "Confidence in a clear trend should be high"


def test_detect_ranging_regime(regime_detector, ranging_data):
    """
    Tests that a sideways market is classified as ranging.
    """
    # Act
    regime, confidence = regime_detector.detect_regime(ranging_data)

    # Assert
    assert regime == MarketRegime.RANGING
    assert confidence > 0.5, "Confidence in a ranging market should be reasonable"


def test_detect_high_volatility_regime(regime_detector, high_volatility_data):
    """
    Tests that an erratic market is classified as high volatility.

    Note: Your internal logic uses a fixed percentile for volatility.
    This test assumes the volatility of the fixture data is high enough
    to cross the 80th percentile threshold in the _classify_regime method.
    """
    # Act
    regime, confidence = regime_detector.detect_regime(high_volatility_data)

    # Assert
    assert regime == MarketRegime.HIGH_VOLATILITY
    assert confidence > 0.7, "Confidence in high volatility should be high"


def test_short_data_returns_default_regime(regime_detector):
    """
    Tests that the detector handles insufficient data gracefully.
    """
    # Arrange: Create data shorter than the lookback period (50)
    short_data = pd.DataFrame({"close": np.linspace(100, 105, 30)})

    # Act
    regime, confidence = regime_detector.detect_regime(short_data)

    # Assert
    # It should return the default value without erroring out
    assert regime == MarketRegime.RANGING
    assert confidence == 0.5


def test_internal_trend_strength_calculation(regime_detector):
    """
    Tests the internal '_calculate_trend_strength' helper method directly.
    """
    # Arrange: A perfect, noiseless upward line
    prices = pd.Series(np.linspace(100, 110, 50))  # 50 days, increases by 10%

    # Act
    # Accessing a "private" method for unit testing is acceptable to validate core logic
    trend_strength = regime_detector._calculate_trend_strength(prices)

    # Assert
    # For a perfect line, R-squared is 1. The result is just the normalized slope.
    # Slope is (110-100)/50 = 0.2. Mean is 105. Normalized slope is 0.2/105 ~= 0.0019
    assert trend_strength > 0.0018
    assert trend_strength < 0.0020
    assert np.isclose(trend_strength, (0.2 / 105.0))
