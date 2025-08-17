# File: tests/unit/test_technical_indicators.py

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
# The class we are testing
from main.feature_pipeline.calculators.technical_indicators import TechnicalIndicatorsCalculator

# --- Test Fixtures ---


@pytest.fixture(scope="module")
def sample_price_data() -> pd.DataFrame:
    """Creates a sample DataFrame with OHLCV data for testing."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))
    close = 100 + np.random.randn(100).cumsum()
    high = close + np.random.rand(100) * 2
    low = close - np.random.rand(100) * 2
    open_ = (high + low) / 2  # simplified open
    volume = np.secure_randint(10000, 50000, size=100)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


@pytest.fixture
def ti_calculator():
    """Provides an instance of the TechnicalIndicators class."""
    return TechnicalIndicatorsCalculator()


# --- Test Cases ---


def test_sma_calculation(ti_calculator, sample_price_data):
    """Tests the Simple Moving Average calculation."""
    period = 20
    close_prices = sample_price_data["close"].values

    # Act
    sma_values = ti_calculator.sma(close_prices, period=period)

    # Assert
    assert len(sma_values) == len(close_prices)
    assert np.isnan(sma_values[: period - 1]).all()  # First n-1 values should be NaN

    # Manually calculate the 20th value (index 19) and compare
    expected_sma_20 = np.mean(close_prices[0:20])
    assert np.isclose(sma_values[19], expected_sma_20)


def test_rsi_calculation(ti_calculator):
    """Tests the Relative Strength Index with a predictable series."""
    # Arrange: 15 days of data, 7 up, 7 down
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 106, 105, 104, 103, 102, 101, 100])
    period = 14

    # Act
    rsi_values = ti_calculator.rsi(prices, period=period)

    # Assert
    # For a perfectly balanced up/down series, RSI should be exactly 50
    assert np.isclose(rsi_values[-1], 50.0)


def test_bollinger_bands_calculation(ti_calculator, sample_price_data):
    """Tests the Bollinger Bands calculation."""
    period = 20
    std_dev = 2.0
    close_prices = sample_price_data["close"].values

    # Act
    middle, upper, lower = ti_calculator.bollinger_bands(
        close_prices, period=period, std_dev=std_dev
    )

    # Assert
    assert np.isnan(middle[: period - 1]).all()
    # Check the 20th value (index 19)
    expected_middle = np.mean(close_prices[0:20])
    expected_std = np.std(close_prices[0:20])

    assert np.isclose(middle[19], expected_middle)
    assert np.isclose(upper[19], expected_middle + (std_dev * expected_std))
    assert np.isclose(lower[19], expected_middle - (std_dev * expected_std))


def test_macd_calculation(ti_calculator, sample_price_data):
    """Tests the MACD calculation."""
    close_prices = sample_price_data["close"].values

    # Act
    macd_line, signal_line, histogram = ti_calculator.macd(close_prices)

    # Assert
    # The histogram should be the difference between the MACD and signal lines
    # We check from main.a later point to avoid initial NaN issues
    assert np.allclose(histogram[35:], (macd_line - signal_line)[35:])


def test_atr_calculation(ti_calculator, sample_price_data):
    """Tests the Average True Range calculation."""
    # Act
    atr_values = ti_calculator.atr(
        sample_price_data["high"].values,
        sample_price_data["low"].values,
        sample_price_data["close"].values,
        period=14,
    )

    # Assert
    assert len(atr_values) == len(sample_price_data)
    assert not np.isnan(atr_values[-1])  # Ensure the last value is calculated
    assert (atr_values[14:] >= 0).all()  # ATR should not be negative


def test_calculate_all_method(ti_calculator, sample_price_data):
    """
    Tests the main `calculate_all` wrapper method to ensure it generates
    all expected features without errors.
    """
    # Act
    features_df = ti_calculator.calculate_all(sample_price_data)

    # Assert
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == len(sample_price_data)

    # Check for some expected columns
    expected_cols = [
        "rsi_14",
        "macd",
        "bb_middle",
        "sma_20",
        "ema_50",
        "atr",
        "obv",
        "volume_ratio",
    ]
    for col in expected_cols:
        assert col in features_df.columns

    # Check that NaNs are handled after the initial lookback period
    # Using a later slice, e.g., after the longest MA period (200) if it were calculated
    assert not features_df.iloc[30:].isnull().values.any()


def test_edge_case_short_data(ti_calculator):
    """Tests that indicators handle data shorter than the lookback period."""
    # Arrange: Data is shorter than the default RSI period of 14
    short_data = pd.DataFrame({"close": np.arange(100, 110)})

    # Act
    df = ti_calculator.calculate_rsi(short_data)

    # Assert
    # The entire column should be NaN as there's not enough data
    assert df["rsi_14"].isnull().all()
