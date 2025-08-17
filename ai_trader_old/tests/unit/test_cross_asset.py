# File: tests/unit/test_cross_asset.py

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
# The class we are testing
# Note: The original file has `class CrossAssetFeatures`, which seems to be a typo for a calculator.
# We will test it as is. If you rename the class to CrossAssetCalculator, update the import here.
from main.feature_pipeline.calculators.cross_asset import CrossAssetFeatures

# --- Test Fixtures ---


@pytest.fixture(scope="module")
def cross_asset_calculator():
    """Provides an instance of the CrossAssetFeatures class."""
    return CrossAssetFeatures(lookback_periods=[20, 60])


@pytest.fixture(scope="module")
def multi_asset_data() -> dict[str, pd.DataFrame]:
    """
    Creates a dictionary of DataFrames with pre-defined relationships:
    - SPY (stocks) and TLT (bonds) will be negatively correlated.
    - SPY and QQQ (more stocks) will be positively correlated.
    - GLD (gold) will be uncorrelated with SPY.
    """
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=100))

    # Create a base series
    base_series = np.random.randn(100).cumsum()

    # Create asset price series based on the base series
    spy_prices = 100 + base_series
    qqq_prices = (
        150 + base_series * 0.8 + np.random.randn(100) * 0.1
    )  # Strong positive corr with SPY
    tlt_prices = (
        120 - base_series * 0.9 + np.random.randn(100) * 0.1
    )  # Strong negative corr with SPY
    gld_prices = 75 + np.random.randn(100).cumsum()  # Uncorrelated with SPY
    hyg_prices = 90 + base_series * 0.7  # High-yield bonds, correlated with stocks

    data = {
        "SPY": pd.DataFrame({"close": spy_prices}, index=dates),
        "QQQ": pd.DataFrame({"close": qqq_prices}, index=dates),
        "TLT": pd.DataFrame({"close": tlt_prices}, index=dates),
        "GLD": pd.DataFrame({"close": gld_prices}, index=dates),
        "HYG": pd.DataFrame({"close": hyg_prices}, index=dates),
        "IEF": pd.DataFrame({"close": tlt_prices + 10}, index=dates),  # Mock safe asset
    }
    return data


# --- Test Cases ---


def test_prepare_returns_data(cross_asset_calculator, multi_asset_data):
    """Tests that the returns data is prepared correctly."""
    # Act
    returns_df = cross_asset_calculator._prepare_returns_data(multi_asset_data)

    # Assert
    assert isinstance(returns_df, pd.DataFrame)
    assert "SPY" in returns_df.columns and "TLT" in returns_df.columns
    assert len(returns_df) == len(multi_asset_data["SPY"])
    # The first value of pct_change is always NaN
    assert pd.isna(returns_df.iloc[0]["SPY"])


def test_rolling_correlations(cross_asset_calculator, multi_asset_data):
    """Tests the calculation of rolling correlations against predefined relationships."""
    # Arrange
    target_symbol = "SPY"
    returns_data = cross_asset_calculator._prepare_returns_data(multi_asset_data)
    target_df = multi_asset_data[target_symbol].copy()

    # Act
    features_df = cross_asset_calculator._calculate_rolling_correlations(
        target_df, returns_data, target_symbol
    )

    # Assert
    # We check the last calculated value after the lookback period has been established.

    # Correlation with bonds (TLT) should be strongly negative
    corr_bonds_col = "corr_bonds_20d"
    assert corr_bonds_col in features_df.columns
    assert features_df[corr_bonds_col].iloc[-1] < -0.7

    # Correlation with equities (QQQ) should be strongly positive
    # Note: The calculator correctly excludes the target symbol from main.its own group.
    corr_equities_col = "corr_equities_20d"
    assert corr_equities_col in features_df.columns
    assert features_df[corr_equities_col].iloc[-1] > 0.7

    # Correlation with commodities (GLD) should be weak
    corr_commodities_col = "corr_commodities_20d"
    assert corr_commodities_col in features_df.columns
    assert (
        abs(features_df[corr_commodities_col].iloc[-1]) < 0.5
    )  # A lenient threshold for random data


def test_risk_on_off_indicator(cross_asset_calculator, multi_asset_data):
    """Tests the risk-on/risk-off indicator calculation."""
    # Arrange
    target_symbol = "SPY"
    returns_data = cross_asset_calculator._prepare_returns_data(multi_asset_data)
    target_df = multi_asset_data[target_symbol].copy()

    # In our fixture, SPY and HYG (risk-on) trend up, while TLT and IEF (risk-off) trend down.
    # Therefore, the risk-on score should be positive.

    # Act
    features_df = cross_asset_calculator._calculate_risk_indicators(target_df, returns_data)

    # Assert
    assert "risk_on_score" in features_df.columns
    assert "risk_regime" in features_df.columns

    # The score should be positive, indicating a risk-on environment
    assert features_df["risk_on_score"].iloc[-1] > 0
    # The regime should be classified as risk-on (1)
    assert features_df["risk_regime"].iloc[-1] == 1


def test_handles_missing_assets_gracefully(cross_asset_calculator, multi_asset_data):
    """
    Tests that the calculator does not crash if some assets from main.its internal
    lists (e.g., volatility assets) are missing from main.the input data.
    """
    # Arrange: Remove assets that are in the calculator's internal lists but not essential for all calcs
    # In this case, VXX, USO, DXY etc. are missing from main.our fixture.

    # Act & Assert
    try:
        # The main public method should run without raising an error
        features_df = cross_asset_calculator.calculate_features(
            multi_asset_data, target_symbol="SPY"
        )
        # Check that it still produced some features it could calculate
        assert "corr_bonds_20d" in features_df.columns
        assert not features_df.empty
    except Exception as e:
        pytest.fail(f"Calculator failed when assets were missing: {e}")
