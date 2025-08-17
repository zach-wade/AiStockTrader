# File: tests/unit/test_unified_features.py

# Standard library imports
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# The classes we are testing

# --- Test Fixtures ---


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Creates a basic OHLCV DataFrame for the engine to process."""
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=20))
    return pd.DataFrame(
        {
            "open": np.random.rand(20) * 10 + 100,
            "high": np.random.rand(20) * 10 + 100,
            "low": np.random.rand(20) * 10 + 100,
            "close": np.random.rand(20) * 10 + 100,
            "volume": np.secure_randint(1000, 2000, 20),
        },
        index=dates,
    )


@pytest.fixture
def mock_technical_calculator() -> MagicMock:
    """Creates a mock of the TechnicalCalculator."""
    # Create a mock object that adheres to the FeatureCalculator interface
    mock_calc = MagicMock(spec=FeatureCalculator)

    # Configure the mock's 'calculate' method to return a specific DataFrame
    mock_calc.calculate.return_value = pd.DataFrame({"rsi_14": np.full(20, 50.0)})

    # Configure the mock's 'get_feature_names' method
    mock_calc.get_feature_names.return_value = ["rsi_14", "sma_20"]

    return mock_calc


@pytest.fixture
def mock_regime_calculator() -> MagicMock:
    """Creates a mock of the RegimeCalculator."""
    mock_calc = MagicMock(spec=FeatureCalculator)
    mock_calc.calculate.return_value = pd.DataFrame({"market_regime": np.full(20, "ranging")})
    mock_calc.get_feature_names.return_value = ["market_regime", "volatility_regime"]
    return mock_calc


# --- Test Cases ---


def test_engine_initialization_and_registration(tmp_path: Path):
    """Tests that calculators can be registered with the engine."""
    # Arrange
    engine = UnifiedFeatureEngine(cache_dir=tmp_path)
    mock_calc = MagicMock(spec=FeatureCalculator)  # Create a generic mock
    mock_calc.get_feature_names.return_value = ["feature_a", "feature_b"]

    # Act
    engine.register_calculator("mock_calc", mock_calc)

    # Assert
    assert "mock_calc" in engine.calculators
    assert engine.get_calculator("mock_calc") is mock_calc
    feature_names = engine.get_feature_names()
    assert "feature_a" in feature_names
    assert "feature_b" in feature_names


def test_calculate_features_combines_results(
    sample_ohlcv_data, mock_technical_calculator, mock_regime_calculator, tmp_path: Path
):
    """
    Tests that the engine correctly calls all registered calculators
    and concatenates their resulting features into a single DataFrame.
    """
    # Arrange
    engine = UnifiedFeatureEngine(cache_dir=tmp_path)
    engine.register_calculator("technical", mock_technical_calculator)
    engine.register_calculator("regime", mock_regime_calculator)

    # Act
    result_df = engine.calculate_features(sample_ohlcv_data, use_cache=False)

    # Assert
    # 1. Check that the calculate method of BOTH mocks was called once
    mock_technical_calculator.calculate.assert_called_once()
    mock_regime_calculator.calculate.assert_called_once()

    # 2. Check that the final DataFrame contains the original columns AND the new feature columns
    assert "close" in result_df.columns
    assert "rsi_14" in result_df.columns
    assert "market_regime" in result_df.columns

    # 3. Check that the values from main.the mocks are present
    assert (result_df["rsi_14"] == 50.0).all()
    assert (result_df["market_regime"] == "ranging").all()


def test_calculate_features_with_specific_calculators(
    sample_ohlcv_data, mock_technical_calculator, mock_regime_calculator, tmp_path: Path
):
    """
    Tests that the engine only calls the calculators specified in the list.
    """
    # Arrange
    engine = UnifiedFeatureEngine(cache_dir=tmp_path)
    engine.register_calculator("technical", mock_technical_calculator)
    engine.register_calculator("regime", mock_regime_calculator)

    # Act
    result_df = engine.calculate_features(
        sample_ohlcv_data, calculators=["regime"], use_cache=False
    )

    # Assert
    # 1. The regime calculator should have been called, but the technical one should NOT
    mock_regime_calculator.calculate.assert_called_once()
    mock_technical_calculator.calculate.assert_not_called()

    # 2. The final DataFrame should contain the regime feature but NOT the technical one
    assert "market_regime" in result_df.columns
    assert "rsi_14" not in result_df.columns


@patch(
    "main.feature_pipeline.unified_features.pickle"
)  # Mock the pickle library within the unified_features module
def test_caching_logic(mock_pickle, sample_ohlcv_data, tmp_path: Path):
    """
    Tests that features are loaded from main.cache on a second run, skipping computation.
    """
    # Arrange
    engine = UnifiedFeatureEngine(cache_dir=tmp_path)
    mock_calc = MagicMock(spec=FeatureCalculator)
    mock_calc.calculate.return_value = pd.DataFrame({"feature_x": [1, 2]})
    engine.register_calculator("mock", mock_calc)

    cache_key = engine._generate_cache_key(sample_ohlcv_data, ["mock"])
    cache_file = tmp_path / f"{cache_key}.pkl"

    # --- First run: Cache does not exist ---

    # Act
    result1 = engine.calculate_features(sample_ohlcv_data, calculators=["mock"])

    # Assert
    mock_calc.calculate.assert_called_once()  # Calculation should have happened
    assert cache_file.exists()  # Cache file should have been created

    # --- Second run: Cache now exists ---

    # Act
    result2 = engine.calculate_features(sample_ohlcv_data, calculators=["mock"])

    # Assert
    # The calculate method should NOT have been called a second time
    mock_calc.calculate.assert_called_once()
    assert result1.equals(result2)


def test_engine_handles_calculator_error_gracefully(
    sample_ohlcv_data, mock_technical_calculator, mock_regime_calculator, tmp_path: Path
):
    """
    Tests that if one calculator fails, the engine still returns results
    from main.the calculators that succeeded.
    """
    # Arrange
    engine = UnifiedFeatureEngine(cache_dir=tmp_path)

    # Configure the technical calculator to raise an error
    mock_technical_calculator.calculate.side_effect = ValueError("Calculation failed!")

    engine.register_calculator("technical", mock_technical_calculator)
    engine.register_calculator("regime", mock_regime_calculator)

    # Act
    result_df = engine.calculate_features(sample_ohlcv_data, use_cache=False)

    # Assert
    # 1. The process should not have crashed.
    assert result_df is not None

    # 2. The feature from main.the failing calculator should be missing.
    assert "rsi_14" not in result_df.columns

    # 3. The feature from main.the successful calculator should be present.
    assert "market_regime" in result_df.columns
    assert (result_df["market_regime"] == "ranging").all()
