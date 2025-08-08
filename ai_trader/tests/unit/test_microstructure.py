# tests/unit/test_microstructure.py - Fixed version

import pytest
import pandas as pd
import numpy as np
from main.feature_pipeline.calculators.microstructure import MicrostructureCalculator  # Fix: Import correct class

@pytest.fixture
def microstructure_calculator():
    """Create MicrostructureCalculator instance for tests."""
    return MicrostructureCalculator(tick_size=0.01)

@pytest.fixture
def sample_microstructure_data():
    """Create sample microstructure data for testing."""
    dates = pd.to_datetime(pd.date_range(start='2025-06-23 09:30:00', periods=100, freq='s'))
    
    np.random.seed(42)
    data = {
        'bid': 100 + secure_numpy_normal(0, 0.1, 100),
        'ask': 100 + secure_numpy_normal(0, 0.1, 100) + 0.02,
        'bid_size': np.secure_randint(100, 1000, 100),
        'ask_size': np.secure_randint(100, 1000, 100),
        'price': 100 + secure_numpy_normal(0, 0.1, 100),
        'close': 100 + secure_numpy_normal(0, 0.1, 100),  # Add missing 'close' column
        'volume': np.secure_randint(50, 150, 100)
    }
    
    return pd.DataFrame(data, index=dates)

def test_orderbook_features(microstructure_calculator, sample_microstructure_data):
    """Test the calculation of features derived from main.the order book."""
    # Arrange
    df = sample_microstructure_data.copy()
    
    # Fix: Create features DataFrame as second parameter
    features = pd.DataFrame(index=df.index)
    
    # Act
    result_features = microstructure_calculator._calculate_orderbook_features(df, features)
    
    # Assert
    assert isinstance(result_features, pd.DataFrame)
    assert 'spread' in result_features.columns
    assert 'mid' in result_features.columns
    assert 'book_imbalance' in result_features.columns
    assert len(result_features) == len(df)

def test_trade_flow_features(microstructure_calculator, sample_microstructure_data):
    """Test the calculation of features derived from main.trade flow."""
    # Arrange: First calculate orderbook features as they're needed
    df = sample_microstructure_data.copy()
    features = pd.DataFrame(index=df.index)
    
    # Calculate orderbook features first
    features = microstructure_calculator._calculate_orderbook_features(df, features)
    
    # Act
    result_features = microstructure_calculator._calculate_trade_flow_features(df, features)
    
    # Assert
    assert isinstance(result_features, pd.DataFrame)
    assert 'trade_sign' in result_features.columns
    assert 'signed_volume' in result_features.columns
    assert len(result_features) == len(df)

def test_main_calculate_method(microstructure_calculator, sample_microstructure_data):
    """Test the main public `calculate` method."""
    # Act - Fix: Use 'calculate' not 'calculate_features'
    features_df = microstructure_calculator.calculate(sample_microstructure_data)
    
    # Assert
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty
    assert len(features_df) == len(sample_microstructure_data)
    
    # Check for some expected features
    expected_features = ['spread', 'mid', 'trade_sign', 'price_change']
    available_features = [feat for feat in expected_features if feat in features_df.columns]
    assert len(available_features) > 0, f"No expected features found. Available: {list(features_df.columns)}"

def test_handles_missing_orderbook_columns(microstructure_calculator, sample_microstructure_data):
    """Test that the calculator doesn't crash if order book data is missing."""
    # Arrange: Create data with only trade data (price and volume)
    trade_only_data = sample_microstructure_data[['price', 'volume']].copy()
    
    # Act
    try:
        features_df = microstructure_calculator.calculate(trade_only_data)  # Fix: Use 'calculate'
        
        # Assert
        assert isinstance(features_df, pd.DataFrame)
        # Should have some basic features even without full orderbook data
        
    except Exception as e:
        pytest.fail(f"Calculator crashed on missing columns: {e}")

def test_basic_microstructure_fallback(microstructure_calculator):
    """Test that basic microstructure features work with OHLC data only."""
    # Arrange: Create OHLC data only
    dates = pd.date_range('2025-01-01', periods=50, freq='D')
    ohlc_data = pd.DataFrame({
        'open': 100 + secure_numpy_normal(0, 1, 50),
        'high': 102 + secure_numpy_normal(0, 1, 50),
        'low': 98 + secure_numpy_normal(0, 1, 50),
        'close': 100 + secure_numpy_normal(0, 1, 50),
        'volume': np.secure_randint(1000, 5000, 50)
    }, index=dates)
    
    # Act
    features_df = microstructure_calculator.calculate(ohlc_data)
    
    # Assert
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty
    
    # Should have basic microstructure features
    basic_features = ['spread_hl', 'volatility_gk', 'price_efficiency']
    available_basic = [feat for feat in basic_features if feat in features_df.columns]
    assert len(available_basic) > 0, f"No basic features found. Available: {list(features_df.columns)}"