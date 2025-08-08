
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from main.feature_pipeline.feature_store import FeatureStore

@pytest.fixture
def sample_features_df():
    """Sample features DataFrame with required columns including 'close'"""
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'rsi_14': np.random.rand(10) * 100,
        'sma_20': np.random.rand(10) * 100,
        'volume_ratio': np.random.rand(10) * 2,
        'close': 100 + np.random.rand(10) * 10,  # Add missing 'close' column
        'open': 99 + np.random.rand(10) * 10,    # Add OHLC columns for feature computation
        'high': 101 + np.random.rand(10) * 10,
        'low': 97 + np.random.rand(10) * 10,
        'volume': np.secure_randint(1000, 5000, 10)
    }, index=dates)

@pytest.mark.asyncio
async def test_save_and_load_features(temp_feature_store: FeatureStore, sample_features_df: pd.DataFrame):
    """Test saving and loading features with corrected method signature."""
    # Arrange
    symbol = 'TEST_STOCK'
    feature_hash = "testhash123"
    
    # Create a mock feature set
    from main.feature_pipeline.feature_store import FeatureSet
    from datetime import datetime
    
    feature_set = FeatureSet(
        name="test_set",
        version="v1",
        created_at=datetime.now(),
        features=list(sample_features_df.columns),
        description="Test feature set",
        dependencies=[],
        hash=feature_hash
    )
    
    # Act: Save the features with correct signature
    all_features = {symbol: sample_features_df}
    await temp_feature_store._save_features(feature_set, all_features)
    
    # Verify file was created
    expected_path = temp_feature_store._get_feature_path(symbol, feature_hash)
    assert expected_path.exists()

@pytest.mark.asyncio
async def test_metadata_creation_and_loading(temp_feature_store: FeatureStore, sample_features_df: pd.DataFrame):
    """Test feature set metadata creation with proper data."""
    # Arrange
    feature_set_name = "test_set"
    data_dict = {'TEST': sample_features_df}  # sample_features_df now has 'close' column
    
    # Act: Create a new feature set
    await temp_feature_store.create_feature_set(
        name=feature_set_name,
        data=data_dict,
        feature_functions=['returns', 'volatility'],
        description="A test set"
    )
    
    # Assert
    assert feature_set_name in temp_feature_store.feature_sets
    feature_set = temp_feature_store.feature_sets[feature_set_name]
    assert feature_set.name == feature_set_name
    assert len(feature_set.features) > 0

@pytest.fixture
def temp_feature_store(tmp_path):
    """Create a temporary FeatureStore for testing"""
    from main.feature_pipeline.feature_store import FeatureStore
    return FeatureStore(tmp_path)