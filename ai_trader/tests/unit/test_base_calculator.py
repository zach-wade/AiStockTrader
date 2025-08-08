# tests/unit/test_base_calculator.py

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from main.feature_pipeline.calculators.base_calculator import BaseFeatureCalculator


# Concrete Implementation for Testing
class ConcreteTestCalculator(BaseFeatureCalculator):
    """Concrete implementation of BaseFeatureCalculator for testing."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple test features."""
        features = pd.DataFrame(index=data.index)
        if 'close' in data.columns:
            features['test_sma'] = data['close'].rolling(5).mean()
            features['test_std'] = data['close'].rolling(10).std()
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate that required columns exist."""
        required = self.get_required_columns()
        return all(col in data.columns for col in required)
    
    def get_required_columns(self) -> list:
        """Return required columns."""
        return ['close', 'volume']


class StrictValidationCalculator(BaseFeatureCalculator):
    """Calculator with strict validation for testing."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features that might fail."""
        if len(data) < 10:
            raise ValueError("Insufficient data")
        features = pd.DataFrame(index=data.index)
        features['strict_feature'] = data['close'] * 2
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Strict validation."""
        return 'close' in data.columns and len(data) >= 10
    
    def get_required_columns(self) -> list:
        """Return required columns."""
        return ['close']


class AlwaysFailingCalculator(BaseFeatureCalculator):
    """Calculator that always fails for testing error handling."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Always raise an exception."""
        raise RuntimeError("Calculator always fails")
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Always fail validation."""
        return False
    
    def get_required_columns(self) -> list:
        """Return required columns."""
        return ['close']


# Test Fixtures
@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D', tz=timezone.utc)
    np.random.seed(42)
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
        'open': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
        'high': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 50)) + np.abs(secure_numpy_normal(0, 0.01, 50)),
        'low': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 50)) - np.abs(secure_numpy_normal(0, 0.01, 50)),
        'volume': np.secure_randint(10000, 100000, 50)
    }, index=dates)
    
    return data


@pytest.fixture
def problematic_data():
    """Data with various quality issues for testing."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D', tz=timezone.utc)
    
    # Create data with issues
    close_prices = [100, 101, np.nan, 103, 104, np.inf, 106, 107, 108, -np.inf,
                   110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    
    data = pd.DataFrame({
        'close': close_prices,
        'volume': [1000] * 20,
        'timestamp': dates
    }, index=dates)
    
    # Add duplicates
    data = pd.concat([data, data.iloc[[5]]])  # Duplicate row
    
    # Make index unsorted
    data = data.sort_index(ascending=False)
    
    return data


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return {
        'strict_validation': True,
        'handle_missing': 'interpolate',
        'fill_nan': True,
        'clip_percentile': 5,
        'version': '1.0.0'
    }


@pytest.fixture
def test_calculator(default_config):
    """Test calculator instance."""
    return ConcreteTestCalculator(default_config)


# Test Base Calculator Initialization
class TestBaseCalculatorInit:
    """Test BaseFeatureCalculator initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        calculator = ConcreteTestCalculator()
        
        assert calculator.config == {}
        assert calculator.name == 'ConcreteTestCalculator'
        assert calculator._cache == {}
        assert 'calculator_name' in calculator._metadata
        assert calculator._metadata['calculator_name'] == 'ConcreteTestCalculator'
        assert calculator.strict_validation is True  # default
        assert calculator.handle_missing == 'interpolate'  # default
    
    def test_init_with_custom_config(self, default_config):
        """Test initialization with custom configuration."""
        calculator = ConcreteTestCalculator(default_config)
        
        assert calculator.config == default_config
        assert calculator.strict_validation is True
        assert calculator.handle_missing == 'interpolate'
        assert calculator._metadata['version'] == '1.0.0'
    
    def test_init_performance_tracking(self, test_calculator):
        """Test that performance tracking is initialized."""
        assert hasattr(test_calculator, '_calculation_times')
        assert hasattr(test_calculator, '_feature_importance')
        assert test_calculator._calculation_times == []
        assert test_calculator._feature_importance == {}


# Test Abstract Method Implementation
class TestAbstractMethods:
    """Test that abstract methods are properly implemented."""
    
    def test_concrete_calculator_implements_required_methods(self, test_calculator):
        """Test that concrete calculator implements all required methods."""
        # Should not raise NotImplementedError
        assert hasattr(test_calculator, 'calculate')
        assert hasattr(test_calculator, 'validate_inputs')
        assert hasattr(test_calculator, 'get_required_columns')
        
        # Methods should be callable
        assert callable(test_calculator.calculate)
        assert callable(test_calculator.validate_inputs)
        assert callable(test_calculator.get_required_columns)
    
    def test_abstract_methods_raise_error_in_base_class(self):
        """Test that abstract methods raise errors in base class."""
        # This would fail at class definition time due to ABC
        # but we can test that the methods are marked as abstract
        from main.feature_pipeline.calculators.base_calculator import BaseFeatureCalculator
        
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseFeatureCalculator()


# Test Input Validation
class TestInputValidation:
    """Test input validation functionality."""
    
    def test_validate_inputs_success(self, test_calculator, sample_market_data):
        """Test successful input validation."""
        result = test_calculator.validate_inputs(sample_market_data)
        assert result is True
    
    def test_validate_inputs_missing_columns(self, test_calculator):
        """Test validation failure with missing columns."""
        incomplete_data = pd.DataFrame({'close': [100, 101, 102]})
        result = test_calculator.validate_inputs(incomplete_data)
        assert result is False
    
    def test_validate_inputs_empty_data(self, test_calculator):
        """Test validation with empty DataFrame."""
        empty_data = pd.DataFrame()
        result = test_calculator.validate_inputs(empty_data)
        assert result is False
    
    def test_get_required_columns(self, test_calculator):
        """Test getting required columns."""
        required = test_calculator.get_required_columns()
        assert isinstance(required, list)
        assert 'close' in required
        assert 'volume' in required


# Test Data Preprocessing
class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_preprocess_data_basic(self, test_calculator, sample_market_data):
        """Test basic data preprocessing."""
        result = test_calculator.preprocess_data(sample_market_data)
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        # Should have same shape (no missing data in sample)
        assert result.shape == sample_market_data.shape
        # Should be sorted by index
        assert result.index.is_monotonic_increasing
    
    def test_preprocess_data_handles_missing_values(self, test_calculator):
        """Test handling of missing values."""
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, np.nan, 104],
            'volume': [1000, 1500, np.nan, 2000, 2500]
        })
        
        result = test_calculator.preprocess_data(data_with_nan)
        
        # Should interpolate missing values
        assert not result['close'].isna().any()
        assert not result['volume'].isna().any()
    
    def test_preprocess_data_forward_fill_option(self, default_config):
        """Test forward fill option for missing data."""
        config = default_config.copy()
        config['handle_missing'] = 'forward_fill'
        calculator = ConcreteTestCalculator(config)
        
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, np.nan, 103],
            'volume': [1000, 1500, np.nan, 2000]
        })
        
        result = calculator.preprocess_data(data_with_nan)
        
        # Should forward fill
        assert result['close'].iloc[1] == 100  # Forward filled
        assert result['close'].iloc[2] == 100  # Forward filled
    
    def test_preprocess_data_drop_option(self, default_config):
        """Test drop option for missing data."""
        config = default_config.copy()
        config['handle_missing'] = 'drop'
        calculator = ConcreteTestCalculator(config)
        
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, 103],
            'volume': [1000, 1500, 2000, 2500]
        })
        
        result = calculator.preprocess_data(data_with_nan)
        
        # Should drop rows with NaN
        assert len(result) < len(data_with_nan)
        assert not result.isna().any().any()
    
    def test_preprocess_data_removes_duplicates(self, test_calculator, problematic_data):
        """Test removal of duplicate index values."""
        result = test_calculator.preprocess_data(problematic_data)
        
        # Should remove duplicates
        assert not result.index.duplicated().any()
    
    def test_preprocess_data_sorts_index(self, test_calculator, problematic_data):
        """Test that index is sorted."""
        result = test_calculator.preprocess_data(problematic_data)
        
        # Should be sorted
        assert result.index.is_monotonic_increasing


# Test Feature Postprocessing
class TestFeaturePostprocessing:
    """Test feature postprocessing functionality."""
    
    def test_postprocess_features_removes_infinite(self, test_calculator):
        """Test removal of infinite values."""
        features = pd.DataFrame({
            'feature1': [1, 2, np.inf, 4, 5],
            'feature2': [1, -np.inf, 3, 4, 5]
        })
        
        result = test_calculator.postprocess_features(features)
        
        # Infinite values should be replaced with NaN
        assert not np.isinf(result).any().any()
        assert pd.isna(result['feature1'].iloc[2])
        assert pd.isna(result['feature2'].iloc[1])
    
    def test_postprocess_features_fills_nan(self, test_calculator):
        """Test NaN filling."""
        features = pd.DataFrame({
            'feature1': [1, np.nan, 3, 4, 5],
            'feature2': [np.nan, 2, 3, np.nan, 5]
        })
        
        result = test_calculator.postprocess_features(features)
        
        # Should fill NaN values
        assert not result.isna().any().any()
    
    def test_postprocess_features_clips_extremes(self, default_config):
        """Test clipping of extreme values."""
        config = default_config.copy()
        config['clip_percentile'] = 10  # Clip at 10th and 90th percentile
        calculator = ConcreteTestCalculator(config)
        
        # Create data with extreme values
        values = list(range(100)) + [1000, -1000]  # Extreme outliers
        features = pd.DataFrame({'feature': values})
        
        result = calculator.postprocess_features(features)
        
        # Extreme values should be clipped
        assert result['feature'].max() < 1000
        assert result['feature'].min() > -1000
    
    def test_postprocess_features_adds_metadata(self, test_calculator):
        """Test that metadata is added to features."""
        features = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        
        result = test_calculator.postprocess_features(features)
        
        # Should have metadata
        assert hasattr(result, 'attrs')
        assert 'calculator_name' in result.attrs
        assert result.attrs['calculator_name'] == 'ConcreteTestCalculator'


# Test Validation Pipeline
class TestValidationPipeline:
    """Test the complete validation pipeline."""
    
    def test_calculate_with_validation_success(self, test_calculator, sample_market_data):
        """Test successful calculation with validation."""
        result = test_calculator.calculate_with_validation(sample_market_data)
        
        # Should return calculated features
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'test_sma' in result.columns
        assert 'test_std' in result.columns
        # Should track timing
        assert len(test_calculator._calculation_times) > 0
    
    def test_calculate_with_validation_input_failure_strict(self):
        """Test input validation failure in strict mode."""
        calculator = StrictValidationCalculator({'strict_validation': True})
        insufficient_data = pd.DataFrame({'close': [100, 101]})  # Less than 10 rows
        
        with pytest.raises(ValueError, match="Input validation failed"):
            calculator.calculate_with_validation(insufficient_data)
    
    def test_calculate_with_validation_input_failure_non_strict(self):
        """Test input validation failure in non-strict mode."""
        calculator = StrictValidationCalculator({'strict_validation': False})
        insufficient_data = pd.DataFrame({'close': [100, 101]})
        
        result = calculator.calculate_with_validation(insufficient_data)
        
        # Should return empty DataFrame instead of raising
        assert result.empty
        assert len(result.index) == len(insufficient_data.index)
    
    def test_calculate_with_validation_calculation_failure_strict(self):
        """Test calculation failure in strict mode."""
        calculator = AlwaysFailingCalculator({'strict_validation': True})
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        with pytest.raises(RuntimeError, match="Calculator always fails"):
            calculator.calculate_with_validation(data)
    
    def test_calculate_with_validation_calculation_failure_non_strict(self, caplog):
        """Test calculation failure in non-strict mode."""
        calculator = AlwaysFailingCalculator({'strict_validation': False})
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        result = calculator.calculate_with_validation(data)
        
        # Should return empty DataFrame and log error
        assert result.empty
        assert "Error in AlwaysFailingCalculator calculation" in caplog.text
    
    def test_calculate_with_validation_output_validation(self, test_calculator, sample_market_data, caplog):
        """Test output validation warnings."""
        # Mock the calculate method to return mismatched index
        original_calculate = test_calculator.calculate
        
        def bad_calculate(data):
            features = pd.DataFrame({'bad_feature': [1, 2, 3]})  # Wrong index
            return features
        
        test_calculator.calculate = bad_calculate
        
        result = test_calculator.calculate_with_validation(sample_market_data)
        
        # Should log warning about index mismatch
        assert "Feature index doesn't match input index" in caplog.text
        
        # Restore original method
        test_calculator.calculate = original_calculate


# Test Performance Monitoring
class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def test_calculation_times_tracking(self, test_calculator, sample_market_data):
        """Test that calculation times are tracked."""
        # Perform multiple calculations
        test_calculator.calculate_with_validation(sample_market_data)
        test_calculator.calculate_with_validation(sample_market_data)
        
        # Should have tracked timing
        assert len(test_calculator._calculation_times) == 2
        assert all(t > 0 for t in test_calculator._calculation_times)
    
    def test_get_calculation_stats(self, test_calculator, sample_market_data):
        """Test getting calculation statistics."""
        # Perform calculations to generate stats
        for _ in range(5):
            test_calculator.calculate_with_validation(sample_market_data)
        
        stats = test_calculator.get_calculation_stats()
        
        assert 'mean_time' in stats
        assert 'std_time' in stats
        assert 'min_time' in stats
        assert 'max_time' in stats
        assert 'total_calculations' in stats
        assert stats['total_calculations'] == 5
    
    def test_get_calculation_stats_no_data(self, test_calculator):
        """Test getting stats when no calculations performed."""
        stats = test_calculator.get_calculation_stats()
        assert stats == {}


# Test Feature Importance
class TestFeatureImportance:
    """Test feature importance functionality."""
    
    def test_set_and_get_feature_importance(self, test_calculator):
        """Test setting and getting feature importance."""
        importance = {'feature1': 0.8, 'feature2': 0.6, 'feature3': 0.4}
        
        test_calculator.set_feature_importance(importance)
        result = test_calculator.get_feature_importance()
        
        assert result == importance
    
    def test_feature_importance_empty_by_default(self, test_calculator):
        """Test that feature importance is empty by default."""
        importance = test_calculator.get_feature_importance()
        assert importance == {}


# Test Caching
class TestCaching:
    """Test caching functionality."""
    
    def test_cache_initialization(self, test_calculator):
        """Test that cache is initialized."""
        assert hasattr(test_calculator, '_cache')
        assert test_calculator._cache == {}
    
    def test_reset_cache(self, test_calculator, caplog):
        """Test cache reset functionality."""
        # Add something to cache
        test_calculator._cache['test_key'] = 'test_value'
        
        test_calculator.reset_cache()
        
        assert test_calculator._cache == {}
        assert "Cache cleared for ConcreteTestCalculator" in caplog.text


# Test Configuration Persistence
class TestConfigurationPersistence:
    """Test configuration saving and loading."""
    
    def test_save_config(self, test_calculator, tmp_path):
        """Test saving configuration to file."""
        config_file = tmp_path / "test_config.json"
        
        # Set some state
        test_calculator.set_feature_importance({'feature1': 0.5})
        
        test_calculator.save_config(str(config_file))
        
        # File should exist and contain valid JSON
        assert config_file.exists()
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        assert 'calculator' in config_data
        assert 'config' in config_data
        assert 'metadata' in config_data
        assert 'feature_importance' in config_data
        assert config_data['calculator'] == 'ConcreteTestCalculator'
    
    def test_load_config(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.json"
        
        # Create config file
        config_data = {
            'calculator': 'ConcreteTestCalculator',
            'config': {'test_param': 'test_value'},
            'metadata': {'version': '2.0.0'},
            'feature_importance': {'feature1': 0.7}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load calculator
        calculator = ConcreteTestCalculator.load_config(str(config_file))
        
        assert calculator.config['test_param'] == 'test_value'
        assert calculator._metadata['version'] == '2.0.0'
        assert calculator._feature_importance['feature1'] == 0.7


# Test Feature Combination
class TestFeatureCombination:
    """Test feature combination functionality."""
    
    def test_combine_features_basic(self, test_calculator):
        """Test basic feature combination."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        
        df1 = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]}, index=dates)
        df2 = pd.DataFrame({'feature2': [6, 7, 8, 9, 10]}, index=dates)
        
        result = test_calculator.combine_features(df1, df2)
        
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert len(result) == 5
    
    def test_combine_features_mismatched_indices(self, test_calculator):
        """Test combination with mismatched indices."""
        dates1 = pd.date_range('2024-01-01', periods=5, freq='D')
        dates2 = pd.date_range('2024-01-03', periods=5, freq='D')
        
        df1 = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]}, index=dates1)
        df2 = pd.DataFrame({'feature2': [6, 7, 8, 9, 10]}, index=dates2)
        
        result = test_calculator.combine_features(df1, df2)
        
        # Should only include common indices
        assert len(result) == 3  # 3 overlapping dates
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
    
    def test_combine_features_empty_input(self, test_calculator):
        """Test combination with empty input."""
        result = test_calculator.combine_features()
        assert result.empty
    
    def test_combine_features_removes_duplicates(self, test_calculator):
        """Test that duplicate columns are removed."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        
        df1 = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'common': [1, 1, 1, 1, 1]}, index=dates)
        df2 = pd.DataFrame({'feature2': [6, 7, 8, 9, 10], 'common': [2, 2, 2, 2, 2]}, index=dates)
        
        result = test_calculator.combine_features(df1, df2)
        
        # Should not have duplicate columns
        assert list(result.columns).count('common') == 1


# Test String Representations
class TestStringRepresentations:
    """Test string representation methods."""
    
    def test_repr(self, test_calculator):
        """Test __repr__ method."""
        repr_str = repr(test_calculator)
        assert 'ConcreteTestCalculator' in repr_str
        assert 'config=' in repr_str
    
    def test_str(self, test_calculator):
        """Test __str__ method."""
        str_repr = str(test_calculator)
        assert str_repr == 'ConcreteTestCalculator'


if __name__ == "__main__":
    pytest.main([__file__])