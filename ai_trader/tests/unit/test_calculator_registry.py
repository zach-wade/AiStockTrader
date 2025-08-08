# tests/unit/test_calculator_registry.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, List, Any

from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.feature_pipeline.calculators.base_calculator import BaseFeatureCalculator
from omegaconf import DictConfig, OmegaConf


# Mock Calculator Classes for Testing
class MockTechnicalCalculator(BaseFeatureCalculator):
    """Mock technical calculator for testing."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock technical features."""
        features = pd.DataFrame(index=data.index)
        features['sma_10'] = data['close'].rolling(10).mean()
        features['rsi'] = np.secure_uniform(0, 100, len(data))
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate inputs."""
        required_cols = ['close', 'volume']
        return all(col in data.columns for col in required_cols)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ['close', 'volume']


class MockFailingCalculator(BaseFeatureCalculator):
    """Mock calculator that always fails for testing error handling."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Always raises an exception."""
        raise ValueError("Mock calculator failure")
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Always fails validation."""
        return False
    
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ['close']


class MockStatisticalCalculator(BaseFeatureCalculator):
    """Mock statistical calculator for testing."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock statistical features."""
        features = pd.DataFrame(index=data.index)
        features['volatility'] = data['close'].rolling(20).std()
        features['skewness'] = data['close'].rolling(30).skew()
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate inputs."""
        return 'close' in data.columns and len(data) >= 30
    
    def get_required_columns(self) -> List[str]:
        """Get required columns."""
        return ['close']


# Test Fixtures
@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D', tz=timezone.utc)
    np.random.seed(42)  # For reproducible tests
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 100)),
        'open': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 100)),
        'high': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 100)) + np.abs(secure_numpy_normal(0, 0.01, 100)),
        'low': 100 + np.cumsum(secure_numpy_normal(0, 0.02, 100)) - np.abs(secure_numpy_normal(0, 0.01, 100)),
        'volume': np.secure_randint(10000, 100000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        'features.enabled': True,
        'features.calculators': ['technical', 'statistical'],
        'features.parallel_processing': False
    }.get(key, default)
    return config


@pytest.fixture
def mock_engine_with_mocks(mock_config):
    """Create UnifiedFeatureEngine with mock calculators."""
    engine = UnifiedFeatureEngine.__new__(UnifiedFeatureEngine)
    engine.config = mock_config
    engine.calculators = {
        'technical': MockTechnicalCalculator(),
        'statistical': MockStatisticalCalculator(),
        'failing': MockFailingCalculator()
    }
    return engine


# Test UnifiedFeatureEngine Initialization
class TestUnifiedFeatureEngineInit:
    """Test UnifiedFeatureEngine initialization and registry setup."""
    
    @patch('main.feature_pipeline.unified_feature_engine.TechnicalIndicators')
    @patch('main.feature_pipeline.unified_feature_engine.MarketRegimeCalculator')
    def test_init_loads_calculators(self, mock_regime, mock_technical, mock_config):
        """Test that initialization loads calculator instances."""
        # Setup mocks
        mock_technical.return_value = MockTechnicalCalculator()
        mock_regime.return_value = MockStatisticalCalculator()
        
        engine = UnifiedFeatureEngine(mock_config)
        
        # Should have loaded calculators
        assert 'technical' in engine.calculators
        assert 'regime' in engine.calculators
        mock_technical.assert_called_once_with(mock_config)
        mock_regime.assert_called_once_with(mock_config)
    
    @patch('main.feature_pipeline.unified_feature_engine.TechnicalIndicators')
    def test_init_handles_calculator_loading_errors(self, mock_technical, mock_config, caplog):
        """Test graceful handling of calculator loading errors."""
        # Make one calculator fail to load
        mock_technical.side_effect = ImportError("Module not found")
        
        engine = UnifiedFeatureEngine(mock_config)
        
        # Should log warning and continue with fallback
        assert "Failed to load some calculators" in caplog.text
        # Should have at least the fallback technical calculator
        assert 'technical' in engine.calculators
    
    def test_init_creates_calculator_instances(self, mock_config):
        """Test that all calculator instances are created properly."""
        with patch.object(UnifiedFeatureEngine, '_load_calculators') as mock_load:
            mock_load.return_value = {
                'technical': MockTechnicalCalculator(),
                'statistical': MockStatisticalCalculator()
            }
            
            engine = UnifiedFeatureEngine(mock_config)
            
            assert len(engine.calculators) == 2
            assert isinstance(engine.calculators['technical'], BaseFeatureCalculator)
            assert isinstance(engine.calculators['statistical'], BaseFeatureCalculator)


# Test Calculator Registry Functionality
class TestCalculatorRegistry:
    """Test calculator registry and discovery functionality."""
    
    def test_get_available_calculators(self, mock_engine_with_mocks):
        """Test getting list of available calculators."""
        engine = mock_engine_with_mocks
        
        available = list(engine.calculators.keys())
        
        assert 'technical' in available
        assert 'statistical' in available
        assert 'failing' in available
        assert len(available) == 3
    
    def test_calculator_instances_are_correct_type(self, mock_engine_with_mocks):
        """Test that calculator instances are of correct type."""
        engine = mock_engine_with_mocks
        
        for name, calculator in engine.calculators.items():
            assert isinstance(calculator, BaseFeatureCalculator)
            assert hasattr(calculator, 'calculate')
            assert hasattr(calculator, 'validate_inputs')
            assert hasattr(calculator, 'get_required_columns')
    
    def test_calculator_configuration_passing(self, mock_config):
        """Test that configuration is passed to calculators."""
        with patch('main.feature_pipeline.unified_feature_engine.TechnicalIndicators') as mock_calc:
            mock_calc.return_value = MockTechnicalCalculator()
            
            engine = UnifiedFeatureEngine(mock_config)
            
            # Calculator should be initialized with config
            mock_calc.assert_called_with(mock_config)
    
    def test_calculator_registry_completeness(self, mock_config):
        """Test that all expected calculators are registered."""
        expected_calculators = [
            'technical', 'regime', 'cross_sectional', 'sentiment',
            'news_features', 'microstructure', 'options', 'insider',
            'sector', 'statistical'
        ]
        
        with patch.object(UnifiedFeatureEngine, '_load_calculators') as mock_load:
            # Mock all calculators as successfully loaded
            mock_calculators = {name: MockTechnicalCalculator() for name in expected_calculators}
            mock_load.return_value = mock_calculators
            
            engine = UnifiedFeatureEngine(mock_config)
            
            for expected in expected_calculators:
                assert expected in engine.calculators


# Test Feature Calculation
class TestFeatureCalculation:
    """Test feature calculation through the registry."""
    
    def test_calculate_for_dataframe_basic(self, mock_engine_with_mocks, sample_market_data):
        """Test basic feature calculation."""
        engine = mock_engine_with_mocks
        
        result = engine.calculate_for_dataframe(sample_market_data)
        
        # Should return DataFrame with original and new features
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_market_data)
        # Should have original columns
        assert 'close' in result.columns
        # Should have new features from mock calculators
        assert 'sma_10' in result.columns
        assert 'rsi' in result.columns
        assert 'volatility' in result.columns
    
    def test_calculate_for_dataframe_empty_data(self, mock_engine_with_mocks):
        """Test calculation with empty DataFrame."""
        engine = mock_engine_with_mocks
        empty_data = pd.DataFrame()
        
        result = engine.calculate_for_dataframe(empty_data)
        
        assert result.empty
    
    def test_calculate_for_dataframe_handles_calculator_errors(self, mock_engine_with_mocks, sample_market_data, caplog):
        """Test that calculator errors don't stop processing."""
        engine = mock_engine_with_mocks
        
        result = engine.calculate_for_dataframe(sample_market_data)
        
        # Should log error for failing calculator but continue
        assert "Error applying feature calculator 'failing'" in caplog.text
        # Should still have results from working calculators
        assert 'sma_10' in result.columns
        assert 'volatility' in result.columns
    
    def test_calculate_features_with_specific_calculators(self, mock_engine_with_mocks, sample_market_data):
        """Test calculating features with specific calculator selection."""
        engine = mock_engine_with_mocks
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical']  # Only technical calculator
        )
        
        # Should only have features from technical calculator
        assert 'sma_10' in result.columns
        assert 'rsi' in result.columns
        # Should not have features from other calculators
        assert 'volatility' not in result.columns
    
    def test_calculate_features_with_validation(self, mock_engine_with_mocks, sample_market_data, caplog):
        """Test feature calculation with validation."""
        engine = mock_engine_with_mocks
        
        # Add validate_inputs method to mock
        engine.calculators['technical'].validate_inputs = Mock(return_value=True)
        engine.calculators['statistical'].validate_inputs = Mock(return_value=False)
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST'
        )
        
        # Should skip statistical calculator due to validation failure
        assert "Skipping statistical for TEST - validation failed" in caplog.text
        assert 'sma_10' in result.columns  # From technical (passed validation)
    
    def test_calculate_features_adds_metadata(self, mock_engine_with_mocks, sample_market_data):
        """Test that metadata is added to results."""
        engine = mock_engine_with_mocks
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical']
        )
        
        # Should have metadata
        assert 'symbol' in result.attrs
        assert result.attrs['symbol'] == 'TEST'
        assert 'feature_timestamp' in result.attrs
        assert 'calculators_used' in result.attrs
        assert 'technical' in result.attrs['calculators_used']
    
    def test_calculate_features_empty_data_warning(self, mock_engine_with_mocks, caplog):
        """Test warning for empty data."""
        engine = mock_engine_with_mocks
        empty_data = pd.DataFrame()
        
        result = engine.calculate_features(empty_data, 'TEST')
        
        assert "Empty data provided for TEST" in caplog.text
        assert result.empty
    
    def test_calculate_features_all_calculators_by_default(self, mock_engine_with_mocks, sample_market_data):
        """Test that all calculators are used by default."""
        engine = mock_engine_with_mocks
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST'
        )
        
        # Should use all available calculators (except failing one that errors)
        calculators_used = result.attrs.get('calculators_used', [])
        assert 'technical' in calculators_used
        assert 'statistical' in calculators_used


# Test Error Handling and Edge Cases
class TestErrorHandling:
    """Test error handling in calculator registry."""
    
    def test_calculator_instantiation_failure(self, mock_config, caplog):
        """Test handling of calculator instantiation failures."""
        with patch('main.feature_pipeline.unified_feature_engine.TechnicalIndicators') as mock_tech:
            with patch('main.feature_pipeline.unified_feature_engine.MarketRegimeCalculator') as mock_regime:
                # Make most calculators fail
                mock_tech.side_effect = Exception("Init failed")
                mock_regime.side_effect = Exception("Init failed")
                
                engine = UnifiedFeatureEngine(mock_config)
                
                # Should handle errors and provide fallback
                assert "Failed to load some calculators" in caplog.text
                # Should have at least the fallback technical calculator
                assert 'technical' in engine.calculators
    
    def test_missing_calculator_graceful_handling(self, mock_engine_with_mocks, sample_market_data):
        """Test graceful handling of missing calculators."""
        engine = mock_engine_with_mocks
        
        # Request non-existent calculator
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical', 'nonexistent']
        )
        
        # Should only process existing calculators
        assert 'sma_10' in result.columns  # From technical
        calculators_used = result.attrs.get('calculators_used', [])
        assert 'technical' in calculators_used
        assert 'nonexistent' not in calculators_used
    
    def test_calculator_method_missing(self, mock_engine_with_mocks, sample_market_data, caplog):
        """Test handling when calculator methods are missing."""
        engine = mock_engine_with_mocks
        
        # Remove method from calculator
        del engine.calculators['technical'].validate_inputs
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical']
        )
        
        # Should still work, just skip validation
        assert 'sma_10' in result.columns
    
    def test_feature_merge_index_mismatch(self, mock_engine_with_mocks, sample_market_data):
        """Test handling of index mismatches during feature merging."""
        engine = mock_engine_with_mocks
        
        # Mock calculator that returns mismatched index
        def bad_calculate(data):
            # Return features with different index
            bad_index = pd.date_range('2025-01-01', periods=len(data), freq='D')
            features = pd.DataFrame({'bad_feature': [1] * len(data)}, index=bad_index)
            return features
        
        engine.calculators['technical'].calculate = bad_calculate
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical']
        )
        
        # Should handle gracefully (reindex will add NaNs or skip)
        assert isinstance(result, pd.DataFrame)


# Test Performance and Caching
class TestPerformanceAndCaching:
    """Test performance monitoring and caching functionality."""
    
    def test_calculation_timing_tracking(self, mock_engine_with_mocks, sample_market_data):
        """Test that calculation timing is tracked."""
        engine = mock_engine_with_mocks
        
        # Add timing tracking to mock calculator
        calc = engine.calculators['technical']
        original_calculate = calc.calculate
        
        def timed_calculate(data):
            import time
            time.sleep(0.01)  # Small delay for timing
            return original_calculate(data)
        
        calc.calculate = timed_calculate
        
        result = engine.calculate_features(
            data=sample_market_data,
            symbol='TEST',
            calculators=['technical']
        )
        
        # Should have timing information if available
        assert isinstance(result, pd.DataFrame)
    
    def test_multiple_calculation_calls(self, mock_engine_with_mocks, sample_market_data):
        """Test multiple calculation calls work correctly."""
        engine = mock_engine_with_mocks
        
        # First calculation
        result1 = engine.calculate_features(sample_market_data, 'TEST1')
        
        # Second calculation with different symbol
        result2 = engine.calculate_features(sample_market_data, 'TEST2')
        
        # Both should succeed
        assert not result1.empty
        assert not result2.empty
        assert result1.attrs['symbol'] == 'TEST1'
        assert result2.attrs['symbol'] == 'TEST2'
    
    def test_calculator_state_isolation(self, mock_engine_with_mocks, sample_market_data):
        """Test that calculator state is properly isolated between calls."""
        engine = mock_engine_with_mocks
        
        # Add state to calculator
        engine.calculators['technical']._test_state = 0
        
        def stateful_calculate(data):
            self = engine.calculators['technical']
            self._test_state += 1
            features = pd.DataFrame(index=data.index)
            features['state_feature'] = self._test_state
            return features
        
        engine.calculators['technical'].calculate = stateful_calculate
        
        # Multiple calls should see state changes
        result1 = engine.calculate_features(sample_market_data, 'TEST1', ['technical'])
        result2 = engine.calculate_features(sample_market_data, 'TEST2', ['technical'])
        
        assert result1['state_feature'].iloc[0] == 1
        assert result2['state_feature'].iloc[0] == 2


# Test Configuration Integration
class TestConfigurationIntegration:
    """Test integration with configuration system."""
    
    @patch('main.feature_pipeline.unified_feature_engine.TechnicalIndicators')
    def test_config_passed_to_calculators(self, mock_tech, mock_config):
        """Test that configuration is properly passed to calculators."""
        mock_tech.return_value = MockTechnicalCalculator()
        
        engine = UnifiedFeatureEngine(mock_config)
        
        # Each calculator should receive the config
        mock_tech.assert_called_with(mock_config)
    
    def test_engine_stores_config_reference(self, mock_config):
        """Test that engine stores configuration reference."""
        with patch.object(UnifiedFeatureEngine, '_load_calculators'):
            engine = UnifiedFeatureEngine(mock_config)
            
            assert engine.config is mock_config


if __name__ == "__main__":
    pytest.main([__file__])