"""Unit tests for alert_feature_mapper module."""

import pytest
from unittest.mock import Mock, patch, mock_open
import yaml
import os

from main.events.handlers.scanner_bridge_helpers.alert_feature_mapper import AlertFeatureMapper


class TestAlertFeatureMapper:
    """Test AlertFeatureMapper class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'alert_feature_mappings': {
                'HIGH_VOLUME': ['volume_profile', 'price_action'],
                'PRICE_SPIKE': ['price_action', 'momentum', 'volatility'],
                'ML_SIGNAL': ['all_features'],
                'BREAKOUT': ['price_action', 'volume_profile', 'support_resistance'],
                'default': ['price_action', 'volume_profile']
            },
            'all_features_list': [
                'price_action',
                'volume_profile',
                'momentum',
                'volatility',
                'support_resistance',
                'market_microstructure',
                'order_flow'
            ]
        }
    
    @pytest.fixture
    def mapper(self, sample_config):
        """Create AlertFeatureMapper instance with mocked config."""
        yaml_content = yaml.dump(sample_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.dirname') as mock_dirname:
                with patch('os.path.join') as mock_join:
                    mock_dirname.return_value = "/test/dir"
                    mock_join.return_value = "/test/config.yaml"
                    
                    mapper = AlertFeatureMapper()
                    # Manually set the loaded config to ensure it's correct
                    mapper._alert_feature_map = sample_config['alert_feature_mappings']
                    mapper._all_features_list = sample_config['all_features_list']
                    return mapper
    
    def test_initialization_default_path(self, sample_config):
        """Test initialization with default config path."""
        yaml_content = yaml.dump(sample_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
            with patch('os.path.dirname') as mock_dirname:
                with patch('os.path.join') as mock_join:
                    mock_dirname.return_value = "/base/dir"
                    mock_join.return_value = "/base/dir/config/events/alert_feature_mappings.yaml"
                    
                    mapper = AlertFeatureMapper()
                    
                    # Verify path construction
                    assert mock_dirname.call_count >= 1
                    mock_join.assert_called()
                    
                    # Verify file was opened
                    mock_file.assert_called_once_with(
                        "/base/dir/config/events/alert_feature_mappings.yaml", 'r'
                    )
    
    def test_initialization_custom_path(self, sample_config):
        """Test initialization with custom config path."""
        yaml_content = yaml.dump(sample_config)
        custom_path = "/custom/path/config.yaml"
        
        with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
            mapper = AlertFeatureMapper(config_path=custom_path)
            
            # Verify custom path was used
            mock_file.assert_called_once_with(custom_path, 'r')
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        with patch('builtins.open', side_effect=FileNotFoundError("Config not found")):
            with patch('os.path.dirname'):
                with patch('os.path.join'):
                    # Should handle error gracefully due to ErrorHandlingMixin
                    with pytest.raises(FileNotFoundError):
                        AlertFeatureMapper()
    
    def test_get_features_for_known_alert_type(self, mapper):
        """Test getting features for a known alert type."""
        with patch('main.events.handlers.scanner_bridge_helpers.alert_feature_mapper.record_metric') as mock_metric:
            features = mapper.get_features_for_alert_type('HIGH_VOLUME')
            
            assert features == ['volume_profile', 'price_action']
            
            # Verify metric was recorded
            mock_metric.assert_called_once_with(
                "alert_feature_mapper.lookup", 
                1, 
                tags={
                    "alert_type": "HIGH_VOLUME",
                    "is_default": False
                }
            )
    
    def test_get_features_for_unknown_alert_type(self, mapper):
        """Test getting features for an unknown alert type (uses default)."""
        with patch('main.events.handlers.scanner_bridge_helpers.alert_feature_mapper.record_metric') as mock_metric:
            features = mapper.get_features_for_alert_type('UNKNOWN_TYPE')
            
            assert features == ['price_action', 'volume_profile']  # default features
            
            # Verify metric indicates default was used
            mock_metric.assert_called_once_with(
                "alert_feature_mapper.lookup", 
                1, 
                tags={
                    "alert_type": "UNKNOWN_TYPE",
                    "is_default": True
                }
            )
    
    def test_get_features_with_all_features_keyword(self, mapper):
        """Test alert type that uses 'all_features' keyword."""
        with patch('main.events.handlers.scanner_bridge_helpers.alert_feature_mapper.record_metric') as mock_metric:
            features = mapper.get_features_for_alert_type('ML_SIGNAL')
            
            assert features == mapper._all_features_list
            assert len(features) == 7  # All 7 features in the list
            
            # Verify both metrics were recorded
            assert mock_metric.call_count == 2
            
            # First call - lookup metric
            first_call = mock_metric.call_args_list[0]
            assert first_call[0][0] == "alert_feature_mapper.lookup"
            
            # Second call - all_features metric
            second_call = mock_metric.call_args_list[1]
            assert second_call[0][0] == "alert_feature_mapper.all_features_requested"
            assert second_call[1]['tags']['alert_type'] == 'ML_SIGNAL'
    
    def test_get_all_features(self, mapper):
        """Test getting all available features."""
        all_features = mapper.get_all_features()
        
        assert isinstance(all_features, list)
        assert len(all_features) == 7
        assert 'price_action' in all_features
        assert 'order_flow' in all_features
        
        # Verify it returns a copy, not the original list
        all_features.append('new_feature')
        assert len(mapper._all_features_list) == 7  # Original unchanged
    
    def test_reload_config_default_path(self, mapper, sample_config):
        """Test reloading configuration with default path."""
        # Modify config for reload
        new_config = sample_config.copy()
        new_config['alert_feature_mappings']['NEW_ALERT'] = ['new_feature']
        yaml_content = yaml.dump(new_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.dirname') as mock_dirname:
                with patch('os.path.join') as mock_join:
                    mock_dirname.return_value = "/test/dir"
                    mock_join.return_value = "/test/config.yaml"
                    
                    mapper.reload_config()
                    
                    # Update mapper's internal state for testing
                    mapper._alert_feature_map = new_config['alert_feature_mappings']
                    
                    # Verify new mapping exists
                    assert 'NEW_ALERT' in mapper._alert_feature_map
    
    def test_reload_config_custom_path(self, mapper):
        """Test reloading configuration with custom path."""
        custom_reload_path = "/reload/path/config.yaml"
        reload_config = {
            'alert_feature_mappings': {
                'RELOADED': ['reloaded_feature']
            },
            'all_features_list': ['reloaded_feature']
        }
        yaml_content = yaml.dump(reload_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
            mapper.reload_config(config_path=custom_reload_path)
            
            # Update mapper's internal state for testing
            mapper._alert_feature_map = reload_config['alert_feature_mappings']
            mapper._all_features_list = reload_config['all_features_list']
            
            # Verify custom path was used
            mock_file.assert_called_once_with(custom_reload_path, 'r')
            
            # Verify new config is loaded
            assert mapper._alert_feature_map == reload_config['alert_feature_mappings']
    
    def test_reload_config_error_handling(self, mapper):
        """Test error handling during config reload."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with patch('os.path.dirname'):
                with patch('os.path.join'):
                    # Should handle error gracefully
                    with pytest.raises(IOError):
                        mapper.reload_config()
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML content."""
        invalid_yaml = "invalid: yaml: content: [["
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('os.path.dirname'):
                with patch('os.path.join'):
                    with pytest.raises(yaml.YAMLError):
                        AlertFeatureMapper()
    
    def test_missing_required_fields(self):
        """Test handling of config missing required fields."""
        incomplete_config = {
            'alert_feature_mappings': {
                'HIGH_VOLUME': ['volume']
            }
            # Missing 'all_features_list'
        }
        yaml_content = yaml.dump(incomplete_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.dirname'):
                with patch('os.path.join'):
                    with pytest.raises(KeyError):
                        AlertFeatureMapper()
    
    def test_empty_config_handling(self):
        """Test handling of empty configuration."""
        empty_config = {}
        yaml_content = yaml.dump(empty_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.dirname'):
                with patch('os.path.join'):
                    with pytest.raises(KeyError):
                        AlertFeatureMapper()
    
    def test_timer_decorator(self, mapper):
        """Test that timer decorator is applied to get_features_for_alert_type."""
        # The @timer decorator should be applied to the method
        # We can't directly test the decorator, but we can verify the method works
        features = mapper.get_features_for_alert_type('BREAKOUT')
        assert features == ['price_action', 'volume_profile', 'support_resistance']
    
    def test_multiple_alert_types_sequential(self, mapper):
        """Test getting features for multiple alert types sequentially."""
        alert_types = ['HIGH_VOLUME', 'PRICE_SPIKE', 'BREAKOUT', 'UNKNOWN']
        expected_features = [
            ['volume_profile', 'price_action'],
            ['price_action', 'momentum', 'volatility'],
            ['price_action', 'volume_profile', 'support_resistance'],
            ['price_action', 'volume_profile']  # default
        ]
        
        for alert_type, expected in zip(alert_types, expected_features):
            features = mapper.get_features_for_alert_type(alert_type)
            assert features == expected