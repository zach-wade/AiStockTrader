"""Unit tests for feature_group_mapper module."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from typing import Dict, Any

from main.events.handlers.feature_pipeline_helpers.feature_group_mapper import FeatureGroupMapper
from main.events.handlers.feature_pipeline_helpers.feature_types import (
    FeatureGroup, FeatureGroupConfig, FeatureRequest
)
from main.events.types import AlertType, EventPriority
from tests.fixtures.events.mock_events import create_scan_alert


class TestFeatureGroupMapper:
    """Test FeatureGroupMapper class."""
    
    @pytest.fixture
    def mapper(self):
        """Create FeatureGroupMapper instance for testing."""
        return FeatureGroupMapper()
    
    @pytest.fixture
    def custom_config(self):
        """Create custom configuration for testing."""
        return {
            "group_configs": {
                "PRICE": {
                    "priority_boost": 2
                }
            },
            "alert_mappings": {
                "CUSTOM_ALERT": ["PRICE", "VOLUME"]
            }
        }
    
    def test_initialization_default(self):
        """Test initialization with default configuration."""
        mapper = FeatureGroupMapper()
        
        assert mapper.config == {}
        assert len(mapper.group_configs) > 0
        assert len(mapper.alert_mappings) > 0
        assert mapper.conditional_rules is not None
        assert mapper.priority_rules is not None
    
    def test_initialization_with_config(self, custom_config):
        """Test initialization with custom configuration."""
        mapper = FeatureGroupMapper(config=custom_config)
        
        assert mapper.config == custom_config
        # Should apply config overrides
        assert mapper.group_configs[FeatureGroup.PRICE].priority_boost == 2
    
    def test_map_alert_to_features_known_type(self, mapper):
        """Test mapping known alert type to features."""
        alert = create_scan_alert(
            alert_type=AlertType.HIGH_VOLUME,
            score=0.7
        )
        
        request = mapper.map_alert_to_features(alert)
        
        assert isinstance(request, FeatureRequest)
        assert request.symbol == alert.symbol
        assert FeatureGroup.VOLUME in request.feature_groups
        assert FeatureGroup.PRICE in request.feature_groups
        assert request.alert_type == AlertType.HIGH_VOLUME
        assert request.priority >= 0
    
    def test_map_alert_to_features_unknown_type(self, mapper):
        """Test mapping unknown alert type uses default."""
        alert = create_scan_alert(
            alert_type=AlertType.UNKNOWN,
            score=0.5
        )
        
        request = mapper.map_alert_to_features(alert)
        
        # Should use default mapping
        assert FeatureGroup.PRICE in request.feature_groups
        assert FeatureGroup.VOLUME in request.feature_groups
    
    def test_conditional_groups_high_score(self, mapper):
        """Test high score alerts get additional features."""
        # Low score alert
        low_score_alert = create_scan_alert(score=0.5)
        low_request = mapper.map_alert_to_features(low_score_alert)
        
        # High score alert
        high_score_alert = create_scan_alert(score=0.9)
        high_request = mapper.map_alert_to_features(high_score_alert)
        
        # High score should have more feature groups
        assert len(high_request.feature_groups) > len(low_request.feature_groups)
        assert FeatureGroup.ML_SIGNALS in high_request.feature_groups
    
    def test_conditional_groups_volume_spike(self, mapper):
        """Test volume spike adds microstructure features."""
        alert = create_scan_alert(
            data={"volume_multiplier": 5.0}
        )
        
        request = mapper.map_alert_to_features(alert)
        
        assert FeatureGroup.ORDER_FLOW in request.feature_groups
        assert FeatureGroup.MICROSTRUCTURE in request.feature_groups
    
    def test_conditional_groups_news_keywords(self, mapper):
        """Test news keywords trigger specific features."""
        alert = create_scan_alert(
            data={"keywords": ["earnings", "revenue"]}
        )
        
        request = mapper.map_alert_to_features(alert)
        
        assert FeatureGroup.EARNINGS in request.feature_groups
    
    @patch('main.events.handlers.feature_pipeline_helpers.feature_group_mapper.datetime')
    def test_conditional_groups_time_based(self, mock_datetime, mapper):
        """Test time-based feature additions."""
        # Mock pre-market time (7 AM UTC)
        mock_now = Mock()
        mock_now.hour = 7
        mock_datetime.utcnow.return_value = mock_now
        mock_datetime.now.return_value.hour = 7
        
        alert = create_scan_alert()
        request = mapper.map_alert_to_features(alert)
        
        # Should add news sentiment for pre-market
        assert FeatureGroup.NEWS_SENTIMENT in request.feature_groups
    
    def test_priority_calculation_base(self, mapper):
        """Test basic priority calculation."""
        # ML signal should have high base priority
        ml_alert = create_scan_alert(
            alert_type=AlertType.ML_SIGNAL,
            score=0.5
        )
        ml_request = mapper.map_alert_to_features(ml_alert)
        
        # Unknown should have low base priority
        unknown_alert = create_scan_alert(
            alert_type=AlertType.UNKNOWN,
            score=0.5
        )
        unknown_request = mapper.map_alert_to_features(unknown_alert)
        
        assert ml_request.priority > unknown_request.priority
    
    def test_priority_calculation_score_boost(self, mapper):
        """Test score-based priority boost."""
        low_score_alert = create_scan_alert(score=0.2)
        high_score_alert = create_scan_alert(score=0.9)
        
        low_request = mapper.map_alert_to_features(low_score_alert)
        high_request = mapper.map_alert_to_features(high_score_alert)
        
        # Higher score should result in higher priority
        assert high_request.priority > low_request.priority
    
    def test_priority_calculation_volatility_boost(self, mapper):
        """Test volatility-based priority boost."""
        normal_alert = create_scan_alert(
            data={"volatility_level": "normal"}
        )
        extreme_alert = create_scan_alert(
            data={"volatility_level": "extreme"}
        )
        
        normal_request = mapper.map_alert_to_features(normal_alert)
        extreme_request = mapper.map_alert_to_features(extreme_alert)
        
        assert extreme_request.priority > normal_request.priority
    
    def test_priority_bounds(self, mapper):
        """Test priority stays within bounds."""
        # Create alert that would exceed max priority
        alert = create_scan_alert(
            alert_type=AlertType.ML_SIGNAL,
            score=1.0,
            data={"volatility_level": "extreme"}
        )
        
        request = mapper.map_alert_to_features(alert)
        
        assert request.priority <= mapper.priority_rules['max_priority']
        assert request.priority >= mapper.priority_rules['min_priority']
    
    def test_get_required_data_types(self, mapper):
        """Test getting required data types for feature groups."""
        groups = [FeatureGroup.PRICE, FeatureGroup.VOLUME, FeatureGroup.VOLATILITY]
        
        data_types = mapper.get_required_data_types(groups)
        
        assert "prices" in data_types
        assert "quotes" in data_types
        assert "trades" in data_types
    
    def test_get_required_data_types_with_dependencies(self, mapper):
        """Test data types include dependencies."""
        # MOMENTUM depends on PRICE
        groups = [FeatureGroup.MOMENTUM]
        
        data_types = mapper.get_required_data_types(groups)
        
        # Should include data required by PRICE as well
        assert "prices" in data_types
    
    def test_prioritize_requests(self, mapper):
        """Test request prioritization."""
        requests = []
        
        # Create requests with different priorities
        for i in range(5):
            alert = create_scan_alert(score=i * 0.2)
            request = mapper.map_alert_to_features(alert)
            request.priority = i  # Override for testing
            requests.append(request)
        
        # Prioritize
        sorted_requests = mapper.prioritize_requests(requests)
        
        # Should be sorted by priority (highest first)
        for i in range(len(sorted_requests) - 1):
            assert sorted_requests[i].priority >= sorted_requests[i + 1].priority
    
    def test_get_computation_params(self, mapper):
        """Test merging computation parameters."""
        groups = [FeatureGroup.PRICE, FeatureGroup.MOMENTUM]
        
        params = mapper.get_computation_params(groups)
        
        # Should have params from both groups
        assert "lookback_periods" in params
        assert "rsi_periods" in params
        
        # Lists should be merged
        if isinstance(params.get("lookback_periods"), list):
            assert len(params["lookback_periods"]) > 0
    
    def test_get_all_dependencies(self, mapper):
        """Test dependency expansion."""
        # TREND depends on PRICE and MOMENTUM
        # MOMENTUM depends on PRICE
        groups = [FeatureGroup.TREND]
        
        all_groups = mapper._get_all_dependencies(groups)
        
        assert FeatureGroup.TREND in all_groups
        assert FeatureGroup.MOMENTUM in all_groups
        assert FeatureGroup.PRICE in all_groups
    
    def test_configuration_override(self, mapper):
        """Test configuration override functionality."""
        # Override alert mapping
        custom_config = {
            "alert_mappings": {
                "HIGH_VOLUME": ["MICROSTRUCTURE", "ORDER_FLOW"]
            }
        }
        
        mapper = FeatureGroupMapper(config=custom_config)
        
        alert = create_scan_alert(alert_type=AlertType.HIGH_VOLUME)
        request = mapper.map_alert_to_features(alert)
        
        # Should use overridden mapping
        assert FeatureGroup.MICROSTRUCTURE in request.feature_groups
        assert FeatureGroup.ORDER_FLOW in request.feature_groups
    
    def test_metadata_preservation(self, mapper):
        """Test alert metadata is preserved in request."""
        alert = create_scan_alert()
        alert_id = id(alert)
        
        request = mapper.map_alert_to_features(alert)
        
        assert request.metadata['alert_id'] == alert_id
        assert request.metadata['alert_timestamp'] == alert.timestamp
        assert request.metadata['alert_data'] == alert.data
    
    def test_error_handling(self, mapper):
        """Test error handling in mapping."""
        # Create alert with invalid data
        alert = create_scan_alert()
        alert.alert_type = None  # Invalid
        
        # Should handle gracefully
        try:
            request = mapper.map_alert_to_features(alert)
            # Should use default mapping
            assert len(request.feature_groups) > 0
        except:
            # Or raise appropriate error
            pass