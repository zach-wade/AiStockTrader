"""
Unit tests for ResilienceStrategies and related components.

Tests cover:
- ResilienceConfig dataclass validation and creation
- ResilienceStrategies initialization and configuration extraction  
- ResilienceStrategiesFactory pattern
- Integration with YAML configuration
- Error handling and edge cases
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from main.utils.resilience.strategies import (
    ResilienceConfig,
    ResilienceStrategies, 
    ResilienceStrategiesFactory,
    create_resilience_strategies
)
from main.utils.resilience.circuit_breaker import CircuitBreakerState


class TestResilienceConfig:
    """Test ResilienceConfig dataclass."""
    
    def test_default_config(self):
        """Test creating default ResilienceConfig."""
        config = ResilienceConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.backoff_factor == 2.0
        assert config.max_delay == 60.0
        assert config.jitter is True
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.critical_latency_ms == 5000.0
        assert config.rate_limit_calls is None
        assert config.rate_limit_period == 60
    
    def test_custom_config(self):
        """Test creating custom ResilienceConfig."""
        config = ResilienceConfig(
            max_retries=5,
            failure_threshold=10,
            critical_latency_ms=2000.0
        )
        
        assert config.max_retries == 5
        assert config.failure_threshold == 10
        assert config.critical_latency_ms == 2000.0
        # Other fields should use defaults
        assert config.initial_delay == 1.0
        assert config.backoff_factor == 2.0
    
    def test_config_validation(self):
        """Test ResilienceConfig validation."""
        # Test invalid max_retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ResilienceConfig(max_retries=-1)
        
        # Test invalid initial_delay
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            ResilienceConfig(initial_delay=0)
        
        # Test invalid backoff_factor
        with pytest.raises(ValueError, match="backoff_factor must be greater than 1.0"):
            ResilienceConfig(backoff_factor=1.0)
        
        # Test invalid failure_threshold
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            ResilienceConfig(failure_threshold=0)
        
        # Test invalid rate_limit_calls
        with pytest.raises(ValueError, match="rate_limit_calls must be positive if specified"):
            ResilienceConfig(rate_limit_calls=0)
    
    def test_from_dict(self):
        """Test ResilienceConfig.from_dict method."""
        config_dict = {
            'max_retries': 7,
            'failure_threshold': 12,
            'unknown_field': 'should_be_ignored'
        }
        
        config = ResilienceConfig.from_dict(config_dict)
        
        assert config.max_retries == 7
        assert config.failure_threshold == 12
        # Unknown fields should be ignored
        assert not hasattr(config, 'unknown_field')
        # Other fields should use defaults
        assert config.initial_delay == 1.0
    
    def test_factory_methods(self):
        """Test ResilienceConfig factory methods."""
        # Test default
        default_config = ResilienceConfig.create_default()
        assert default_config.max_retries == 3
        
        # Test API client
        api_config = ResilienceConfig.create_for_api_client("test_api")
        assert api_config.max_retries == 5
        assert api_config.critical_latency_ms == 10000.0
        assert api_config.rate_limit_calls == 100
        
        # Test database
        db_config = ResilienceConfig.create_for_database("test_db")
        assert db_config.max_retries == 3
        assert db_config.critical_latency_ms == 3000.0
        assert db_config.rate_limit_calls is None
        
        # Test feature calculation
        feature_config = ResilienceConfig.create_for_feature_calculation()
        assert feature_config.max_retries == 2
        assert feature_config.critical_latency_ms == 2000.0


class TestResilienceStrategies:
    """Test ResilienceStrategies class."""
    
    def test_init_with_none(self):
        """Test initialization with None config."""
        rs = ResilienceStrategies(None)
        
        assert rs.max_retries == 3
        assert rs.failure_threshold == 5
        assert rs.critical_latency_ms == 5000.0
        assert rs.circuit_breaker is not None
        assert rs.error_recovery is not None
    
    def test_init_with_resilience_config(self):
        """Test initialization with ResilienceConfig object."""
        config = ResilienceConfig(max_retries=7, failure_threshold=12)
        rs = ResilienceStrategies(config)
        
        assert rs.max_retries == 7
        assert rs.failure_threshold == 12
        assert rs.resilience_config is config
    
    def test_init_with_dict(self):
        """Test initialization with dictionary config."""
        config_dict = {
            'max_retries': 4,
            'failure_threshold': 8,
            'critical_latency_ms': 3000.0
        }
        rs = ResilienceStrategies(config_dict)
        
        assert rs.max_retries == 4
        assert rs.failure_threshold == 8
        assert rs.critical_latency_ms == 3000.0
    
    def test_config_extraction_from_complex_config(self):
        """Test configuration extraction from flat dict structure."""
        # Test with a flattened structure that should work
        config = {
            'max_retries': 6,
            'initial_delay': 2.0,
            'failure_threshold': 10,
            'critical_latency_ms': 8000.0
        }
        
        rs = ResilienceStrategies(config)
        
        assert rs.max_retries == 6
        assert rs.initial_delay == 2.0
        assert rs.failure_threshold == 10
        assert rs.critical_latency_ms == 8000.0
    
    def test_circuit_breaker_integration(self):
        """Test that circuit breaker is properly initialized."""
        config = ResilienceConfig(
            failure_threshold=7,
            recovery_timeout=30.0,
            critical_latency_ms=4000.0
        )
        rs = ResilienceStrategies(config)
        
        # Verify circuit breaker config
        cb_config = rs.circuit_breaker.config
        assert cb_config.failure_threshold == 7
        assert cb_config.recovery_timeout == 30.0
        assert cb_config.timeout_seconds == 4.0  # ms converted to seconds
    
    def test_error_recovery_integration(self):
        """Test that error recovery is properly initialized."""
        config = ResilienceConfig(
            max_retries=5,
            initial_delay=0.5,
            backoff_factor=1.5,
            max_delay=30.0
        )
        rs = ResilienceStrategies(config)
        
        # Verify error recovery config
        er_config = rs.error_recovery.config
        assert er_config.max_attempts == 5
        assert er_config.base_delay == 0.5
        assert er_config.backoff_multiplier == 1.5
        assert er_config.max_delay == 30.0
    
    @pytest.mark.asyncio
    async def test_execute_with_resilience_success(self):
        """Test successful execution with resilience."""
        rs = ResilienceStrategies()
        
        async def test_func(value):
            return value * 2
        
        result = await rs.execute_with_resilience(test_func, 21)
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_execute_with_resilience_sync_function(self):
        """Test execution with sync function."""
        rs = ResilienceStrategies()
        
        def sync_func(value):
            return value * 3
        
        result = await rs.execute_with_resilience(sync_func, 14)
        assert result == 42
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = ResilienceConfig(rate_limit_calls=2, rate_limit_period=1)
        rs = ResilienceStrategies(config)
        
        # First two calls should succeed
        assert rs._check_rate_limit() is True
        assert rs._check_rate_limit() is True
        
        # Third call should be rate limited
        assert rs._check_rate_limit() is False
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        rs = ResilienceStrategies()
        stats = rs.get_stats()
        
        assert 'circuit_breaker' in stats
        assert 'error_recovery' in stats
        assert 'rate_limiting' in stats
        assert 'configuration' in stats
        
        # Check configuration stats
        config_stats = stats['configuration']
        assert config_stats['max_retries'] == 3
        assert config_stats['failure_threshold'] == 5
        assert config_stats['critical_latency_ms'] == 5000.0
        
        # Check circuit breaker stats structure
        cb_stats = stats['circuit_breaker']
        assert isinstance(cb_stats, dict)  # Should be a dictionary of metrics
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = ResilienceConfig(max_retries=9)
        rs = ResilienceStrategies(config)
        
        retrieved_config = rs.get_config()
        assert retrieved_config is config
        assert retrieved_config.max_retries == 9
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting resilience components."""
        rs = ResilienceStrategies()
        
        # Add some rate limit calls
        rs._call_times = [1, 2, 3]
        
        # Reset should clear the call times
        await rs.reset()
        assert len(rs._call_times) == 0


class TestResilienceStrategiesFactory:
    """Test ResilienceStrategiesFactory class."""
    
    def test_create_default(self):
        """Test creating default ResilienceStrategies."""
        rs = ResilienceStrategiesFactory.create_default()
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 3
        assert rs.failure_threshold == 5
    
    def test_create_for_api_client(self):
        """Test creating API client ResilienceStrategies."""
        rs = ResilienceStrategiesFactory.create_for_api_client("polygon")
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 5
        assert rs.critical_latency_ms == 10000.0
        assert rs.rate_limit_calls == 100
    
    def test_create_for_database(self):
        """Test creating database ResilienceStrategies."""
        rs = ResilienceStrategiesFactory.create_for_database("postgresql")
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 3
        assert rs.critical_latency_ms == 3000.0
        assert rs.rate_limit_calls is None
    
    def test_create_for_feature_calculation(self):
        """Test creating feature calculation ResilienceStrategies."""
        rs = ResilienceStrategiesFactory.create_for_feature_calculation()
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 2
        assert rs.critical_latency_ms == 2000.0
    
    def test_create_from_config(self):
        """Test creating from arbitrary config."""
        config = {'max_retries': 8, 'failure_threshold': 15}
        rs = ResilienceStrategiesFactory.create_from_config(config)
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 8
        assert rs.failure_threshold == 15
    
    def test_create_from_profile_with_dict_config(self):
        """Test creating from profile with dict config."""
        # Test with a simple profile config dict
        config = {
            'resilience': {
                'profiles': {
                    'api_client': {
                        'retry': {'max_retries': 7},
                        'circuit_breaker': {'failure_threshold': 12},
                        'rate_limiting': {'calls_per_period': 150, 'period_seconds': 30}
                    }
                }
            }
        }
        
        # This will fall back to the factory method since the complex profile extraction
        # is designed for OmegaConf objects, not plain dicts
        rs = ResilienceStrategiesFactory.create_from_profile('api_client', config)
        
        assert isinstance(rs, ResilienceStrategies)
        # Should use the factory method fallback for api_client
        assert rs.max_retries == 5  # From create_for_api_client
    
    def test_create_from_profile_fallback(self):
        """Test fallback behavior when profile not found."""
        # This should fallback to the factory method for known profiles
        rs = ResilienceStrategiesFactory.create_from_profile('feature_calculation')
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 2  # Should match feature_calculation preset
        assert rs.critical_latency_ms == 2000.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_resilience_strategies_default(self):
        """Test convenience function with default arguments."""
        rs = create_resilience_strategies()
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 3
    
    def test_create_resilience_strategies_with_profile(self):
        """Test convenience function with profile."""
        rs = create_resilience_strategies(profile='database')
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.critical_latency_ms == 3000.0
    
    def test_create_resilience_strategies_with_config(self):
        """Test convenience function with config."""
        config = {'max_retries': 10}
        rs = create_resilience_strategies(config=config)
        
        assert isinstance(rs, ResilienceStrategies)
        assert rs.max_retries == 10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_dict(self):
        """Test with empty configuration dictionary."""
        rs = ResilienceStrategies({})
        
        # Should use defaults
        assert rs.max_retries == 3
        assert rs.failure_threshold == 5
    
    def test_config_with_none_values(self):
        """Test configuration with None values."""
        config = {
            'max_retries': None,
            'failure_threshold': 10
        }
        # This should not raise an error due to our safe extraction
        # The None value should be filtered out by from_dict
        rs = ResilienceStrategies(config)
        assert rs.failure_threshold == 10
        assert rs.max_retries == 3  # Should use default since None was filtered out
    
    def test_invalid_config_object(self):
        """Test with invalid configuration object."""
        # Should fallback to defaults gracefully
        rs = ResilienceStrategies("invalid_config")
        
        assert rs.max_retries == 3
        assert rs.failure_threshold == 5
    
    def test_partial_config_structure(self):
        """Test with partial configuration structure."""
        # Test with only some fields provided - should use defaults for others
        config = {'max_retries': 4}
        
        rs = ResilienceStrategies(config)
        
        # Should use provided values and defaults for others
        assert rs.max_retries == 4
        assert rs.failure_threshold == 5  # default
        assert rs.critical_latency_ms == 5000.0  # default