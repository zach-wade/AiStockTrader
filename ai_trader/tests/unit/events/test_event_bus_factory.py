"""Unit tests for EventBusFactory."""

import pytest
from unittest.mock import Mock, patch

from main.interfaces.events import IEventBus
from main.events.core import EventBusFactory, EventBusConfig
from main.utils.resilience import CircuitBreakerConfig


@pytest.mark.unit
@pytest.mark.events
class TestEventBusFactory:
    """Test EventBusFactory functionality."""
    
    def test_create_default(self):
        """Test creating event bus with default config."""
        bus = EventBusFactory.create()
        assert isinstance(bus, IEventBus)
    
    def test_create_with_config(self):
        """Test creating event bus with custom config."""
        config = EventBusConfig(
            max_queue_size=500,
            max_workers=5,
            enable_history=False,
            enable_dlq=True
        )
        bus = EventBusFactory.create(config)
        assert isinstance(bus, IEventBus)
    
    def test_create_from_dict(self):
        """Test creating event bus from config dictionary."""
        config_dict = {
            'max_queue_size': 1000,
            'max_workers': 10,
            'enable_history': True,
            'custom_setting': 'value'  # Should go to custom_config
        }
        bus = EventBusFactory.create_from_dict(config_dict)
        assert isinstance(bus, IEventBus)
    
    def test_create_test_instance(self):
        """Test creating test-optimized event bus."""
        bus = EventBusFactory.create_test_instance()
        assert isinstance(bus, IEventBus)
    
    def test_config_from_dict(self):
        """Test EventBusConfig.from_dict method."""
        config_dict = {
            'max_queue_size': 2000,
            'max_workers': 20,
            'enable_history': False,
            'circuit_breaker_config': {
                'failure_threshold': 10,
                'recovery_timeout': 60.0
            },
            'custom_key': 'custom_value'
        }
        
        config = EventBusConfig.from_dict(config_dict)
        
        assert config.max_queue_size == 2000
        assert config.max_workers == 20
        assert config.enable_history is False
        assert isinstance(config.circuit_breaker_config, CircuitBreakerConfig)
        assert config.circuit_breaker_config.failure_threshold == 10
        assert config.custom_config['custom_key'] == 'custom_value'
    
    def test_register_custom_implementation(self):
        """Test registering custom event bus implementation."""
        # Create mock implementation
        class CustomEventBus:
            def __init__(self, config):
                self.config = config
        
        # Register it
        EventBusFactory.register_implementation('custom', CustomEventBus)
        
        # Verify it's registered
        impls = EventBusFactory.get_implementations()
        assert 'custom' in impls
        assert impls['custom'] == CustomEventBus
        
        # Create instance with custom implementation
        bus = EventBusFactory.create(implementation='custom')
        assert isinstance(bus, CustomEventBus)
        
        # Clean up
        EventBusFactory.unregister_implementation('custom')
    
    def test_unregister_implementation(self):
        """Test unregistering implementation."""
        # Register and unregister
        EventBusFactory.register_implementation('temp', Mock)
        EventBusFactory.unregister_implementation('temp')
        
        # Verify it's gone
        impls = EventBusFactory.get_implementations()
        assert 'temp' not in impls
    
    def test_cannot_unregister_default(self):
        """Test that default implementation cannot be unregistered."""
        with pytest.raises(ValueError, match="Cannot unregister default"):
            EventBusFactory.unregister_implementation('default')
    
    def test_unknown_implementation_error(self):
        """Test error when using unknown implementation."""
        with pytest.raises(ValueError, match="Unknown implementation"):
            EventBusFactory.create(implementation='nonexistent')
    
    def test_duplicate_registration_error(self):
        """Test error when registering duplicate implementation."""
        EventBusFactory.register_implementation('test_dup', Mock)
        
        with pytest.raises(ValueError, match="already registered"):
            EventBusFactory.register_implementation('test_dup', Mock)
        
        # Clean up
        EventBusFactory.unregister_implementation('test_dup')