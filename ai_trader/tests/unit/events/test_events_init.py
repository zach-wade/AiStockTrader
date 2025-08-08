"""Unit tests for events/__init__.py module."""

import pytest
import importlib
import inspect
from types import ModuleType


class TestEventsInit:
    """Test the main events module initialization."""
    
    def test_module_imports(self):
        """Test that the events module can be imported."""
        import main.events as events
        
        assert isinstance(events, ModuleType)
        assert hasattr(events, '__all__')
        assert hasattr(events, '__doc__')
        
    def test_all_exports_present(self):
        """Test that all items in __all__ are actually exported."""
        import main.events as events
        
        for export_name in events.__all__:
            assert hasattr(events, export_name), f"Export '{export_name}' not found in module"
            
    def test_event_types_exports(self):
        """Test that all event type exports are available."""
        import main.events as events
        
        # Core types
        assert hasattr(events, 'Event')
        assert hasattr(events, 'EventType')
        assert hasattr(events, 'AlertType')
        assert hasattr(events, 'ScanAlert')
        
        # Event classes
        event_classes = [
            'ScannerAlertEvent',
            'FeatureRequestEvent',
            'FeatureComputedEvent',
            'ErrorEvent',
            'OrderEvent',
            'FillEvent',
            'MarketEvent'
        ]
        
        for event_class in event_classes:
            assert hasattr(events, event_class)
            # Verify they are classes
            assert inspect.isclass(getattr(events, event_class))
            
    def test_event_bus_di_exports(self):
        """Test that event bus DI components are exported."""
        import main.events as events
        
        # Factory and config
        assert hasattr(events, 'EventBusFactory')
        assert hasattr(events, 'EventBusConfig')
        assert inspect.isclass(events.EventBusFactory)
        assert inspect.isclass(events.EventBusConfig)
        
        # Registry
        assert hasattr(events, 'EventBusRegistry')
        assert hasattr(events, 'get_global_registry')
        assert inspect.isclass(events.EventBusRegistry)
        assert callable(events.get_global_registry)
        
        # Provider
        assert hasattr(events, 'DefaultEventBusProvider')
        assert hasattr(events, 'get_default_provider')
        assert inspect.isclass(events.DefaultEventBusProvider)
        assert callable(events.get_default_provider)
            
    def test_scanner_bridge_export(self):
        """Test that scanner bridge is exported."""
        import main.events as events
        
        # Only the bridge class, no singleton functions
        assert hasattr(events, 'ScannerFeatureBridge')
        assert inspect.isclass(events.ScannerFeatureBridge)
            
            
    def test_no_unexpected_exports(self):
        """Test that no internal/private items are accidentally exported."""
        import main.events as events
        
        # Get all public attributes
        public_attrs = [attr for attr in dir(events) if not attr.startswith('_')]
        
        # Remove standard module attributes
        standard_attrs = {'__all__', '__doc__', '__file__', '__name__', '__package__', '__path__'}
        public_attrs = [attr for attr in public_attrs if attr not in standard_attrs]
        
        # All public attributes should be in __all__
        for attr in public_attrs:
            assert attr in events.__all__, f"Public attribute '{attr}' not in __all__"
            
    def test_all_list_completeness(self):
        """Test that __all__ list matches documented exports."""
        import main.events as events
        
        expected_exports = {
            # Event types
            'Event', 'EventType', 'AlertType', 'ScanAlert',
            
            # Event classes
            'ScannerAlertEvent', 'FeatureRequestEvent', 'FeatureComputedEvent',
            'ErrorEvent', 'OrderEvent', 'FillEvent', 'MarketEvent',
            
            # Event bus
            'EventBus', 'get_event_bus', 'initialize_event_bus',
            'cleanup_event_bus', 'is_event_bus_running', 'restart_event_bus',
            
            # Convenience functions
            'subscribe_to_event', 'unsubscribe_from_event', 'publish_event',
            
            # Scanner-feature bridge
            'ScannerFeatureBridge', 'get_scanner_feature_bridge',
            'initialize_scanner_feature_bridge', 'cleanup_scanner_feature_bridge',
            'is_bridge_running', 'restart_scanner_feature_bridge', 'get_bridge_stats'
        }
        
        actual_exports = set(events.__all__)
        
        # Check both directions
        assert actual_exports == expected_exports, (
            f"Missing: {expected_exports - actual_exports}, "
            f"Extra: {actual_exports - expected_exports}"
        )
        
    def test_submodule_imports(self):
        """Test that submodules are properly structured."""
        # These imports should work
        from main.events import event_types
        from main.events import event_bus
        from main.events import event_bus_initializer
        from main.events import scanner_feature_bridge
        # scanner_feature_bridge_initializer removed - use ScannerFeatureBridge directly
        
        # Verify they are modules
        assert isinstance(event_types, ModuleType)
        assert isinstance(event_bus, ModuleType)
        assert isinstance(event_bus_initializer, ModuleType)
        assert isinstance(scanner_feature_bridge, ModuleType)
        assert isinstance(scanner_feature_bridge_initializer, ModuleType)
        
    def test_import_from_events(self):
        """Test that we can import directly from events."""
        # These should all work
        from main.events import Event, EventType, EventBus
        from main.events import ScannerAlertEvent, FeatureRequestEvent
        from main.events import get_event_bus, initialize_event_bus
        from main.events import ScannerFeatureBridge, get_scanner_feature_bridge
        
        # Verify types
        assert inspect.isclass(Event)
        assert inspect.isclass(EventType)
        assert inspect.isclass(EventBus)
        assert inspect.isclass(ScannerFeatureBridge)
        assert callable(get_event_bus)
        assert callable(initialize_event_bus)
        
    def test_module_documentation(self):
        """Test that module has proper documentation."""
        import main.events as events
        
        assert events.__doc__ is not None
        assert len(events.__doc__) > 50  # Should have substantial documentation
        assert "event-driven" in events.__doc__.lower()
        assert "event bus" in events.__doc__.lower()
        
    def test_circular_import_safety(self):
        """Test that there are no circular import issues."""
        # This should not raise ImportError
        import main.events
        importlib.reload(main.events)
        
        # Import in different order
        from main.events import EventBus
        from main.events import Event
        from main.events import ScannerFeatureBridge
        
        # All should be properly initialized
        assert EventBus is not None
        assert Event is not None
        assert ScannerFeatureBridge is not None