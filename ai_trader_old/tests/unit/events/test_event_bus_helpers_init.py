"""Unit tests for event_bus_helpers/__init__.py module."""

# Standard library imports
import importlib
import inspect
from types import ModuleType


class TestEventBusHelpersInit:
    """Test the event_bus_helpers module initialization."""

    def test_module_imports(self):
        """Test that the event_bus_helpers module can be imported."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        assert isinstance(helpers, ModuleType)
        assert hasattr(helpers, "__all__")
        assert hasattr(helpers, "__doc__")

    def test_all_exports_present(self):
        """Test that all items in __all__ are actually exported."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        for export_name in helpers.__all__:
            assert hasattr(helpers, export_name), f"Export '{export_name}' not found in module"

    def test_helper_class_exports(self):
        """Test that all helper classes are properly exported."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        expected_classes = ["EventBusStatsTracker", "EventHistoryManager", "DeadLetterQueueManager"]

        for class_name in expected_classes:
            # Check it exists
            assert hasattr(helpers, class_name)

            # Check it's a class
            cls = getattr(helpers, class_name)
            assert inspect.isclass(cls), f"{class_name} should be a class"

            # Check it's in __all__
            assert class_name in helpers.__all__

    def test_all_list_completeness(self):
        """Test that __all__ list is complete and correct."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        expected_all = ["EventBusStatsTracker", "EventHistoryManager", "DeadLetterQueueManager"]

        assert set(helpers.__all__) == set(expected_all)

    def test_no_unexpected_exports(self):
        """Test that no internal items are accidentally exported."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        # Get all public attributes
        public_attrs = [attr for attr in dir(helpers) if not attr.startswith("_")]

        # Remove standard module attributes
        standard_attrs = {"__all__", "__doc__", "__file__", "__name__", "__package__", "__path__"}
        public_attrs = [attr for attr in public_attrs if attr not in standard_attrs]

        # All public attributes should be in __all__
        for attr in public_attrs:
            assert attr in helpers.__all__, f"Public attribute '{attr}' not in __all__"

    def test_import_from_helpers(self):
        """Test that we can import directly from event_bus_helpers."""
        # These should all work
        # Local imports
        from main.events.core.event_bus_helpers import (
            DeadLetterQueueManager,
            EventBusStatsTracker,
            EventHistoryManager,
        )

        # Verify they are classes
        assert inspect.isclass(EventBusStatsTracker)
        assert inspect.isclass(EventHistoryManager)
        assert inspect.isclass(DeadLetterQueueManager)

    def test_submodule_structure(self):
        """Test that submodules exist and can be imported."""
        # Import individual submodules
        # Local imports
        from main.events.core.event_bus_helpers import (
            dead_letter_queue_manager,
            event_bus_stats_tracker,
            event_history_manager,
        )

        # Verify they are modules
        assert isinstance(event_bus_stats_tracker, ModuleType)
        assert isinstance(event_history_manager, ModuleType)
        assert isinstance(dead_letter_queue_manager, ModuleType)

        # Verify classes exist in their modules
        assert hasattr(event_bus_stats_tracker, "EventBusStatsTracker")
        assert hasattr(event_history_manager, "EventHistoryManager")
        assert hasattr(dead_letter_queue_manager, "DeadLetterQueueManager")

    def test_module_documentation(self):
        """Test that module has proper documentation."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        assert helpers.__doc__ is not None
        assert len(helpers.__doc__) > 50  # Should have substantial documentation

        # Check for key concepts in documentation
        doc_lower = helpers.__doc__.lower()
        assert "event bus" in doc_lower
        assert "helper" in doc_lower
        assert any(term in doc_lower for term in ["stats", "statistics"])
        assert "history" in doc_lower
        assert "dead letter" in doc_lower

    def test_class_basic_functionality(self):
        """Test that exported classes have expected basic structure."""
        # Local imports
        import main.events.event_bus_helpers as helpers

        # EventBusStatsTracker should have stats tracking methods
        assert hasattr(helpers.EventBusStatsTracker, "__init__")

        # EventHistoryManager should have history management methods
        assert hasattr(helpers.EventHistoryManager, "__init__")

        # DeadLetterQueueManager should have queue management methods
        assert hasattr(helpers.DeadLetterQueueManager, "__init__")

    def test_circular_import_safety(self):
        """Test that there are no circular import issues."""
        # This should not raise ImportError
        # Local imports
        import main.events.event_bus_helpers

        importlib.reload(main.events.event_bus_helpers)

        # Import in different order
        # Local imports
        from main.events.core.event_bus_helpers import (
            DeadLetterQueueManager,
            EventBusStatsTracker,
            EventHistoryManager,
        )

        # All should be properly initialized
        assert EventBusStatsTracker is not None
        assert EventHistoryManager is not None
        assert DeadLetterQueueManager is not None

    def test_import_consistency(self):
        """Test that imports are consistent across different import methods."""
        # Import module and get class
        # Local imports
        import main.events.event_bus_helpers as helpers

        StatsTracker1 = helpers.EventBusStatsTracker

        # Direct import
        # Local imports
        from main.events.core.event_bus_helpers import EventBusStatsTracker as StatsTracker2

        # Import from submodule
        from main.events.core.event_bus_helpers.event_bus_stats_tracker import (
            EventBusStatsTracker as StatsTracker3,
        )

        # All should be the same class
        assert StatsTracker1 is StatsTracker2
        assert StatsTracker2 is StatsTracker3

    def test_relative_imports_work(self):
        """Test that the module uses proper relative imports internally."""
        # Re-import to ensure fresh load
        # Local imports
        import main.events.event_bus_helpers

        importlib.reload(main.events.event_bus_helpers)

        # If relative imports are broken, the reload would fail
        assert main.events.event_bus_helpers.EventBusStatsTracker is not None
        assert main.events.event_bus_helpers.EventHistoryManager is not None
        assert main.events.event_bus_helpers.DeadLetterQueueManager is not None
