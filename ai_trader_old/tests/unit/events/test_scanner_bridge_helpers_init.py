"""Unit tests for scanner_bridge_helpers/__init__.py module."""

# Standard library imports
import importlib
import inspect
from types import ModuleType


class TestScannerBridgeHelpersInit:
    """Test the scanner_bridge_helpers module initialization."""

    def test_module_imports(self):
        """Test that the scanner_bridge_helpers module can be imported."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        assert isinstance(helpers, ModuleType)
        assert hasattr(helpers, "__all__")
        assert hasattr(helpers, "__doc__")

    def test_all_exports_present(self):
        """Test that all items in __all__ are actually exported."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        for export_name in helpers.__all__:
            assert hasattr(helpers, export_name), f"Export '{export_name}' not found in module"

    def test_helper_class_exports(self):
        """Test that all helper classes are properly exported."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        expected_classes = [
            "AlertFeatureMapper",
            "BridgeStatsTracker",
            "FeatureRequestBatcher",
            "PriorityCalculator",
            "RequestDispatcher",
        ]

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
        import main.events.scanner_bridge_helpers as helpers

        expected_all = [
            "AlertFeatureMapper",
            "BridgeStatsTracker",
            "FeatureRequestBatcher",
            "PriorityCalculator",
            "RequestDispatcher",
        ]

        assert set(helpers.__all__) == set(expected_all)

    def test_no_unexpected_exports(self):
        """Test that no internal items are accidentally exported."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        # Get all public attributes
        public_attrs = [attr for attr in dir(helpers) if not attr.startswith("_")]

        # Remove standard module attributes
        standard_attrs = {"__all__", "__doc__", "__file__", "__name__", "__package__", "__path__"}
        public_attrs = [attr for attr in public_attrs if attr not in standard_attrs]

        # All public attributes should be in __all__
        for attr in public_attrs:
            assert attr in helpers.__all__, f"Public attribute '{attr}' not in __all__"

    def test_import_from_helpers(self):
        """Test that we can import directly from scanner_bridge_helpers."""
        # These should all work
        # Local imports
        from main.events.scanner_bridge_helpers import (
            AlertFeatureMapper,
            BridgeStatsTracker,
            FeatureRequestBatcher,
            PriorityCalculator,
            RequestDispatcher,
        )

        # Verify they are classes
        assert inspect.isclass(AlertFeatureMapper)
        assert inspect.isclass(BridgeStatsTracker)
        assert inspect.isclass(FeatureRequestBatcher)
        assert inspect.isclass(PriorityCalculator)
        assert inspect.isclass(RequestDispatcher)

    def test_submodule_structure(self):
        """Test that submodules exist and can be imported."""
        # Import individual submodules
        # Local imports
        from main.events.scanner_bridge_helpers import (
            alert_feature_mapper,
            bridge_stats_tracker,
            feature_request_batcher,
            priority_calculator,
            request_dispatcher,
        )

        # Verify they are modules
        assert isinstance(alert_feature_mapper, ModuleType)
        assert isinstance(bridge_stats_tracker, ModuleType)
        assert isinstance(feature_request_batcher, ModuleType)
        assert isinstance(priority_calculator, ModuleType)
        assert isinstance(request_dispatcher, ModuleType)

        # Verify classes exist in their modules
        assert hasattr(alert_feature_mapper, "AlertFeatureMapper")
        assert hasattr(bridge_stats_tracker, "BridgeStatsTracker")
        assert hasattr(feature_request_batcher, "FeatureRequestBatcher")
        assert hasattr(priority_calculator, "PriorityCalculator")
        assert hasattr(request_dispatcher, "RequestDispatcher")

    def test_module_documentation(self):
        """Test that module has proper documentation."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        assert helpers.__doc__ is not None
        assert len(helpers.__doc__) > 50  # Should have substantial documentation

        # Check for key concepts in documentation
        doc_lower = helpers.__doc__.lower()
        assert "scanner" in doc_lower
        assert "feature" in doc_lower
        assert "bridge" in doc_lower
        assert "helper" in doc_lower

    def test_class_basic_functionality(self):
        """Test that exported classes have expected basic structure."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        # All classes should have __init__ method
        for class_name in helpers.__all__:
            cls = getattr(helpers, class_name)
            assert hasattr(cls, "__init__"), f"{class_name} should have __init__ method"

    def test_circular_import_safety(self):
        """Test that there are no circular import issues."""
        # This should not raise ImportError
        # Local imports
        import main.events.scanner_bridge_helpers

        importlib.reload(main.events.scanner_bridge_helpers)

        # Import in different order
        # Local imports
        from main.events.scanner_bridge_helpers import (
            AlertFeatureMapper,
            BridgeStatsTracker,
            RequestDispatcher,
        )

        # All should be properly initialized
        assert AlertFeatureMapper is not None
        assert BridgeStatsTracker is not None
        assert RequestDispatcher is not None

    def test_import_consistency(self):
        """Test that imports are consistent across different import methods."""
        # Import module and get class
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        Mapper1 = helpers.AlertFeatureMapper

        # Direct import
        # Import from submodule
        # Local imports
        from main.events.handlers.scanner_bridge_helpers.alert_feature_mapper import (
            AlertFeatureMapper as Mapper3,
        )
        from main.events.scanner_bridge_helpers import AlertFeatureMapper as Mapper2

        # All should be the same class
        assert Mapper1 is Mapper2
        assert Mapper2 is Mapper3

    def test_feature_request_batch_availability(self):
        """Test that FeatureRequestBatch is available through the batcher module."""
        # Local imports
        from main.events.handlers.scanner_bridge_helpers.feature_request_batcher import (
            FeatureRequestBatch,
        )

        # Should be available and be a class
        assert FeatureRequestBatch is not None
        assert inspect.isclass(FeatureRequestBatch) or hasattr(
            FeatureRequestBatch, "__dataclass_fields__"
        )

    def test_relative_imports_work(self):
        """Test that the module uses proper relative imports internally."""
        # Re-import to ensure fresh load
        # Local imports
        import main.events.scanner_bridge_helpers

        importlib.reload(main.events.scanner_bridge_helpers)

        # If relative imports are broken, the reload would fail
        assert main.events.scanner_bridge_helpers.AlertFeatureMapper is not None
        assert main.events.scanner_bridge_helpers.BridgeStatsTracker is not None
        assert main.events.scanner_bridge_helpers.FeatureRequestBatcher is not None
        assert main.events.scanner_bridge_helpers.PriorityCalculator is not None
        assert main.events.scanner_bridge_helpers.RequestDispatcher is not None

    def test_class_inheritance(self):
        """Test that classes inherit from expected base classes."""
        # Local imports
        import main.events.scanner_bridge_helpers as helpers

        # Most classes should inherit from ErrorHandlingMixin
        # This is based on the refactoring that was done
        for class_name in ["AlertFeatureMapper", "PriorityCalculator", "RequestDispatcher"]:
            cls = getattr(helpers, class_name)
            # Check if it has error handling methods (from ErrorHandlingMixin)
            instance = cls.__new__(cls)  # Create uninitialized instance
            assert hasattr(instance, "_handle_error") or hasattr(
                cls, "_handle_error"
            ), f"{class_name} should inherit from ErrorHandlingMixin"
