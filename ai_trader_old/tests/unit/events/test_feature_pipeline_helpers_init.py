"""Unit tests for feature_pipeline_helpers/__init__.py module."""

# Standard library imports
import importlib
import inspect
from types import ModuleType


class TestFeaturePipelineHelpersInit:
    """Test the feature_pipeline_helpers module initialization."""

    def test_module_imports(self):
        """Test that the feature_pipeline_helpers module can be imported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        assert isinstance(helpers, ModuleType)
        assert hasattr(helpers, "__all__")
        assert hasattr(helpers, "__doc__")

    def test_all_exports_present(self):
        """Test that all items in __all__ are actually exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        for export_name in helpers.__all__:
            assert hasattr(helpers, export_name), f"Export '{export_name}' not found in module"

    def test_main_class_exports(self):
        """Test that main helper classes are properly exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        main_classes = [
            "FeatureComputationWorker",
            "FeatureGroupMapper",
            "FeatureHandlerStatsTracker",
            "RequestQueueManager",
            "DeduplicationTracker",
        ]

        for class_name in main_classes:
            # Check it exists
            assert hasattr(helpers, class_name)

            # Check it's a class
            cls = getattr(helpers, class_name)
            assert inspect.isclass(cls), f"{class_name} should be a class"

            # Check it's in __all__
            assert class_name in helpers.__all__

    def test_type_exports(self):
        """Test that type definitions are properly exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        type_exports = [
            "FeatureGroup",
            "FeatureGroupConfig",
            "FeatureRequest",
            "QueuedRequest",
            "QueueStats",
        ]

        for type_name in type_exports:
            # Check it exists
            assert hasattr(helpers, type_name)

            # Check it's in __all__
            assert type_name in helpers.__all__

            # For enums and dataclasses
            obj = getattr(helpers, type_name)
            assert obj is not None

    def test_config_function_exports(self):
        """Test that configuration functions are properly exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        config_functions = [
            "initialize_group_configs",
            "initialize_alert_mappings",
            "get_conditional_group_rules",
            "get_priority_calculation_rules",
        ]

        for func_name in config_functions:
            # Check it exists
            assert hasattr(helpers, func_name)

            # Check it's callable
            func = getattr(helpers, func_name)
            assert callable(func), f"{func_name} should be callable"

            # Check it's in __all__
            assert func_name in helpers.__all__

    def test_all_list_completeness(self):
        """Test that __all__ list is complete and correct."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        expected_all = {
            # Main classes
            "FeatureComputationWorker",
            "FeatureGroupMapper",
            "FeatureHandlerStatsTracker",
            "RequestQueueManager",
            "DeduplicationTracker",
            # Types
            "FeatureGroup",
            "FeatureGroupConfig",
            "FeatureRequest",
            "QueuedRequest",
            "QueueStats",
            # Config functions
            "initialize_group_configs",
            "initialize_alert_mappings",
            "get_conditional_group_rules",
            "get_priority_calculation_rules",
        }

        assert set(helpers.__all__) == expected_all

    def test_no_unexpected_exports(self):
        """Test that no internal items are accidentally exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        # Get all public attributes
        public_attrs = [attr for attr in dir(helpers) if not attr.startswith("_")]

        # Remove standard module attributes
        standard_attrs = {"__all__", "__doc__", "__file__", "__name__", "__package__", "__path__"}
        public_attrs = [attr for attr in public_attrs if attr not in standard_attrs]

        # All public attributes should be in __all__
        for attr in public_attrs:
            assert attr in helpers.__all__, f"Public attribute '{attr}' not in __all__"

    def test_import_from_helpers(self):
        """Test that we can import directly from feature_pipeline_helpers."""
        # Classes
        # Types
        # Functions
        # Local imports
        from main.events.feature_pipeline_helpers import (
            FeatureComputationWorker,
            FeatureGroup,
            FeatureGroupMapper,
            initialize_group_configs,
        )

        # Verify imports worked
        assert inspect.isclass(FeatureComputationWorker)
        assert inspect.isclass(FeatureGroupMapper)
        assert FeatureGroup is not None
        assert callable(initialize_group_configs)

    def test_submodule_structure(self):
        """Test that submodules exist and can be imported."""
        # Import individual submodules
        # Import split modules
        # Local imports
        from main.events.feature_pipeline_helpers import (
            feature_computation_worker,
            feature_types,
            queue_types,
        )

        # Verify they are modules
        assert isinstance(feature_computation_worker, ModuleType)
        assert isinstance(feature_types, ModuleType)
        assert isinstance(queue_types, ModuleType)

    def test_split_module_exports(self):
        """Test that types from split modules are correctly re-exported."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        # These types come from feature_types.py
        assert helpers.FeatureGroup is not None
        assert helpers.FeatureGroupConfig is not None
        assert helpers.FeatureRequest is not None

        # These types come from queue_types.py
        assert helpers.QueuedRequest is not None
        assert helpers.QueueStats is not None

        # These functions come from feature_config.py
        assert callable(helpers.initialize_group_configs)
        assert callable(helpers.get_conditional_group_rules)

    def test_module_documentation(self):
        """Test that module has proper documentation."""
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        assert helpers.__doc__ is not None
        assert len(helpers.__doc__) > 50  # Should have substantial documentation

        # Check for key concepts in documentation
        doc_lower = helpers.__doc__.lower()
        assert "feature" in doc_lower
        assert "pipeline" in doc_lower
        assert any(term in doc_lower for term in ["scanner", "bridge"])

    def test_feature_group_enum_values(self):
        """Test that FeatureGroup enum has expected values."""
        # Local imports
        from main.events.feature_pipeline_helpers import FeatureGroup

        # Should be an enum with various feature groups
        assert hasattr(FeatureGroup, "PRICE")
        assert hasattr(FeatureGroup, "VOLUME")
        # Don't test all values as they may change

    def test_circular_import_safety(self):
        """Test that there are no circular import issues."""
        # This should not raise ImportError
        # Local imports
        import main.events.feature_pipeline_helpers

        importlib.reload(main.events.feature_pipeline_helpers)

        # Import in different order
        # Local imports
        from main.events.feature_pipeline_helpers import (
            FeatureComputationWorker,
            FeatureGroup,
            QueuedRequest,
        )

        # All should be properly initialized
        assert QueuedRequest is not None
        assert FeatureComputationWorker is not None
        assert FeatureGroup is not None

    def test_import_consistency(self):
        """Test that imports are consistent across different import methods."""
        # Import module and get class
        # Local imports
        import main.events.feature_pipeline_helpers as helpers

        Worker1 = helpers.FeatureComputationWorker

        # Direct import
        # Local imports
        from main.events.feature_pipeline_helpers import FeatureComputationWorker as Worker2

        # Import from submodule
        from main.events.handlers.feature_pipeline_helpers.feature_computation_worker import (
            FeatureComputationWorker as Worker3,
        )

        # All should be the same class
        assert Worker1 is Worker2
        assert Worker2 is Worker3

    def test_type_reexport_consistency(self):
        """Test that re-exported types are the same as original."""
        # Import from main module
        # Local imports
        from main.events.feature_pipeline_helpers import FeatureGroup as FG1
        from main.events.feature_pipeline_helpers import QueuedRequest as QR1

        # Import from split modules
        from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureGroup as FG2
        from main.events.handlers.feature_pipeline_helpers.queue_types import QueuedRequest as QR2

        # Should be the same objects
        assert FG1 is FG2
        assert QR1 is QR2
