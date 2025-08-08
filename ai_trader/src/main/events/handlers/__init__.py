"""
Event handlers for the AI Trader system.

This module contains event handlers that can depend on external modules:
- ScannerFeatureBridge: Bridges scanner alerts to feature pipeline

Note: 
- FeaturePipelineHandler should be imported directly from
  main.events.handlers.feature_pipeline_handler to avoid circular imports.
- EventDrivenEngine should be imported directly from
  main.events.handlers.event_driven_engine as it has external dependencies.
"""

from .scanner_feature_bridge import ScannerFeatureBridge

__all__ = [
    'ScannerFeatureBridge',
    # 'FeaturePipelineHandler' - Import directly to avoid circular imports
    # 'EventDrivenEngine' - Import directly due to external dependencies
]