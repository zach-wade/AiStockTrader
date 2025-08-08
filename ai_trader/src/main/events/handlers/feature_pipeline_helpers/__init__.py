"""
Feature pipeline event helpers.

This module provides helper components for the scanner-to-feature pipeline bridge:
- FeatureComputationWorker: Processes feature computation requests
- FeatureGroupMapper: Maps scanner alerts to required feature groups
- FeatureHandlerStatsTracker: Tracks feature computation statistics
- RequestQueueManager: Manages the queue of feature computation requests
"""

from .feature_computation_worker import FeatureComputationWorker
from .feature_group_mapper import FeatureGroupMapper
from .feature_handler_stats_tracker import FeatureHandlerStatsTracker
from .request_queue_manager import RequestQueueManager

# Export types from split modules
from .feature_types import FeatureGroup, FeatureGroupConfig, FeatureRequest
from .queue_types import QueuedRequest, QueueStats
from .deduplication_tracker import DeduplicationTracker
from .feature_config import (
    initialize_group_configs,
    initialize_alert_mappings,
    get_conditional_group_rules,
    get_priority_calculation_rules
)

__all__ = [
    'FeatureComputationWorker',
    'FeatureGroupMapper',
    'FeatureHandlerStatsTracker',
    'RequestQueueManager',
    # Types
    'FeatureGroup',
    'FeatureGroupConfig',
    'FeatureRequest',
    'QueuedRequest',
    'QueueStats',
    'DeduplicationTracker',
    # Config functions
    'initialize_group_configs',
    'initialize_alert_mappings',
    'get_conditional_group_rules',
    'get_priority_calculation_rules',
]