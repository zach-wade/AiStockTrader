# File: src/main/events/feature_pipeline_helpers/feature_computation_worker.py

import asyncio
import yaml
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd

# Corrected absolute imports
from main.interfaces.events import IEventBus, Event, EventType
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
from main.events.core.event_bus_helpers.event_bus_stats_tracker import EventBusStatsTracker # To update stats
from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import record_metric, timer

logger = get_logger(__name__)

class FeatureComputationWorker(ErrorHandlingMixin):
    """
    A worker responsible for processing a single feature computation request.
    It interacts with the FeatureOrchestrator to compute features and
    publishes completion or error events back to the Event Bus.
    """

    def __init__(self, 
                 worker_id: int,
                 feature_orchestrator: FeatureOrchestrator,
                 event_bus: IEventBus,
                 stats_tracker: EventBusStatsTracker, # To update worker-specific stats
                 feature_group_mapper: Any # FeatureGroupMapper instance
                ):
        """
        Initializes the FeatureComputationWorker.

        Args:
            worker_id: Unique ID for the worker.
            feature_orchestrator: The FeatureOrchestrator instance for computing features.
            event_bus: The EventBus instance to publish events.
            stats_tracker: The EventBusStatsTracker for updating worker-related statistics.
            feature_group_mapper: The FeatureGroupMapper instance for categorizing features.
        """
        self.worker_id = worker_id
        self.feature_orchestrator = feature_orchestrator
        self.event_bus = event_bus
        self.stats_tracker = stats_tracker
        self.feature_group_mapper = feature_group_mapper
        
        # Load feature group mappings configuration
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, 'config', 'events', 'feature_group_mappings.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self.feature_group_config = yaml.safe_load(f)
        except Exception as e:
            self.handle_error(e, "loading feature group mappings")
            # Provide empty config if loading fails
            self.feature_group_config = {}
        
        logger.info(f"Feature worker {worker_id} initialized with feature group mappings.")

    async def process_request(self, request: Dict[str, Any]):
        """
        Processes a single feature computation request.
        Coordinates feature grouping, computation, and event publishing.

        Args:
            request: A dictionary containing the feature request details:
                     'symbols' (List[str]), 'features' (List[str]), 'event' (original Event).
        """
        symbols = request['symbols']
        features_list = request['features'] # Renamed to avoid confusion with `features` param in `_compute_features`
        original_event = request['event'] # Renamed for clarity
        
        logger.info(f"Worker {self.worker_id} processing: {len(symbols)} symbols, {len(features_list)} features.")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Group features by type for efficient computation using the mapper
            feature_groups = self.feature_group_mapper.group_features_by_type(features_list)
            
            results: Dict[str, Dict[str, pd.DataFrame]] = {} # {symbol: {feature_type: DataFrame}}
            for symbol in symbols:
                symbol_features: Dict[str, pd.DataFrame] = {}
                
                for feature_type, group_features in feature_groups.items():
                    # Compute features using orchestrator
                    computed_df = await self._compute_features_for_group(
                        symbol, feature_type, group_features
                    )
                    
                    if computed_df is not None and not computed_df.empty:
                        symbol_features[feature_type] = computed_df
                
                if symbol_features:
                    results[symbol] = symbol_features
            
            computation_time_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Publish completion event
            completion_event = Event(
                event_type=EventType.FEATURE_COMPUTED,
                source='feature_pipeline_worker',
                data={
                    'symbols_processed': list(results.keys()),
                    'requested_features': features_list,
                    'results_available': len(results) > 0,
                    'computation_time_seconds': computation_time_seconds
                },
                metadata={
                    'request_source_event_id': original_event.data.get('id'), # If event has an ID
                    'request_correlation_id': original_event.correlation_id,
                    'worker_id': self.worker_id
                },
                correlation_id=original_event.correlation_id
            )
            await self.event_bus.publish(completion_event)
            
            # Update stats via tracker
            self.stats_tracker.increment_processed()
            self.stats_tracker.increment_features_computed(len(results) * len(features_list)) # Rough estimate
            
            logger.info(f"Worker {self.worker_id} completed request for {len(symbols)} symbols in {computation_time_seconds:.2f}s.")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to process feature request for symbols {symbols}: {e}", exc_info=True)
            self.stats_tracker.increment_failed() # Update failed stats

            # Publish error event to bus
            error_event = Event(
                event_type=EventType.ERROR_OCCURRED,
                source='feature_pipeline_worker',
                data={
                    'error_type': 'feature_computation_failed',
                    'error_message': str(e),
                    'symbols': symbols,
                    'features': features_list,
                    'worker_id': self.worker_id
                },
                correlation_id=original_event.correlation_id
            )
            await self.event_bus.publish(error_event)

    async def _compute_features_for_group(self, symbol: str, feature_group_name: str,
                                         features_in_group: List[str]) -> Optional[pd.DataFrame]:
        """
        Computes features for a given symbol and a specific feature group.
        Maps the feature group name to the appropriate feature_orchestrator method/feature_sets.
        """
        try:
            # Map feature group name to actual feature sets for the orchestrator
            # This is a key mapping logic. This can be complex or simple.
            # Assuming `feature_orchestrator.compute_features` can take `feature_sets` as a list of strings.
            # The group name itself might correspond to a feature set, or we need a more detailed map.
            
            # Get feature sets from configuration
            feature_sets_to_request: List[str] = []
            
            group_mappings = self.feature_group_config.get('feature_group_mappings', {})
            if feature_group_name in group_mappings:
                group_config = group_mappings[feature_group_name]
                
                # Check if we should use features in group directly
                if isinstance(group_config, dict) and group_config.get('use_features_in_group'):
                    feature_sets_to_request = features_in_group
                else:
                    # Use the mapped feature sets
                    feature_sets_to_request = group_config
            else:
                logger.warning(f"Unknown feature group '{feature_group_name}'. No feature sets configured for it.")
                return None  # No features to compute for this unknown group

            if not feature_sets_to_request:
                logger.debug(f"No specific feature sets defined for group '{feature_group_name}'. Skipping computation.")
                return None
            
            # Call the central FeatureOrchestrator
            with timer() as t:
                computed_df = await self.feature_orchestrator.compute_features(
                    symbols=[symbol],
                    feature_sets=feature_sets_to_request # Orchestrator consumes feature sets
                )
            
            record_metric("feature_worker.orchestrator_compute_time",
                         t.elapsed_seconds,
                         metric_type="histogram",
                         tags={
                             "symbol": symbol,
                             "feature_group": feature_group_name,
                             "feature_count": len(feature_sets_to_request)
                         })
            
            if computed_df is not None:
                record_metric("feature_worker.features_computed",
                             len(computed_df.columns) if hasattr(computed_df, 'columns') else 0,
                             tags={
                                 "symbol": symbol,
                                 "feature_group": feature_group_name
                             })
            
            return computed_df
            
        except Exception as e:
            logger.error(f"Error computing {feature_group_name} features for {symbol}: {e}", exc_info=True)
            record_metric("feature_worker.computation_error", 1, tags={
                "worker_id": self.worker_id,
                "symbol": symbol,
                "feature_group": feature_group_name,
                "error_type": type(e).__name__
            })
            return None