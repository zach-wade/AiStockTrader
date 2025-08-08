# File: src/main/events/feature_pipeline_handler.py

"""
Feature Pipeline Event Handler.

Orchestrates feature computation requests received from the event bus,
managing a pool of workers and delegating tasks to specialized helpers
for request queuing, computation, and statistics tracking.
"""

import asyncio
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timezone
import pandas as pd # Used by FeatureOrchestrator methods, not directly in this orchestrator

# Corrected absolute imports
from main.interfaces.events import IEventBus, Event, EventType
from main.events.types import FeatureRequestEvent, FeatureComputedEvent
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator # Feature computation logic
from main.config.config_manager import get_config # Global config access
from main.utils.core import get_logger
from main.utils.monitoring import record_metric, timer

# Import the new feature pipeline helper classes
from main.events.handlers.feature_pipeline_helpers.request_queue_manager import RequestQueueManager
from main.events.handlers.feature_pipeline_helpers.feature_computation_worker import FeatureComputationWorker
from main.events.handlers.feature_pipeline_helpers.feature_group_mapper import FeatureGroupMapper
from main.events.handlers.feature_pipeline_helpers.feature_handler_stats_tracker import FeatureHandlerStatsTracker

logger = get_logger(__name__)


class FeaturePipelineHandler:
    """
    Orchestrates feature computation requests received from the event bus.
    Manages worker lifecycle, dispatches requests, and tracks internal statistics.
    Delegates specific tasks to specialized helper components.
    """
    
    def __init__(self, event_bus: IEventBus,
                 feature_service: Optional[Any] = None,
                 feature_orchestrator: Optional[FeatureOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Feature Pipeline Handler and its composing helper components.
        
        Args:
            event_bus: The event bus instance to use (dependency injection).
            feature_service: Optional. A feature service for computing features (used in tests).
            feature_orchestrator: Optional. An initialized FeatureOrchestrator instance.
                                  If None, a default instance is created.
            config: Optional. Configuration for the handler. If None, `get_config()` is used.
        """
        self.config = config
        self.event_bus = event_bus
        self.feature_service = feature_service
        
        # Feature orchestrator (shared dependency)
        self.feature_orchestrator = feature_orchestrator
        if not self.feature_orchestrator:
            # Only get config if needed and not provided
            if self.config is None:
                self.config = get_config()
            self.feature_orchestrator = FeatureOrchestrator(self.config, event_bus=self.event_bus)
        
        # Initialize helper components
        self._request_queue_manager = RequestQueueManager()
        self._feature_group_mapper = FeatureGroupMapper()
        self._stats_tracker = FeatureHandlerStatsTracker()
        
        # Worker configuration
        self._num_workers = 4  # Default
        if self.config:
            self._num_workers = self.config.get('feature_pipeline', {}).get('event_workers', 4)
        self._workers: List[asyncio.Task] = []
        
        logger.info("🔧 Feature Pipeline Handler initialized with helper components.")
    
    async def start(self):
        """
        Starts the Feature Pipeline Handler, subscribing to relevant events
        and launching worker tasks.
        """
        # Subscribe to feature request events
        await self.event_bus.subscribe(
            EventType.FEATURE_REQUEST,
            self._handle_feature_request_event # Pass the handler method
        )
        
        # Start worker tasks
        for i in range(self._num_workers):
            # Each worker gets its own FeatureComputationWorker instance
            worker_instance = FeatureComputationWorker(
                worker_id=i,
                feature_orchestrator=self.feature_orchestrator,
                event_bus=self.event_bus,
                stats_tracker=self._stats_tracker,
                feature_group_mapper=self._feature_group_mapper
            )
            worker_task = asyncio.create_task(
                self._run_worker_loop(worker_instance)
            )
            self._workers.append(worker_task)
        
        logger.info(f"🚀 Feature Pipeline Handler started with {self._num_workers} worker tasks.")
    
    async def stop(self):
        """
        Stops the Feature Pipeline Handler, canceling all worker tasks
        and unsubscribing from events.
        """
        # Unsubscribe from events (important to clean up)
        await self.event_bus.unsubscribe(
            EventType.FEATURE_REQUEST,
            self._handle_feature_request_event
        )

        # Cancel all workers
        for worker_task in self._workers:
            worker_task.cancel()
        
        # Wait for workers to finish, handling CancelledError
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True) # return_exceptions=True so one cancelled worker doesn't stop others
        
        logger.info("🛑 Feature Pipeline Handler stopped.")
    
    async def _handle_feature_request_event(self, event: Event):
        """
        Callback method to handle incoming feature request events from the Event Bus.
        Delegates queuing logic to RequestQueueManager.
        """
        try:
            symbols = event.data.get('symbols', [])
            features = event.data.get('features', [])
            priority = event.data.get('priority', 5)
            
            await self._request_queue_manager.add_request(
                symbols=symbols,
                features=features,
                event=event,
                priority=priority
            )
            self._stats_tracker.increment_requests_received() # Update stats
            
        except Exception as e:
            logger.error(f"Error in _handle_feature_request_event for event {event.correlation_id}: {e}", exc_info=True)
            record_metric("feature_pipeline.request_error", 1, tags={
                "error_type": type(e).__name__
            })
            # Consider publishing an ERROR_OCCURRED event here if request is fundamentally malformed
    
    @timer
    async def _run_worker_loop(self, worker_instance: FeatureComputationWorker):
        """
        The main loop for each feature worker task.
        Continuously gets requests from the queue and processes them.
        """
        logger.debug(f"Worker loop {worker_instance.worker_id} started.")
        while True:
            try:
                # Get request from queue (blocks until a request is available)
                _priority, _timestamp, request_data = await self._request_queue_manager.get_next_request()
                
                # Process the request using the dedicated worker instance
                record_metric("feature_pipeline.worker_processing", 1, tags={
                    "worker_id": worker_instance.worker_id,
                    "symbols": len(request_data.get('symbols', [])),
                    "features": len(request_data.get('features', []))
                })
                
                await worker_instance.process_request(request_data)
                
                self._request_queue_manager.mark_request_done() # Mark task as done in the queue
                
            except asyncio.CancelledError:
                logger.info(f"Worker loop {worker_instance.worker_id} cancelled.")
                break # Exit loop on cancellation
            except Exception as e:
                logger.error(f"Unhandled exception in worker loop {worker_instance.worker_id}: {e}", exc_info=True)
                record_metric("feature_pipeline.worker_error", 1, tags={
                    "worker_id": worker_instance.worker_id,
                    "error_type": type(e).__name__
                })
                # Note: `_process_request` in worker_instance handles errors for individual requests,
                # so this outer catch is for unhandled worker-level errors.
                self._stats_tracker.increment_computation_errors() # Track general worker errors

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves current statistics for the Feature Pipeline Handler.
        Delegates to the FeatureHandlerStatsTracker.
        """
        queue_size = self._request_queue_manager.get_queue_size()
        active_workers = len([w for w in self._workers if not w.done()])
        
        # Record gauge metrics
        record_metric("feature_pipeline.queue_size", queue_size, metric_type="gauge")
        record_metric("feature_pipeline.active_workers", active_workers, metric_type="gauge")
        
        return self._stats_tracker.get_stats(
            queue_size=queue_size,
            active_workers=active_workers
        )