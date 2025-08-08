# File: src/main/events/scanner_bridge_helpers/request_dispatcher.py

from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Optional

# Corrected absolute imports
from main.interfaces.events import IEventBus
from main.events.types import FeatureRequestEvent # Only need FeatureRequestEvent for publishing
from main.events.handlers.scanner_bridge_helpers.feature_request_batcher import FeatureRequestBatch # For type hinting batch
from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import record_metric

logger = get_logger(__name__)

class RequestDispatcher(ErrorHandlingMixin):
    """
    Dispatches batched feature computation requests as events onto the Event Bus.
    
    Now inherits from ErrorHandlingMixin for better error handling.
    """

    def __init__(self, event_bus: IEventBus):
        """
        Initializes the RequestDispatcher.

        Args:
            event_bus: The EventBus instance to publish events to.
        """
        self.event_bus = event_bus
        logger.debug("RequestDispatcher initialized.")

    async def send_feature_request_batch(self, request_batch: FeatureRequestBatch):
        """
        Converts a FeatureRequestBatch into a FeatureRequestEvent and publishes it to the Event Bus.

        Args:
            request_batch: The FeatureRequestBatch object to dispatch.
        """
        if not request_batch.symbols:
            logger.warning("Attempted to send an empty feature request batch. Skipping.")
            return
        
        try:
            # Convert set of features to list for the event data
            features_list = list(request_batch.features)

            # Create FeatureRequestEvent
            feature_event = FeatureRequestEvent(
                symbols=list(request_batch.symbols), # Convert set of symbols to list
                features=features_list,
                requester='scanner_bridge', # Source of the request
                priority=request_batch.priority
            )
            
            # Add metadata from the batch
            feature_event.metadata = {
                'scanner_sources': list(request_batch.scanner_sources),
                'batch_size_symbols': len(request_batch.symbols),
                'request_age_seconds': (datetime.now(timezone.utc) - request_batch.created_at).total_seconds(),
                'correlation_ids': list(request_batch.correlation_ids) if request_batch.correlation_ids else []
            }
        
            # Publish event to the bus
            await self.event_bus.publish(feature_event)
            
            # Record metrics
            record_metric("scanner_bridge.request_dispatched", 1, tags={
                "symbols": len(request_batch.symbols),
                "features": len(features_list),
                "priority": request_batch.priority
            })
            
            logger.info(f"Dispatched FeatureRequestEvent: {len(request_batch.symbols)} symbols, "
                       f"{len(features_list)} features, priority {request_batch.priority}.")
        except Exception as e:
            self.handle_error(e, f"dispatching feature request batch")
            raise  # Re-raise to maintain behavior