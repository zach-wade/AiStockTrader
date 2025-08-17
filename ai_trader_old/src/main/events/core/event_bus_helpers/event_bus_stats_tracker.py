# File: src/main/events/event_bus_helpers/event_bus_stats_tracker.py

# Standard library imports
from collections import defaultdict
from typing import Any

# Local imports
# Corrected absolute import for EventType
from main.events.types import EventType
from main.utils.core import get_logger
from main.utils.monitoring import MetricsCollector, record_metric

logger = get_logger(__name__)


class EventBusStatsTracker:
    """
    Tracks and provides statistics about event bus operations,
    including published, processed, and failed events.

    Now uses utils.monitoring.MetricsCollector for consistency.
    """

    def __init__(self):
        """Initializes the EventBusStatsTracker with MetricsCollector."""
        self.metrics = MetricsCollector()
        self._subscribers_by_type = defaultdict(int)
        logger.debug("EventBusStatsTracker initialized with MetricsCollector.")

    def increment_published(self):
        """Increments the count of published events."""
        self.metrics.increment_counter("event_bus.events_published")
        record_metric("event_bus.published", 1)

    def increment_processed(self):
        """Increments the count of successfully processed events."""
        self.metrics.increment_counter("event_bus.events_processed")
        record_metric("event_bus.processed", 1)

    def increment_failed(self):
        """Increments the count of events that failed processing by a handler."""
        self.metrics.increment_counter("event_bus.events_failed")
        record_metric("event_bus.failed", 1)

    def record_processing_time(self, processing_time: float):
        """Records the time taken to process an event."""
        self.metrics.record_histogram("event_bus.processing_time", processing_time)

    def record_queue_size(self, size: int):
        """Records the current size of the event queue."""
        self.metrics.set_gauge("event_bus.queue_size", size)

    def update_subscriber_count(self, event_type: EventType, count: int):
        """Updates the count of active subscribers for a given event type."""
        self._subscribers_by_type[event_type.value] = count
        self.metrics.set_gauge(f"event_bus.subscribers.{event_type.value}", count)

    def get_stats(
        self, queue_size: int = 0, history_size: int = 0, dlq_size: int = 0
    ) -> dict[str, Any]:
        """
        Retrieves current event bus statistics.

        Args:
            queue_size: Current size of the event processing queue.
            history_size: Current number of events in history.
            dlq_size: Current number of events in the Dead Letter Queue.

        Returns:
            A dictionary containing various event bus statistics.
        """
        # Get metrics from collector
        published_stats = self.metrics.get_metric_stats("event_bus.events_published") or {}
        processed_stats = self.metrics.get_metric_stats("event_bus.events_processed") or {}
        failed_stats = self.metrics.get_metric_stats("event_bus.events_failed") or {}

        # Handle mock objects or non-dict values
        def get_metric_value(stats, key="latest", default=0):
            if isinstance(stats, dict):
                value = stats.get(key, default)
                # Ensure we get a numeric value
                try:
                    return int(value) if value is not None else default
                except (TypeError, ValueError):
                    return default
            return default

        return {
            "events_published": get_metric_value(published_stats),
            "events_processed": get_metric_value(processed_stats),
            "events_failed": get_metric_value(failed_stats),
            "subscribers_by_type": dict(self._subscribers_by_type),
            "queue_size": queue_size,
            "history_size": history_size,
            "dead_letter_size": dlq_size,
            "total_subscribers": sum(self._subscribers_by_type.values()),
            "processing_time": (
                self.metrics.get_metric_value("event_bus.processing_time")
                if hasattr(self.metrics, "get_metric_value")
                else 0
            ),
        }
