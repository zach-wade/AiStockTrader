# File: src/main/events/event_bus_helpers/event_history_manager.py

# Standard library imports
from collections import deque
from datetime import datetime

# Local imports
# Import from interfaces layer
from main.interfaces.events import Event, EventType
from main.utils.core import get_logger
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


class EventHistoryManager:
    """
    Manages the storage and retrieval of historical events for replay purposes.
    Maintains a bounded history using collections.deque for O(1) operations.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initializes the EventHistoryManager.

        Args:
            max_history: The maximum number of events to keep in history.
        """
        self._history = deque(maxlen=max_history)
        self._max_history = max_history
        logger.debug(f"EventHistoryManager initialized with max_history: {max_history}")

    def add_event(self, event: Event):
        """
        Adds an event to the history. If history size exceeds max_history,
        the oldest event is automatically removed by deque.
        """
        self._history.append(event)

        # Handle string or enum event types
        event_type_str = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        logger.debug(
            f"Added event to history: {event_type_str}. Current history size: {len(self._history)}"
        )

        # Record metrics
        record_metric("event_history.event_added", 1, tags={"event_type": event_type_str})
        record_metric("event_history.size", len(self._history), metric_type="gauge")

    def get_history(self, event_type: EventType | None = None, limit: int = 100) -> list[Event]:
        """
        Retrieves historical events, optionally filtered by event type and limited by count.

        Args:
            event_type: Optional. The type of event to filter by.
            limit: The maximum number of historical events to return (from the most recent).

        Returns:
            A list of historical Event objects.
        """
        # Convert deque to list for filtering and slicing
        history_list = list(self._history)

        if event_type:
            history_list = [e for e in history_list if e.event_type == event_type]
            event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
            record_metric(
                "event_history.query_filtered",
                1,
                tags={"event_type": event_type_str, "results": len(history_list)},
            )
        else:
            record_metric("event_history.query_all", 1, tags={"results": len(history_list)})

        return history_list[-limit:]  # Return the most recent `limit` events

    def get_history_size(self) -> int:
        """Returns the current number of events in history."""
        return len(self._history)

    def get_events_for_replay(
        self,
        event_type: EventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Event]:
        """
        Retrieves historical events that match criteria for replay.

        Args:
            event_type: Optional. The type of events to retrieve for replay.
            start_time: Optional. The start of the time range for replay (inclusive).
            end_time: Optional. The end of the time range for replay (inclusive).

        Returns:
            A list of Event objects matching the replay criteria.
        """
        # Efficient filtering directly on deque without converting to list first
        events_to_replay = [
            e
            for e in self._history
            if (not event_type or e.event_type == event_type)
            and (not start_time or e.timestamp >= start_time)
            and (not end_time or e.timestamp <= end_time)
        ]
        if event_type:
            event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
            logger.debug(f"Found {len(events_to_replay)} {event_type_str} events for replay.")
        else:
            event_type_str = "all"
            logger.debug(f"Found {len(events_to_replay)} events for replay (all types).")

        # Record replay query metrics
        record_metric(
            "event_history.replay_query",
            1,
            tags={
                "event_type": event_type_str,
                "has_start_time": bool(start_time),
                "has_end_time": bool(end_time),
                "results": len(events_to_replay),
            },
        )

        return events_to_replay
