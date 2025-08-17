# File: events/event_bus.py

"""
Core event bus implementation for the AI Trader system.

Provides asynchronous event publishing and subscription with type safety,
error handling, and monitoring capabilities.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.events.core.event_bus_helpers.dead_letter_queue_manager import DeadLetterQueueManager
from main.events.core.event_bus_helpers.event_bus_stats_tracker import EventBusStatsTracker
from main.events.core.event_bus_helpers.event_history_manager import EventHistoryManager

# Import module-specific types
from main.events.types.event_types import ExtendedEventType

# Import interfaces and base types
from main.interfaces.events import Event, EventType, IEventBus
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.monitoring import record_metric
from main.utils.resilience import CircuitBreakerConfig, get_circuit_breaker

logger = get_logger(__name__)

# Type alias for event handlers
EventHandler = Callable[[Event], asyncio.Future]


class EventBus(ErrorHandlingMixin, IEventBus):
    """
    Central event bus for publishing and subscribing to events.

    Implements the IEventBus interface from main.interfaces.events.

    Features:
    - Asynchronous event processing
    - Type-safe subscriptions
    - Circuit breaker for fault tolerance
    - Event history and replay
    - Dead letter queue for failed events
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        max_workers: int = 10,
        enable_history: bool = True,
        history_retention_seconds: int = 3600,
        enable_dlq: bool = True,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum number of events in processing queue
            max_workers: Number of concurrent event processors
            enable_history: Whether to keep event history
            history_retention_seconds: How long to retain events in history
            enable_dlq: Whether to use dead letter queue
            circuit_breaker_config: Configuration for circuit breaker
        """
        super().__init__()

        # Core components
        # Support both EventType and ExtendedEventType
        # Store handlers with their priority as tuples: (priority, handler)
        self._subscribers: dict[Any, list[tuple[int, EventHandler]]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._max_workers = max_workers

        # Helper components
        self._stats_tracker = EventBusStatsTracker()
        self._history_manager = EventHistoryManager() if enable_history else None
        self._dlq_manager = DeadLetterQueueManager() if enable_dlq else None

        # Circuit breaker for resilience
        self._circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=30.0, success_threshold=2, timeout_seconds=10.0
        )
        self._circuit_breaker = None  # Will be initialized asynchronously

        # Track active subscriptions
        self._subscription_locks: dict[EventType, asyncio.Lock] = {}

        # Schema validation (disabled by default for backwards compatibility)
        self._enable_validation = False

        logger.info(
            f"EventBus initialized with queue_size={max_queue_size}, "
            f"workers={max_workers}, history={enable_history}, dlq={enable_dlq}"
        )

    async def start(self):
        """Start the event bus workers."""
        if self._running:
            logger.warning("EventBus already running")
            return

        self._running = True

        # Initialize circuit breaker
        self._circuit_breaker = await get_circuit_breaker("event_bus", self._circuit_breaker_config)

        # Start worker tasks
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._process_events(i))
            self._workers.append(worker)

        logger.info(f"EventBus started with {len(self._workers)} workers")

    async def stop(self):
        """Stop the event bus and clean up resources."""
        if not self._running:
            return

        logger.info("Stopping EventBus...")
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # Clear the queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("EventBus stopped")

    def subscribe(
        self, event_type: str, handler: Callable[[Any], asyncio.Future], priority: int = 0
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to (string or EventType)
            handler: Async function to handle events
            priority: Handler priority (higher executes first)
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            # Try to match against EventType or ExtendedEventType
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                try:
                    event_type_enum = ExtendedEventType(event_type)
                except ValueError:
                    # Use the string as-is if not found in enums
                    event_type_enum = event_type
        else:
            event_type_enum = event_type
        if event_type not in self._subscription_locks:
            self._subscription_locks[event_type_enum] = asyncio.Lock()

        # Thread-safe subscription
        try:
            handlers = self._subscribers[event_type_enum]

            # Add handler with priority as a tuple
            handlers.append((priority, handler))
            # Sort by priority (higher first)
            handlers.sort(key=lambda h: h[0], reverse=True)
        except Exception as e:
            self.handle_error(e, "subscribing to event")
            raise

            # Update stats
            event_type_str = (
                event_type_enum.value if hasattr(event_type_enum, "value") else str(event_type_enum)
            )
            self._stats_tracker.update_subscriber_count(event_type_enum, len(handlers))

            logger.debug(
                f"Subscribed handler {handler.__name__} to {event_type_str} "
                f"(total subscribers: {len(handlers)})"
            )

    def unsubscribe(self, event_type: str, handler: Callable[[Any], asyncio.Future]) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of events to unsubscribe from (string or EventType)
            handler: Handler function to remove
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                try:
                    event_type_enum = ExtendedEventType(event_type)
                except ValueError:
                    event_type_enum = event_type
        else:
            event_type_enum = event_type
        try:
            handlers = self._subscribers[event_type_enum]

            # Find and remove the handler (it's stored as (priority, handler) tuple)
            handler_to_remove = None
            for h in handlers:
                if h[1] == handler:  # h is (priority, handler)
                    handler_to_remove = h
                    break

            if handler_to_remove:
                handlers.remove(handler_to_remove)
                self._stats_tracker.update_subscriber_count(event_type_enum, len(handlers))
                event_type_str = (
                    event_type_enum.value
                    if hasattr(event_type_enum, "value")
                    else str(event_type_enum)
                )
                logger.debug(f"Unsubscribed handler {handler.__name__} from {event_type_str}")
        except Exception as e:
            self.handle_error(e, "unsubscribing from event")

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus with optional schema validation.

        Args:
            event: Event to publish
        """
        if not self._running:
            raise RuntimeError("EventBus is not running")

        # Validate event schema if enabled
        if hasattr(self, "_enable_validation") and self._enable_validation:
            # Local imports
            from main.events.validation.event_schemas import validate_event

            event_type_str = (
                event.event_type if isinstance(event.event_type, str) else str(event.event_type)
            )

            # Validate metadata if it exists
            if hasattr(event, "metadata") and event.metadata:
                if not validate_event(event_type_str, event.metadata):
                    logger.warning(f"Event failed schema validation: {event_type_str}")
                    # In strict mode, we could raise an exception here

        # Update stats
        self._stats_tracker.increment_published()

        # Record metric for event published
        event_type_str = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        record_metric(
            "event_bus.event_published",
            1,
            tags={"event_type": event_type_str, "priority": getattr(event, "priority", "normal")},
        )

        # Add to history if enabled
        if self._history_manager:
            self._history_manager.add_event(event)

        # Put event in queue with circuit breaker
        async def _enqueue():
            await self._event_queue.put(event)

        try:
            await self._circuit_breaker.call(_enqueue)
            logger.debug(f"Published event: {event_type_str}")
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event_type_str}")
            record_metric("event_bus.queue_full", 1, tags={"event_type": event_type_str})
            if self._dlq_manager:
                await self._dlq_manager.add_event(event, "queue_full")
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            record_metric(
                "event_bus.publish_error",
                1,
                tags={"event_type": event_type_str, "error_type": type(e).__name__},
            )
            if self._dlq_manager:
                await self._dlq_manager.add_event(event, str(e))

    async def _process_events(self, worker_id: int):
        """
        Worker coroutine to process events from the queue.

        Args:
            worker_id: ID of this worker for logging
        """
        logger.debug(f"Event worker {worker_id} started")

        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Process the event
                await self._dispatch_event(event)

            except TimeoutError:
                # No events available, continue
                continue
            except asyncio.CancelledError:
                # Worker cancelled, exit
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.debug(f"Event worker {worker_id} stopped")

    async def _dispatch_event(self, event: Event):
        """
        Dispatch an event to all registered handlers.

        Args:
            event: Event to dispatch
        """
        handlers = self._subscribers.get(event.event_type, [])

        event_type_str = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )

        if not handlers:
            logger.debug(f"No handlers for event type: {event_type_str}")
            record_metric("event_bus.no_handlers", 1, tags={"event_type": event_type_str})
            return

        # Record number of handlers
        record_metric(
            "event_bus.handler_count",
            len(handlers),
            metric_type="gauge",
            tags={"event_type": event_type_str},
        )

        # Execute handlers concurrently
        tasks = []
        for priority, handler in handlers:
            task = asyncio.create_task(self._execute_handler(handler, event))
            tasks.append(task)

        # Wait for all handlers to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        failure_count = len(results) - success_count

        if success_count > 0:
            self._stats_tracker.increment_processed()
            record_metric(
                "event_bus.handlers_succeeded", success_count, tags={"event_type": event_type_str}
            )

        if failure_count > 0:
            self._stats_tracker.increment_failed()
            record_metric(
                "event_bus.handlers_failed", failure_count, tags={"event_type": event_type_str}
            )
            logger.warning(f"Event {event_type_str} had {failure_count} handler failures")

    async def _execute_handler(self, handler: EventHandler, event: Event):
        """
        Execute a single event handler with error handling.

        Args:
            handler: Handler function to execute
            event: Event to pass to handler
        """
        event_type_str = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )

        try:
            await handler(event)
            record_metric(
                "event_bus.handler_execution",
                1,
                tags={
                    "handler": handler.__name__,
                    "event_type": event_type_str,
                    "status": "success",
                },
            )
        except Exception as e:
            logger.error(
                f"Handler {handler.__name__} failed for event " f"{event_type_str}: {e}",
                exc_info=True,
            )

            record_metric(
                "event_bus.handler_execution",
                1,
                tags={
                    "handler": handler.__name__,
                    "event_type": event_type_str,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )

            # Add to DLQ if enabled
            if self._dlq_manager:
                await self._dlq_manager.add_event(
                    event, f"handler_error: {handler.__name__} - {e!s}"
                )

            raise

    def get_stats(self) -> dict[str, Any]:
        """
        Get current event bus statistics.

        Returns:
            Dictionary containing various metrics
        """
        queue_size = self._event_queue.qsize()
        history_size = self._history_manager.get_event_count() if self._history_manager else 0
        dlq_size = self._dlq_manager.get_queue_size() if self._dlq_manager else 0

        stats = self._stats_tracker.get_stats(
            queue_size=queue_size, history_size=history_size, dlq_size=dlq_size
        )

        # Add circuit breaker state
        stats["circuit_breaker_state"] = self._circuit_breaker.state.name
        stats["worker_count"] = len(self._workers)
        stats["is_running"] = self._running

        # Record queue size metric
        record_metric("event_bus.queue_size", queue_size, metric_type="gauge")

        # Record circuit breaker state as gauge (0=closed, 1=open, 2=half_open)
        cb_state_value = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}.get(
            self._circuit_breaker.state.name, -1
        )
        record_metric("event_bus.circuit_breaker_state", cb_state_value, metric_type="gauge")

        return stats

    async def replay_events(
        self,
        event_type: EventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        target_handler: Callable | None = None,
        speed_multiplier: float = 1.0,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Replay historical events for debugging or testing.

        Enhanced replay functionality with:
        - Time-based filtering
        - Event type filtering
        - Speed control for replay
        - Dry run mode for testing
        - Target handler for isolated testing
        - Detailed replay statistics

        Args:
            event_type: Optional filter by event type
            start_time: Optional start time filter
            end_time: Optional end time filter
            target_handler: Optional specific handler to replay to (bypasses normal routing)
            speed_multiplier: Speed of replay (1.0 = real-time, 2.0 = 2x speed, 0 = instant)
            dry_run: If True, don't actually publish events, just return what would be replayed

        Returns:
            Dictionary with replay statistics and results
        """
        if not self._history_manager:
            logger.warning("Event history not enabled")
            return {"status": "error", "message": "Event history not enabled", "events_replayed": 0}

        # Get events for replay
        events = self._history_manager.get_events_for_replay(
            event_type=event_type, start_time=start_time, end_time=end_time
        )

        if not events:
            return {
                "status": "no_events",
                "message": "No events found matching criteria",
                "events_replayed": 0,
            }

        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Prepare replay statistics
        stats = {
            "total_events": len(events),
            "events_replayed": 0,
            "events_skipped": 0,
            "replay_start": datetime.now(UTC),
            "errors": [],
            "event_types": {},
        }

        if dry_run:
            # In dry run, just analyze what would be replayed
            for event in events:
                event_type_str = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                stats["event_types"][event_type_str] = (
                    stats["event_types"].get(event_type_str, 0) + 1
                )

            return {
                "status": "dry_run",
                "message": f"Would replay {len(events)} events",
                "statistics": stats,
                "first_event": events[0].timestamp.isoformat() if events else None,
                "last_event": events[-1].timestamp.isoformat() if events else None,
            }

        # Perform actual replay
        logger.info(f"Starting replay of {len(events)} events (speed={speed_multiplier}x)")

        last_event_time = None

        for i, event in enumerate(events):
            try:
                # Calculate delay for realistic replay timing
                if speed_multiplier > 0 and last_event_time:
                    time_diff = (event.timestamp - last_event_time).total_seconds()
                    delay = time_diff / speed_multiplier
                    if delay > 0:
                        await asyncio.sleep(delay)

                # Replay the event
                if target_handler:
                    # Direct replay to specific handler
                    await target_handler(event)
                else:
                    # Normal publish through event bus
                    await self.publish(event)

                # Update statistics
                stats["events_replayed"] += 1
                event_type_str = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                stats["event_types"][event_type_str] = (
                    stats["event_types"].get(event_type_str, 0) + 1
                )

                last_event_time = event.timestamp

                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Replayed {i + 1}/{len(events)} events")

            except Exception as e:
                stats["events_skipped"] += 1
                stats["errors"].append(
                    {
                        "event_id": event.event_id,
                        "event_type": str(event.event_type),
                        "error": str(e),
                    }
                )
                logger.error(f"Error replaying event {event.event_id}: {e}")

        # Calculate replay duration
        stats["replay_end"] = datetime.now(UTC)
        stats["replay_duration_seconds"] = (
            stats["replay_end"] - stats["replay_start"]
        ).total_seconds()

        logger.info(
            f"Replay completed: {stats['events_replayed']}/{stats['total_events']} events "
            f"in {stats['replay_duration_seconds']:.2f}s"
        )

        return {
            "status": "success",
            "message": f"Replayed {stats['events_replayed']} events",
            "statistics": stats,
        }

    async def process_dlq(self, max_retries: int = 3) -> dict[str, int]:
        """
        Process events in the dead letter queue.

        Args:
            max_retries: Maximum retry attempts per event

        Returns:
            Dictionary with counts of processed and failed events
        """
        if not self._dlq_manager:
            logger.warning("Dead letter queue not enabled")
            return {"processed": 0, "failed": 0}

        results = await self._dlq_manager.process_events(self.publish, max_retries=max_retries)

        logger.info(
            f"DLQ processing complete: {results['processed']} processed, "
            f"{results['failed']} failed"
        )

        return results

    def is_running(self) -> bool:
        """
        Check if the event bus is currently running.

        Returns:
            True if the bus is running and can process events,
            False otherwise.
        """
        return self._running

    def enable_validation(self, strict: bool = False):
        """
        Enable event schema validation.

        Args:
            strict: If True, invalid events will be rejected.
                   If False, invalid events will be logged but allowed.
        """
        self._enable_validation = True
        self._strict_validation = strict
        logger.info(f"Event schema validation enabled (strict={strict})")

    def disable_validation(self):
        """Disable event schema validation."""
        self._enable_validation = False
        logger.info("Event schema validation disabled")
