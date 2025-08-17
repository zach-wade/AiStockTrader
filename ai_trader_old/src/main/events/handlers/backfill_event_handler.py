"""
Backfill Event Handler

Handles BackfillRequested events from the event bus and executes
the actual backfill operations using the historical backfill system.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from typing import Any

# Local imports
from main.interfaces.events import IEventBus
from main.utils.core import AsyncCircuitBreaker, ErrorHandlingMixin, async_retry, get_logger, timer


@dataclass
class BackfillTask:
    """Represents a backfill task to be executed."""

    backfill_id: str
    symbol: str
    layer: int
    data_types: list
    start_date: str
    end_date: str
    priority: str
    attempt: int = 0

    def get_dedup_key(self) -> str:
        """Generate deduplication key for this task."""
        key_data = f"{self.symbol}:{self.layer}:{self.start_date}:{self.end_date}"
        return hashlib.md5(key_data.encode()).hexdigest()


class BackfillEventHandler(ErrorHandlingMixin):
    """
    Handles BackfillRequested events and executes backfill operations.

    This handler bridges the event-driven scheduling system with the
    actual backfill execution logic, providing:
    - Automatic backfill execution when events are received
    - Concurrency control to prevent overload
    - Deduplication to avoid redundant backfills
    - Progress tracking and completion notifications
    - Error handling with retries and circuit breaker
    """

    def __init__(self, event_bus: IEventBus, config: dict[str, Any] | None = None):
        """
        Initialize the backfill event handler.

        Args:
            event_bus: Event bus for subscribing to events
            config: Handler configuration
        """
        super().__init__()
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Configuration
        handler_config = self.config.get("backfill_handler", {})
        self.enabled = handler_config.get("enabled", True)
        self.max_concurrent = handler_config.get("max_concurrent_backfills", 3)
        self.retry_attempts = handler_config.get("retry_attempts", 3)
        self.retry_delay = handler_config.get("retry_delay_seconds", 60)
        self.dedup_window_minutes = handler_config.get("deduplication_window_minutes", 60)

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._active_tasks: set[str] = set()
        self._completed_tasks: dict[str, datetime] = {}

        # Circuit breaker for protection
        self._circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=handler_config.get("circuit_breaker_threshold", 5),
            recovery_timeout=handler_config.get("circuit_breaker_timeout", 300),
            expected_exception=Exception,
        )

        # Statistics
        self._stats = {
            "received": 0,
            "executed": 0,
            "succeeded": 0,
            "failed": 0,
            "deduplicated": 0,
            "retried": 0,
        }

        self.logger.info(
            f"BackfillEventHandler initialized "
            f"(enabled={self.enabled}, max_concurrent={self.max_concurrent})"
        )

    async def initialize(self) -> None:
        """Initialize the handler and subscribe to events."""
        if not self.enabled:
            self.logger.info("BackfillEventHandler is disabled, skipping initialization")
            return

        try:
            # Subscribe to BackfillRequested events (subscribe is synchronous)
            self.event_bus.subscribe("BackfillRequested", self.handle_backfill_requested)

            self.logger.info("BackfillEventHandler subscribed to BackfillRequested events")

        except Exception as e:
            self.logger.error(f"Failed to initialize BackfillEventHandler: {e}")
            raise

    async def handle_backfill_requested(self, event) -> None:
        """
        Handle a BackfillRequested event.

        Args:
            event: The backfill request event (Event object)
        """
        self._stats["received"] += 1

        try:
            # Extract event data from Event object
            if hasattr(event, "metadata"):
                event_data = event.metadata
            elif hasattr(event, "data"):
                event_data = event.data
            else:
                # Fallback for dict-like events
                event_data = event.get("data", event) if isinstance(event, dict) else event

            # Create backfill task
            task = BackfillTask(
                backfill_id=event_data.get("backfill_id"),
                symbol=event_data.get("symbol"),
                layer=event_data.get("layer"),
                data_types=event_data.get("data_types", ["market_data"]),
                start_date=event_data.get("start_date"),
                end_date=event_data.get("end_date"),
                priority=event_data.get("priority", "normal"),
            )

            # Check for deduplication
            if self._is_duplicate(task):
                self._stats["deduplicated"] += 1
                self.logger.debug(
                    f"Skipping duplicate backfill for {task.symbol} " f"(layer={task.layer})"
                )
                return

            # Execute with concurrency control
            await self._execute_backfill_with_control(task)

        except Exception as e:
            self._stats["failed"] += 1
            self.logger.error(f"Error handling backfill request: {e}")
            await self._publish_failure_event(event_data, str(e))

    async def _execute_backfill_with_control(self, task: BackfillTask) -> None:
        """Execute backfill with concurrency control and circuit breaker."""
        async with self._semaphore:
            # Check circuit breaker
            if self._circuit_breaker.state == "open":
                self.logger.warning(f"Circuit breaker open, skipping backfill for {task.symbol}")
                return

            # Mark as active
            dedup_key = task.get_dedup_key()
            self._active_tasks.add(dedup_key)

            try:
                # Execute with retries
                await self._execute_backfill_with_retry(task)

            finally:
                # Mark as completed
                self._active_tasks.discard(dedup_key)
                self._completed_tasks[dedup_key] = datetime.now(UTC)

                # Clean old completed tasks
                self._cleanup_completed_tasks()

    @async_retry(max_attempts=3, delay=60.0)
    async def _execute_backfill_with_retry(self, task: BackfillTask) -> None:
        """Execute backfill with retry logic."""
        task.attempt += 1

        if task.attempt > 1:
            self._stats["retried"] += 1
            self.logger.info(f"Retrying backfill for {task.symbol} " f"(attempt {task.attempt})")

        try:
            with timer(f"backfill_{task.symbol}_{task.layer}"):
                await self._execute_backfill(task)

            self._stats["succeeded"] += 1

        except Exception:
            self._stats["failed"] += 1
            # Circuit breaker handles failures automatically in its call method
            raise

    async def _execute_backfill(self, task: BackfillTask) -> None:
        """
        Execute the actual backfill operation with comprehensive metrics.

        Args:
            task: The backfill task to execute
        """
        self._stats["executed"] += 1

        # Start timing
        start_time = datetime.now(UTC)

        self.logger.info(
            f"Executing backfill: {task.backfill_id} " f"for {task.symbol} (layer={task.layer})"
        )

        # Import here to avoid circular dependency
        # Local imports
        from main.app.historical_backfill import run_historical_backfill
        from main.utils.monitoring import record_metric

        # Calculate lookback days
        start_date = datetime.fromisoformat(task.start_date.replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(task.end_date.replace("Z", "+00:00"))
        lookback_days = (end_date - start_date).days

        # Create backfill configuration
        backfill_config = {
            "stages": task.data_types,
            "symbols": [task.symbol],
            "lookback_days": lookback_days,
            "layer": task.layer,
            "backfill_id": task.backfill_id,
            "priority": task.priority,
            "test_mode": False,
        }

        # Execute backfill with timing
        try:
            result = await run_historical_backfill(backfill_config)

            # Calculate duration
            end_time = datetime.now(UTC)
            duration_seconds = (end_time - start_time).total_seconds()

            # Record success metrics
            records_processed = result.get("total_records", 0)

            # Record duration histogram
            record_metric(
                "backfill.duration_seconds",
                duration_seconds,
                metric_type="histogram",
                tags={
                    "symbol": task.symbol,
                    "layer": str(task.layer),
                    "data_types": ",".join(task.data_types),
                    "status": "success",
                    "priority": task.priority,
                },
            )

            # Record processing rate
            if duration_seconds > 0:
                records_per_second = records_processed / duration_seconds
                record_metric(
                    "backfill.records_per_second",
                    records_per_second,
                    metric_type="gauge",
                    tags={
                        "symbol": task.symbol,
                        "layer": str(task.layer),
                        "data_types": ",".join(task.data_types),
                    },
                )

            # Record data volume
            record_metric(
                "backfill.records_processed",
                records_processed,
                metric_type="counter",
                tags={
                    "symbol": task.symbol,
                    "layer": str(task.layer),
                    "data_types": ",".join(task.data_types),
                },
            )

            # Track by data type for granular analysis
            for data_type in task.data_types:
                type_records = result.get(f"{data_type}_records", 0)
                if type_records > 0:
                    record_metric(
                        f"backfill.{data_type}.records",
                        type_records,
                        metric_type="counter",
                        tags={"symbol": task.symbol, "layer": str(task.layer)},
                    )

            # Log performance summary
            self.logger.info(
                f"Backfill completed: {task.backfill_id} | "
                f"Duration: {duration_seconds:.2f}s | "
                f"Records: {records_processed} | "
                f"Rate: {records_processed/duration_seconds:.1f} rec/s"
            )

        except Exception as e:
            # Record failure metrics
            end_time = datetime.now(UTC)
            duration_seconds = (end_time - start_time).total_seconds()

            record_metric(
                "backfill.duration_seconds",
                duration_seconds,
                metric_type="histogram",
                tags={
                    "symbol": task.symbol,
                    "layer": str(task.layer),
                    "data_types": ",".join(task.data_types),
                    "status": "failed",
                    "error_type": type(e).__name__,
                },
            )

            record_metric(
                "backfill.failures",
                1,
                metric_type="counter",
                tags={
                    "symbol": task.symbol,
                    "layer": str(task.layer),
                    "error_type": type(e).__name__,
                },
            )

            raise

        # Publish completion event
        await self._publish_completion_event(task, result)

    async def _publish_completion_event(self, task: BackfillTask, result: dict[str, Any]) -> None:
        """Publish a BackfillCompleted event."""
        try:
            # Local imports
            from main.interfaces.events.event_types import BackfillCompletedEvent

            event = BackfillCompletedEvent(
                backfill_id=task.backfill_id,
                symbol=task.symbol,
                layer=task.layer,
                success=result.get("success", False),
                records_processed=result.get("total_records", 0),
                duration_seconds=result.get("duration_seconds", 0),
                errors=result.get("errors", []),
            )

            await self.event_bus.publish(event)

        except Exception as e:
            self.logger.error(f"Failed to publish completion event: {e}")

    async def _publish_failure_event(self, event_data: dict[str, Any], error: str) -> None:
        """Publish a backfill failure event."""
        try:
            # Local imports
            from main.interfaces.events import Event

            failure_event = Event(
                event_type="BackfillFailed",
                metadata={
                    "backfill_id": event_data.get("backfill_id"),
                    "symbol": event_data.get("symbol"),
                    "error": error,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            await self.event_bus.publish(failure_event)

        except Exception as e:
            self.logger.error(f"Failed to publish failure event: {e}")

    def _is_duplicate(self, task: BackfillTask) -> bool:
        """Check if this backfill is a duplicate."""
        dedup_key = task.get_dedup_key()

        # Check if currently active
        if dedup_key in self._active_tasks:
            return True

        # Check if recently completed
        if dedup_key in self._completed_tasks:
            completed_time = self._completed_tasks[dedup_key]
            age_minutes = (datetime.now(UTC) - completed_time).total_seconds() / 60

            if age_minutes < self.dedup_window_minutes:
                return True

        return False

    def _cleanup_completed_tasks(self) -> None:
        """Remove old entries from completed tasks."""
        cutoff_time = datetime.now(UTC) - timedelta(minutes=self.dedup_window_minutes)

        keys_to_remove = [
            key for key, timestamp in self._completed_tasks.items() if timestamp < cutoff_time
        ]

        for key in keys_to_remove:
            del self._completed_tasks[key]

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "statistics": self._stats.copy(),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "circuit_breaker_state": self._circuit_breaker.state,
            "configuration": {
                "enabled": self.enabled,
                "max_concurrent": self.max_concurrent,
                "retry_attempts": self.retry_attempts,
                "dedup_window_minutes": self.dedup_window_minutes,
            },
        }
