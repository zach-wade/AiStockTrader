"""
Dead Letter Queue (DLQ) manager for failed events.

This module manages events that fail processing:
- Stores failed events with error information
- Provides retry mechanisms with backoff
- Tracks failure patterns
- Enables manual inspection and reprocessing
- Implements TTL for automatic cleanup
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    ensure_utc,
    secure_dumps,
    secure_loads
)
from main.utils.database import (
    DatabasePool,
    batch_upsert,
    transaction_context,
    execute_with_retry
)
from main.utils.monitoring import record_metric, timer

from main.events.types import Event, EventType

logger = get_logger(__name__)


@dataclass
class FailedEvent:
    """Represents a failed event in the DLQ."""
    event: Event
    error_message: str
    error_type: str
    failure_count: int = 1
    first_failure: datetime = field(default_factory=datetime.utcnow)
    last_failure: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def increment_failure(self, error_message: str) -> None:
        """Increment failure count and update error info."""
        self.failure_count += 1
        self.last_failure = datetime.utcnow()
        self.error_message = error_message


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 300.0  # 5 minutes
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(
            self.initial_delay_seconds * (self.backoff_multiplier ** attempt),
            self.max_delay_seconds
        )
        
        if self.jitter:
            # Add random jitter (�25%)
            from main.utils.core import secure_uniform
            jitter_factor = 0.75 + (secure_uniform(0, 0.5))
            delay *= jitter_factor
            
        return delay


class DeadLetterQueueManager(ErrorHandlingMixin):
    """
    Manages failed events in a dead letter queue.
    
    Features:
    - Persistent storage of failed events
    - Configurable retry policies
    - Failure pattern analysis
    - Manual and automatic reprocessing
    - TTL-based cleanup
    - Metrics and monitoring
    """
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        retry_policy: Optional[RetryPolicy] = None,
        ttl_seconds: int = 86400,  # 24 hours default
        max_queue_size: int = 10000
    ):
        """
        Initialize DLQ manager.
        
        Args:
            db_pool: Optional database pool for persistence
            retry_policy: Retry configuration
            ttl_seconds: Time-to-live for failed events
            max_queue_size: Maximum queue size
        """
        super().__init__()
        self.db_pool = db_pool
        self.retry_policy = retry_policy or RetryPolicy()
        self.ttl_seconds = ttl_seconds
        self.max_queue_size = max_queue_size
        
        # In-memory storage (can be backed by database)
        self._queue: List[FailedEvent] = []
        self._event_index: Dict[str, FailedEvent] = {}
        
        # Failure tracking
        self._failure_counts: Dict[EventType, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        # Retry tracking
        self._retry_tasks: Dict[str, asyncio.Task] = {}
        
        logger.debug(
            f"DeadLetterQueueManager initialized with ttl={ttl_seconds}s, "
            f"max_size={max_queue_size}"
        )
    
    async def add_event(
        self,
        event: Event,
        error_message: str,
        error_type: Optional[str] = None
    ) -> bool:
        """
        Add a failed event to the DLQ.
        
        Args:
            event: The failed event
            error_message: Error description
            error_type: Optional error classification
            
        Returns:
            Success status
        """
        try:
            # Check queue size
            if len(self._queue) >= self.max_queue_size:
                await self._evict_oldest()
            
            # Create or update failed event
            event_id = self._get_event_id(event)
            
            if event_id in self._event_index:
                # Update existing failed event
                failed_event = self._event_index[event_id]
                failed_event.increment_failure(error_message)
            else:
                # Create new failed event
                failed_event = FailedEvent(
                    event=event,
                    error_message=error_message,
                    error_type=error_type or "unknown"
                )
                
                self._queue.append(failed_event)
                self._event_index[event_id] = failed_event
            
            # Update tracking
            self._failure_counts[event.event_type] += 1
            self._error_counts[error_type or "unknown"] += 1
            
            # Persist if database available
            if self.db_pool:
                await self._persist_failed_event(failed_event)
            
            # Record metrics
            record_metric(
                'dlq.event_added',
                1,
                tags={
                    'event_type': event.event_type.value,
                    'error_type': error_type or "unknown"
                }
            )
            
            logger.warning(
                f"Added event to DLQ: {event.event_type.value} - {error_message}"
            )
            
            return True
        except Exception as e:
            self.handle_error(e, "adding event to DLQ")
            return False
    
    async def process_events(
        self,
        publish_func: Callable[[Event], asyncio.Future],
        max_retries: Optional[int] = None,
        event_filter: Optional[Callable[[FailedEvent], bool]] = None
    ) -> Dict[str, int]:
        """
        Process events in the DLQ with retry logic.
        
        Args:
            publish_func: Function to republish events
            max_retries: Override default max retries
            event_filter: Optional filter for events to process
            
        Returns:
            Dictionary with processing statistics
        """
        max_retries = max_retries or self.retry_policy.max_retries
        
        processed = 0
        succeeded = 0
        failed = 0
        
        # Get events to process
        events_to_process = [
            fe for fe in self._queue
            if (not event_filter or event_filter(fe)) and
            fe.failure_count <= max_retries
        ]
        
        logger.info(f"Processing {len(events_to_process)} events from DLQ")
        
        for failed_event in events_to_process:
            processed += 1
            
            try:
                # Calculate retry delay
                delay = self.retry_policy.get_delay(failed_event.failure_count - 1)
                
                # Wait before retry
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Attempt to republish
                await publish_func(failed_event.event)
                
                # Success - remove from DLQ
                await self.remove_event(failed_event.event)
                succeeded += 1
                
                logger.info(
                    f"Successfully reprocessed event from DLQ: "
                    f"{failed_event.event.event_type.value}"
                )
                
            except Exception as e:
                failed += 1
                
                # Update failure info
                await self.add_event(
                    failed_event.event,
                    str(e),
                    type(e).__name__
                )
                
                logger.error(
                    f"Failed to reprocess event from DLQ: {e}",
                    exc_info=True
                )
        
        # Record metrics
        record_metric('dlq.batch_processed', processed)
        record_metric('dlq.batch_succeeded', succeeded)
        record_metric('dlq.batch_failed', failed)
        
        return {
            'processed': processed,
            'succeeded': succeeded,
            'failed': failed
        }
    
    async def remove_event(self, event: Event) -> bool:
        """
        Remove an event from the DLQ.
        
        Args:
            event: Event to remove
            
        Returns:
            Success status
        """
        event_id = self._get_event_id(event)
        
        if event_id in self._event_index:
            failed_event = self._event_index[event_id]
            
            # Remove from queue and index
            self._queue.remove(failed_event)
            del self._event_index[event_id]
            
            # Cancel any retry task
            if event_id in self._retry_tasks:
                self._retry_tasks[event_id].cancel()
                del self._retry_tasks[event_id]
            
            # Remove from persistence
            if self.db_pool:
                await self._remove_persisted_event(event_id)
            
            return True
        
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def get_failed_events(
        self,
        event_type: Optional[EventType] = None,
        error_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[FailedEvent]:
        """
        Get failed events with optional filters.
        
        Args:
            event_type: Filter by event type
            error_type: Filter by error type
            limit: Maximum events to return
            
        Returns:
            List of failed events
        """
        filtered = self._queue
        
        if event_type:
            filtered = [
                fe for fe in filtered
                if fe.event.event_type == event_type
            ]
        
        if error_type:
            filtered = [
                fe for fe in filtered
                if fe.error_type == error_type
            ]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        return {
            'total_events': len(self._queue),
            'by_event_type': dict(self._failure_counts),
            'by_error_type': dict(self._error_counts),
            'oldest_failure': (
                min(fe.first_failure for fe in self._queue)
                if self._queue else None
            ),
            'newest_failure': (
                max(fe.last_failure for fe in self._queue)
                if self._queue else None
            )
        }
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired events based on TTL.
        
        Returns:
            Number of events removed
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
        expired = [
            fe for fe in self._queue
            if fe.last_failure < cutoff_time
        ]
        
        removed = 0
        for failed_event in expired:
            if await self.remove_event(failed_event.event):
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired events from DLQ")
            record_metric('dlq.expired_removed', removed)
        
        return removed
    
    async def schedule_retry(
        self,
        failed_event: FailedEvent,
        publish_func: Callable[[Event], asyncio.Future]
    ) -> None:
        """
        Schedule automatic retry for a failed event.
        
        Args:
            failed_event: The failed event
            publish_func: Function to republish
        """
        event_id = self._get_event_id(failed_event.event)
        
        # Cancel existing retry if any
        if event_id in self._retry_tasks:
            self._retry_tasks[event_id].cancel()
        
        # Calculate delay
        delay = self.retry_policy.get_delay(failed_event.failure_count - 1)
        
        # Create retry task
        async def retry_task():
            await asyncio.sleep(delay)
            
            try:
                await publish_func(failed_event.event)
                await self.remove_event(failed_event.event)
            except Exception as e:
                logger.error(f"Scheduled retry failed: {e}")
                await self.add_event(
                    failed_event.event,
                    str(e),
                    type(e).__name__
                )
        
        task = asyncio.create_task(retry_task())
        self._retry_tasks[event_id] = task
    
    def _get_event_id(self, event: Event) -> str:
        """Generate unique ID for event."""
        # Use event data to create ID
        key_parts = [
            event.event_type.value,
            str(event.timestamp),
            str(event.data.get('id', ''))
        ]
        return ':'.join(key_parts)
    
    async def _evict_oldest(self) -> None:
        """Evict oldest event when queue is full."""
        if self._queue:
            oldest = min(self._queue, key=lambda fe: fe.first_failure)
            await self.remove_event(oldest.event)
            
            logger.warning(
                f"Evicted oldest event from DLQ: {oldest.event.event_type.value}"
            )
    
    async def _persist_failed_event(self, failed_event: FailedEvent) -> None:
        """Persist failed event to database."""
        if not self.db_pool:
            return
        
        # Prepare data for batch_upsert
        event_data = {
            'event_id': self._get_event_id(failed_event.event),
            'event_type': failed_event.event.event_type.value,
            'event_data': secure_dumps(failed_event.event.data),
            'error_message': failed_event.error_message,
            'error_type': failed_event.error_type,
            'failure_count': failed_event.failure_count,
            'first_failure': failed_event.first_failure,
            'last_failure': failed_event.last_failure,
            'metadata': secure_dumps(failed_event.metadata)
        }
        
        # Use batch_upsert with transaction context for reliability
        async with transaction_context(self.db_pool) as conn:
            await batch_upsert(
                conn,
                'event_dlq',
                [event_data],
                conflict_columns=['event_id'],
                update_columns=['error_message', 'failure_count', 'last_failure']
            )
    
    async def _remove_persisted_event(self, event_id: str) -> None:
        """Remove event from database."""
        if not self.db_pool:
            return
        
        # Use execute_with_retry for resilient database operations
        async def delete_operation(conn):
            return await conn.execute(
                "DELETE FROM event_dlq WHERE event_id = $1",
                event_id
            )
        
        await execute_with_retry(
            self.db_pool,
            delete_operation,
            max_retries=3
        )
    
    async def load_from_persistence(self) -> int:
        """
        Load failed events from database.
        
        Returns:
            Number of events loaded
        """
        if not self.db_pool:
            return 0
        
        async with transaction_context(self.db_pool) as conn:
            query = """
                SELECT * FROM event_dlq
                WHERE last_failure > $1
                ORDER BY last_failure DESC
                LIMIT $2
            """
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
            rows = await conn.fetch(query, cutoff_time, self.max_queue_size)
            
            loaded = 0
            for row in rows:
                # Reconstruct event
                event_data = secure_loads(row['event_data'])
                event = Event(
                    event_type=EventType(row['event_type']),
                    data=event_data,
                    timestamp=event_data.get('timestamp', row['first_failure'])
                )
                
                # Create failed event
                failed_event = FailedEvent(
                    event=event,
                    error_message=row['error_message'],
                    error_type=row['error_type'],
                    failure_count=row['failure_count'],
                    first_failure=row['first_failure'],
                    last_failure=row['last_failure'],
                    metadata=secure_loads(row['metadata'])
                )
                
                self._queue.append(failed_event)
                self._event_index[row['event_id']] = failed_event
                loaded += 1
            
            logger.info(f"Loaded {loaded} failed events from persistence")
            return loaded