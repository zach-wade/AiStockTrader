# File: risk_management/pre_trade/unified_limit_checker/events.py

"""
Event handling for the Unified Limit Checker.

Provides event-driven architecture for limit checking lifecycle events including
violations, resolutions, and status changes.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import uuid

from main.utils.core import get_logger, ensure_utc, AsyncCircuitBreaker
from main.interfaces.events import IEventBus, Event, EventType as MainEventType
from .types import (
    LimitType,
    ViolationSeverity,
    LimitAction,
    ComparisonOperator
)
from .models import LimitViolation, LimitCheckResult

logger = get_logger(__name__)


class LimitEventType(Enum):
    """Types of limit checker events."""
    VIOLATION_DETECTED = "violation_detected"
    VIOLATION_RESOLVED = "violation_resolved"
    CHECK_COMPLETED = "check_completed"
    CHECK_FAILED = "check_failed"
    THRESHOLD_UPDATED = "threshold_updated"
    LIMIT_ENABLED = "limit_enabled"
    LIMIT_DISABLED = "limit_disabled"
    ACTION_TAKEN = "action_taken"
    ESCALATION_TRIGGERED = "escalation_triggered"


@dataclass
class LimitEvent(Event):
    """Base class for limit checker events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    limit_id: str = ""
    limit_type: Optional[LimitType] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViolationEvent(LimitEvent):
    """Event raised when a limit violation is detected."""
    violation: Optional[LimitViolation] = None
    severity: Optional[ViolationSeverity] = None
    current_value: float = 0.0
    threshold_value: float = 0.0
    recommended_action: Optional[LimitAction] = None
    
    def __post_init__(self):
        self.event_type = MainEventType.ERROR_OCCURRED
        if self.violation:
            self.limit_id = self.violation.limit_id
        self.limit_type = self.context.get('limit_type')


@dataclass
class ResolutionEvent(LimitEvent):
    """Event raised when a violation is resolved."""
    violation_id: str = ""
    resolution_notes: Optional[str] = None
    duration_seconds: Optional[float] = None
    action_taken: Optional[LimitAction] = None
    
    def __post_init__(self):
        self.event_type = MainEventType.ERROR_OCCURRED  # Using existing event type


@dataclass
class CheckEvent(LimitEvent):
    """Event raised when a limit check is completed."""
    check_result: Optional[LimitCheckResult] = None
    check_duration_ms: float = 0.0
    passed: bool = False
    
    def __post_init__(self):
        self.event_type = MainEventType.ERROR_OCCURRED  # Using existing event type
        if self.check_result:
            self.limit_id = self.check_result.limit_id
            self.passed = self.check_result.passed


class EventStatsTracker:
    """Tracks statistics for event handling."""
    
    def __init__(self):
        self.events_by_type: Dict[str, int] = {}
        self.events_by_severity: Dict[ViolationSeverity, int] = {}
        self.total_events = 0
        self.failed_events = 0
        
    def record_event(self, event: LimitEvent):
        """Record an event for statistics."""
        self.total_events += 1
        
        # Track by event type
        event_type_str = getattr(event, 'event_type', 'unknown')
        if isinstance(event_type_str, Enum):
            event_type_str = event_type_str.value
        self.events_by_type[event_type_str] = self.events_by_type.get(event_type_str, 0) + 1
        
        # Track violations by severity
        if isinstance(event, ViolationEvent):
            severity = event.severity
            self.events_by_severity[severity] = self.events_by_severity.get(severity, 0) + 1
    
    def record_failure(self):
        """Record a failed event."""
        self.failed_events += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'total_events': self.total_events,
            'failed_events': self.failed_events,
            'events_by_type': dict(self.events_by_type),
            'events_by_severity': {
                k.value: v for k, v in self.events_by_severity.items()
            }
        }


class EventBufferManager:
    """Manages event buffering for batch processing."""
    
    def __init__(self, buffer_size: int = 100, flush_interval: float = 5.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[LimitEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        
    async def add_event(self, event: LimitEvent) -> bool:
        """Add event to buffer."""
        async with self._lock:
            self.buffer.append(event)
            
            if len(self.buffer) >= self.buffer_size:
                return True  # Signal that buffer should be flushed
        
        return False
    
    async def flush(self) -> List[LimitEvent]:
        """Flush the buffer and return events."""
        async with self._lock:
            events = self.buffer.copy()
            self.buffer.clear()
            return events
    
    def start_periodic_flush(self, flush_callback: Callable):
        """Start periodic buffer flushing."""
        async def periodic_flush():
            while True:
                await asyncio.sleep(self.flush_interval)
                events = await self.flush()
                if events:
                    await flush_callback(events)
        
        self._flush_task = asyncio.create_task(periodic_flush())
    
    def stop_periodic_flush(self):
        """Stop periodic flushing."""
        if self._flush_task:
            self._flush_task.cancel()


class EventManager:
    """
    Manages event handling for the limit checker.
    
    Provides subscription-based event distribution with async support.
    """
    
    def __init__(self, 
                 event_bus: Optional[IEventBus] = None,
                 buffer_size: int = 100,
                 enable_stats: bool = True):
        """
        Initialize event manager.
        
        Args:
            event_bus: Optional main event bus for integration
            buffer_size: Size of event buffer
            enable_stats: Whether to track statistics
        """
        self.event_bus = event_bus
        self.buffer_size = buffer_size
        self.enable_stats = enable_stats
        
        # Subscribers by event type
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # Global subscribers (receive all events)
        self._global_subscribers: List[Callable] = []
        
        # Helper components
        self.stats_tracker = EventStatsTracker() if enable_stats else None
        self.buffer_manager = EventBufferManager(buffer_size)
        
        # Circuit breaker for resilience
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=10,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        logger.info("EventManager initialized")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type}")
    
    def subscribe_all(self, handler: Callable):
        """Subscribe to all events."""
        self._global_subscribers.append(handler)
        logger.debug("Added global subscriber")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from a specific event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type}")
    
    def unsubscribe_all(self, handler: Callable):
        """Unsubscribe from all events."""
        try:
            self._global_subscribers.remove(handler)
            logger.debug("Removed global subscriber")
        except ValueError:
            logger.warning("Global handler not found")
    
    async def emit(self, event: LimitEvent):
        """
        Emit an event to all subscribers.
        
        Args:
            event: The event to emit
        """
        if not self._running:
            logger.warning("EventManager not running, event discarded")
            return
        
        # Track statistics
        if self.stats_tracker:
            self.stats_tracker.record_event(event)
        
        # Add to buffer
        should_flush = await self.buffer_manager.add_event(event)
        
        if should_flush:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush the event buffer and process events."""
        events = await self.buffer_manager.flush()
        
        for event in events:
            # Create tasks for async processing
            task = asyncio.create_task(self._process_event(event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
    
    async def _process_event(self, event: LimitEvent):
        """Process a single event."""
        try:
            # Send to main event bus if available
            if self.event_bus:
                await self.circuit_breaker.call(
                    self.event_bus.emit_event,
                    event
                )
            
            # Get event type string
            event_type_str = getattr(event, 'event_type', 'unknown')
            if isinstance(event_type_str, Enum):
                event_type_str = event_type_str.value
            
            # Notify specific subscribers
            if event_type_str in self._subscribers:
                for handler in self._subscribers[event_type_str]:
                    await self._call_handler(handler, event)
            
            # Notify global subscribers
            for handler in self._global_subscribers:
                await self._call_handler(handler, event)
                
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            if self.stats_tracker:
                self.stats_tracker.record_failure()
    
    async def _call_handler(self, handler: Callable, event: LimitEvent):
        """Call an event handler safely."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
    
    async def start(self):
        """Start the event manager."""
        self._running = True
        
        # Start periodic buffer flushing
        self.buffer_manager.start_periodic_flush(
            lambda events: asyncio.create_task(self._process_events_batch(events))
        )
        
        logger.info("EventManager started")
    
    async def stop(self):
        """Stop the event manager."""
        self._running = False
        
        # Stop periodic flushing
        self.buffer_manager.stop_periodic_flush()
        
        # Flush remaining events
        await self._flush_buffer()
        
        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("EventManager stopped")
    
    async def _process_events_batch(self, events: List[LimitEvent]):
        """Process a batch of events."""
        for event in events:
            await self._process_event(event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        if not self.stats_tracker:
            return {}
        
        stats = self.stats_tracker.get_stats()
        stats['subscriber_count'] = sum(len(handlers) for handlers in self._subscribers.values())
        stats['global_subscriber_count'] = len(self._global_subscribers)
        stats['buffer_size'] = len(self.buffer_manager.buffer)
        
        return stats
    
    def clear_subscribers(self):
        """Clear all subscribers."""
        self._subscribers.clear()
        self._global_subscribers.clear()
        logger.info("All subscribers cleared")


def create_event_manager_with_defaults(event_bus: Optional[IEventBus] = None) -> EventManager:
    """
    Create an event manager with default configuration.
    
    Args:
        event_bus: Optional main event bus for integration
        
    Returns:
        Configured EventManager instance
    """
    manager = EventManager(
        event_bus=event_bus,
        buffer_size=1000,
        enable_stats=True
    )
    
    # Add default logging handler
    async def log_violations(event: LimitEvent):
        if isinstance(event, ViolationEvent):
            logger.warning(
                f"Limit violation detected: {event.violation.limit_name} "
                f"(severity: {event.severity.value}, "
                f"current: {event.current_value}, threshold: {event.threshold_value})"
            )
        elif isinstance(event, ResolutionEvent):
            logger.info(
                f"Violation resolved: {event.violation_id} "
                f"(duration: {event.duration_seconds}s)"
            )
    
    manager.subscribe_all(log_violations)
    
    return manager


# Export convenience function for backward compatibility
create_limit_event_manager = create_event_manager_with_defaults