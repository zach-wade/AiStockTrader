"""
Event Manager

Core event processing and callback management functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

from .types import CallbackInfo, Event, EventResult, EventStatus, CallbackPriority

logger = logging.getLogger(__name__)


class CallbackManager:
    """
    Manages callbacks and event handling.
    
    Provides registration, execution, and management of callbacks
    with support for priorities, filtering, and async execution.
    """
    
    def __init__(self, max_concurrent_callbacks: int = 10):
        """
        Initialize callback manager.
        
        Args:
            max_concurrent_callbacks: Maximum concurrent callback executions
        """
        self.max_concurrent_callbacks = max_concurrent_callbacks
        self.callbacks: Dict[str, List[CallbackInfo]] = defaultdict(list)
        self.global_callbacks: List[CallbackInfo] = []
        self.event_filters: Dict[str, Callable[[Event], bool]] = {}
        
        # Execution control
        self._semaphore = asyncio.Semaphore(max_concurrent_callbacks)
        self._execution_stats = defaultdict(int)
        self._event_history: List[EventResult] = []
        
        # Middleware
        self._before_callbacks: List[Callable[[Event], None]] = []
        self._after_callbacks: List[Callable[[EventResult], None]] = []
        
        logger.info("Callback manager initialized")
    
    def register_callback(self, 
                         event_type: str,
                         callback: Callable,
                         priority: CallbackPriority = CallbackPriority.NORMAL,
                         event_filter: Optional[Callable[[Any], bool]] = None,
                         weak_ref: bool = False,
                         max_retries: int = 0,
                         retry_delay: float = 1.0,
                         **metadata) -> str:
        """
        Register a callback for specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function
            priority: Callback priority
            event_filter: Optional filter function
            weak_ref: Use weak reference for callback
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            **metadata: Additional metadata
            
        Returns:
            Registration ID
        """
        callback_info = CallbackInfo(
            callback=callback,
            priority=priority,
            event_filter=event_filter,
            weak_ref=weak_ref,
            max_retries=max_retries,
            retry_delay=retry_delay,
            metadata=metadata
        )
        
        self.callbacks[event_type].append(callback_info)
        
        # Sort by priority
        self.callbacks[event_type].sort(key=lambda x: x.priority.value)
        
        registration_id = f"{event_type}_{len(self.callbacks[event_type])}"
        logger.debug(f"Registered callback for '{event_type}' with priority {priority.name}")
        
        return registration_id
    
    def register_global_callback(self, 
                                callback: Callable,
                                priority: CallbackPriority = CallbackPriority.NORMAL,
                                event_filter: Optional[Callable[[Event], bool]] = None,
                                **metadata) -> str:
        """
        Register a global callback that receives all events.
        
        Args:
            callback: Callback function
            priority: Callback priority
            event_filter: Optional filter function
            **metadata: Additional metadata
            
        Returns:
            Registration ID
        """
        callback_info = CallbackInfo(
            callback=callback,
            priority=priority,
            event_filter=event_filter,
            metadata=metadata
        )
        
        self.global_callbacks.append(callback_info)
        self.global_callbacks.sort(key=lambda x: x.priority.value)
        
        registration_id = f"global_{len(self.global_callbacks)}"
        logger.debug(f"Registered global callback with priority {priority.name}")
        
        return registration_id
    
    def unregister_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Unregister a callback.
        
        Args:
            event_type: Event type
            callback: Callback function to remove
            
        Returns:
            True if callback was removed
        """
        if event_type not in self.callbacks:
            return False
        
        original_count = len(self.callbacks[event_type])
        self.callbacks[event_type] = [
            cb for cb in self.callbacks[event_type] 
            if cb.get_callback() != callback
        ]
        
        removed = len(self.callbacks[event_type]) < original_count
        if removed:
            logger.debug(f"Unregistered callback for '{event_type}'")
        
        return removed
    
    def clear_callbacks(self, event_type: Optional[str] = None):
        """
        Clear callbacks for event type or all callbacks.
        
        Args:
            event_type: Event type to clear (all if None)
        """
        if event_type:
            if event_type in self.callbacks:
                count = len(self.callbacks[event_type])
                self.callbacks[event_type].clear()
                logger.info(f"Cleared {count} callbacks for '{event_type}'")
        else:
            total_count = sum(len(cbs) for cbs in self.callbacks.values())
            total_count += len(self.global_callbacks)
            
            self.callbacks.clear()
            self.global_callbacks.clear()
            
            logger.info(f"Cleared all {total_count} callbacks")
    
    def add_event_filter(self, event_type: str, filter_func: Callable[[Event], bool]):
        """Add event filter for specific event type."""
        self.event_filters[event_type] = filter_func
        logger.debug(f"Added event filter for '{event_type}'")
    
    def remove_event_filter(self, event_type: str):
        """Remove event filter for specific event type."""
        if event_type in self.event_filters:
            del self.event_filters[event_type]
            logger.debug(f"Removed event filter for '{event_type}'")
    
    def add_before_callback(self, callback: Callable[[Event], None]):
        """Add middleware callback executed before event processing."""
        self._before_callbacks.append(callback)
        logger.debug("Added before callback middleware")
    
    def add_after_callback(self, callback: Callable[[EventResult], None]):
        """Add middleware callback executed after event processing."""
        self._after_callbacks.append(callback)
        logger.debug("Added after callback middleware")
    
    async def emit_event(self, 
                        event_type: str,
                        data: Any,
                        source: Optional[str] = None,
                        event_id: Optional[str] = None,
                        **metadata) -> EventResult:
        """
        Emit an event and execute callbacks.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            event_id: Optional event ID
            **metadata: Additional metadata
            
        Returns:
            EventResult with execution details
        """
        # Create event
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            event_id=event_id,
            metadata=metadata
        )
        
        # Check event filter
        if event_type in self.event_filters:
            if not self.event_filters[event_type](event):
                logger.debug(f"Event '{event_type}' filtered out")
                return EventResult(
                    event=event,
                    status=EventStatus.CANCELLED
                )
        
        return await self._process_event(event)
    
    async def _process_event(self, event: Event) -> EventResult:
        """Process event through all applicable callbacks."""
        start_time = asyncio.get_event_loop().time()
        
        result = EventResult(
            event=event,
            status=EventStatus.PROCESSING
        )
        
        try:
            # Execute before middleware
            for before_callback in self._before_callbacks:
                try:
                    if asyncio.iscoroutinefunction(before_callback):
                        await before_callback(event)
                    else:
                        before_callback(event)
                except Exception as e:
                    logger.error(f"Error in before callback: {e}")
            
            # Get applicable callbacks
            callbacks = self._get_applicable_callbacks(event)
            
            # Execute callbacks
            await self._execute_callbacks(callbacks, event, result)
            
            # Set final status
            if result.callbacks_failed > 0:
                result.status = EventStatus.FAILED
            else:
                result.status = EventStatus.COMPLETED
            
        except Exception as e:
            result.status = EventStatus.FAILED
            result.errors.append(f"Event processing error: {e}")
            logger.error(f"Error processing event: {e}")
        
        finally:
            result.execution_time = asyncio.get_event_loop().time() - start_time
            
            # Execute after middleware
            for after_callback in self._after_callbacks:
                try:
                    if asyncio.iscoroutinefunction(after_callback):
                        await after_callback(result)
                    else:
                        after_callback(result)
                except Exception as e:
                    logger.error(f"Error in after callback: {e}")
            
            # Store result
            self._event_history.append(result)
            if len(self._event_history) > 1000:  # Keep last 1000 events
                self._event_history.pop(0)
            
            # Update stats
            self._execution_stats[event.event_type] += 1
            
            logger.debug(f"Event '{event.event_type}' processed in {result.execution_time:.3f}s")
        
        return result
    
    def _get_applicable_callbacks(self, event: Event) -> List[CallbackInfo]:
        """Get callbacks applicable to the event."""
        callbacks = []
        
        # Event-specific callbacks
        if event.event_type in self.callbacks:
            callbacks.extend(self.callbacks[event.event_type])
        
        # Global callbacks
        callbacks.extend(self.global_callbacks)
        
        # Filter valid callbacks
        valid_callbacks = []
        for cb in callbacks:
            if not cb.enabled:
                continue
            
            if not cb.is_valid():
                logger.debug("Removing invalid callback (weak reference expired)")
                continue
            
            # Apply event filter if present
            if cb.event_filter and not cb.event_filter(event.data):
                continue
            
            valid_callbacks.append(cb)
        
        # Sort by priority
        valid_callbacks.sort(key=lambda x: x.priority.value)
        
        return valid_callbacks
    
    async def _execute_callbacks(self, 
                                callbacks: List[CallbackInfo], 
                                event: Event,
                                result: EventResult):
        """Execute callbacks with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_callbacks)
        
        async def execute_callback(callback_info: CallbackInfo):
            async with semaphore:
                return await self._execute_single_callback(callback_info, event)
        
        # Execute callbacks concurrently
        tasks = [execute_callback(cb) for cb in callbacks]
        
        if tasks:
            callback_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for callback_result in callback_results:
                if isinstance(callback_result, Exception):
                    result.callbacks_failed += 1
                    result.errors.append(str(callback_result))
                else:
                    result.callbacks_executed += 1
                    if callback_result is not None:
                        result.results.append(callback_result)
    
    async def _execute_single_callback(self, 
                                      callback_info: CallbackInfo, 
                                      event: Event) -> Any:
        """Execute a single callback with retry logic."""
        callback = callback_info.get_callback()
        
        if callback is None:
            raise ValueError("Callback is no longer valid")
        
        last_error = None
        
        for attempt in range(callback_info.max_retries + 1):
            try:
                if callback_info.is_async:
                    return await callback(event)
                else:
                    return callback(event)
            
            except Exception as e:
                last_error = e
                
                if attempt < callback_info.max_retries:
                    logger.warning(f"Callback failed (attempt {attempt + 1}): {e}")
                    if callback_info.retry_delay > 0:
                        await asyncio.sleep(callback_info.retry_delay)
                else:
                    logger.error(f"Callback failed after {callback_info.max_retries + 1} attempts: {e}")
                    raise
        
        if last_error:
            raise last_error
    
    def get_callback_count(self, event_type: Optional[str] = None) -> int:
        """Get number of registered callbacks."""
        if event_type:
            return len(self.callbacks.get(event_type, []))
        else:
            return sum(len(cbs) for cbs in self.callbacks.values()) + len(self.global_callbacks)
    
    def get_event_types(self) -> List[str]:
        """Get list of event types with registered callbacks."""
        return list(self.callbacks.keys())
    
    def get_execution_stats(self) -> Dict[str, int]:
        """Get event execution statistics."""
        return dict(self._execution_stats)
    
    def get_event_history(self, limit: int = 100) -> List[EventResult]:
        """Get recent event history."""
        return self._event_history[-limit:]
    
    def cleanup_expired_callbacks(self):
        """Remove callbacks with expired weak references."""
        total_removed = 0
        
        for event_type, callbacks in self.callbacks.items():
            original_count = len(callbacks)
            self.callbacks[event_type] = [cb for cb in callbacks if cb.is_valid()]
            total_removed += original_count - len(self.callbacks[event_type])
        
        # Clean global callbacks
        original_count = len(self.global_callbacks)
        self.global_callbacks = [cb for cb in self.global_callbacks if cb.is_valid()]
        total_removed += original_count - len(self.global_callbacks)
        
        if total_removed > 0:
            logger.info(f"Cleaned up {total_removed} expired callbacks")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get callback manager health status."""
        return {
            'total_callbacks': self.get_callback_count(),
            'event_types': len(self.get_event_types()),
            'global_callbacks': len(self.global_callbacks),
            'total_events_processed': sum(self._execution_stats.values()),
            'recent_events': len(self._event_history),
            'max_concurrent_callbacks': self.max_concurrent_callbacks
        }