"""
Event System Types

Data classes and enums for the event system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime


class CallbackPriority(Enum):
    """Callback execution priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CallbackInfo:
    """Information about a registered callback."""
    callback: Callable
    priority: CallbackPriority
    event_filter: Optional[Callable[[Any], bool]] = None
    is_async: bool = False
    weak_ref: bool = False
    max_retries: int = 0
    retry_delay: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize callback info."""
        import asyncio
        import weakref
        
        self.is_async = asyncio.iscoroutinefunction(self.callback)
        if self.weak_ref and hasattr(self.callback, '__self__'):
            # Create weak reference for bound methods
            self._weak_callback = weakref.WeakMethod(self.callback)
        else:
            self._weak_callback = None
    
    def get_callback(self) -> Optional[Callable]:
        """Get the callback function, handling weak references."""
        if self._weak_callback is not None:
            return self._weak_callback()
        return self.callback
    
    def is_valid(self) -> bool:
        """Check if callback is still valid."""
        return self.get_callback() is not None


@dataclass
class Event:
    """Event data structure."""
    event_type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'event_id': self.event_id,
            'metadata': self.metadata
        }


@dataclass
class EventResult:
    """Result of event processing."""
    event: Event
    status: EventStatus
    callbacks_executed: int = 0
    callbacks_failed: int = 0
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'event': self.event.to_dict(),
            'status': self.status.value,
            'callbacks_executed': self.callbacks_executed,
            'callbacks_failed': self.callbacks_failed,
            'execution_time': self.execution_time,
            'errors': self.errors,
            'results': self.results
        }