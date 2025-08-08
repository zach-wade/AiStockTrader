"""
Circuit breaker event definitions.

This module defines events emitted by the circuit breaker system
for monitoring, logging, and integration with other components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from .types import BreakerType, BreakerStatus, BreakerPriority


class EventType(Enum):
    """Circuit breaker event types."""
    BREAKER_TRIPPED = "breaker_tripped"
    BREAKER_RESET = "breaker_reset"
    BREAKER_WARNING = "breaker_warning"
    BREAKER_TEST = "breaker_test"
    COOLDOWN_START = "cooldown_start"
    COOLDOWN_END = "cooldown_end"
    THRESHOLD_UPDATED = "threshold_updated"
    BREAKER_DISABLED = "breaker_disabled"
    BREAKER_ENABLED = "breaker_enabled"


@dataclass
class CircuitBreakerEvent:
    """Base class for circuit breaker events."""
    event_id: str
    event_type: EventType
    breaker_name: str
    breaker_type: BreakerType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: BreakerPriority = BreakerPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'breaker_name': self.breaker_name,
            'breaker_type': self.breaker_type.value,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'metadata': self.metadata
        }


@dataclass
class BreakerTrippedEvent(CircuitBreakerEvent):
    """Event emitted when a circuit breaker trips."""
    trip_reason: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    affected_symbols: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    cooldown_seconds: int = 300
    auto_reset: bool = True
    
    def __post_init__(self):
        """Set event type."""
        self.event_type = EventType.BREAKER_TRIPPED
        
        # Add trip details to metadata
        self.metadata.update({
            'trip_reason': self.trip_reason,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'breach_percentage': ((self.current_value - self.threshold_value) / 
                                self.threshold_value * 100),
            'affected_symbols': self.affected_symbols,
            'actions_taken': self.actions_taken,
            'cooldown_seconds': self.cooldown_seconds,
            'auto_reset': self.auto_reset
        })


@dataclass
class BreakerResetEvent(CircuitBreakerEvent):
    """Event emitted when a circuit breaker resets."""
    reset_reason: str = ""
    was_manual: bool = False
    trip_duration_seconds: Optional[float] = None
    conditions_cleared: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set event type."""
        self.event_type = EventType.BREAKER_RESET
        
        # Add reset details to metadata
        self.metadata.update({
            'reset_reason': self.reset_reason,
            'was_manual': self.was_manual,
            'trip_duration_seconds': self.trip_duration_seconds,
            'conditions_cleared': self.conditions_cleared
        })


@dataclass
class BreakerWarningEvent(CircuitBreakerEvent):
    """Event emitted when approaching circuit breaker threshold."""
    warning_level: float = 0.0  # Percentage of threshold reached (e.g., 0.8 = 80%)
    current_value: float = 0.0
    threshold_value: float = 0.0
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    estimated_time_to_breach: Optional[float] = None  # Seconds
    
    def __post_init__(self):
        """Set event type."""
        self.event_type = EventType.BREAKER_WARNING
        self.priority = BreakerPriority.HIGH
        
        # Add warning details to metadata
        self.metadata.update({
            'warning_level': self.warning_level,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'trend': self.trend,
            'estimated_time_to_breach': self.estimated_time_to_breach,
            'percentage_of_threshold': self.warning_level * 100
        })


@dataclass
class CooldownEvent(CircuitBreakerEvent):
    """Event for cooldown period changes."""
    cooldown_type: str = "start"  # "start" or "end"
    cooldown_duration: int = 0  # Seconds
    remaining_time: Optional[int] = None
    
    def __post_init__(self):
        """Set event type based on cooldown type."""
        if self.cooldown_type == "start":
            self.event_type = EventType.COOLDOWN_START
        else:
            self.event_type = EventType.COOLDOWN_END
        
        self.metadata.update({
            'cooldown_duration': self.cooldown_duration,
            'remaining_time': self.remaining_time
        })


@dataclass
class ThresholdUpdateEvent(CircuitBreakerEvent):
    """Event emitted when breaker thresholds are updated."""
    old_threshold: float = 0.0
    new_threshold: float = 0.0
    update_reason: str = ""
    updated_by: Optional[str] = None  # User or system component
    
    def __post_init__(self):
        """Set event type."""
        self.event_type = EventType.THRESHOLD_UPDATED
        
        self.metadata.update({
            'old_threshold': self.old_threshold,
            'new_threshold': self.new_threshold,
            'threshold_change_pct': ((self.new_threshold - self.old_threshold) / 
                                   self.old_threshold * 100),
            'update_reason': self.update_reason,
            'updated_by': self.updated_by
        })


@dataclass
class BreakerStatusChangeEvent(CircuitBreakerEvent):
    """Event for breaker enable/disable status changes."""
    enabled: bool = True
    change_reason: str = ""
    changed_by: Optional[str] = None
    
    def __post_init__(self):
        """Set event type based on status."""
        self.event_type = (EventType.BREAKER_ENABLED if self.enabled 
                          else EventType.BREAKER_DISABLED)
        
        self.metadata.update({
            'enabled': self.enabled,
            'change_reason': self.change_reason,
            'changed_by': self.changed_by
        })


class CircuitBreakerEventBuilder:
    """Builder for creating circuit breaker events."""
    
    @staticmethod
    def build_trip_event(breaker_name: str,
                        breaker_type: BreakerType,
                        trip_reason: str,
                        current_value: float,
                        threshold_value: float,
                        **kwargs) -> BreakerTrippedEvent:
        """Build a breaker tripped event."""
        event_id = f"TRIP_{breaker_name}_{datetime.utcnow().timestamp():.0f}"
        
        return BreakerTrippedEvent(
            event_id=event_id,
            breaker_name=breaker_name,
            breaker_type=breaker_type,
            trip_reason=trip_reason,
            current_value=current_value,
            threshold_value=threshold_value,
            **kwargs
        )
    
    @staticmethod
    def build_reset_event(breaker_name: str,
                         breaker_type: BreakerType,
                         reset_reason: str,
                         **kwargs) -> BreakerResetEvent:
        """Build a breaker reset event."""
        event_id = f"RESET_{breaker_name}_{datetime.utcnow().timestamp():.0f}"
        
        return BreakerResetEvent(
            event_id=event_id,
            breaker_name=breaker_name,
            breaker_type=breaker_type,
            reset_reason=reset_reason,
            **kwargs
        )
    
    @staticmethod
    def build_warning_event(breaker_name: str,
                           breaker_type: BreakerType,
                           warning_level: float,
                           current_value: float,
                           threshold_value: float,
                           **kwargs) -> BreakerWarningEvent:
        """Build a breaker warning event."""
        event_id = f"WARN_{breaker_name}_{datetime.utcnow().timestamp():.0f}"
        
        return BreakerWarningEvent(
            event_id=event_id,
            breaker_name=breaker_name,
            breaker_type=breaker_type,
            warning_level=warning_level,
            current_value=current_value,
            threshold_value=threshold_value,
            **kwargs
        )


# Event priority levels for different scenarios
EVENT_PRIORITIES = {
    EventType.BREAKER_TRIPPED: BreakerPriority.CRITICAL,
    EventType.BREAKER_RESET: BreakerPriority.HIGH,
    EventType.BREAKER_WARNING: BreakerPriority.HIGH,
    EventType.BREAKER_TEST: BreakerPriority.LOW,
    EventType.COOLDOWN_START: BreakerPriority.MEDIUM,
    EventType.COOLDOWN_END: BreakerPriority.MEDIUM,
    EventType.THRESHOLD_UPDATED: BreakerPriority.MEDIUM,
    EventType.BREAKER_DISABLED: BreakerPriority.HIGH,
    EventType.BREAKER_ENABLED: BreakerPriority.MEDIUM
}