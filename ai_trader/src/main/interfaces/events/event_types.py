"""
Core event type definitions.

This module contains the base event types and enums that are used
throughout the system. By defining these in the interfaces layer,
we avoid circular dependencies while maintaining type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List
import uuid

from .time_utils import ensure_utc


class EventType(Enum):
    """
    Core event types used across the AI Trader system.
    
    These represent the fundamental event categories that
    components can publish and subscribe to.
    """
    # Market events
    MARKET_DATA = "market_data"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    
    # Trading events  
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Scanner events
    SCANNER_ALERT = "scanner_alert"
    
    # Feature pipeline events
    FEATURE_REQUEST = "feature_request"
    FEATURE_COMPUTED = "feature_computed"
    
    # Risk events
    RISK_ALERT = "risk_alert"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    
    # System events
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SYSTEM_STATUS = "system_status"
    
    # Data pipeline events
    DATA_INGESTED = "data_ingested"
    DATA_PROCESSED = "data_processed"
    DATA_VALIDATED = "data_validated"
    DATA_WRITTEN = "data_written"
    
    # Symbol qualification and backfill events
    SYMBOL_QUALIFIED = "symbol_qualified"
    SYMBOL_PROMOTED = "symbol_promoted"
    DATA_GAP_DETECTED = "data_gap_detected"
    BACKFILL_SCHEDULED = "backfill_scheduled"
    BACKFILL_COMPLETED = "backfill_completed"


class EventPriority(Enum):
    """Event priority levels for processing order."""
    LOW = 1
    NORMAL = 5
    HIGH = 7
    CRITICAL = 10


@dataclass
class Event:
    """
    Base event class that all events inherit from.
    
    This provides common fields that all events share, ensuring
    consistency across the event system.
    """
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    def __lt__(self, other):
        """Compare events by timestamp for heap queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp < other.timestamp
    
    def __le__(self, other):
        """Compare events by timestamp for heap queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp <= other.timestamp
    
    def __gt__(self, other):
        """Compare events by timestamp for heap queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp > other.timestamp
    
    def __ge__(self, other):
        """Compare events by timestamp for heap queue ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp >= other.timestamp
    
    def __eq__(self, other):
        """Compare events by timestamp and event_id for equality."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp == other.timestamp and self.event_id == other.event_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization.
        
        Returns:
            Dictionary representation of the event
        """
        from main.utils.core import to_json
        import json
        
        # Use dataclass asdict but handle special types
        result = {}
        for field_name, field_info in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            if isinstance(value, datetime):
                result[field_name] = value.isoformat()
            elif isinstance(value, Enum):
                result[field_name] = value.value
            elif hasattr(value, 'to_dict'):
                result[field_name] = value.to_dict()
            else:
                # For complex types, use our JSON encoder
                try:
                    json.dumps(value)  # Test if directly serializable
                    result[field_name] = value
                except (TypeError, ValueError):
                    # Use our custom encoder for complex types
                    result[field_name] = json.loads(to_json(value))
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """
        Create event from dictionary.
        
        Args:
            data: Dictionary containing event data
            
        Returns:
            Event instance
        """
        from main.utils.core import parse_iso_datetime
        
        # Prepare data for dataclass construction
        kwargs = {}
        fields = {f.name: f for f in cls.__dataclass_fields__.values()}
        
        for field_name, field_value in data.items():
            if field_name not in fields:
                continue
                
            field = fields[field_name]
            
            # Handle datetime fields
            if field.type == datetime and isinstance(field_value, str):
                kwargs[field_name] = parse_iso_datetime(field_value)
            
            # Handle EventType enum
            elif field.type == EventType and isinstance(field_value, str):
                kwargs[field_name] = EventType(field_value)
            
            # Handle EventPriority enum
            elif field.type == EventPriority and isinstance(field_value, (str, int)):
                if isinstance(field_value, str):
                    # Find enum by name
                    kwargs[field_name] = EventPriority[field_value]
                else:
                    # Find enum by value
                    kwargs[field_name] = EventPriority(field_value)
            
            else:
                kwargs[field_name] = field_value
        
        return cls(**kwargs)
    
    def to_json(self) -> str:
        """
        Convert event to JSON string.
        
        Returns:
            JSON string representation
        """
        from main.utils.core import to_json
        return to_json(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """
        Create event from JSON string.
        
        Args:
            json_str: JSON string containing event data
            
        Returns:
            Event instance
        """
        from main.utils.core import from_json
        data = from_json(json_str)
        return cls.from_dict(data)


# Specific event types commonly used across modules

@dataclass
class MarketEvent(Event):
    """Event for market data updates."""
    symbol: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = field(default=EventType.MARKET_DATA)


@dataclass 
class OrderEvent(Event):
    """Event for order-related actions."""
    order_id: str = ""
    symbol: str = ""
    quantity: float = 0.0
    side: str = ""  # 'buy' or 'sell'
    order_type: str = ""  # 'market', 'limit', etc.
    price: Optional[float] = None
    status: Optional[str] = None
    event_type: EventType = field(default=EventType.ORDER_PLACED)


@dataclass
class ScannerAlertEvent(Event):
    """Event for scanner alerts."""
    symbol: str = ""
    alert_type: str = ""
    score: float = 0.0
    scanner_name: str = ""
    event_type: EventType = field(default=EventType.SCANNER_ALERT)


@dataclass
class FeatureRequestEvent(Event):
    """Event for requesting feature computation."""
    symbols: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    requester: str = ""
    priority: EventPriority = EventPriority.NORMAL
    event_type: EventType = field(default=EventType.FEATURE_REQUEST)


@dataclass
class FeatureComputedEvent(Event):
    """Event for completed feature computation."""
    symbol: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    event_type: EventType = field(default=EventType.FEATURE_COMPUTED)


@dataclass
class ErrorEvent(Event):
    """Event for system errors."""
    error_type: str = ""
    message: str = ""
    component: str = ""
    stack_trace: Optional[str] = None
    recoverable: bool = True
    event_type: EventType = field(default=EventType.ERROR)


@dataclass
class SystemStatusEvent(Event):
    """Event for system status updates."""
    component: str = ""
    status: str = ""  # 'healthy', 'degraded', 'down'
    metrics: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = field(default=EventType.SYSTEM_STATUS)


@dataclass
class DataWrittenEvent(Event):
    """Event for data written to storage."""
    data: Dict[str, Any] = field(default_factory=dict)
    storage_tier: str = "hot"  # 'hot' or 'cold'
    table_name: str = ""
    record_ids: List[str] = field(default_factory=list)
    write_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    row_count: int = 0
    event_type: EventType = field(default=EventType.DATA_WRITTEN)


# Symbol qualification and backfill events

@dataclass
class SymbolQualifiedEvent(Event):
    """Event triggered when a symbol is qualified for a layer."""
    symbol: str = ""
    layer: int = 0  # DataLayer value (0-3)
    qualification_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = field(default=EventType.SYMBOL_QUALIFIED)


@dataclass
class SymbolPromotedEvent(Event):
    """Event triggered when a symbol is promoted to a higher layer."""
    symbol: str = ""
    from_layer: int = 0
    to_layer: int = 1
    promotion_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = field(default=EventType.SYMBOL_PROMOTED)


@dataclass
class DataGapDetectedEvent(Event):
    """Event triggered when a data gap is detected."""
    symbol: str = ""
    data_type: str = ""  # 'market_data', 'news', etc.
    gap_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    gap_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    gap_size_hours: float = 0.0
    priority: EventPriority = EventPriority.HIGH
    event_type: EventType = field(default=EventType.DATA_GAP_DETECTED)


@dataclass
class BackfillScheduledEvent(Event):
    """Event triggered when a backfill is scheduled."""
    backfill_id: str = ""
    symbol: str = ""
    layer: int = 0
    data_types: List[str] = field(default_factory=list)
    start_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType = field(default=EventType.BACKFILL_SCHEDULED)


@dataclass
class BackfillCompletedEvent(Event):
    """Event triggered when a backfill is completed."""
    backfill_id: str = ""
    symbol: str = ""
    success: bool = True
    records_fetched: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    event_type: EventType = field(default=EventType.BACKFILL_COMPLETED)