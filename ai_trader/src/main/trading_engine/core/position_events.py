"""
Position Events System

Event definitions for position-related changes in the trading system.
Integrates with the existing events infrastructure.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass

from main.models.common import Position, Order, OrderSide


class PositionEventType(Enum):
    """Types of position events."""
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_INCREASED = "position_increased"
    POSITION_DECREASED = "position_decreased"
    POSITION_REVERSED = "position_reversed"
    FILL_PROCESSED = "fill_processed"
    POSITION_UPDATED = "position_updated"


@dataclass
class PositionEvent(ABC):
    """Base class for all position events."""
    symbol: str
    event_type: PositionEventType
    timestamp: datetime
    old_position: Optional[Position]
    new_position: Optional[Position]
    trigger_order_id: Optional[str] = None
    realized_pnl: Optional[Decimal] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    @abstractmethod
    def get_event_data(self) -> Dict[str, Any]:
        """Get event data for serialization."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        base_data = {
            'symbol': self.symbol,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'trigger_order_id': self.trigger_order_id,
            'realized_pnl': float(self.realized_pnl) if self.realized_pnl else None,
            'metadata': self.metadata
        }
        
        # Add position data
        if self.old_position:
            base_data['old_position'] = {
                'quantity': self.old_position.quantity,
                'avg_entry_price': self.old_position.avg_entry_price,
                'market_value': self.old_position.market_value,
                'unrealized_pnl': self.old_position.unrealized_pnl
            }
        
        if self.new_position:
            base_data['new_position'] = {
                'quantity': self.new_position.quantity,
                'avg_entry_price': self.new_position.avg_entry_price,
                'market_value': self.new_position.market_value,
                'unrealized_pnl': self.new_position.unrealized_pnl
            }
        
        # Add event-specific data
        base_data.update(self.get_event_data())
        
        return base_data


@dataclass
class PositionOpenedEvent(PositionEvent):
    """Event fired when a new position is opened."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_OPENED
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'initial_quantity': self.new_position.quantity if self.new_position else 0,
            'entry_price': self.new_position.avg_entry_price if self.new_position else 0
        }


@dataclass
class PositionClosedEvent(PositionEvent):
    """Event fired when a position is completely closed."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_CLOSED
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'closed_quantity': abs(self.old_position.quantity) if self.old_position else 0,
            'exit_price': float(self.metadata.get('fill_price', 0)) if self.metadata else 0,
            'total_realized_pnl': float(self.realized_pnl) if self.realized_pnl else 0,
            'hold_duration_seconds': self._calculate_hold_duration()
        }
    
    def _calculate_hold_duration(self) -> float:
        """Calculate how long position was held."""
        if self.old_position and self.old_position.timestamp:
            return (self.timestamp - self.old_position.timestamp).total_seconds()
        return 0.0


@dataclass
class PositionIncreasedEvent(PositionEvent):
    """Event fired when position size is increased."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_INCREASED
    
    def get_event_data(self) -> Dict[str, Any]:
        quantity_added = 0
        if self.old_position and self.new_position:
            quantity_added = abs(self.new_position.quantity) - abs(self.old_position.quantity)
        
        return {
            'quantity_added': quantity_added,
            'fill_price': float(self.metadata.get('fill_price', 0)) if self.metadata else 0,
            'new_avg_entry_price': self.new_position.avg_entry_price if self.new_position else 0
        }


@dataclass
class PositionDecreasedEvent(PositionEvent):
    """Event fired when position size is decreased."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_DECREASED
    
    def get_event_data(self) -> Dict[str, Any]:
        quantity_reduced = 0
        if self.old_position and self.new_position:
            quantity_reduced = abs(self.old_position.quantity) - abs(self.new_position.quantity)
        
        return {
            'quantity_reduced': quantity_reduced,
            'fill_price': float(self.metadata.get('fill_price', 0)) if self.metadata else 0,
            'partial_realized_pnl': float(self.realized_pnl) if self.realized_pnl else 0
        }


@dataclass
class PositionReversedEvent(PositionEvent):
    """Event fired when position side is reversed (long to short or vice versa)."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_REVERSED
    
    def get_event_data(self) -> Dict[str, Any]:
        old_side = self.old_position.side if self.old_position else 'unknown'
        new_side = self.new_position.side if self.new_position else 'unknown'
        
        return {
            'old_side': old_side,
            'new_side': new_side,
            'fill_price': float(self.metadata.get('fill_price', 0)) if self.metadata else 0,
            'reversal_realized_pnl': float(self.realized_pnl) if self.realized_pnl else 0
        }


@dataclass
class FillProcessedEvent(PositionEvent):
    """Event fired when an order fill is processed."""
    fill_price: Decimal = Decimal('0')
    fill_quantity: Decimal = Decimal('0')
    commission: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.FILL_PROCESSED
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'fill_price': float(self.fill_price),
            'fill_quantity': float(self.fill_quantity),
            'commission': float(self.commission) if self.commission else 0,
            'fees': float(self.fees) if self.fees else 0,
            'total_cost': float(self.commission or 0) + float(self.fees or 0)
        }


@dataclass
class PositionUpdatedEvent(PositionEvent):
    """Event fired when position is updated (e.g., price changes)."""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = PositionEventType.POSITION_UPDATED
    
    def get_event_data(self) -> Dict[str, Any]:
        pnl_change = 0
        if self.old_position and self.new_position:
            pnl_change = self.new_position.unrealized_pnl - self.old_position.unrealized_pnl
        
        return {
            'price_change': float(self.metadata.get('price_change', 0)) if self.metadata else 0,
            'new_unrealized_pnl': self.new_position.unrealized_pnl if self.new_position else 0,
            'pnl_change': pnl_change
        }


# Event factory functions for easier creation
def create_position_opened_event(
    symbol: str,
    new_position: Position,
    trigger_order_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PositionOpenedEvent:
    """Create a position opened event."""
    return PositionOpenedEvent(
        symbol=symbol,
        event_type=PositionEventType.POSITION_OPENED,
        timestamp=datetime.now(timezone.utc),
        old_position=None,
        new_position=new_position,
        trigger_order_id=trigger_order_id,
        metadata=metadata or {}
    )


def create_position_closed_event(
    symbol: str,
    old_position: Position,
    realized_pnl: Decimal,
    trigger_order_id: Optional[str] = None,
    fill_price: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PositionClosedEvent:
    """Create a position closed event."""
    event_metadata = metadata or {}
    if fill_price:
        event_metadata['fill_price'] = fill_price
    
    return PositionClosedEvent(
        symbol=symbol,
        event_type=PositionEventType.POSITION_CLOSED,
        timestamp=datetime.now(timezone.utc),
        old_position=old_position,
        new_position=None,
        trigger_order_id=trigger_order_id,
        realized_pnl=realized_pnl,
        metadata=event_metadata
    )


def create_fill_processed_event(
    symbol: str,
    old_position: Optional[Position],
    new_position: Optional[Position],
    fill_price: Decimal,
    fill_quantity: Decimal,
    realized_pnl: Optional[Decimal] = None,
    commission: Optional[Decimal] = None,
    fees: Optional[Decimal] = None,
    trigger_order_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> FillProcessedEvent:
    """Create a fill processed event."""
    return FillProcessedEvent(
        symbol=symbol,
        event_type=PositionEventType.FILL_PROCESSED,
        timestamp=datetime.now(timezone.utc),
        old_position=old_position,
        new_position=new_position,
        fill_price=fill_price,
        fill_quantity=fill_quantity,
        trigger_order_id=trigger_order_id,
        realized_pnl=realized_pnl,
        commission=commission,
        fees=fees,
        metadata=metadata or {}
    )


class PriceUpdateEvent(PositionEvent):
    """Event when market price updates affect position values."""
    
    price_changes: Dict[str, float]  # Symbol -> new price
    
    def get_event_data(self) -> Dict[str, Any]:
        """Get event data for serialization."""
        return {
            "price_changes": self.price_changes,
            "affected_positions": len(self.price_changes)
        }


class BrokerSyncEvent(PositionEvent):
    """Event when positions are synchronized with broker."""
    
    broker_positions: Dict[str, Position]  # Broker-reported positions
    discrepancies: Dict[str, Dict[str, Any]]  # Any discrepancies found
    
    def get_event_data(self) -> Dict[str, Any]:
        """Get event data for serialization."""
        return {
            "broker_position_count": len(self.broker_positions),
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": self.discrepancies
        }


class RiskLimitBreachEvent(PositionEvent):
    """Event when a risk limit is breached."""
    
    limit_type: str  # Type of limit breached
    limit_value: float  # The limit value
    current_value: float  # Current value that breached the limit
    severity: str  # "warning", "critical", etc.
    
    def get_event_data(self) -> Dict[str, Any]:
        """Get event data for serialization."""
        return {
            "limit_type": self.limit_type,
            "limit_value": self.limit_value,
            "current_value": self.current_value,
            "severity": self.severity,
            "breach_percentage": ((self.current_value / self.limit_value) - 1) * 100
        }



def create_position_event(
    event_type: PositionEventType,
    symbol: str,
    old_position: Optional[Position] = None,
    new_position: Optional[Position] = None,
    trigger_order_id: Optional[str] = None,
    realized_pnl: Optional[Decimal] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PositionEvent:
    """Generic factory function to create position events."""
    
    # Map event types to their corresponding classes
    event_class_map = {
        PositionEventType.POSITION_OPENED: PositionOpenedEvent,
        PositionEventType.POSITION_CLOSED: PositionClosedEvent,
        PositionEventType.POSITION_INCREASED: PositionIncreasedEvent,
        PositionEventType.POSITION_DECREASED: PositionDecreasedEvent,
        PositionEventType.POSITION_REVERSED: PositionReversedEvent,
        PositionEventType.FILL_PROCESSED: FillProcessedEvent,
        PositionEventType.POSITION_UPDATED: PositionUpdatedEvent,
    }
    
    event_class = event_class_map.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")
    
    # Create base event args
    event_args = {
        "symbol": symbol,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc),
        "old_position": old_position,
        "new_position": new_position,
        "trigger_order_id": trigger_order_id,
        "realized_pnl": realized_pnl,
        "metadata": metadata or {}
    }
    
    # Add any additional kwargs specific to the event type
    if event_type == PositionEventType.FILL_PROCESSED and "fill_price" in kwargs:
        event_args.update({
            "fill_price": kwargs.get("fill_price", Decimal("0")),
            "fill_quantity": kwargs.get("fill_quantity", Decimal("0")),
            "commission": kwargs.get("commission"),
            "fees": kwargs.get("fees")
        })
    
    return event_class(**event_args)
