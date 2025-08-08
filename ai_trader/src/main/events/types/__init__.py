"""
Event type definitions for the AI Trader system.

This module contains all event type definitions specific to the events module.
Core event types are imported from interfaces to avoid circular dependencies.
"""

# Import core types from interfaces
from main.interfaces.events import (
    Event,
    EventType,
    EventPriority,
    MarketEvent,
    OrderEvent,
    ScannerAlertEvent,
    FeatureRequestEvent,
    FeatureComputedEvent,
    ErrorEvent,
)

# Import module-specific types
from .event_types import (
    AlertType,
    ScanAlert,
    FillEvent,
    RiskEvent,
    PositionEvent,
)

__all__ = [
    # Core types from interfaces
    'Event',
    'EventType',
    'EventPriority',
    'MarketEvent',
    'OrderEvent',
    'ScannerAlertEvent',
    'FeatureRequestEvent',
    'FeatureComputedEvent',
    'ErrorEvent',
    
    # Module-specific types
    'AlertType',
    'ScanAlert',
    'FillEvent',
    'RiskEvent',
    'PositionEvent',
]