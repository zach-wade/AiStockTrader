"""
Event type definitions for the AI Trader system.

This module contains all event type definitions specific to the events module.
Core event types are imported from interfaces to avoid circular dependencies.
"""

# Local imports
# Import core types from interfaces
from main.interfaces.events import (
    ErrorEvent,
    Event,
    EventPriority,
    EventType,
    FeatureComputedEvent,
    FeatureRequestEvent,
    MarketEvent,
    OrderEvent,
    ScannerAlertEvent,
)

# Import module-specific types
from .event_types import AlertType, FillEvent, PositionEvent, RiskEvent, ScanAlert

__all__ = [
    # Core types from interfaces
    "Event",
    "EventType",
    "EventPriority",
    "MarketEvent",
    "OrderEvent",
    "ScannerAlertEvent",
    "FeatureRequestEvent",
    "FeatureComputedEvent",
    "ErrorEvent",
    # Module-specific types
    "AlertType",
    "ScanAlert",
    "FillEvent",
    "RiskEvent",
    "PositionEvent",
]
