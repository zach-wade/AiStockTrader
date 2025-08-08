"""
Event interfaces for the AI Trader system.

This module defines the contracts for event-driven architecture components,
enabling loose coupling between event publishers and subscribers.
"""

from .event_bus import (
    IEventPublisher,
    IEventSubscriber,
    IEventBus,
)

from .event_bus_provider import (
    IEventBusProvider,
)

from .event_types import (
    Event,
    EventType,
    EventPriority,
    MarketEvent,
    OrderEvent,
    ScannerAlertEvent,
    FeatureRequestEvent,
    FeatureComputedEvent,
    ErrorEvent,
    SystemStatusEvent,
    DataWrittenEvent,
)

from .event_handlers import (
    EventHandler,
    AsyncEventHandler,
)

__all__ = [
    # Event bus interfaces
    "IEventPublisher",
    "IEventSubscriber",
    "IEventBus",
    "IEventBusProvider",
    
    # Event types
    "Event",
    "EventType",
    "EventPriority",
    "MarketEvent",
    "OrderEvent",
    "ScannerAlertEvent",
    "FeatureRequestEvent",
    "FeatureComputedEvent",
    "ErrorEvent",
    "SystemStatusEvent",
    "DataWrittenEvent",
    
    # Handler types
    "EventHandler",
    "AsyncEventHandler",
]