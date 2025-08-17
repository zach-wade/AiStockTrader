# File: events/__init__.py

"""
Event-driven architecture for the AI Trader system.

This module provides a comprehensive event bus implementation with:
- Type-safe event definitions and publishing
- Asynchronous event processing with worker pools
- Scanner to feature pipeline bridging
- Dead letter queue for failed events
- Event history and replay capabilities
- Performance metrics and monitoring

The event system enables loose coupling between components while maintaining
high performance and reliability.

Architecture:
============
The events module is organized into three main subdirectories:

1. core/
   - Event bus infrastructure (EventBus, EventBusFactory, EventBusRegistry)
   - Core event processing logic
   - Helper classes for stats tracking, history management, and DLQ

2. types/
   - Event type definitions (EventType enums, AlertType, etc.)
   - Event dataclasses specific to this module (ScanAlert, FillEvent, RiskEvent)
   - Extensions to base event types from interfaces

3. handlers/
   - Event handlers that can depend on external modules
   - ScannerFeatureBridge - bridges scanner alerts to feature pipeline
   - FeaturePipelineHandler - handles feature computation requests
   - EventDrivenEngine - coordinates event-driven processing

Import Guidelines:
==================
To avoid circular dependencies, follow these import patterns:

From core infrastructure:
```python
from main.events.core import EventBusFactory, EventBusConfig, EventBusRegistry
```

From event types:
```python
from main.events.types import AlertType, ScanAlert, FillEvent, RiskEvent
```

From handlers (be careful of circular imports):
```python
from main.events.handlers import ScannerFeatureBridge
from main.events.handlers.feature_pipeline_handler import FeaturePipelineHandler
```

Base event types come from interfaces:
```python
from main.interfaces.events import Event, EventType, EventPriority, MarketEvent, OrderEvent
```

Migration Notes:
================
The events module has been restructured to eliminate circular dependencies:
- Removed all re-exports from __init__.py
- Direct imports are now required from submodules
- FeaturePipelineHandler must be imported directly to avoid circular imports
- Deprecated singleton patterns (get_event_bus) - use EventBusFactory instead

Example Usage:
==============
```python
# Create event bus
from main.events.core import EventBusFactory, EventBusConfig
config = EventBusConfig(max_workers=5, enable_history=True)
event_bus = EventBusFactory.create(config)

# Subscribe to events
from main.interfaces.events import EventType
event_bus.subscribe(EventType.SCANNER_ALERT, my_handler)

# Publish events
from main.events.types import ScanAlert, AlertType
alert = ScanAlert(
    symbol="AAPL",
    alert_type=AlertType.VOLUME_SPIKE,
    score=0.85,
    message="High volume detected"
)
await event_bus.publish(alert)
```
"""

# This file intentionally left empty to avoid circular imports.
# All imports should be done directly from the appropriate submodules.
