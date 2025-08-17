"""
Event handler protocol definitions.

This module defines the protocols for event handlers, establishing
the contract for how components should handle events.
"""

# Standard library imports
from typing import Protocol, TypeVar, runtime_checkable

# Type variable for event types
TEvent = TypeVar("TEvent")


@runtime_checkable
class EventHandler(Protocol[TEvent]):
    """
    Protocol for synchronous event handlers.

    Note: Most handlers in the async event system should be async,
    but this is provided for compatibility.
    """

    def __call__(self, event: TEvent) -> None:
        """
        Handle an event synchronously.

        Args:
            event: The event to handle.
        """
        ...


@runtime_checkable
class AsyncEventHandler(Protocol[TEvent]):
    """
    Protocol for asynchronous event handlers.

    This is the preferred handler type for the event system,
    allowing non-blocking event processing.
    """

    async def __call__(self, event: TEvent) -> None:
        """
        Handle an event asynchronously.

        Args:
            event: The event to handle.

        Note:
            Handlers should not raise exceptions. Any errors
            should be logged and handled internally.
        """
        ...
