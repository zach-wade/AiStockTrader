"""
Event bus interface definitions.

This module defines the protocols for event publishing and subscribing,
establishing the contract that all event bus implementations must follow.
"""

# Standard library imports
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IEventPublisher(Protocol):
    """
    Interface for publishing events to an event bus.

    This protocol defines the minimal contract for event publishers,
    allowing components to publish events without depending on a
    specific event bus implementation.
    """

    @abstractmethod
    async def publish(self, event: Any) -> None:
        """
        Publish an event to the event bus.

        Args:
            event: The event to publish. Should be a dataclass or object
                  with an event_type attribute.

        Raises:
            EventPublishError: If the event cannot be published.
        """
        ...


@runtime_checkable
class IEventSubscriber(Protocol):
    """
    Interface for subscribing to events from an event bus.

    This protocol defines how components can register handlers
    to receive events of specific types.
    """

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Subscribe a handler to events of a specific type.

        Args:
            event_type: The type of events to subscribe to.
            handler: An async callable that will receive events.
                    Should accept a single event parameter.

        Raises:
            SubscriptionError: If subscription fails.
        """
        ...

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Unsubscribe a handler from events of a specific type.

        Args:
            event_type: The type of events to unsubscribe from.
            handler: The handler to remove.

        Raises:
            SubscriptionError: If unsubscription fails.
        """
        ...

    def subscribe_all(self, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Subscribe a handler to all event types.

        Args:
            handler: An async callable that will receive all events.

        Note:
            This is an optional method. Implementations may raise
            NotImplementedError if not supported.
        """
        raise NotImplementedError("subscribe_all is optional")


@runtime_checkable
class IEventBus(IEventPublisher, IEventSubscriber, Protocol):
    """
    Complete event bus interface combining publishing and subscribing.

    This protocol represents a full event bus that can both publish
    events and manage subscriptions. It also includes lifecycle
    management methods.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Start the event bus.

        This method should initialize any required resources and
        begin processing events.

        Raises:
            EventBusError: If the bus cannot be started.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the event bus gracefully.

        This method should:
        - Stop accepting new events
        - Process any pending events
        - Clean up resources
        - Notify subscribers of shutdown

        Raises:
            EventBusError: If the bus cannot be stopped cleanly.
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the event bus is currently running.

        Returns:
            True if the bus is running and can process events,
            False otherwise.
        """
        ...

    def get_stats(self) -> dict:
        """
        Get statistics about the event bus operation.

        Returns:
            Dictionary containing metrics like:
            - Total events published
            - Events by type
            - Active subscriptions
            - Processing times

        Note:
            This is an optional method. Implementations may return
            an empty dict or raise NotImplementedError.
        """
        return {}

    async def wait_for_shutdown(self) -> None:
        """
        Wait for the event bus to shut down.

        This method blocks until the event bus is stopped.

        Note:
            This is an optional method. Implementations may raise
            NotImplementedError if not supported.
        """
        raise NotImplementedError("wait_for_shutdown is optional")
