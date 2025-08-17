"""
Event bus provider interface for dependency injection.

This module defines the contract for providing event bus instances,
enabling flexible dependency injection and testing scenarios.
"""

# Standard library imports
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from .event_bus import IEventBus


@runtime_checkable
class IEventBusProvider(Protocol):
    """
    Interface for providing event bus instances.

    This protocol enables dependency injection by abstracting
    how event bus instances are created and retrieved.
    """

    @abstractmethod
    def get_event_bus(self, name: str | None = None) -> IEventBus:
        """
        Get an event bus instance.

        Args:
            name: Optional name for named instances.
                  If None, returns the default instance.

        Returns:
            An IEventBus instance.

        Raises:
            EventBusNotFoundError: If named instance doesn't exist.
            EventBusCreationError: If instance cannot be created.
        """
        ...

    @abstractmethod
    def register_event_bus(self, event_bus: IEventBus, name: str | None = None) -> None:
        """
        Register an event bus instance.

        Args:
            event_bus: The event bus instance to register.
            name: Optional name for the instance.
                  If None, registers as default instance.

        Raises:
            EventBusAlreadyExistsError: If name is already registered.
        """
        ...

    def has_event_bus(self, name: str | None = None) -> bool:
        """
        Check if an event bus is registered.

        Args:
            name: Name to check. If None, checks for default instance.

        Returns:
            True if the named instance exists, False otherwise.

        Note:
            This is an optional method. Implementations may raise
            NotImplementedError if not supported.
        """
        raise NotImplementedError("has_event_bus is optional")

    def unregister_event_bus(self, name: str | None = None) -> None:
        """
        Unregister an event bus instance.

        Args:
            name: Name to unregister. If None, unregisters default instance.

        Raises:
            EventBusNotFoundError: If named instance doesn't exist.

        Note:
            This is an optional method. Implementations may raise
            NotImplementedError if not supported.
        """
        raise NotImplementedError("unregister_event_bus is optional")
