"""
Event bus registry for managing named event bus instances.

This module provides a registry for storing and retrieving named
event bus instances, supporting multi-bus architectures.
"""

# Standard library imports
import logging
from threading import Lock

# Local imports
from main.events.core.event_bus_factory import EventBusConfig, EventBusFactory
from main.interfaces.events import IEventBus, IEventBusProvider

logger = logging.getLogger(__name__)


class EventBusNotFoundError(Exception):
    """Raised when a requested event bus is not found."""

    pass


class EventBusAlreadyExistsError(Exception):
    """Raised when trying to register an existing event bus name."""

    pass


class EventBusRegistry(IEventBusProvider):
    """
    Registry for managing multiple named event bus instances.

    This class implements the IEventBusProvider interface and manages
    a collection of named event bus instances, supporting scenarios
    where different components need different event buses.
    """

    def __init__(self, auto_create: bool = True):
        """
        Initialize the registry.

        Args:
            auto_create: If True, automatically creates event buses
                        when requested but not found.
        """
        self._instances: dict[str | None, IEventBus] = {}
        self._configs: dict[str | None, EventBusConfig] = {}
        self._lock = Lock()
        self._auto_create = auto_create

        logger.info(f"EventBusRegistry initialized with auto_create={auto_create}")

    def get_event_bus(self, name: str | None = None) -> IEventBus:
        """
        Get an event bus instance by name.

        Args:
            name: Name of the event bus. If None, returns default instance.

        Returns:
            The requested IEventBus instance.

        Raises:
            EventBusNotFoundError: If not found and auto_create is False.
        """
        with self._lock:
            if name in self._instances:
                return self._instances[name]

            if self._auto_create:
                logger.info(f"Auto-creating event bus: {name or 'default'}")
                # Use stored config if available
                config = self._configs.get(name)
                event_bus = EventBusFactory.create(config)
                self._instances[name] = event_bus
                return event_bus

            raise EventBusNotFoundError(
                f"Event bus '{name or 'default'}' not found and auto_create is disabled"
            )

    def register_event_bus(
        self, event_bus: IEventBus, name: str | None = None, config: EventBusConfig | None = None
    ) -> None:
        """
        Register an event bus instance.

        Args:
            event_bus: The event bus instance to register.
            name: Name for the instance. If None, registers as default.
            config: Optional configuration to store with the instance.

        Raises:
            EventBusAlreadyExistsError: If name is already registered.
        """
        with self._lock:
            if name in self._instances:
                raise EventBusAlreadyExistsError(
                    f"Event bus '{name or 'default'}' already registered"
                )

            self._instances[name] = event_bus
            if config:
                self._configs[name] = config

            logger.info(f"Registered event bus: {name or 'default'}")

    def has_event_bus(self, name: str | None = None) -> bool:
        """
        Check if an event bus is registered.

        Args:
            name: Name to check. If None, checks for default instance.

        Returns:
            True if the named instance exists, False otherwise.
        """
        return name in self._instances

    def unregister_event_bus(self, name: str | None = None) -> None:
        """
        Unregister an event bus instance.

        Args:
            name: Name to unregister. If None, unregisters default instance.

        Raises:
            EventBusNotFoundError: If named instance doesn't exist.
        """
        with self._lock:
            if name not in self._instances:
                raise EventBusNotFoundError(f"Event bus '{name or 'default'}' not found")

            del self._instances[name]
            self._configs.pop(name, None)

            logger.info(f"Unregistered event bus: {name or 'default'}")

    def register_config(self, name: str | None, config: EventBusConfig) -> None:
        """
        Register a configuration for future event bus creation.

        Args:
            name: Name to associate with the configuration.
            config: Configuration to store.
        """
        with self._lock:
            self._configs[name] = config
            logger.info(f"Registered config for event bus: {name or 'default'}")

    def get_all_names(self) -> set[str | None]:
        """Get names of all registered event buses."""
        return set(self._instances.keys())

    def clear(self) -> None:
        """Remove all registered event buses."""
        with self._lock:
            count = len(self._instances)
            self._instances.clear()
            self._configs.clear()
            logger.info(f"Cleared {count} event buses from registry")

    async def stop_all(self) -> None:
        """Stop all registered event buses."""
        event_buses = list(self._instances.values())

        for event_bus in event_buses:
            try:
                if event_bus.is_running():
                    await event_bus.stop()
                    logger.info(f"Stopped event bus: {event_bus}")
            except Exception as e:
                logger.error(f"Error stopping event bus: {e}")


# Global registry instance
_global_registry = EventBusRegistry(auto_create=True)


def get_global_registry() -> EventBusRegistry:
    """Get the global event bus registry."""
    return _global_registry
