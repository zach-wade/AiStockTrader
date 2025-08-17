"""
Simple dependency injection container for AI Trader.

This module provides a lightweight DI container for managing
dependencies and their lifecycles in the application.
"""

# Standard library imports
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
import logging
from typing import Any, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifecycle(Enum):
    """Dependency lifecycle types."""

    SINGLETON = "singleton"  # Single instance for entire app lifetime
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # Single instance per scope/request


@dataclass
class Registration:
    """Registration entry for a dependency."""

    interface: type
    implementation: type | Callable | Any
    lifecycle: Lifecycle
    factory: Callable | None = None
    instance: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DIContainer:
    """
    Simple dependency injection container.

    Supports constructor injection, factory methods, and different lifecycles.
    """

    def __init__(self):
        """Initialize the container."""
        self._registrations: dict[type, Registration] = {}
        self._scoped_instances: dict[type, Any] = {}
        logger.info("DIContainer initialized")

    def register(
        self,
        interface: type[T],
        implementation: type[T] | Callable[..., T] | T,
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
        factory: Callable[..., T] | None = None,
        **metadata,
    ) -> None:
        """
        Register a dependency.

        Args:
            interface: The interface/base type to register.
            implementation: The implementation (class, instance, or factory).
            lifecycle: How instances should be managed.
            factory: Optional factory function.
            **metadata: Additional metadata for the registration.

        Raises:
            ValueError: If interface is already registered.
        """
        if interface in self._registrations:
            raise ValueError(f"Interface {interface} is already registered")

        registration = Registration(
            interface=interface,
            implementation=implementation,
            lifecycle=lifecycle,
            factory=factory,
            metadata=metadata,
        )

        # If implementation is an instance, store it
        if not inspect.isclass(implementation) and not callable(implementation):
            registration.instance = implementation
            registration.lifecycle = Lifecycle.SINGLETON

        self._registrations[interface] = registration
        logger.debug(f"Registered {interface.__name__} with lifecycle {lifecycle.value}")

    def register_singleton(
        self, interface: type[T], implementation: type[T] | T, **metadata
    ) -> None:
        """Register a singleton dependency."""
        self.register(interface, implementation, Lifecycle.SINGLETON, **metadata)

    def register_transient(self, interface: type[T], implementation: type[T], **metadata) -> None:
        """Register a transient dependency."""
        self.register(interface, implementation, Lifecycle.TRANSIENT, **metadata)

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
        lifecycle: Lifecycle = Lifecycle.TRANSIENT,
        **metadata,
    ) -> None:
        """Register a factory function."""
        self.register(interface, factory, lifecycle, factory=factory, **metadata)

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a dependency.

        Args:
            interface: The interface to resolve.

        Returns:
            An instance of the requested type.

        Raises:
            ValueError: If interface is not registered.
        """
        if interface not in self._registrations:
            raise ValueError(f"Interface {interface} is not registered")

        registration = self._registrations[interface]

        # Handle different lifecycles
        if registration.lifecycle == Lifecycle.SINGLETON:
            if registration.instance is None:
                registration.instance = self._create_instance(registration)
            return registration.instance

        elif registration.lifecycle == Lifecycle.SCOPED:
            if interface not in self._scoped_instances:
                self._scoped_instances[interface] = self._create_instance(registration)
            return self._scoped_instances[interface]

        else:  # TRANSIENT
            return self._create_instance(registration)

    def _create_instance(self, registration: Registration) -> Any:
        """Create an instance based on registration."""
        # If we have a factory, use it
        if registration.factory:
            return self._call_with_injection(registration.factory)

        # If implementation is already an instance, return it
        if not inspect.isclass(registration.implementation):
            if callable(registration.implementation):
                return self._call_with_injection(registration.implementation)
            return registration.implementation

        # Create instance with constructor injection
        return self._call_with_injection(registration.implementation)

    def _call_with_injection(self, func: Callable) -> Any:
        """Call a function/constructor with dependency injection."""
        sig = inspect.signature(func)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Try to resolve by type annotation
            if param.annotation != param.empty:
                param_type = param.annotation

                # Handle Optional types
                if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
                    # Get the non-None type from Optional
                    args = param_type.__args__
                    param_type = next((arg for arg in args if arg is not type(None)), None)

                if param_type and param_type in self._registrations:
                    kwargs[param_name] = self.resolve(param_type)
                    continue

            # Use default if available
            if param.default != param.empty:
                continue

            # Skip if we can't resolve
            logger.debug(f"Cannot resolve parameter {param_name} for {func}")

        return func(**kwargs)

    def has_registration(self, interface: type) -> bool:
        """Check if an interface is registered."""
        return interface in self._registrations

    def unregister(self, interface: type) -> None:
        """Unregister an interface."""
        if interface in self._registrations:
            del self._registrations[interface]
            self._scoped_instances.pop(interface, None)
            logger.debug(f"Unregistered {interface.__name__}")

    def clear_scoped(self) -> None:
        """Clear all scoped instances."""
        self._scoped_instances.clear()
        logger.debug("Cleared scoped instances")

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()
        self._scoped_instances.clear()
        logger.info("Cleared all registrations")


# Global container instance
_global_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container."""
    return _global_container


def configure_dependencies() -> DIContainer:
    """
    Configure default dependencies for the application.

    Returns:
        Configured DI container.
    """
    container = get_container()

    # Register event bus dependencies
    # Local imports
    from main.events.core import EventBusFactory, get_global_registry
    from main.interfaces.events import IEventBus, IEventBusProvider

    # Register event bus provider as singleton
    container.register_singleton(IEventBusProvider, get_global_registry())

    # Register event bus factory
    container.register_factory(
        IEventBus, lambda: EventBusFactory.create(), lifecycle=Lifecycle.SINGLETON
    )

    # Register backtesting dependencies
    # Local imports
    from main.backtesting.factories import BacktestEngineFactory
    from main.interfaces.backtesting import IBacktestEngineFactory

    container.register_singleton(IBacktestEngineFactory, BacktestEngineFactory())

    # Register database dependencies
    # Local imports
    from main.config.config_manager import get_config
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.interfaces.database import IAsyncDatabase, IDatabaseFactory

    # Register database factory as singleton
    container.register_singleton(IDatabaseFactory, DatabaseFactory())

    # Register database instances with factory methods
    config = get_config()

    container.register_factory(
        IAsyncDatabase,
        lambda: container.resolve(IDatabaseFactory).create_async_database(config),
        lifecycle=Lifecycle.SINGLETON,
    )

    # Register repository factory
    # Local imports
    from main.data_pipeline.storage.repositories.repository_factory import get_repository_factory
    from main.interfaces.repositories import IRepositoryFactory

    container.register_factory(
        IRepositoryFactory,
        lambda: get_repository_factory(container.resolve(IAsyncDatabase)),
        lifecycle=Lifecycle.SINGLETON,
    )

    logger.info("Default dependencies configured")
    return container
