"""
Event bus factory for creating configured event bus instances.

This module provides factory methods for creating event bus instances
with different configurations, supporting dependency injection patterns.
"""

# Standard library imports
from dataclasses import dataclass, field
import logging
from typing import Any

# Local imports
from main.events.core.event_bus import EventBus
from main.interfaces.events import IEventBus
from main.utils.resilience import CircuitBreakerConfig

logger = logging.getLogger(__name__)


@dataclass
class EventBusConfig:
    """Configuration for event bus instances."""

    # Queue configuration
    max_queue_size: int = 10000
    max_workers: int = 10

    # Feature toggles
    enable_history: bool = True
    enable_dlq: bool = True
    enable_metrics: bool = True

    # History configuration
    history_retention_seconds: int = 3600

    # Circuit breaker configuration
    circuit_breaker_config: CircuitBreakerConfig | None = None

    # Custom configuration
    custom_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EventBusConfig":
        """Create configuration from dictionary."""
        # Extract known fields
        known_fields = {
            "max_queue_size",
            "max_workers",
            "enable_history",
            "enable_dlq",
            "enable_metrics",
            "history_retention_seconds",
        }

        config_args = {k: v for k, v in config_dict.items() if k in known_fields}

        # Handle circuit breaker config
        if "circuit_breaker_config" in config_dict:
            cb_config = config_dict["circuit_breaker_config"]
            if isinstance(cb_config, dict):
                config_args["circuit_breaker_config"] = CircuitBreakerConfig(**cb_config)
            else:
                config_args["circuit_breaker_config"] = cb_config

        # Store remaining as custom config
        custom_config = {
            k: v
            for k, v in config_dict.items()
            if k not in known_fields and k != "circuit_breaker_config"
        }
        if custom_config:
            config_args["custom_config"] = custom_config

        return cls(**config_args)


class EventBusFactory:
    """
    Factory for creating event bus instances.

    Provides methods for creating event buses with different
    configurations and supports custom implementations.
    """

    # Registry of event bus implementations
    _implementations: dict[str, type[IEventBus]] = {"default": EventBus}

    @classmethod
    def create(
        cls, config: EventBusConfig | None = None, implementation: str = "default"
    ) -> IEventBus:
        """
        Create an event bus instance.

        Args:
            config: Configuration for the event bus.
                    If None, uses default configuration.
            implementation: Name of the implementation to use.
                          Defaults to 'default' (EventBus).

        Returns:
            Configured IEventBus instance.

        Raises:
            ValueError: If implementation is not registered.
        """
        if config is None:
            config = EventBusConfig()

        if implementation not in cls._implementations:
            raise ValueError(
                f"Unknown implementation: {implementation}. "
                f"Available: {list(cls._implementations.keys())}"
            )

        impl_class = cls._implementations[implementation]

        logger.info(f"Creating event bus with implementation={implementation}, " f"config={config}")

        # Create instance with configuration
        if implementation == "default":
            # Default EventBus constructor
            return impl_class(
                max_queue_size=config.max_queue_size,
                max_workers=config.max_workers,
                enable_history=config.enable_history,
                history_retention_seconds=config.history_retention_seconds,
                enable_dlq=config.enable_dlq,
                circuit_breaker_config=config.circuit_breaker_config,
            )
        else:
            # Custom implementations should accept config
            return impl_class(config)

    @classmethod
    def create_from_dict(
        cls, config_dict: dict[str, Any], implementation: str = "default"
    ) -> IEventBus:
        """
        Create an event bus instance from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary.
            implementation: Implementation to use.

        Returns:
            Configured IEventBus instance.
        """
        config = EventBusConfig.from_dict(config_dict)
        return cls.create(config, implementation)

    @classmethod
    def create_test_instance(cls) -> IEventBus:
        """
        Create an event bus instance suitable for testing.

        Returns:
            Event bus with test-friendly configuration.
        """
        test_config = EventBusConfig(
            max_queue_size=100,
            max_workers=2,
            enable_history=False,
            enable_dlq=False,
            enable_metrics=False,
        )
        return cls.create(test_config)

    @classmethod
    def register_implementation(cls, name: str, implementation: type[IEventBus]) -> None:
        """
        Register a custom event bus implementation.

        Args:
            name: Name to register the implementation under.
            implementation: Event bus implementation class.

        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._implementations:
            raise ValueError(f"Implementation '{name}' already registered")

        cls._implementations[name] = implementation
        logger.info(f"Registered event bus implementation: {name}")

    @classmethod
    def unregister_implementation(cls, name: str) -> None:
        """
        Unregister an event bus implementation.

        Args:
            name: Name of implementation to unregister.

        Raises:
            ValueError: If name is not registered or is 'default'.
        """
        if name == "default":
            raise ValueError("Cannot unregister default implementation")

        if name not in cls._implementations:
            raise ValueError(f"Implementation '{name}' not registered")

        del cls._implementations[name]
        logger.info(f"Unregistered event bus implementation: {name}")

    @classmethod
    def get_implementations(cls) -> dict[str, type[IEventBus]]:
        """Get all registered implementations."""
        return cls._implementations.copy()
