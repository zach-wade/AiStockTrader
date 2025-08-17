"""Application factory for creating event-driven applications."""

# Standard library imports
import asyncio
from typing import Any

# Local imports
from main.interfaces.events import IEventBus


def create_event_driven_app(
    config: dict[str, Any] | None = None, event_bus: IEventBus | None = None
) -> dict[str, Any]:
    """
    Create an event-driven application with all necessary components.

    Args:
        config: Application configuration
        event_bus: Optional event bus instance (creates new if not provided)

    Returns:
        Dictionary containing app components
    """
    if event_bus is None:
        # Lazy import to avoid circular dependency
        # Local imports
        from main.events.core import EventBusFactory

        event_bus = EventBusFactory.create()

    if config is None:
        config = {}

    # Create application context
    app_context = {"event_bus": event_bus, "config": config, "components": {}, "running": False}

    async def start_app():
        """Start the application."""
        if not app_context["running"]:
            await event_bus.start()
            app_context["running"] = True

    async def stop_app():
        """Stop the application."""
        if app_context["running"]:
            await event_bus.stop()
            app_context["running"] = False

    app_context["start"] = start_app
    app_context["stop"] = stop_app

    return app_context


async def run_event_driven_app(app_context: dict[str, Any]):
    """
    Run an event-driven application.

    Args:
        app_context: Application context from create_event_driven_app
    """
    try:
        await app_context["start"]()
        # Keep running until interrupted
        while app_context["running"]:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await app_context["stop"]()
