"""Application factory for creating event-driven applications."""

from typing import Dict, Any, Optional
import asyncio
from main.interfaces.events import IEventBus


def create_event_driven_app(
    config: Optional[Dict[str, Any]] = None,
    event_bus: Optional[IEventBus] = None
) -> Dict[str, Any]:
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
        from main.events.core import EventBusFactory
        event_bus = EventBusFactory.create()
    
    if config is None:
        config = {}
    
    # Create application context
    app_context = {
        'event_bus': event_bus,
        'config': config,
        'components': {},
        'running': False
    }
    
    async def start_app():
        """Start the application."""
        if not app_context['running']:
            await event_bus.start()
            app_context['running'] = True
    
    async def stop_app():
        """Stop the application."""
        if app_context['running']:
            await event_bus.stop()
            app_context['running'] = False
    
    app_context['start'] = start_app
    app_context['stop'] = stop_app
    
    return app_context


async def run_event_driven_app(app_context: Dict[str, Any]):
    """
    Run an event-driven application.
    
    Args:
        app_context: Application context from create_event_driven_app
    """
    try:
        await app_context['start']()
        # Keep running until interrupted
        while app_context['running']:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await app_context['stop']()