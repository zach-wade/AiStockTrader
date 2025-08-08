"""
Event Decorators

Decorators for event handling and callback registration.
"""

import asyncio
from typing import Callable, Any
from contextlib import asynccontextmanager

from .manager import CallbackManager
from .types import Event


def callback(event_type: str, **kwargs):
    """
    Decorator to register a method as a callback.
    
    Args:
        event_type: Event type to listen for
        **kwargs: Additional callback options
    """
    def decorator(func):
        if not hasattr(func, '_callback_registrations'):
            func._callback_registrations = []
        
        func._callback_registrations.append((event_type, kwargs))
        return func
    
    return decorator


def event_handler(event_type: str, **kwargs):
    """
    Decorator to mark a method as an event handler.
    
    Args:
        event_type: Event type to handle
        **kwargs: Additional handler options
    """
    return callback(event_type, **kwargs)


def auto_register_callbacks(cls):
    """
    Class decorator to automatically register callback methods.
    
    Looks for methods decorated with @callback and registers them.
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # Register callbacks if the class has a callback manager
        if hasattr(self, '_callback_manager'):
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if hasattr(attr, '_callback_registrations'):
                    for event_type, options in attr._callback_registrations:
                        self._callback_manager.register_callback(event_type, attr, **options)
    
    cls.__init__ = new_init
    return cls


@asynccontextmanager
async def event_context(callback_manager: CallbackManager, event_type: str, data: Any = None):
    """
    Context manager for event lifecycle management.
    
    Emits start and end events around the context block.
    """
    start_event = f"{event_type}_start"
    end_event = f"{event_type}_end"
    
    await callback_manager.emit_event(start_event, data)
    
    try:
        yield
    finally:
        await callback_manager.emit_event(end_event, data)