"""Events package."""

from .types import (
    CallbackPriority,
    EventStatus,
    CallbackInfo,
    Event,
    EventResult
)

from .manager import CallbackManager

from .mixin import CallbackMixin

from .decorators import (
    callback,
    event_handler,
    auto_register_callbacks,
    event_context
)

from .global_manager import (
    get_global_callback_manager,
    on,
    off,
    emit,
    emit_and_wait
)

__all__ = [
    # Types
    'CallbackPriority',
    'EventStatus',
    'CallbackInfo',
    'Event',
    'EventResult',
    
    # Manager
    'CallbackManager',
    
    # Mixin
    'CallbackMixin',
    
    # Decorators
    'callback',
    'event_handler',
    'auto_register_callbacks',
    'event_context',
    
    # Global functions
    'get_global_callback_manager',
    'on',
    'off',
    'emit',
    'emit_and_wait'
]