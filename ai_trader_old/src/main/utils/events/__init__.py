"""Events package."""

from .decorators import auto_register_callbacks, callback, event_context, event_handler
from .global_manager import emit, emit_and_wait, get_global_callback_manager, off, on
from .manager import CallbackManager
from .mixin import CallbackMixin
from .types import CallbackInfo, CallbackPriority, Event, EventResult, EventStatus

__all__ = [
    # Types
    "CallbackPriority",
    "EventStatus",
    "CallbackInfo",
    "Event",
    "EventResult",
    # Manager
    "CallbackManager",
    # Mixin
    "CallbackMixin",
    # Decorators
    "callback",
    "event_handler",
    "auto_register_callbacks",
    "event_context",
    # Global functions
    "get_global_callback_manager",
    "on",
    "off",
    "emit",
    "emit_and_wait",
]
