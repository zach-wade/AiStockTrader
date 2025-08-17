"""
Global Event Manager

Global callback manager instance and convenience functions.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from typing import Any

from .manager import CallbackManager
from .types import EventResult

# Global callback manager instance
_global_callback_manager = CallbackManager()


def get_global_callback_manager() -> CallbackManager:
    """Get the global callback manager instance."""
    return _global_callback_manager


def on(event_type: str, callback: Callable, **kwargs) -> str:
    """Register callback with global manager."""
    return _global_callback_manager.register_callback(event_type, callback, **kwargs)


def off(event_type: str, callback: Callable) -> bool:
    """Unregister callback from global manager."""
    return _global_callback_manager.unregister_callback(event_type, callback)


def emit(event_type: str, data: Any = None, **kwargs) -> asyncio.Task:
    """Emit event using global manager."""
    return asyncio.create_task(_global_callback_manager.emit_event(event_type, data, **kwargs))


async def emit_and_wait(event_type: str, data: Any = None, **kwargs) -> EventResult:
    """Emit event and wait for completion using global manager."""
    return await _global_callback_manager.emit_event(event_type, data, **kwargs)
