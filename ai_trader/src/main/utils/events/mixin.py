"""
Callback Mixin

Mixin class that provides callback functionality to any class.
"""

import asyncio
from typing import Callable, Any, Optional

from .manager import CallbackManager
from .types import EventResult


class CallbackMixin:
    """
    Mixin class that provides callback functionality to any class.
    
    Adds event emission and callback management capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback_manager = CallbackManager()
    
    def on(self, event_type: str, callback: Callable, **kwargs) -> str:
        """Register callback for event type."""
        return self._callback_manager.register_callback(event_type, callback, **kwargs)
    
    def once(self, event_type: str, callback: Callable, **kwargs) -> str:
        """Register callback that executes only once."""
        original_callback = callback
        
        def once_wrapper(event):
            result = original_callback(event)
            self.off(event_type, once_wrapper)
            return result
        
        async def async_once_wrapper(event):
            result = await original_callback(event)
            self.off(event_type, async_once_wrapper)
            return result
        
        wrapper = async_once_wrapper if asyncio.iscoroutinefunction(callback) else once_wrapper
        return self._callback_manager.register_callback(event_type, wrapper, **kwargs)
    
    def off(self, event_type: str, callback: Callable) -> bool:
        """Unregister callback."""
        return self._callback_manager.unregister_callback(event_type, callback)
    
    def emit(self, event_type: str, data: Any = None, **kwargs) -> asyncio.Task:
        """Emit event asynchronously."""
        return asyncio.create_task(
            self._callback_manager.emit_event(event_type, data, **kwargs)
        )
    
    async def emit_and_wait(self, event_type: str, data: Any = None, **kwargs) -> EventResult:
        """Emit event and wait for completion."""
        return await self._callback_manager.emit_event(event_type, data, **kwargs)
    
    def add_middleware(self, before: Optional[Callable] = None, after: Optional[Callable] = None):
        """Add middleware callbacks."""
        if before:
            self._callback_manager.add_before_callback(before)
        if after:
            self._callback_manager.add_after_callback(after)
    
    def get_callback_manager(self) -> CallbackManager:
        """Get the callback manager instance."""
        return self._callback_manager