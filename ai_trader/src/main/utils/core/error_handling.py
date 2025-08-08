"""
Error Handling Mixin Module

This module provides error handling utilities including circuit breakers,
configuration classes, and error handling mixins for robust error management.
"""

import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

from main.utils.resilience import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class ErrorHandlingMixin:
    """
    Mixin class providing common error handling functionality.
    
    This mixin provides standardized error handling, logging, and recovery
    mechanisms that can be used across different classes.
    """
    
    def __init__(self):
        """Initialize error handling mixin."""
        self._error_count = 0
        self._last_error = None
        self._error_callbacks: Dict[str, Callable] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def handle_error(self, error: Exception, context: str = None, 
                    reraise: bool = False) -> bool:
        """
        Handle an error with standardized logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            reraise: Whether to reraise the exception after handling
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        self._error_count += 1
        self._last_error = error
        
        # Log the error with full traceback
        error_msg = f"Error in {context or 'unknown context'}: {error}"
        self._logger.error(error_msg, exc_info=True)
        
        # Also print to stderr to ensure we see it
        import sys
        print(f"ERROR: {error_msg}", file=sys.stderr)
        
        # Call registered error callbacks
        for callback_name, callback in self._error_callbacks.items():
            try:
                callback(error, context)
            except Exception as callback_error:
                self._logger.error(f"Error in callback {callback_name}: {callback_error}")
        
        # Reraise if requested
        if reraise:
            raise error
        
        return True
    
    def register_error_callback(self, name: str, callback: Callable[[Exception, str], None]):
        """
        Register a callback to be called when errors occur.
        
        Args:
            name: Name of the callback
            callback: Function to call when errors occur
        """
        self._error_callbacks[name] = callback
    
    def unregister_error_callback(self, name: str):
        """
        Unregister an error callback.
        
        Args:
            name: Name of the callback to remove
        """
        self._error_callbacks.pop(name, None)
    
    def get_error_count(self) -> int:
        """Get the total number of errors encountered."""
        return self._error_count
    
    def get_last_error(self) -> Optional[Exception]:
        """Get the last error that occurred."""
        return self._last_error
    
    def reset_error_count(self):
        """Reset the error count."""
        self._error_count = 0
        self._last_error = None
    
    def with_error_handling(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call, or None if an error occurred
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context=f"calling {func.__name__}")
            return None
    
    @contextmanager
    def _handle_error(self, context: str):
        """
        Context manager for error handling.
        
        Args:
            context: Description of the operation being performed
            
        Example:
            with self._handle_error("initializing component"):
                # code that might raise exceptions
        """
        try:
            yield
        except Exception as e:
            self.handle_error(e, context=context, reraise=True)

# Export the classes
__all__ = ['CircuitBreaker', 'CircuitBreakerConfig', 'ErrorHandlingMixin']