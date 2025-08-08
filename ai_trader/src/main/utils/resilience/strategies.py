"""
Resilience Strategies Module

Combines circuit breaker and error recovery patterns for comprehensive resilience.
This module provides the ResilienceStrategies class that was referenced but missing
in the codebase.
"""

import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Union
import asyncio
from functools import wraps

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .error_recovery import ErrorRecoveryManager, RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResilienceStrategies:
    """
    Combines multiple resilience patterns into a unified strategy.
    
    This class integrates:
    - Circuit breaker for fail-fast behavior
    - Retry logic with exponential backoff
    - Error recovery management
    - Rate limiting capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize resilience strategies with configuration.
        
        Args:
            config: Configuration dictionary with resilience settings
        """
        self.config = config or {}
        
        # Extract configurations
        self.max_retries = self.config.get('max_retries', 3)
        self.initial_delay = self.config.get('initial_delay', 1.0)
        self.backoff_factor = self.config.get('backoff_factor', 2.0)
        self.max_delay = self.config.get('max_delay', 60.0)
        self.jitter = self.config.get('jitter', True)
        
        # Circuit breaker configuration
        self.failure_threshold = self.config.get('failure_threshold', 5)
        self.recovery_timeout = self.config.get('recovery_timeout', 60)
        self.critical_latency_ms = self.config.get('critical_latency_ms', 5000.0)
        
        # Rate limiting (optional)
        self.rate_limit_calls = self.config.get('rate_limit_calls', None)
        self.rate_limit_period = self.config.get('rate_limit_period', 60)
        
        # Initialize components
        self._init_circuit_breaker()
        self._init_error_recovery()
        
        # Rate limiter state
        self._call_times = []
        
        logger.info(f"ResilienceStrategies initialized with max_retries={self.max_retries}, "
                   f"failure_threshold={self.failure_threshold}")
    
    def _init_circuit_breaker(self):
        """Initialize circuit breaker with configuration."""
        cb_config = CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            timeout_seconds=float(self.recovery_timeout),
            critical_latency_ms=self.critical_latency_ms
        )
        self.circuit_breaker = CircuitBreaker(config=cb_config)
    
    def _init_error_recovery(self):
        """Initialize error recovery manager."""
        retry_config = RetryConfig(
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_factor=self.backoff_factor,
            max_delay=self.max_delay,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=self.jitter
        )
        self.error_recovery = ErrorRecoveryManager(default_config=retry_config)
    
    def _check_rate_limit(self) -> bool:
        """
        Check if rate limit allows another call.
        
        Returns:
            True if call is allowed, False if rate limited
        """
        if self.rate_limit_calls is None:
            return True
        
        import time
        current_time = time.time()
        
        # Remove old calls outside the period
        self._call_times = [t for t in self._call_times 
                          if current_time - t < self.rate_limit_period]
        
        # Check if we can make another call
        if len(self._call_times) < self.rate_limit_calls:
            self._call_times.append(current_time)
            return True
        
        return False
    
    async def execute_with_resilience(self, 
                                    func: Callable[..., T], 
                                    *args, 
                                    **kwargs) -> T:
        """
        Execute a function with full resilience strategies.
        
        Applies:
        1. Rate limiting check
        2. Circuit breaker protection
        3. Retry with backoff on failure
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Exception: If all resilience strategies fail
        """
        # Check rate limit
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        # Define the wrapped function for circuit breaker
        async def wrapped():
            # If func is async, await it
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
        
        # Execute with circuit breaker and retry
        try:
            # First try with circuit breaker
            result = await self.circuit_breaker.call(wrapped)
            return result
        except Exception as e:
            # If circuit breaker is open or call failed, try with retry logic
            logger.warning(f"Initial call failed: {e}. Attempting retry...")
            
            # Use error recovery manager for retries
            retry_config = RetryConfig(
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                backoff_factor=self.backoff_factor,
                max_delay=self.max_delay,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                jitter=self.jitter
            )
            
            result = await self.error_recovery.execute_with_retry(
                wrapped,
                retry_config=retry_config
            )
            
            # If retry succeeded, reset circuit breaker failure count
            self.circuit_breaker.on_success()
            
            return result
    
    def with_resilience(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to add resilience strategies to a function.
        
        Args:
            func: Function to wrap with resilience
            
        Returns:
            Wrapped function with resilience strategies
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.execute_with_resilience(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create an event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.execute_with_resilience(func, *args, **kwargs)
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all resilience components.
        
        Returns:
            Dictionary with resilience statistics
        """
        return {
            'circuit_breaker': self.circuit_breaker.get_stats(),
            'error_recovery': {
                'max_retries': self.max_retries,
                'initial_delay': self.initial_delay,
                'backoff_factor': self.backoff_factor
            },
            'rate_limiting': {
                'enabled': self.rate_limit_calls is not None,
                'calls_limit': self.rate_limit_calls,
                'period_seconds': self.rate_limit_period,
                'current_calls': len(self._call_times)
            }
        }
    
    def reset(self):
        """Reset all resilience components to initial state."""
        self.circuit_breaker.reset()
        self._call_times.clear()
        logger.info("ResilienceStrategies reset to initial state")