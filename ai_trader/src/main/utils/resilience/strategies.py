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
from dataclasses import dataclass

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .error_recovery import ErrorRecoveryManager, RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ResilienceConfig:
    """Configuration for resilience strategies with type safety and validation."""
    
    # Retry configuration
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True
    
    # Circuit breaker configuration
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    critical_latency_ms: float = 5000.0
    
    # Rate limiting configuration (optional)
    rate_limit_calls: Optional[int] = None
    rate_limit_period: int = 60
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay is not None and self.initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if self.backoff_factor is not None and self.backoff_factor <= 1.0:
            raise ValueError("backoff_factor must be greater than 1.0")
        if self.max_delay is not None and self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.failure_threshold is not None and self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout is not None and self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.critical_latency_ms is not None and self.critical_latency_ms <= 0:
            raise ValueError("critical_latency_ms must be positive")
        if self.rate_limit_calls is not None and self.rate_limit_calls <= 0:
            raise ValueError("rate_limit_calls must be positive if specified")
        if self.rate_limit_period is not None and self.rate_limit_period <= 0:
            raise ValueError("rate_limit_period must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResilienceConfig':
        """Create ResilienceConfig from dictionary with safe field extraction."""
        # Only include known fields to avoid TypeError
        known_fields = {
            'max_retries', 'initial_delay', 'backoff_factor', 'max_delay', 'jitter',
            'failure_threshold', 'recovery_timeout', 'critical_latency_ms',
            'rate_limit_calls', 'rate_limit_period'
        }
        
        # Filter out None values and unknown fields
        filtered_config = {
            k: v for k, v in config_dict.items() 
            if k in known_fields and v is not None
        }
        return cls(**filtered_config)
    
    @classmethod
    def create_default(cls) -> 'ResilienceConfig':
        """Create default resilience configuration."""
        return cls()
    
    @classmethod  
    def create_for_api_client(cls, service_name: str) -> 'ResilienceConfig':
        """Create resilience config optimized for API clients."""
        return cls(
            max_retries=5,
            initial_delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0,
            failure_threshold=3,
            recovery_timeout=120.0,
            critical_latency_ms=10000.0,
            rate_limit_calls=100,
            rate_limit_period=60
        )
    
    @classmethod
    def create_for_database(cls, db_name: str) -> 'ResilienceConfig':
        """Create resilience config optimized for database operations."""
        return cls(
            max_retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
            max_delay=10.0,
            failure_threshold=5,
            recovery_timeout=60.0,
            critical_latency_ms=3000.0,
            rate_limit_calls=None  # No rate limiting for DB
        )
    
    @classmethod
    def create_for_feature_calculation(cls) -> 'ResilienceConfig':
        """Create resilience config optimized for feature calculation."""
        return cls(
            max_retries=2,
            initial_delay=1.0,
            backoff_factor=1.8,
            max_delay=15.0,
            failure_threshold=3,
            recovery_timeout=30.0,
            critical_latency_ms=2000.0,
            rate_limit_calls=None
        )


class ResilienceStrategies:
    """
    Combines multiple resilience patterns into a unified strategy.
    
    This class integrates:
    - Circuit breaker for fail-fast behavior
    - Retry logic with exponential backoff
    - Error recovery management
    - Rate limiting capabilities
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ResilienceConfig, Any]] = None):
        """
        Initialize resilience strategies with configuration.
        
        Args:
            config: Configuration (ResilienceConfig, dict, or complex config object)
        """
        # Handle different config types
        if isinstance(config, ResilienceConfig):
            # Use ResilienceConfig directly
            self.resilience_config = config
        else:
            # Extract and convert to ResilienceConfig
            resilience_dict = self._extract_resilience_config(config)
            self.resilience_config = ResilienceConfig.from_dict(resilience_dict)
        
        # Set instance attributes from validated config
        self.max_retries = self.resilience_config.max_retries
        self.initial_delay = self.resilience_config.initial_delay
        self.backoff_factor = self.resilience_config.backoff_factor
        self.max_delay = self.resilience_config.max_delay
        self.jitter = self.resilience_config.jitter
        
        # Circuit breaker configuration
        self.failure_threshold = self.resilience_config.failure_threshold
        self.recovery_timeout = self.resilience_config.recovery_timeout
        self.critical_latency_ms = self.resilience_config.critical_latency_ms
        
        # Rate limiting configuration
        self.rate_limit_calls = self.resilience_config.rate_limit_calls
        self.rate_limit_period = self.resilience_config.rate_limit_period
        
        # Initialize components
        self._init_circuit_breaker()
        self._init_error_recovery()
        
        # Rate limiter state
        self._call_times = []
        
        logger.info(f"ResilienceStrategies initialized with max_retries={self.max_retries}, "
                   f"failure_threshold={self.failure_threshold}")
    
    def _extract_resilience_config(self, config: Any) -> Dict[str, Any]:
        """
        Extract resilience configuration from complex config objects.
        
        Handles OmegaConf DictConfig, regular dicts, and other config types.
        
        Args:
            config: Configuration object of any type
            
        Returns:
            Dict with resilience-specific settings
        """
        if config is None:
            return {}
            
        # If it's already a dict, return as-is if it has resilience keys
        if isinstance(config, dict):
            # Check if it already contains resilience keys (direct resilience config)
            resilience_keys = {'max_retries', 'failure_threshold', 'critical_latency_ms'}
            if any(key in config for key in resilience_keys):
                return config
        
        # Try to extract resilience section from complex config objects
        resilience_config = {}
        
        # Handle OmegaConf DictConfig or similar objects
        try:
            # First try to access resilience section directly
            if hasattr(config, 'resilience'):
                resilience_section = getattr(config, 'resilience')
                if hasattr(resilience_section, '_content') or isinstance(resilience_section, dict):
                    # Convert to dict if it's an OmegaConf node  
                    resilience_dict = dict(resilience_section) if hasattr(resilience_section, 'items') else {}
                    
                    # Flatten the YAML structure to match ResilienceConfig fields
                    if 'retry' in resilience_dict:
                        retry_config = resilience_dict['retry']
                        if isinstance(retry_config, dict):
                            resilience_config.update(retry_config)
                    
                    if 'circuit_breaker' in resilience_dict:
                        cb_config = resilience_dict['circuit_breaker']
                        if isinstance(cb_config, dict):
                            resilience_config.update(cb_config)
                    
                    if 'rate_limiting' in resilience_dict:
                        rate_config = resilience_dict['rate_limiting']
                        if isinstance(rate_config, dict):
                            # Map YAML rate limiting keys to internal keys
                            if 'calls_per_period' in rate_config:
                                resilience_config['rate_limit_calls'] = rate_config['calls_per_period']
                            if 'period_seconds' in rate_config:
                                resilience_config['rate_limit_period'] = rate_config['period_seconds']
                    
                    # If we found a nested structure, return the flattened config
                    if resilience_config:
                        return resilience_config
                        
                    # Otherwise treat the whole resilience section as flat config
                    resilience_config = resilience_dict
                    
            # If no resilience section, try to find relevant keys in the root config
            elif hasattr(config, '_content') or hasattr(config, 'keys'):
                # This is likely an OmegaConf DictConfig
                config_dict = dict(config) if hasattr(config, 'items') else {}
                
                # Look for resilience-related keys at the root level
                resilience_keys = {
                    'max_retries', 'initial_delay', 'backoff_factor', 'max_delay', 'jitter',
                    'failure_threshold', 'recovery_timeout', 'critical_latency_ms',
                    'rate_limit_calls', 'rate_limit_period'
                }
                
                resilience_config = {k: v for k, v in config_dict.items() if k in resilience_keys}
                
        except (AttributeError, TypeError):
            # Fallback: treat as dict-like object
            try:
                if hasattr(config, '__getitem__'):
                    # Try to access as dict
                    if 'resilience' in config:
                        resilience_section = config['resilience']
                        
                        # Handle nested YAML structure
                        if isinstance(resilience_section, dict):
                            if 'retry' in resilience_section:
                                resilience_config.update(resilience_section['retry'])
                            if 'circuit_breaker' in resilience_section:
                                resilience_config.update(resilience_section['circuit_breaker'])
                            if 'rate_limiting' in resilience_section:
                                rate_config = resilience_section['rate_limiting']
                                if 'calls_per_period' in rate_config:
                                    resilience_config['rate_limit_calls'] = rate_config['calls_per_period']
                                if 'period_seconds' in rate_config:
                                    resilience_config['rate_limit_period'] = rate_config['period_seconds']
                        else:
                            resilience_config = dict(resilience_section)
            except (KeyError, TypeError):
                pass
        
        return resilience_config
    
    def _init_circuit_breaker(self):
        """Initialize circuit breaker with configuration."""
        # Convert critical_latency_ms to timeout_seconds (ms to seconds)
        timeout_seconds = self.critical_latency_ms / 1000.0
        
        cb_config = CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
            timeout_seconds=timeout_seconds
        )
        self.circuit_breaker = CircuitBreaker(config=cb_config)
    
    def _init_error_recovery(self):
        """Initialize error recovery manager."""
        retry_config = RetryConfig(
            max_attempts=self.max_retries,
            base_delay=self.initial_delay,
            backoff_multiplier=self.backoff_factor,
            max_delay=self.max_delay,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=self.jitter
        )
        self.error_recovery = ErrorRecoveryManager(config=retry_config)
    
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
                max_attempts=self.max_retries,
                base_delay=self.initial_delay,
                backoff_multiplier=self.backoff_factor,
                max_delay=self.max_delay,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
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
            'circuit_breaker': self.circuit_breaker.get_metrics(),
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
            },
            'configuration': {
                'max_retries': self.resilience_config.max_retries,
                'initial_delay': self.resilience_config.initial_delay,
                'backoff_factor': self.resilience_config.backoff_factor,
                'max_delay': self.resilience_config.max_delay,
                'failure_threshold': self.resilience_config.failure_threshold,
                'recovery_timeout': self.resilience_config.recovery_timeout,
                'critical_latency_ms': self.resilience_config.critical_latency_ms
            }
        }
    
    def get_config(self) -> ResilienceConfig:
        """
        Get the current resilience configuration.
        
        Returns:
            Current ResilienceConfig instance
        """
        return self.resilience_config
    
    async def reset(self):
        """Reset all resilience components to initial state."""
        await self.circuit_breaker.reset()
        self._call_times.clear()
        logger.info("ResilienceStrategies reset to initial state")


class ResilienceStrategiesFactory:
    """Factory for creating ResilienceStrategies instances with predefined configurations."""
    
    @staticmethod
    def create_default() -> ResilienceStrategies:
        """Create ResilienceStrategies with default configuration."""
        config = ResilienceConfig.create_default()
        return ResilienceStrategies(config)
    
    @staticmethod
    def create_for_api_client(service_name: str) -> ResilienceStrategies:
        """Create ResilienceStrategies optimized for API client usage."""
        config = ResilienceConfig.create_for_api_client(service_name)
        return ResilienceStrategies(config)
    
    @staticmethod
    def create_for_database(db_name: str) -> ResilienceStrategies:
        """Create ResilienceStrategies optimized for database operations."""
        config = ResilienceConfig.create_for_database(db_name)
        return ResilienceStrategies(config)
    
    @staticmethod
    def create_for_feature_calculation() -> ResilienceStrategies:
        """Create ResilienceStrategies optimized for feature calculation."""
        config = ResilienceConfig.create_for_feature_calculation()
        return ResilienceStrategies(config)
    
    @staticmethod
    def create_from_config(config: Union[ResilienceConfig, Dict[str, Any], Any]) -> ResilienceStrategies:
        """Create ResilienceStrategies from any configuration source."""
        return ResilienceStrategies(config)
    
    @staticmethod
    def create_from_profile(profile_name: str, config: Any = None) -> ResilienceStrategies:
        """
        Create ResilienceStrategies from a named profile in the configuration.
        
        Args:
            profile_name: Name of the profile (e.g., 'api_client', 'database', 'feature_calculation')
            config: Optional configuration object containing profiles section
            
        Returns:
            ResilienceStrategies configured for the specified profile
        """
        if config is None:
            # Load default system config
            try:
                from main.config import get_config_manager
                manager = get_config_manager()
                config = manager.load_config('defaults/system')
            except ImportError:
                # Fallback to default config if config manager not available
                return ResilienceStrategiesFactory.create_default()
        
        # Extract profile configuration
        profile_config = {}
        
        try:
            if hasattr(config, 'resilience') and hasattr(config.resilience, 'profiles'):
                profiles = config.resilience.profiles
                if hasattr(profiles, profile_name):
                    profile_section = getattr(profiles, profile_name)
                    
                    # Flatten profile structure similar to main config extraction
                    if hasattr(profile_section, 'retry'):
                        profile_config.update(dict(profile_section.retry))
                    if hasattr(profile_section, 'circuit_breaker'):
                        profile_config.update(dict(profile_section.circuit_breaker))
                    if hasattr(profile_section, 'rate_limiting'):
                        rate_config = dict(profile_section.rate_limiting)
                        if 'calls_per_period' in rate_config:
                            profile_config['rate_limit_calls'] = rate_config['calls_per_period']
                        if 'period_seconds' in rate_config:
                            profile_config['rate_limit_period'] = rate_config['period_seconds']
                            
        except (AttributeError, TypeError):
            logger.warning(f"Could not find profile '{profile_name}' in configuration, using defaults")
        
        if not profile_config:
            # Fallback to factory method based on profile name
            if profile_name == 'api_client':
                return ResilienceStrategiesFactory.create_for_api_client(profile_name)
            elif profile_name == 'database':
                return ResilienceStrategiesFactory.create_for_database(profile_name)
            elif profile_name == 'feature_calculation':
                return ResilienceStrategiesFactory.create_for_feature_calculation()
            else:
                return ResilienceStrategiesFactory.create_default()
        
        resilience_config = ResilienceConfig.from_dict(profile_config)
        return ResilienceStrategies(resilience_config)


# Global factory instance for convenience
_global_factory = ResilienceStrategiesFactory()


def get_resilience_strategies_factory() -> ResilienceStrategiesFactory:
    """Get the global ResilienceStrategiesFactory instance."""
    return _global_factory


def create_resilience_strategies(
    profile: Optional[str] = None, 
    config: Optional[Union[ResilienceConfig, Dict[str, Any], Any]] = None
) -> ResilienceStrategies:
    """
    Convenience function to create ResilienceStrategies.
    
    Args:
        profile: Optional profile name ('api_client', 'database', 'feature_calculation')  
        config: Optional configuration object
        
    Returns:
        Configured ResilienceStrategies instance
    """
    factory = get_resilience_strategies_factory()
    
    if profile:
        return factory.create_from_profile(profile, config)
    elif config:
        return factory.create_from_config(config)
    else:
        return factory.create_default()