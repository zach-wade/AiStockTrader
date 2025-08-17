"""
UtilityManager - Centralized Management for Utility Classes

This module provides a centralized factory and manager for utility classes used
throughout the trading system including circuit breakers, caches, and monitors.
"""

# Standard library imports
import logging
import threading
from typing import Any, TypeVar

# Local imports
from main.utils.monitoring.memory import get_memory_monitor
from main.utils.resilience import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class UtilityManager:
    """
    Centralized manager for utility classes and shared resources.

    Provides factory methods and singleton management for:
    - Circuit breakers
    - Cache instances
    - Memory monitors
    - Resilience managers
    """

    def __init__(self):
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._resilience_managers: dict[str, ResilienceStrategies] = {}
        self._lock = threading.RLock()
        self._cache_initialized = False

        logger.info("UtilityManager initialized")

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker instance.

        Args:
            name: Unique name for the circuit breaker
            config: Optional configuration (uses default if None)

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._circuit_breakers:
                if config is None:
                    config = CircuitBreakerConfig()

                self._circuit_breakers[name] = CircuitBreaker(config)
                logger.debug(f"Created circuit breaker: {name}")

            return self._circuit_breakers[name]

    def get_resilience_manager(
        self, name: str, config: dict[str, Any] | None = None
    ) -> ResilienceStrategies:
        """
        Get or create a resilience manager instance.

        Args:
            name: Unique name for the resilience manager
            config: Optional configuration dictionary

        Returns:
            ResilienceStrategies instance
        """
        with self._lock:
            if name not in self._resilience_managers:
                self._resilience_managers[name] = ResilienceStrategies(config)
                logger.debug(f"Created resilience manager: {name}")

            return self._resilience_managers[name]

    def get_cache(self, config_dict: dict[str, Any] | None = None):
        """
        Get the global cache instance.

        Args:
            config_dict: Optional configuration override

        Returns:
            MarketDataCache instance
        """
        if not self._cache_initialized:
            with self._lock:
                if not self._cache_initialized:
                    initialize_global_cache(config_dict)
                    self._cache_initialized = True

        return get_global_cache()

    def get_memory_monitor(self):
        """
        Get the global memory monitor instance.

        Returns:
            MemoryMonitor instance
        """
        return get_memory_monitor()

    def create_default_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """
        Create a circuit breaker with default configuration for a service.

        Args:
            service_name: Name of the service

        Returns:
            CircuitBreaker instance with service-appropriate defaults
        """
        # Default configs for different service types
        default_configs = {
            "database": CircuitBreakerConfig(
                failure_threshold=5, timeout_seconds=60.0, critical_latency_ms=2000.0
            ),
            "api": CircuitBreakerConfig(
                failure_threshold=3, timeout_seconds=30.0, critical_latency_ms=5000.0
            ),
            "websocket": CircuitBreakerConfig(
                failure_threshold=3, timeout_seconds=30.0, critical_latency_ms=1000.0
            ),
            "file": CircuitBreakerConfig(
                failure_threshold=2, timeout_seconds=15.0, critical_latency_ms=500.0
            ),
        }

        # Determine service type from name
        service_type = "api"  # default
        for stype in default_configs:
            if stype in service_name.lower():
                service_type = stype
                break

        config = default_configs[service_type]
        return self.get_circuit_breaker(service_name, config)

    def create_default_resilience_manager(self, service_name: str) -> ResilienceStrategies:
        """
        Create a resilience manager with default configuration for a service.

        Args:
            service_name: Name of the service

        Returns:
            ResilienceStrategies instance with service-appropriate defaults
        """
        # Default configs for different service types
        default_configs = {
            "database": {
                "max_retries": 5,
                "initial_delay": 1.0,
                "backoff_factor": 2.0,
                "max_delay": 30.0,
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "jitter": True,
            },
            "api": {
                "max_retries": 3,
                "initial_delay": 2.0,
                "backoff_factor": 2.0,
                "max_delay": 60.0,
                "failure_threshold": 3,
                "recovery_timeout": 30,
                "jitter": True,
                "rate_limit_calls": 100,
                "rate_limit_period": 60,
            },
            "websocket": {
                "max_retries": 3,
                "initial_delay": 1.0,
                "backoff_factor": 1.5,
                "max_delay": 30.0,
                "failure_threshold": 3,
                "recovery_timeout": 30,
                "jitter": False,
            },
            "file": {
                "max_retries": 3,
                "initial_delay": 0.1,
                "backoff_factor": 1.5,
                "max_delay": 5.0,
                "failure_threshold": 2,
                "recovery_timeout": 15,
                "jitter": False,
            },
        }

        # Determine service type from name
        service_type = "api"  # default
        for stype in default_configs:
            if stype in service_name.lower():
                service_type = stype
                break

        config = default_configs[service_type]
        return self.get_resilience_manager(service_name, config)

    def reset_circuit_breaker(self, name: str) -> bool:
        """
        Reset a circuit breaker to closed state.

        Args:
            name: Name of the circuit breaker

        Returns:
            True if reset successfully
        """
        with self._lock:
            if name in self._circuit_breakers:
                breaker = self._circuit_breakers[name]
                breaker.reset()
                logger.info(f"Reset circuit breaker: {name}")
                return True

            logger.warning(f"Circuit breaker not found: {name}")
            return False

    def get_circuit_breaker_stats(self, name: str | None = None) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Args:
            name: Optional specific circuit breaker name

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if name:
                if name in self._circuit_breakers:
                    return {name: self._circuit_breakers[name].get_stats()}
                else:
                    return {}
            else:
                return {
                    name: breaker.get_stats() for name, breaker in self._circuit_breakers.items()
                }

    def get_overall_stats(self) -> dict[str, Any]:
        """
        Get overall utility manager statistics.

        Returns:
            Comprehensive statistics dictionary
        """
        with self._lock:
            return {
                "circuit_breakers": {
                    "count": len(self._circuit_breakers),
                    "instances": list(self._circuit_breakers.keys()),
                    "stats": self.get_circuit_breaker_stats(),
                },
                "resilience_managers": {
                    "count": len(self._resilience_managers),
                    "instances": list(self._resilience_managers.keys()),
                },
                "cache_initialized": self._cache_initialized,
                "memory_monitor_available": True,  # Always available
            }

    def cleanup(self):
        """
        Cleanup all managed resources.
        """
        with self._lock:
            logger.info("Cleaning up UtilityManager")

            # Clear all circuit breakers
            self._circuit_breakers.clear()

            # Clear resilience managers
            self._resilience_managers.clear()

            # Reset cache flag
            self._cache_initialized = False

            logger.info("UtilityManager cleanup completed")


# Global utility manager instance
_utility_manager: UtilityManager | None = None
_manager_lock = threading.Lock()


def get_utility_manager() -> UtilityManager:
    """
    Get the global utility manager instance.

    Returns:
        UtilityManager singleton instance
    """
    global _utility_manager

    if _utility_manager is None:
        with _manager_lock:
            if _utility_manager is None:
                _utility_manager = UtilityManager()

    return _utility_manager


def reset_utility_manager():
    """
    Reset the global utility manager (primarily for testing).
    """
    global _utility_manager

    with _manager_lock:
        if _utility_manager:
            _utility_manager.cleanup()
        _utility_manager = None


# Convenience functions for common operations
def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Convenience function to get a circuit breaker."""
    return get_utility_manager().get_circuit_breaker(name, config)


def get_resilience_manager(name: str, config: dict[str, Any] | None = None) -> ResilienceStrategies:
    """Convenience function to get a resilience manager."""
    return get_utility_manager().get_resilience_manager(name, config)


def get_cache(config_dict: dict[str, Any] | None = None):
    """Convenience function to get the cache."""
    return get_utility_manager().get_cache(config_dict)


def get_memory_monitor():
    """Convenience function to get the memory monitor."""
    return get_utility_manager().get_memory_monitor()


# Circuit breaker factory functions for common use cases
def create_api_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Create a circuit breaker optimized for API calls."""
    return get_utility_manager().create_default_circuit_breaker(f"api_{service_name}")


def create_database_circuit_breaker(db_name: str) -> CircuitBreaker:
    """Create a circuit breaker optimized for database operations."""
    return get_utility_manager().create_default_circuit_breaker(f"database_{db_name}")


def create_websocket_circuit_breaker(ws_name: str) -> CircuitBreaker:
    """Create a circuit breaker optimized for WebSocket connections."""
    return get_utility_manager().create_default_circuit_breaker(f"websocket_{ws_name}")


def create_file_circuit_breaker(operation_name: str) -> CircuitBreaker:
    """Create a circuit breaker optimized for file operations."""
    return get_utility_manager().create_default_circuit_breaker(f"file_{operation_name}")


# Resilience manager factory functions
def create_api_resilience_manager(service_name: str) -> ResilienceStrategies:
    """Create a resilience manager optimized for API calls."""
    return get_utility_manager().create_default_resilience_manager(f"api_{service_name}")


def create_database_resilience_manager(db_name: str) -> ResilienceStrategies:
    """Create a resilience manager optimized for database operations."""
    return get_utility_manager().create_default_resilience_manager(f"database_{db_name}")


def create_websocket_resilience_manager(ws_name: str) -> ResilienceStrategies:
    """Create a resilience manager optimized for WebSocket connections."""
    return get_utility_manager().create_default_resilience_manager(f"websocket_{ws_name}")


def create_file_resilience_manager(operation_name: str) -> ResilienceStrategies:
    """Create a resilience manager optimized for file operations."""
    return get_utility_manager().create_default_resilience_manager(f"file_{operation_name}")
