"""
Circuit Breaker Pattern Implementation

This module provides circuit breaker functionality for fault tolerance and system resilience.
Prevents cascading failures by monitoring service health and temporarily failing fast
when downstream services are experiencing issues.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying recovery
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_seconds: float = 30.0  # Operation timeout
    expected_exception: tuple = (Exception,)  # Exceptions to count as failures

    # Sliding window configuration
    window_size: int = 100  # Size of the sliding window
    minimum_throughput: int = 10  # Minimum requests before considering failure rate


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics and statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED

    # Sliding window for recent requests
    recent_requests: list = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self.recent_requests:
            return 0.0

        failures = sum(1 for success in self.recent_requests if not success)
        return failures / len(self.recent_requests)

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Exception raised when operation times out."""

    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Monitors service calls and opens the circuit when failure threshold is reached.
    Provides fail-fast behavior when circuit is open and automatic recovery testing.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
        self._half_open_success_count = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.

        Args:
            func: Function to call (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function call

        Raises:
            CircuitBreakerError: When circuit is open
            CircuitBreakerTimeoutError: When operation times out
        """
        async with self._lock:
            await self._check_state()

        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker is open. Last failure: {self.metrics.last_failure_time}"
            )

        start_time = time.time()

        try:
            # Execute function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )

            # Record success
            async with self._lock:
                await self._record_success()

            return result

        except TimeoutError:
            async with self._lock:
                await self._record_failure()
            raise CircuitBreakerTimeoutError(
                f"Operation timed out after {self.config.timeout_seconds} seconds"
            )

        except self.config.expected_exception:
            async with self._lock:
                await self._record_failure()
            raise

    async def _check_state(self):
        """Check and update circuit breaker state."""
        current_time = time.time()

        if self.state == CircuitBreakerState.OPEN:
            # Check if we should move to half-open
            if (
                self.metrics.last_failure_time
                and current_time - self.metrics.last_failure_time >= self.config.recovery_timeout
            ):
                self.state = CircuitBreakerState.HALF_OPEN
                self._half_open_success_count = 0
                logger.info("Circuit breaker moved to HALF_OPEN state")

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Half-open state is handled in success/failure recording
            pass

    async def _record_success(self):
        """Record a successful operation."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()

        # Update sliding window
        self.metrics.recent_requests.append(True)
        if len(self.metrics.recent_requests) > self.config.window_size:
            self.metrics.recent_requests.pop(0)

        if self.state == CircuitBreakerState.HALF_OPEN:
            self._half_open_success_count += 1

            if self._half_open_success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.metrics.circuit_closed_count += 1
                self._half_open_success_count = 0
                logger.info("Circuit breaker moved to CLOSED state after successful recovery")

    async def _record_failure(self):
        """Record a failed operation."""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()

        # Update sliding window
        self.metrics.recent_requests.append(False)
        if len(self.metrics.recent_requests) > self.config.window_size:
            self.metrics.recent_requests.pop(0)

        # Check if we should open the circuit
        if self.state == CircuitBreakerState.CLOSED:
            if len(
                self.metrics.recent_requests
            ) >= self.config.minimum_throughput and self.metrics.failure_rate >= (
                self.config.failure_threshold / self.config.window_size
            ):
                self.state = CircuitBreakerState.OPEN
                self.metrics.circuit_opened_count += 1
                logger.warning(
                    f"Circuit breaker opened due to high failure rate: {self.metrics.failure_rate:.2%}"
                )

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state returns to open
            self.state = CircuitBreakerState.OPEN
            self.metrics.circuit_opened_count += 1
            self._half_open_success_count = 0
            logger.warning("Circuit breaker returned to OPEN state after failure during recovery")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": round(self.metrics.success_rate * 100, 2),
            "failure_rate": round(self.metrics.failure_rate * 100, 2),
            "circuit_opened_count": self.metrics.circuit_opened_count,
            "circuit_closed_count": self.metrics.circuit_closed_count,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "window_size": self.config.window_size,
                "minimum_throughput": self.config.minimum_throughput,
            },
        }

    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        return self.state != CircuitBreakerState.OPEN

    async def reset(self):
        """Reset circuit breaker to initial state."""
        async with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self._half_open_success_count = 0
            logger.info("Circuit breaker reset to initial state")


def circuit_breaker(config: CircuitBreakerConfig | None = None):
    """
    Decorator to apply circuit breaker pattern to functions.

    Args:
        config: Circuit breaker configuration

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))

        # Attach circuit breaker for monitoring
        if asyncio.iscoroutinefunction(func):
            async_wrapper._circuit_breaker = breaker
            return async_wrapper
        else:
            sync_wrapper._circuit_breaker = breaker
            return sync_wrapper

    return decorator


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers by name.

    Useful for having different circuit breakers for different services
    or operations with different failure characteristics.
    """

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(config)
                logger.info(f"Created circuit breaker: {name}")
            return self.breakers[name]

    async def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Call function through named circuit breaker."""
        breaker = await self.get_breaker(name)
        return await breaker.call(func, *args, **kwargs)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}

    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._lock:
            for name, breaker in self.breakers.items():
                await breaker.reset()
                logger.info(f"Reset circuit breaker: {name}")

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary of all circuit breakers."""
        total_breakers = len(self.breakers)
        if total_breakers == 0:
            return {"status": "no_breakers", "total": 0}

        states = [breaker.get_state() for breaker in self.breakers.values()]
        open_count = sum(1 for state in states if state == CircuitBreakerState.OPEN)
        half_open_count = sum(1 for state in states if state == CircuitBreakerState.HALF_OPEN)
        closed_count = sum(1 for state in states if state == CircuitBreakerState.CLOSED)

        if open_count == 0:
            status = "healthy"
        elif open_count < total_breakers / 2:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "total": total_breakers,
            "closed": closed_count,
            "open": open_count,
            "half_open": half_open_count,
            "availability": round((closed_count + half_open_count) / total_breakers * 100, 2),
        }


# Global circuit breaker manager instance
_global_manager = CircuitBreakerManager()


async def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get a circuit breaker from the global manager."""
    return await _global_manager.get_breaker(name, config)


async def circuit_breaker_call(name: str, func: Callable, *args, **kwargs) -> Any:
    """Call function through global circuit breaker manager."""
    return await _global_manager.call(name, func, *args, **kwargs)


def get_global_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    return _global_manager
