"""
Circuit Breaker Pattern Implementation

Production-grade circuit breaker for external service calls including
broker APIs, market data providers, and database connections.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    # Failure thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open

    # Time windows
    window_size: int = 10  # Sliding window size for failure tracking
    half_open_max_calls: int = 3  # Max concurrent calls in half-open

    # Recovery settings
    recovery_timeout: float = 300.0  # Time to wait before automatic recovery
    exponential_backoff: bool = True  # Use exponential backoff for timeout
    max_timeout: float = 3600.0  # Maximum timeout in exponential backoff

    # Monitoring
    failure_types: tuple[type[Exception], ...] = (
        Exception,
    )  # Exception types that count as failures

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState) -> None:
        self.name = name
        self.state = state
        super().__init__(f"Circuit breaker '{name}' is {state.value}")


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.

    Features:
    - Thread-safe operation
    - Sliding window failure tracking
    - Exponential backoff recovery
    - Concurrent call limiting in half-open state
    - Comprehensive metrics and monitoring
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration (defaults to CircuitBreakerConfig())
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self._state_changed_at = time.time()

        # Failure tracking (sliding window)
        self._failures: deque[float] = deque(maxlen=self.config.window_size)
        self._successes = 0

        # Half-open state management
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Exponential backoff
        self._failure_count = 0
        self._last_failure_time = 0.0

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_timeouts = 0
        self._state_changes: dict[str, int] = defaultdict(int)

        logger.info(f"Initialized circuit breaker '{name}' with config: {config}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self._state == CircuitState.HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset to half-open."""
        if self._state != CircuitState.OPEN:
            return False

        now = time.time()
        time_since_open = now - self._state_changed_at

        if self.config.exponential_backoff:
            # Exponential backoff: timeout * (2 ^ failure_count)
            backoff_timeout = min(
                self.config.timeout * (2 ** min(self._failure_count, 10)), self.config.max_timeout
            )
            return bool(time_since_open >= backoff_timeout)
        else:
            return time_since_open >= self.config.timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        now = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Remove old failures from sliding window
            cutoff_time = now - 60.0  # 1 minute window
            while self._failures and self._failures[0] < cutoff_time:
                self._failures.popleft()

        self._total_successes += 1
        self._successes += 1

        logger.debug(f"Circuit breaker '{self.name}': Success recorded. State: {self._state}")

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        now = time.time()

        # Check if this exception type should count as a failure
        if not isinstance(exception, self.config.failure_types):
            logger.debug(
                f"Circuit breaker '{self.name}': Exception {type(exception)} not counted as failure"
            )
            return

        self._failures.append(now)
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = now

        # Check if we should open the circuit
        if self._state == CircuitState.CLOSED:
            recent_failures = len(self._failures)
            if recent_failures >= self.config.failure_threshold:
                self._transition_to_open()
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()

        logger.debug(
            f"Circuit breaker '{self.name}': Failure recorded. State: {self._state}, Recent failures: {len(self._failures)}"
        )

    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        if self._state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}': Opening circuit due to failures")
            self._state = CircuitState.OPEN
            self._state_changed_at = time.time()
            self._state_changes["open"] += 1
            self._half_open_calls = 0
            self._half_open_successes = 0

    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        if self._state == CircuitState.OPEN:
            logger.info(f"Circuit breaker '{self.name}': Attempting recovery (half-open)")
            self._state = CircuitState.HALF_OPEN
            self._state_changed_at = time.time()
            self._state_changes["half_open"] += 1
            self._half_open_calls = 0
            self._half_open_successes = 0

    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        if self._state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.name}': Circuit recovered (closed)")
            self._state = CircuitState.CLOSED
            self._state_changed_at = time.time()
            self._state_changes["closed"] += 1
            self._failure_count = 0  # Reset exponential backoff counter
            self._failures.clear()
            self._half_open_calls = 0
            self._half_open_successes = 0

    def _can_execute(self) -> bool:
        """Check if we can execute a call."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                    return True
                return False
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function call through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the wrapped function
        """
        self._total_calls += 1

        if not self._can_execute():
            self._total_timeouts += 1
            raise CircuitBreakerError(self.name, self._state)

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result

        except Exception as e:
            with self._lock:
                self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Execute an async function call through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the wrapped function
        """
        self._total_calls += 1

        if not self._can_execute():
            self._total_timeouts += 1
            raise CircuitBreakerError(self.name, self._state)

        try:
            result = await func(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result

        except Exception as e:
            with self._lock:
                self._record_failure(e)
            raise

    def __call__(self, func: F) -> F:
        """
        Decorator for wrapping functions with circuit breaker.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.call_async(func, *args, **kwargs)

            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.call(func, *args, **kwargs)

            return cast(F, wrapper)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get circuit breaker metrics.

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            now = time.time()
            return {
                "name": self.name,
                "state": self._state.value,
                "state_changed_at": self._state_changed_at,
                "time_in_current_state": now - self._state_changed_at,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "total_timeouts": self._total_timeouts,
                "success_rate": self._total_successes / max(self._total_calls, 1),
                "failure_rate": self._total_failures / max(self._total_calls, 1),
                "recent_failures": len(self._failures),
                "failure_threshold": self.config.failure_threshold,
                "half_open_calls": self._half_open_calls,
                "half_open_successes": self._half_open_successes,
                "state_changes": dict(self._state_changes),
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}': Manual reset")
            self._transition_to_closed()
            self._total_calls = 0
            self._total_failures = 0
            self._total_successes = 0
            self._total_timeouts = 0
            self._state_changes.clear()

    def __str__(self) -> str:
        """String representation."""
        return f"CircuitBreaker(name='{self.name}', state={self._state.value})"


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._instance_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_or_create(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """
        Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration (used only for new breakers)

        Returns:
            CircuitBreaker instance
        """
        with self._instance_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def list_names(self) -> list[str]:
        """List all circuit breaker names."""
        return list(self._breakers.keys())

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._instance_lock:
            return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._instance_lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def clear(self) -> None:
        """Clear all circuit breakers (for testing)."""
        with self._instance_lock:
            self._breakers.clear()
