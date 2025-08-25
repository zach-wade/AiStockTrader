"""
Retry Logic with Exponential Backoff

Production-grade retry mechanisms for API calls and database operations
with configurable backoff strategies, jitter, and circuit breaker integration.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Basic retry settings
    max_retries: int = 3
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier

    # Jitter settings
    jitter: bool = True  # Add random jitter to prevent thundering herd
    jitter_range: float = 0.1  # Jitter as fraction of delay (0.1 = ±10%)

    # Exception handling
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    )

    # Advanced settings
    exponential_base: float = 2.0  # Base for exponential backoff
    timeout_per_attempt: float | None = None  # Timeout for each attempt
    total_timeout: float | None = None  # Total timeout for all attempts

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.backoff_multiplier <= 1.0:
            raise ValueError("backoff_multiplier must be greater than 1.0")
        if self.jitter_range < 0 or self.jitter_range > 1:
            raise ValueError("jitter_range must be between 0 and 1")


class RetryExhaustedException(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: Exception | None, total_time: float) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        self.total_time = total_time
        error_msg = f"Last error: {last_exception}" if last_exception else "No exception recorded"
        super().__init__(
            f"Retry exhausted after {attempts} attempts in {total_time:.2f}s. " f"{error_msg}"
        )


class ExponentialBackoff:
    """Exponential backoff calculator with jitter."""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-based)

        Returns:
            Delay in seconds
        """
        if attempt < 0:
            return 0.0

        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = self.config.initial_delay * (self.config.exponential_base**attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter and delay > 0:
            jitter_amount = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay + jitter)  # Minimum 100ms delay

        return delay

    def get_delays(self, max_attempts: int) -> list[float]:
        """
        Get all delays for a retry sequence.

        Args:
            max_attempts: Maximum number of attempts

        Returns:
            List of delays in seconds
        """
        return [self.get_delay(i) for i in range(max_attempts)]


def is_retryable_exception(exception: Exception, config: RetryConfig) -> bool:
    """
    Check if an exception is retryable based on configuration.

    Args:
        exception: Exception to check
        config: Retry configuration

    Returns:
        True if exception is retryable
    """
    # Check non-retryable exceptions first (takes precedence)
    if isinstance(exception, config.non_retryable_exceptions):
        return False

    # Check retryable exceptions
    if isinstance(exception, config.retryable_exceptions):
        return True

    # Default to non-retryable for unknown exceptions
    return False


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute async function with retry and exponential backoff.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        RetryExhaustedException: When all retries are exhausted
    """
    config = config or RetryConfig()
    backoff = ExponentialBackoff(config)

    start_time = time.time()
    last_exception = None

    for attempt in range(config.max_retries + 1):  # +1 for initial attempt
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_retries + 1} for {func.__name__}")

            if config.timeout_per_attempt:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=config.timeout_per_attempt
                )
            else:
                result = await func(*args, **kwargs)

            if attempt > 0:
                total_time = time.time() - start_time
                logger.info(
                    f"Function {func.__name__} succeeded on attempt {attempt + 1} "
                    f"after {total_time:.2f}s"
                )

            return result

        except Exception as e:
            last_exception = e
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check total timeout
            if config.total_timeout and elapsed_time >= config.total_timeout:
                logger.warning(
                    f"Total timeout ({config.total_timeout}s) reached for {func.__name__}"
                )
                break

            # Check if we should retry
            if attempt >= config.max_retries:
                logger.warning(f"Max retries ({config.max_retries}) reached for {func.__name__}")
                break

            if not is_retryable_exception(e, config):
                logger.warning(f"Non-retryable exception for {func.__name__}: {e}")
                break

            # Calculate delay for next attempt
            delay = backoff.get_delay(attempt)

            # Check if delay would exceed total timeout
            if config.total_timeout and elapsed_time + delay >= config.total_timeout:
                remaining_time = config.total_timeout - elapsed_time
                if remaining_time > 0.1:  # At least 100ms remaining
                    delay = remaining_time
                else:
                    logger.warning(f"Insufficient time remaining for {func.__name__} retry")
                    break

            logger.warning(
                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s"
            )

            await asyncio.sleep(delay)

    total_time = time.time() - start_time
    raise RetryExhaustedException(
        attempts=attempt + 1, last_exception=last_exception, total_time=total_time
    )


def retry_with_backoff_sync(
    func: Callable[..., T], *args: Any, config: RetryConfig | None = None, **kwargs: Any
) -> T:
    """
    Execute sync function with retry and exponential backoff.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        RetryExhaustedException: When all retries are exhausted
    """
    config = config or RetryConfig()
    backoff = ExponentialBackoff(config)

    start_time = time.time()
    last_exception = None

    for attempt in range(config.max_retries + 1):  # +1 for initial attempt
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_retries + 1} for {func.__name__}")

            result = func(*args, **kwargs)

            if attempt > 0:
                total_time = time.time() - start_time
                logger.info(
                    f"Function {func.__name__} succeeded on attempt {attempt + 1} "
                    f"after {total_time:.2f}s"
                )

            return result

        except Exception as e:
            last_exception = e
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check total timeout
            if config.total_timeout and elapsed_time >= config.total_timeout:
                logger.warning(
                    f"Total timeout ({config.total_timeout}s) reached for {func.__name__}"
                )
                break

            # Check if we should retry
            if attempt >= config.max_retries:
                logger.warning(f"Max retries ({config.max_retries}) reached for {func.__name__}")
                break

            if not is_retryable_exception(e, config):
                logger.warning(f"Non-retryable exception for {func.__name__}: {e}")
                break

            # Calculate delay for next attempt
            delay = backoff.get_delay(attempt)

            # Check if delay would exceed total timeout
            if config.total_timeout and elapsed_time + delay >= config.total_timeout:
                remaining_time = config.total_timeout - elapsed_time
                if remaining_time > 0.1:  # At least 100ms remaining
                    delay = remaining_time
                else:
                    logger.warning(f"Insufficient time remaining for {func.__name__} retry")
                    break

            logger.warning(
                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s"
            )

            time.sleep(delay)

    total_time = time.time() - start_time
    raise RetryExhaustedException(
        attempts=attempt + 1, last_exception=last_exception, total_time=total_time
    )


def retry(
    config: RetryConfig | None = None,
    *,
    max_retries: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
) -> Callable[[F], F]:
    """
    Decorator for adding retry logic to functions.

    Args:
        config: Retry configuration
        max_retries: Override max_retries from config
        initial_delay: Override initial_delay from config
        max_delay: Override max_delay from config

    Returns:
        Decorated function
    """
    # Override config with explicit parameters
    if config is None:
        config = RetryConfig()
    else:
        # Create a copy to avoid modifying the original
        config = RetryConfig(
            max_retries=max_retries if max_retries is not None else config.max_retries,
            initial_delay=initial_delay if initial_delay is not None else config.initial_delay,
            max_delay=max_delay if max_delay is not None else config.max_delay,
            backoff_multiplier=config.backoff_multiplier,
            jitter=config.jitter,
            jitter_range=config.jitter_range,
            retryable_exceptions=config.retryable_exceptions,
            non_retryable_exceptions=config.non_retryable_exceptions,
            exponential_base=config.exponential_base,
            timeout_per_attempt=config.timeout_per_attempt,
            total_timeout=config.total_timeout,
        )

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await retry_with_backoff(func, *args, config=config, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return retry_with_backoff_sync(func, *args, config=config, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# Predefined configurations for common use cases
DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=10.0,
    backoff_multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    total_timeout=30.0,
)

API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    total_timeout=120.0,
)

BROKER_RETRY_CONFIG = RetryConfig(
    max_retries=2,  # Conservative for trading operations
    initial_delay=0.5,
    max_delay=5.0,
    backoff_multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    total_timeout=15.0,  # Fast timeout for trading
)

MARKET_DATA_RETRY_CONFIG = RetryConfig(
    max_retries=4,
    initial_delay=0.25,
    max_delay=30.0,
    backoff_multiplier=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    total_timeout=60.0,
)
