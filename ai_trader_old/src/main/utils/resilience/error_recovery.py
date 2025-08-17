"""
Error Recovery and Retry Logic

This module provides comprehensive error recovery mechanisms including retry logic,
exponential backoff, jitter, and resilience patterns for handling transient failures
in distributed systems and external API calls.
"""

# Standard library imports
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging
import time
from typing import Any, TypeVar

# Local imports
from main.utils.core import secure_uniform  # DEPRECATED - use secure_random

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"


class RecoveryAction(Enum):
    """Recovery action types."""

    RETRY = "retry"
    FAIL = "fail"
    FALLBACK = "fallback"
    SKIP = "skip"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_max: float = 0.1
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Exception handling
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    non_retryable_exceptions: tuple[type[Exception], ...] = ()

    # Conditions
    retry_if: Callable[[Exception], bool] | None = None
    stop_if: Callable[[Exception], bool] | None = None

    # Custom delay function
    custom_delay_func: Callable[[int, Exception], float] | None = None


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay_time: float = 0.0
    exception_counts: dict[str, int] = field(default_factory=dict)
    last_success_time: float | None = None
    last_failure_time: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    @property
    def avg_delay_per_attempt(self) -> float:
        """Calculate average delay per attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_delay_time / self.total_attempts


class RetryExhaustedError(Exception):
    """Exception raised when retry attempts are exhausted."""

    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and retry logic.

    Provides flexible retry mechanisms with various backoff strategies,
    exception filtering, and recovery actions.
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize error recovery manager."""
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()
        self._recovery_handlers: dict[type[Exception], Callable] = {}
        self._fallback_handlers: dict[str, Callable] = {}

    async def execute_with_retry(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of successful function execution

        Raises:
            RetryExhaustedError: When all retry attempts are exhausted
        """
        last_exception = None
        total_delay = 0.0

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)

                # Record success
                self.metrics.total_attempts += 1
                self.metrics.successful_attempts += 1
                self.metrics.total_delay_time += total_delay
                self.metrics.last_success_time = time.time()

                logger.debug(f"Function succeeded on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e
                self.metrics.total_attempts += 1
                self.metrics.failed_attempts += 1
                self.metrics.last_failure_time = time.time()

                # Track exception types
                exception_name = type(e).__name__
                self.metrics.exception_counts[exception_name] = (
                    self.metrics.exception_counts.get(exception_name, 0) + 1
                )

                # Check if we should retry
                if not self._should_retry(e, attempt):
                    logger.warning(f"Not retrying after attempt {attempt}: {e}")
                    break

                # Calculate delay
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt, e)
                    total_delay += delay

                    logger.info(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")

        # All attempts exhausted
        self.metrics.total_delay_time += total_delay

        error_msg = f"Failed after {self.config.max_attempts} attempts"
        raise RetryExhaustedError(error_msg, self.config.max_attempts, last_exception)

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception and attempt count."""

        # Check attempt limit
        if attempt >= self.config.max_attempts:
            return False

        # Check non-retryable exceptions
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check custom stop condition
        if self.config.stop_if and self.config.stop_if(exception):
            return False

        # Check retryable exceptions
        if not isinstance(exception, self.config.retryable_exceptions):
            return False

        # Check custom retry condition
        if self.config.retry_if and not self.config.retry_if(exception):
            return False

        return True

    def _calculate_delay(self, attempt: int, exception: Exception) -> float:
        """Calculate delay before next retry attempt."""

        # Use custom delay function if provided
        if self.config.custom_delay_func:
            delay = self.config.custom_delay_func(attempt, exception)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        else:
            delay = self.config.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_max
            delay += secure_uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def register_recovery_handler(
        self, exception_type: type[Exception], handler: Callable[[Exception], Any]
    ):
        """Register a recovery handler for specific exception types."""
        self._recovery_handlers[exception_type] = handler
        logger.debug(f"Registered recovery handler for {exception_type.__name__}")

    def register_fallback_handler(self, name: str, handler: Callable[..., Any]):
        """Register a fallback handler by name."""
        self._fallback_handlers[name] = handler
        logger.debug(f"Registered fallback handler: {name}")

    async def execute_with_fallback(
        self, func: Callable[..., Awaitable[T]], fallback_name: str, *args, **kwargs
    ) -> T:
        """
        Execute function with fallback on failure.

        Args:
            func: Primary function to execute
            fallback_name: Name of registered fallback handler
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from primary function or fallback
        """
        try:
            return await self.execute_with_retry(func, *args, **kwargs)
        except RetryExhaustedError:
            logger.warning(f"Primary function failed, using fallback: {fallback_name}")

            if fallback_name in self._fallback_handlers:
                fallback_func = self._fallback_handlers[fallback_name]
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            else:
                logger.error(f"Fallback handler not found: {fallback_name}")
                raise

    def get_metrics(self) -> dict[str, Any]:
        """Get retry metrics."""
        return {
            "total_attempts": self.metrics.total_attempts,
            "successful_attempts": self.metrics.successful_attempts,
            "failed_attempts": self.metrics.failed_attempts,
            "success_rate": round(self.metrics.success_rate * 100, 2),
            "total_delay_time": round(self.metrics.total_delay_time, 2),
            "avg_delay_per_attempt": round(self.metrics.avg_delay_per_attempt, 2),
            "exception_counts": self.metrics.exception_counts,
            "last_success_time": self.metrics.last_success_time,
            "last_failure_time": self.metrics.last_failure_time,
        }

    def reset_metrics(self):
        """Reset retry metrics."""
        self.metrics = RetryMetrics()
        logger.debug("Retry metrics reset")


def retry(config: RetryConfig | None = None):
    """
    Decorator to add retry logic to functions.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry capability
    """

    def decorator(func: Callable) -> Callable:
        recovery_manager = ErrorRecoveryManager(config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await recovery_manager.execute_with_retry(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            async def async_func():
                return func(*args, **kwargs)

            return asyncio.run(recovery_manager.execute_with_retry(async_func))

        # Attach recovery manager for metrics access
        if asyncio.iscoroutinefunction(func):
            async_wrapper._recovery_manager = recovery_manager
            return async_wrapper
        else:
            sync_wrapper._recovery_manager = recovery_manager
            return sync_wrapper

    return decorator


class BulkRetryManager:
    """
    Manages retry operations for bulk/batch operations.

    Provides strategies for handling partial failures in batch operations,
    including retry of individual items and failure isolation.
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize bulk retry manager."""
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()

    async def execute_bulk_with_retry(
        self,
        items: list[T],
        func: Callable[[T], Awaitable[Any]],
        max_concurrent: int = 10,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """
        Execute bulk operation with retry logic.

        Args:
            items: List of items to process
            func: Function to apply to each item
            max_concurrent: Maximum concurrent operations
            fail_fast: Stop on first failure

        Returns:
            Dictionary with results, failures, and metrics
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        failures = []

        async def process_item(item: T, index: int) -> tuple[int, bool, Any]:
            """Process single item with retry logic."""
            async with semaphore:
                recovery_manager = ErrorRecoveryManager(self.config)
                try:
                    result = await recovery_manager.execute_with_retry(func, item)
                    return index, True, result
                except RetryExhaustedError as e:
                    return index, False, e

        # Execute all items
        tasks = [process_item(item, i) for i, item in enumerate(items)]

        if fail_fast:
            # Stop on first failure
            for coro in asyncio.as_completed(tasks):
                index, success, result = await coro
                if success:
                    results.append((index, result))
                else:
                    failures.append((index, result))
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
        else:
            # Process all items regardless of failures
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            for task_result in completed_tasks:
                if isinstance(task_result, Exception):
                    failures.append((-1, task_result))
                else:
                    index, success, result = task_result
                    if success:
                        results.append((index, result))
                    else:
                        failures.append((index, result))

        return {
            "total_items": len(items),
            "successful_items": len(results),
            "failed_items": len(failures),
            "success_rate": len(results) / len(items) * 100 if items else 0,
            "results": results,
            "failures": failures,
        }


@asynccontextmanager
async def error_recovery_context(config: RetryConfig | None = None):
    """
    Context manager for error recovery operations.

    Provides automatic error recovery within a context block.
    """
    manager = ErrorRecoveryManager(config)
    try:
        yield manager
    except Exception as e:
        logger.error(f"Error in recovery context: {e}")
        raise


# Predefined retry configurations
NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(ValueError, TypeError),
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    backoff_multiplier=1.5,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(Exception,),
    non_retryable_exceptions=(ValueError, TypeError),
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(ConnectionError, TimeoutError),
    non_retryable_exceptions=(ValueError, TypeError, KeyError),
)


# Global recovery manager instance
_global_recovery_manager = ErrorRecoveryManager()


def get_global_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    return _global_recovery_manager


async def retry_call(
    func: Callable[..., Awaitable[T]], *args, config: RetryConfig | None = None, **kwargs
) -> T:
    """
    Convenience function for retry calls.

    Args:
        func: Function to call with retry
        *args: Positional arguments
        config: Retry configuration
        **kwargs: Keyword arguments

    Returns:
        Result of successful function call
    """
    manager = ErrorRecoveryManager(config)
    return await manager.execute_with_retry(func, *args, **kwargs)


class BulkRetryManager:
    """
    Manager for bulk retry operations with configurable retry policies.

    This class handles retrying operations on collections of items, with
    individual retry logic for each item and bulk failure handling.
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize bulk retry manager.

        Args:
            config: Retry configuration to use for operations
        """
        self.config = config
        self.metrics = RetryMetrics()
        self._logger = logger

    async def execute_bulk_operation(
        self,
        items: list[Any],
        operation_func: Callable[[Any], Awaitable[Any]],
        max_concurrent: int = 10,
    ) -> dict[str, Any]:
        """
        Execute an operation on a collection of items with retry logic.

        Args:
            items: List of items to process
            operation_func: Async function to execute for each item
            max_concurrent: Maximum concurrent operations

        Returns:
            Dictionary with results and failure information
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        failures = []

        async def process_item_with_retry(item, index):
            """Process a single item with retry logic."""
            async with semaphore:
                for attempt in range(self.config.max_attempts):
                    try:
                        result = await operation_func(item)
                        self.metrics.successful_attempts += 1
                        return (index, True, result)

                    except Exception as e:
                        self.metrics.failed_attempts += 1
                        self.metrics.total_attempts += 1

                        # Check if exception is retryable
                        if not self._is_retryable_exception(e):
                            return (index, False, e)

                        # If this is the last attempt, return failure
                        if attempt == self.config.max_attempts - 1:
                            return (index, False, e)

                        # Calculate delay for next attempt
                        delay = self._calculate_delay(attempt, e)
                        self.metrics.total_delay_time += delay

                        self._logger.debug(
                            f"Retry attempt {attempt + 1}/{self.config.max_attempts} "
                            f"for item {index} after {delay}s delay: {e}"
                        )

                        await asyncio.sleep(delay)

        # Create tasks for all items
        tasks = [process_item_with_retry(item, i) for i, item in enumerate(items)]

        # Execute all tasks
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in completed_results:
            if isinstance(result, Exception):
                failures.append((-1, result))
            else:
                index, success, data = result
                if success:
                    results.append((index, data))
                else:
                    failures.append((index, data))

        # Update metrics
        self.metrics.last_success_time = time.time() if results else None
        self.metrics.last_failure_time = time.time() if failures else None

        return {
            "total_items": len(items),
            "successful_items": len(results),
            "failed_items": len(failures),
            "success_rate": len(results) / len(items) * 100 if items else 0,
            "results": results,
            "failures": failures,
            "metrics": self.metrics,
        }

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if an exception is retryable based on configuration."""
        # Check non-retryable exceptions first
        if self.config.non_retryable_exceptions:
            if isinstance(exception, self.config.non_retryable_exceptions):
                return False

        # Check retryable exceptions
        if self.config.retryable_exceptions:
            if not isinstance(exception, self.config.retryable_exceptions):
                return False

        # Check custom retry conditions
        if self.config.retry_if and not self.config.retry_if(exception):
            return False

        if self.config.stop_if and self.config.stop_if(exception):
            return False

        return True

    def _calculate_delay(self, attempt: int, exception: Exception) -> float:
        """Calculate delay for retry attempt."""
        if self.config.custom_delay_func:
            return self.config.custom_delay_func(attempt, exception)

        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)
        else:
            delay = self.config.base_delay

        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_max * secure_uniform(0, 1)
            delay += jitter_amount

        # Ensure delay doesn't exceed maximum
        return min(delay, self.config.max_delay)

    def get_metrics(self) -> RetryMetrics:
        """Get current retry metrics."""
        return self.metrics

    def reset_metrics(self):
        """Reset retry metrics."""
        self.metrics = RetryMetrics()
