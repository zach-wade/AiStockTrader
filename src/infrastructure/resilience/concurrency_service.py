"""
Concurrency Service for handling concurrent operations.

Since threading was removed from Portfolio entity, this service provides
basic concurrency utilities without thread locks.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from src.domain.exceptions import DeadlockException, OptimisticLockException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConcurrencyService:
    """
    Service for handling concurrency-related operations.

    Provides retry mechanisms and deadlock detection for database operations.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        Initialize the concurrency service.

        Args:
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def async_retry_on_version_conflict(self, operation: Callable[[], Awaitable[T]]) -> T:
        """
        Retry an async operation on version conflicts (optimistic locking failures).

        Args:
            operation: The async operation to retry

        Returns:
            The result of the operation

        Raises:
            OptimisticLockException: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except OptimisticLockException as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Optimistic lock conflict on attempt {attempt + 1}, retrying..."
                    )
                    await asyncio.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error("All retry attempts exhausted for version conflict")
                    break

        raise last_exception or OptimisticLockException(
            entity_type="Unknown",
            entity_id="Unknown",
            retries=self.max_retries,
            message="Operation failed after retries",
        )

    def detect_deadlock(self, exception: Exception) -> bool:
        """
        Detect if an exception indicates a database deadlock.

        Args:
            exception: The exception to check

        Returns:
            True if the exception indicates a deadlock
        """
        if isinstance(exception, DeadlockException):
            return True

        # Check for PostgreSQL deadlock error codes
        error_message = str(exception).lower()
        deadlock_indicators = [
            "deadlock detected",
            "deadlock_detected",
            "40p01",  # PostgreSQL deadlock error code
            "could not serialize access",
            "serialization failure",
        ]

        return any(indicator in error_message for indicator in deadlock_indicators)

    def retry_on_deadlock(self, operation: Callable[[], T]) -> T:
        """
        Retry a synchronous operation on deadlock detection.

        Args:
            operation: The operation to retry

        Returns:
            The result of the operation

        Raises:
            DeadlockException: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                if self.detect_deadlock(e):
                    last_exception = e
                    if attempt < self.max_retries:
                        logger.warning(f"Deadlock detected on attempt {attempt + 1}, retrying...")
                        # Small delay for deadlock retries
                        import time

                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error("All retry attempts exhausted for deadlock")
                        break
                else:
                    # Re-raise non-deadlock exceptions immediately
                    raise

        if isinstance(last_exception, DeadlockException):
            raise last_exception
        elif last_exception:
            raise DeadlockException(f"Deadlock retry exhausted: {last_exception}")
        else:
            raise DeadlockException("Operation failed due to deadlock")
