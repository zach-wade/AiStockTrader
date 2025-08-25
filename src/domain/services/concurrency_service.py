"""
Concurrency Control Service

Provides thread-safe operations and optimistic locking support for domain entities.
Implements retry mechanisms with exponential backoff for handling version conflicts.
"""

import asyncio
import logging
import random
import threading
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import UUID

from src.domain.exceptions import (
    ConcurrencyException,
    DeadlockException,
    OptimisticLockException,
    PessimisticLockException,
    StaleDataException,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConcurrencyService:
    """
    Service for managing concurrent access to domain entities.

    Provides:
    - Optimistic locking with version checking
    - Retry logic with exponential backoff
    - Pessimistic locking support
    - Deadlock detection and recovery
    - Concurrency metrics tracking
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """
        Initialize the concurrency service.

        Args:
            max_retries: Maximum number of retry attempts for version conflicts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Multiplication factor for exponential backoff
            jitter: Whether to add random jitter to retry delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        # Locks for different entities
        self._entity_locks: dict[str, threading.RLock] = {}
        self._async_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = threading.RLock()

        # Metrics
        self._version_conflicts = 0
        self._successful_retries = 0
        self._failed_retries = 0
        self._deadlocks_detected = 0
        self._lock_timeouts = 0

    def get_entity_key(self, entity_type: str, entity_id: UUID | str) -> str:
        """Generate a unique key for an entity."""
        return f"{entity_type}:{entity_id}"

    def calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt using exponential backoff.

        Args:
            attempt: Current retry attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (self.backoff_factor**attempt), self.max_delay)

        if self.jitter:
            # Add random jitter (±25% of delay)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.01, delay)  # Ensure minimum delay

    @contextmanager
    def entity_lock(
        self,
        entity_type: str,
        entity_id: UUID | str,
        timeout: float | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for pessimistic locking of an entity.

        Args:
            entity_type: Type of entity to lock
            entity_id: ID of entity to lock
            timeout: Lock acquisition timeout in seconds

        Yields:
            None

        Raises:
            PessimisticLockException: If lock cannot be acquired within timeout
        """
        entity_key = self.get_entity_key(entity_type, entity_id)

        # Get or create lock for this entity
        with self._global_lock:
            if entity_key not in self._entity_locks:
                self._entity_locks[entity_key] = threading.RLock()
            lock = self._entity_locks[entity_key]

        acquired = False
        try:
            if timeout is None:
                acquired = lock.acquire()
            else:
                acquired = lock.acquire(timeout=timeout)

            if not acquired:
                self._lock_timeouts += 1
                raise PessimisticLockException(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    timeout=timeout or 0,
                )

            yield

        finally:
            if acquired:
                lock.release()

    @asynccontextmanager
    async def async_entity_lock(
        self,
        entity_type: str,
        entity_id: UUID | str,
        timeout: float | None = None,
    ) -> AsyncGenerator[None, None]:
        """
        Async context manager for pessimistic locking of an entity.

        Args:
            entity_type: Type of entity to lock
            entity_id: ID of entity to lock
            timeout: Lock acquisition timeout in seconds

        Yields:
            None

        Raises:
            PessimisticLockException: If lock cannot be acquired within timeout
        """
        entity_key = self.get_entity_key(entity_type, entity_id)

        # Get or create lock for this entity
        if entity_key not in self._async_locks:
            self._async_locks[entity_key] = asyncio.Lock()
        lock = self._async_locks[entity_key]

        acquired = False
        try:
            if timeout is None:
                await lock.acquire()
                acquired = True
            else:
                try:
                    await asyncio.wait_for(lock.acquire(), timeout=timeout)
                    acquired = True
                except TimeoutError:
                    self._lock_timeouts += 1
                    raise PessimisticLockException(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        timeout=timeout,
                    )

            yield

        finally:
            if acquired:
                lock.release()

    def retry_on_version_conflict(
        self,
        operation: Callable[[], T],
        entity_type: str,
        entity_id: UUID | str,
    ) -> T:
        """
        Execute an operation with automatic retry on version conflicts.

        Args:
            operation: Function to execute
            entity_type: Type of entity being operated on
            entity_id: ID of entity being operated on

        Returns:
            Result of the operation

        Raises:
            OptimisticLockException: If operation fails after max retries
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = operation()

                if attempt > 0:
                    self._successful_retries += 1
                    logger.info(
                        f"Operation succeeded after {attempt} retries for "
                        f"{entity_type} {entity_id}"
                    )

                return result

            except StaleDataException as e:
                last_exception = e
                self._version_conflicts += 1

                if attempt < self.max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Version conflict for {entity_type} {entity_id} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.3f}s..."
                    )
                    threading.Event().wait(delay)
                else:
                    self._failed_retries += 1

            except DeadlockException as e:
                last_exception = e
                self._deadlocks_detected += 1

                if attempt < self.max_retries:
                    # Use longer delay for deadlock recovery
                    delay = self.calculate_backoff_delay(attempt) * 2
                    logger.warning(
                        f"Deadlock detected for {entity_type} {entity_id}. "
                        f"Retrying in {delay:.3f}s..."
                    )
                    threading.Event().wait(delay)
                else:
                    self._failed_retries += 1

        # All retries exhausted
        raise OptimisticLockException(
            entity_type=entity_type,
            entity_id=entity_id,
            retries=self.max_retries,
            message=f"Failed after {self.max_retries + 1} attempts: {last_exception}",
        )

    async def async_retry_on_version_conflict(
        self,
        operation: Callable[[], T],
        entity_type: str,
        entity_id: UUID | str,
    ) -> T:
        """
        Execute an async operation with automatic retry on version conflicts.

        Args:
            operation: Async function to execute
            entity_type: Type of entity being operated on
            entity_id: ID of entity being operated on

        Returns:
            Result of the operation

        Raises:
            OptimisticLockException: If operation fails after max retries
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Handle both sync and async operations
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()

                if attempt > 0:
                    self._successful_retries += 1
                    logger.info(
                        f"Operation succeeded after {attempt} retries for "
                        f"{entity_type} {entity_id}"
                    )

                return result

            except StaleDataException as e:
                last_exception = e
                self._version_conflicts += 1

                if attempt < self.max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Version conflict for {entity_type} {entity_id} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.3f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self._failed_retries += 1

            except DeadlockException as e:
                last_exception = e
                self._deadlocks_detected += 1

                if attempt < self.max_retries:
                    # Use longer delay for deadlock recovery
                    delay = self.calculate_backoff_delay(attempt) * 2
                    logger.warning(
                        f"Deadlock detected for {entity_type} {entity_id}. "
                        f"Retrying in {delay:.3f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self._failed_retries += 1

        # All retries exhausted
        raise OptimisticLockException(
            entity_type=entity_type,
            entity_id=entity_id,
            retries=self.max_retries,
            message=f"Failed after {self.max_retries + 1} attempts: {last_exception}",
        )

    def check_version(
        self,
        entity: Any,
        expected_version: int,
        entity_type: str | None = None,
    ) -> None:
        """
        Check if entity version matches expected version.

        Args:
            entity: Entity to check
            expected_version: Expected version number
            entity_type: Type of entity (for error messages)

        Raises:
            StaleDataException: If versions don't match
        """
        current_version = getattr(entity, "version", 1)

        if current_version != expected_version:
            entity_type = entity_type or entity.__class__.__name__
            entity_id = getattr(entity, "id", "unknown")

            raise StaleDataException(
                entity_type=entity_type,
                entity_id=entity_id,
                expected_version=expected_version,
                actual_version=current_version,
            )

    def increment_version(self, entity: Any) -> int:
        """
        Increment entity version and update timestamp.

        Args:
            entity: Entity to update

        Returns:
            New version number
        """
        current_version = getattr(entity, "version", 1)
        new_version = current_version + 1

        entity.version = new_version

        # Update timestamp if entity has one
        if hasattr(entity, "last_updated"):
            entity.last_updated = datetime.now(UTC)

        return new_version

    def get_metrics(self) -> dict[str, int]:
        """
        Get concurrency metrics.

        Returns:
            Dictionary of metric names to values
        """
        return {
            "version_conflicts": self._version_conflicts,
            "successful_retries": self._successful_retries,
            "failed_retries": self._failed_retries,
            "deadlocks_detected": self._deadlocks_detected,
            "lock_timeouts": self._lock_timeouts,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._version_conflicts = 0
        self._successful_retries = 0
        self._failed_retries = 0
        self._deadlocks_detected = 0
        self._lock_timeouts = 0

    def detect_deadlock(self, error: Exception) -> bool:
        """
        Check if an exception indicates a deadlock.

        Args:
            error: Exception to check

        Returns:
            True if deadlock detected
        """
        error_msg = str(error).lower()

        # Common deadlock indicators
        deadlock_indicators = [
            "deadlock",
            "lock wait timeout",
            "lock timeout exceeded",
            "could not obtain lock",
            "concurrent update",
        ]

        return any(indicator in error_msg for indicator in deadlock_indicators)

    def handle_concurrent_error(
        self,
        error: Exception,
        entity_type: str,
        entity_id: UUID | str,
        operation: str,
    ) -> None:
        """
        Handle a concurrency-related error.

        Args:
            error: The error that occurred
            entity_type: Type of entity
            entity_id: ID of entity
            operation: Operation being performed

        Raises:
            Appropriate concurrency exception
        """
        if isinstance(error, (StaleDataException, OptimisticLockException)):
            raise  # Re-raise as is

        if self.detect_deadlock(error):
            raise DeadlockException(
                message=f"Deadlock detected during {operation} on {entity_type} {entity_id}",
                entities=[(entity_type, entity_id)],
            )

        # Generic concurrency error
        raise ConcurrencyException(
            message=f"Concurrency error during {operation}: {error}",
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
        )
