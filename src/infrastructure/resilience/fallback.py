"""
Fallback Strategies and Graceful Degradation

Production-grade fallback mechanisms for handling service failures
with cache-first patterns, timeout handling, and service degradation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackMode(Enum):
    """Fallback execution modes."""

    FAIL_FAST = "fail_fast"  # Fail immediately on primary failure
    CACHE_FIRST = "cache_first"  # Try cache before primary
    CACHE_FALLBACK = "cache_fallback"  # Use cache only on primary failure
    TIMEOUT_FALLBACK = "timeout_fallback"  # Use fallback on timeout
    DEGRADED_SERVICE = "degraded_service"  # Provide limited functionality


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies."""

    mode: FallbackMode = FallbackMode.CACHE_FALLBACK
    primary_timeout: float = 30.0  # Timeout for primary service
    fallback_timeout: float = 10.0  # Timeout for fallback service
    cache_ttl: float = 300.0  # Cache time-to-live in seconds

    # Health checking
    health_check_interval: float = 60.0  # Health check frequency
    failure_threshold: int = 3  # Failures before marking unhealthy
    recovery_threshold: int = 2  # Successes before marking healthy

    # Metrics
    track_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.primary_timeout <= 0:
            raise ValueError("primary_timeout must be positive")
        if self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be positive")
        if self.cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")


class FallbackResult(Generic[T]):
    """Result of a fallback operation."""

    def __init__(
        self,
        value: T,
        source: str,
        execution_time: float,
        used_fallback: bool = False,
        cache_hit: bool = False,
        primary_error: Exception | None = None,
    ):
        self.value = value
        self.source = source
        self.execution_time = execution_time
        self.used_fallback = used_fallback
        self.cache_hit = cache_hit
        self.primary_error = primary_error

    def __repr__(self) -> str:
        return (
            f"FallbackResult(source='{self.source}', "
            f"time={self.execution_time:.3f}s, "
            f"fallback={self.used_fallback}, cache={self.cache_hit})"
        )


class FallbackStrategy(ABC, Generic[T]):
    """Abstract base class for fallback strategies."""

    def __init__(self, name: str, config: FallbackConfig | None = None) -> None:
        self.name = name
        self.config = config or FallbackConfig()

        # Health tracking
        self.primary_healthy = True
        self.fallback_healthy = True
        self.failure_count = 0
        self.success_count = 0

        # Metrics
        self.metrics: dict[str, int] = defaultdict(int)
        self.last_health_check = 0.0

        logger.info(f"Initialized fallback strategy '{name}' with config: {config}")

    @abstractmethod
    async def execute_primary(self, *args: Any, **kwargs: Any) -> T:
        """Execute primary service call."""
        pass

    @abstractmethod
    async def execute_fallback(self, *args: Any, **kwargs: Any) -> T:
        """Execute fallback service call."""
        pass

    async def execute_with_fallback(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """
        Execute with fallback logic based on configuration.

        Returns:
            FallbackResult containing the result and metadata
        """
        start_time = time.time()

        if self.config.track_metrics:
            self.metrics["total_calls"] += 1

        try:
            if self.config.mode == FallbackMode.CACHE_FIRST:
                return await self._execute_cache_first(*args, **kwargs)
            elif self.config.mode == FallbackMode.CACHE_FALLBACK:
                return await self._execute_cache_fallback(*args, **kwargs)
            elif self.config.mode == FallbackMode.TIMEOUT_FALLBACK:
                return await self._execute_timeout_fallback(*args, **kwargs)
            elif self.config.mode == FallbackMode.DEGRADED_SERVICE:
                return await self._execute_degraded_service(*args, **kwargs)
            else:  # FAIL_FAST
                return await self._execute_fail_fast(*args, **kwargs)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Fallback strategy '{self.name}' failed: {e}")
            if self.config.track_metrics:
                self.metrics["total_failures"] += 1
            raise

    async def _execute_fail_fast(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Execute primary only, fail immediately on error."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.execute_primary(*args, **kwargs), timeout=self.config.primary_timeout
            )
            execution_time = time.time() - start_time
            self._record_success("primary")

            return FallbackResult(value=result, source="primary", execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure("primary", e)
            raise

    async def _execute_cache_first(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Try cache first, then primary, then fallback."""
        start_time = time.time()

        # Try cache first
        cache_result = await self._try_cache(*args, **kwargs)
        if cache_result is not None:
            execution_time = time.time() - start_time
            if self.config.track_metrics:
                self.metrics["cache_hits"] += 1

            return FallbackResult(
                value=cache_result, source="cache", execution_time=execution_time, cache_hit=True
            )

        # Try primary
        try:
            result = await asyncio.wait_for(
                self.execute_primary(*args, **kwargs), timeout=self.config.primary_timeout
            )
            execution_time = time.time() - start_time
            self._record_success("primary")

            # Update cache
            await self._update_cache(result, *args, **kwargs)

            return FallbackResult(value=result, source="primary", execution_time=execution_time)

        except Exception as primary_error:
            self._record_failure("primary", primary_error)

            # Try fallback
            try:
                result = await asyncio.wait_for(
                    self.execute_fallback(*args, **kwargs), timeout=self.config.fallback_timeout
                )
                execution_time = time.time() - start_time
                self._record_success("fallback")

                return FallbackResult(
                    value=result,
                    source="fallback",
                    execution_time=execution_time,
                    used_fallback=True,
                    primary_error=primary_error,
                )

            except Exception as fallback_error:
                execution_time = time.time() - start_time
                self._record_failure("fallback", fallback_error)

                logger.error(
                    f"Both primary and fallback failed for '{self.name}'. "
                    f"Primary: {primary_error}, Fallback: {fallback_error}"
                )
                raise primary_error  # Raise original primary error

    async def _execute_cache_fallback(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Try primary first, use cache/fallback on failure."""
        start_time = time.time()

        # Try primary
        try:
            result = await asyncio.wait_for(
                self.execute_primary(*args, **kwargs), timeout=self.config.primary_timeout
            )
            execution_time = time.time() - start_time
            self._record_success("primary")

            # Update cache
            await self._update_cache(result, *args, **kwargs)

            return FallbackResult(value=result, source="primary", execution_time=execution_time)

        except Exception as primary_error:
            self._record_failure("primary", primary_error)

            # Try cache
            cache_result = await self._try_cache(*args, **kwargs)
            if cache_result is not None:
                execution_time = time.time() - start_time
                if self.config.track_metrics:
                    self.metrics["cache_hits"] += 1

                return FallbackResult(
                    value=cache_result,
                    source="cache",
                    execution_time=execution_time,
                    used_fallback=True,
                    cache_hit=True,
                    primary_error=primary_error,
                )

            # Try fallback
            try:
                result = await asyncio.wait_for(
                    self.execute_fallback(*args, **kwargs), timeout=self.config.fallback_timeout
                )
                execution_time = time.time() - start_time
                self._record_success("fallback")

                return FallbackResult(
                    value=result,
                    source="fallback",
                    execution_time=execution_time,
                    used_fallback=True,
                    primary_error=primary_error,
                )

            except Exception as fallback_error:
                execution_time = time.time() - start_time
                self._record_failure("fallback", fallback_error)

                logger.error(
                    f"Primary, cache, and fallback all failed for '{self.name}'. "
                    f"Primary: {primary_error}, Fallback: {fallback_error}"
                )
                raise primary_error  # Raise original primary error

    async def _execute_timeout_fallback(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Use fallback on primary timeout."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.execute_primary(*args, **kwargs), timeout=self.config.primary_timeout
            )
            execution_time = time.time() - start_time
            self._record_success("primary")

            return FallbackResult(value=result, source="primary", execution_time=execution_time)

        except TimeoutError as timeout_error:
            self._record_failure("primary", timeout_error)

            # Try fallback on timeout
            try:
                result = await asyncio.wait_for(
                    self.execute_fallback(*args, **kwargs), timeout=self.config.fallback_timeout
                )
                execution_time = time.time() - start_time
                self._record_success("fallback")

                return FallbackResult(
                    value=result,
                    source="fallback",
                    execution_time=execution_time,
                    used_fallback=True,
                    primary_error=timeout_error,
                )

            except Exception as fallback_error:
                execution_time = time.time() - start_time
                self._record_failure("fallback", fallback_error)
                raise timeout_error  # Raise original timeout error

        except Exception as e:
            # For non-timeout errors, don't use fallback
            execution_time = time.time() - start_time
            self._record_failure("primary", e)
            raise

    async def _execute_degraded_service(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Provide degraded service functionality."""
        start_time = time.time()

        # Check if primary service is healthy
        if not self.primary_healthy:
            logger.info(f"Primary service unhealthy for '{self.name}', using degraded mode")
            try:
                result = await asyncio.wait_for(
                    self.execute_fallback(*args, **kwargs), timeout=self.config.fallback_timeout
                )
                execution_time = time.time() - start_time

                return FallbackResult(
                    value=result,
                    source="degraded",
                    execution_time=execution_time,
                    used_fallback=True,
                )

            except Exception as e:
                self._record_failure("fallback", e)
                raise

        # Try primary service
        try:
            result = await asyncio.wait_for(
                self.execute_primary(*args, **kwargs), timeout=self.config.primary_timeout
            )
            execution_time = time.time() - start_time
            self._record_success("primary")

            return FallbackResult(value=result, source="primary", execution_time=execution_time)

        except Exception as primary_error:
            self._record_failure("primary", primary_error)

            # Switch to degraded mode
            logger.warning(f"Switching '{self.name}' to degraded mode due to: {primary_error}")
            try:
                result = await asyncio.wait_for(
                    self.execute_fallback(*args, **kwargs), timeout=self.config.fallback_timeout
                )
                execution_time = time.time() - start_time

                return FallbackResult(
                    value=result,
                    source="degraded",
                    execution_time=execution_time,
                    used_fallback=True,
                    primary_error=primary_error,
                )

            except Exception as fallback_error:
                self._record_failure("fallback", fallback_error)
                raise primary_error

    async def _try_cache(self, *args: Any, **kwargs: Any) -> T | None:
        """Try to get result from cache. Override in subclasses."""
        return None

    async def _update_cache(self, value: T, *args: Any, **kwargs: Any) -> None:
        """Update cache with result. Override in subclasses."""
        pass

    def _record_success(self, source: str) -> None:
        """Record successful call."""
        self.success_count += 1

        if source == "primary":
            if not self.primary_healthy and self.success_count >= self.config.recovery_threshold:
                self.primary_healthy = True
                self.failure_count = 0
                logger.info(f"Primary service for '{self.name}' marked as healthy")

        if self.config.track_metrics:
            self.metrics[f"{source}_successes"] += 1

    def _record_failure(self, source: str, error: Exception) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.success_count = 0

        if source == "primary":
            if self.primary_healthy and self.failure_count >= self.config.failure_threshold:
                self.primary_healthy = False
                logger.warning(f"Primary service for '{self.name}' marked as unhealthy")

        if self.config.track_metrics:
            self.metrics[f"{source}_failures"] += 1

        logger.debug(f"Recorded failure for '{self.name}' {source}: {error}")

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy metrics."""
        return {
            "name": self.name,
            "mode": self.config.mode.value,
            "primary_healthy": self.primary_healthy,
            "fallback_healthy": self.fallback_healthy,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "metrics": dict(self.metrics) if self.config.track_metrics else {},
        }


class CacheFirstStrategy(FallbackStrategy[T]):
    """Strategy that prioritizes cache, then primary, then fallback."""

    def __init__(
        self,
        name: str,
        primary_func: Callable[..., Awaitable[T]],
        fallback_func: Callable[..., Awaitable[T]],
        cache_get_func: Callable[..., Awaitable[T | None]] | None = None,
        cache_set_func: Callable[..., Awaitable[None]] | None = None,
        config: FallbackConfig | None = None,
    ):
        super().__init__(name, config or FallbackConfig(mode=FallbackMode.CACHE_FIRST))
        self.primary_func = primary_func
        self.fallback_func = fallback_func
        self.cache_get_func = cache_get_func
        self.cache_set_func = cache_set_func

    async def execute_primary(self, *args: Any, **kwargs: Any) -> T:
        """Execute primary function."""
        return await self.primary_func(*args, **kwargs)

    async def execute_fallback(self, *args: Any, **kwargs: Any) -> T:
        """Execute fallback function."""
        return await self.fallback_func(*args, **kwargs)

    async def _try_cache(self, *args: Any, **kwargs: Any) -> T | None:
        """Try to get result from cache."""
        if self.cache_get_func:
            try:
                return await self.cache_get_func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Cache get failed for '{self.name}': {e}")
        return None

    async def _update_cache(self, value: T, *args: Any, **kwargs: Any) -> None:
        """Update cache with result."""
        if self.cache_set_func:
            try:
                await self.cache_set_func(value, *args, **kwargs)
            except Exception as e:
                logger.debug(f"Cache set failed for '{self.name}': {e}")


class TimeoutStrategy(FallbackStrategy[T]):
    """Strategy that uses fallback specifically for timeout scenarios."""

    def __init__(
        self,
        name: str,
        primary_func: Callable[..., Awaitable[T]],
        fallback_func: Callable[..., Awaitable[T]],
        config: FallbackConfig | None = None,
    ):
        super().__init__(name, config or FallbackConfig(mode=FallbackMode.TIMEOUT_FALLBACK))
        self.primary_func = primary_func
        self.fallback_func = fallback_func

    async def execute_primary(self, *args: Any, **kwargs: Any) -> T:
        """Execute primary function."""
        return await self.primary_func(*args, **kwargs)

    async def execute_fallback(self, *args: Any, **kwargs: Any) -> T:
        """Execute fallback function."""
        return await self.fallback_func(*args, **kwargs)
